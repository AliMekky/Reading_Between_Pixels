#!/usr/bin/env python3
"""
Token-space Integrated Gradients for Llava-NeXT (and similar) + bbox visualization.

What this script does
1) Loads a HF dataset sample-by-sample (filtered by question_id list).
2) Builds an MCQ prompt (with shuffled options like your evaluator code).
3) Runs the model once to get a predicted answer letter (A/B/C/D).
4) Computes Integrated Gradients **over packed image tokens** (NOT pixels):
   - Option 1: teacher forcing on the predicted answer letter
   - Option 2: prefill next-token logprob at generation start
5) Uses your Llava-NeXT packed image token → bbox mapping logic to draw top-K important
   image tokens back onto the original image.

Outputs per sample:
- out_dir/{variant}/{qid}/
    - ig_{mode}.npz          (token_scores + meta)
    - overlay_{mode}.png     (top-K token boxes on image)

Notes / assumptions
- Designed for LlavaNextForConditionalGeneration + LlavaNextProcessor.
- Uses model.get_image_features(...) if available. If your model class doesn’t expose it,
  you’ll need to add a projector-hook fallback (I can add that too).
- The IG attribution target is a **single answer-token** (letter).
- Token-space IG here attributes w.r.t. the packed image embeddings inserted into the LLM,
  so it aligns well with your packed token mapping (base+mosaic+newline).

Run example:
  python ig_token_space_mcq.py \
    --model_id llava-hf/llava-v1.6-mistral-7b-hf \
    --hf_dataset AHAAM/GUIC \
    --ids_file no_overlap_question_ids.txt \
    --variant notext \
    --mode teacher_forced \
    --out_dir ./ig_token_outputs \
    --max_samples 50

"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import log_softmax
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from captum.attr import IntegratedGradients

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution


# ======================================================================================
# Your packed-token mapping logic (copied in so this is truly "full code").
# If you prefer importing, replace these defs with:
#   from your_prev_script import build_packed_image_token_map_for_one_image, ...
# ======================================================================================


def keep_top_bottom_k_within_percentiles(
    grid: np.ndarray,
    k: int,
    clip_percentiles=(5, 95),
    *,
    signed=True,
    fill_value=0.0,
):
    """
    Keep only:
      - lowest k values and highest k values
    computed AFTER restricting values to be within [p5, p95] of the grid distribution.

    Returns:
      masked_grid: same shape as grid, with non-selected entries set to fill_value.
      mask: boolean mask of selected entries (True = kept)
    """
    g = grid.astype(np.float32, copy=True)
    flat = g.reshape(-1)

    # Percentile clipping range
    lo = np.percentile(flat, clip_percentiles[0])
    hi = np.percentile(flat, clip_percentiles[1])

    # Only consider values within [lo, hi] for ranking
    in_band = (flat >= lo) & (flat <= hi) & np.isfinite(flat)
    idx_band = np.nonzero(in_band)[0]

    if k <= 0 or idx_band.size == 0:
        mask = np.zeros_like(flat, dtype=bool)
        return np.full_like(g, fill_value, dtype=np.float32), mask.reshape(g.shape)

    vals_band = flat[idx_band]

    # bottom-k and top-k indices *within the band*
    k_eff = min(k, idx_band.size)

    # bottom-k
    bot_rel = np.argpartition(vals_band, k_eff - 1)[:k_eff]
    # top-k
    top_rel = np.argpartition(vals_band, -(k_eff))[-k_eff:]

    keep_idx = np.unique(np.concatenate([idx_band[bot_rel], idx_band[top_rel]]))

    mask = np.zeros_like(flat, dtype=bool)
    mask[keep_idx] = True

    out = np.full_like(flat, fill_value, dtype=np.float32)
    out[mask] = flat[mask]
    return out.reshape(g.shape), mask.reshape(g.shape)

def overlay_grid_block_on_image(
    img, grid, out_path, title="",
    signed=False, cmap=None, alpha=0.55,
    *,
    show_top_bottom_k: int = 0,          # NEW: 0 disables masking
    clip_percentiles=(5, 95),            # NEW: same idea as robust_normalize
):
    """
    Block (nearest / 'pixelated') overlay: upsample with nearest or simple zoom so
    each token patch shows as a block. This function explicitly creates axes and
    uses the returned mappable for the colorbar so Matplotlib won't complain.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    import numpy as np

    H, W = img.height, img.width
    gh, gw = grid.shape

    # If requested: keep only bottom-k + top-k within [p5, p95]
    if show_top_bottom_k and show_top_bottom_k > 0:
        grid_kept, mask = keep_top_bottom_k_within_percentiles(
            grid,
            k=show_top_bottom_k,
            clip_percentiles=clip_percentiles,
            signed=signed,
            fill_value=0.0,   # value doesn't matter if we mask it
        )
    else:
        grid_kept = grid
        mask = np.ones_like(grid_kept, dtype=bool)

    # Upsample with nearest-like behavior
    scale_h = H / gh
    scale_w = W / gw
    grid_up = zoom(grid_kept, (scale_h, scale_w), order=0)

    # Upsample mask similarly (nearest) so transparency aligns with blocks
    mask_up = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5

    # Normalize kept values for viz
    grid_norm = robust_normalize(grid_up, signed=signed)

    # Mask everything except the selected top/bottom-k regions
    grid_vis = np.ma.array(grid_norm, mask=~mask_up)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(img)

    if signed:
        if cmap is None:
            cmap = "RdBu_r"
        mappable = ax.imshow(grid_vis, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)
    else:
        if cmap is None:
            cmap = "jet"
        mappable = ax.imshow(grid_vis, alpha=alpha, cmap=cmap, vmin=0, vmax=1)

    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def get_anyres_image_grid_shape(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> Tuple[int, int]:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return best_h // tile_size, best_w // tile_size


def image_size_to_num_tiles_plus_base(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> int:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    num_tiles = 0
    for _ in range(0, best_h, tile_size):
        for _ in range(0, best_w, tile_size):
            num_tiles += 1
    num_tiles += 1  # base tile
    return num_tiles


def unpad_params(current_h: int, current_w: int, original_h: int, original_w: int) -> Dict[str, Any]:
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h
    if orig_ar > curr_ar:
        # slice height
        scale = current_w / original_w
        new_h = int(round(original_h * scale, 7))
        pad = (current_h - new_h) // 2
        return {"mode": "slice_h", "scale": scale, "pad": pad, "new_h": new_h, "new_w": current_w}
    else:
        # slice width
        scale = current_h / original_h
        new_w = int(round(original_w * scale, 7))
        pad = (current_w - new_w) // 2
        return {"mode": "slice_w", "scale": scale, "pad": pad, "new_h": current_h, "new_w": new_w}


def bbox_from_unpadded_grid_to_original(
    r0: float, c0: float, r1: float, c1: float,
    current_h: int, current_w: int,
    original_h: int, original_w: int,
) -> Tuple[float, float, float, float]:
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h
    if orig_ar > curr_ar:
        scale = current_w / original_w
    else:
        scale = current_h / original_h

    y0, y1 = r0 / scale, r1 / scale
    x0, x1 = c0 / scale, c1 / scale

    y0 = max(0.0, min(float(original_h), y0))
    y1 = max(0.0, min(float(original_h), y1))
    x0 = max(0.0, min(float(original_w), x0))
    x1 = max(0.0, min(float(original_w), x1))
    return (y0, x0, y1, x1)


def build_packed_image_token_map_for_one_image(
    *,
    model: LlavaNextForConditionalGeneration,
    original_size_hw: Tuple[int, int],      # (H, W)
    num_views_provided: int,                # pixel_values.shape[1]
    include_newline_tokens: bool = True,
) -> Dict[str, Any]:
    cfg = model.config
    vision_cfg = cfg.vision_config

    tile_size = int(vision_cfg.image_size)         # 336
    patch_size = int(vision_cfg.patch_size)        # 14
    patches_per_side = tile_size // patch_size     # 24
    per_tile_patch_tokens = patches_per_side * patches_per_side  # 576

    cls_removed = (cfg.vision_feature_select_strategy == "default")
    per_tile_tokens_after_select = per_tile_patch_tokens if cls_removed else (per_tile_patch_tokens + 1)

    expected_tiles_plus_base = image_size_to_num_tiles_plus_base(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    used_tiles_plus_base = min(num_views_provided, expected_tiles_plus_base)
    non_base_tiles = max(0, used_tiles_plus_base - 1)

    num_tile_h, num_tile_w = get_anyres_image_grid_shape(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    expected_non_base_tiles_for_layout = num_tile_h * num_tile_w

    mosaic_h = num_tile_h * patches_per_side
    mosaic_w = num_tile_w * patches_per_side

    orig_h, orig_w = original_size_hw
    params = unpad_params(mosaic_h, mosaic_w, orig_h, orig_w)
    if params["mode"] == "slice_h":
        unpadded_h, unpadded_w = params["new_h"], mosaic_w
    else:
        unpadded_h, unpadded_w = mosaic_h, params["new_w"]

    row_width_with_newline = unpadded_w + (1 if include_newline_tokens else 0)

    base_tokens = per_tile_tokens_after_select
    mosaic_tokens = unpadded_h * row_width_with_newline
    total_packed_image_tokens = base_tokens + mosaic_tokens

    tokens: List[Dict[str, Any]] = []

    # Base tokens
    for t in range(base_tokens):
        if not cls_removed and t == 0:
            tokens.append({"token_idx": t, "kind": "base_cls", "row": None, "col": None, "bbox": None})
            continue
        patch_t = t if cls_removed else (t - 1)
        r = patch_t // patches_per_side
        c = patch_t % patches_per_side
        bbox = bbox_from_unpadded_grid_to_original(
            r, c, r + 1, c + 1,
            patches_per_side, patches_per_side,
            orig_h, orig_w
        )
        tokens.append({"token_idx": t, "kind": "base_patch", "row": int(r), "col": int(c), "bbox": bbox})

    # Mosaic tokens + newline
    base_offset = base_tokens
    for k in range(mosaic_tokens):
        token_idx = base_offset + k
        row = k // row_width_with_newline
        col = k % row_width_with_newline
        if include_newline_tokens and col == unpadded_w:
            tokens.append({"token_idx": token_idx, "kind": "newline", "row": int(row), "col": None, "bbox": None})
            continue
        bbox = bbox_from_unpadded_grid_to_original(
            row, col, row + 1, col + 1,
            unpadded_h, unpadded_w,
            orig_h, orig_w
        )
        tokens.append({"token_idx": token_idx, "kind": "mosaic_patch", "row": int(row), "col": int(col), "bbox": bbox})

    return {
        "summary": {
            "original_size_hw": original_size_hw,
            "tile_size": tile_size,
            "patch_size": patch_size,
            "patches_per_side": patches_per_side,
            "cls_removed": cls_removed,
            "num_views_provided": num_views_provided,
            "expected_tiles_plus_base_from_image_sizes": expected_tiles_plus_base,
            "used_tiles_plus_base_after_slicing": used_tiles_plus_base,
            "non_base_tiles_used": non_base_tiles,
            "tile_layout_hw": (num_tile_h, num_tile_w),
            "expected_non_base_tiles_for_layout": expected_non_base_tiles_for_layout,
            "mosaic_pre_unpad_hw_in_patches": (mosaic_h, mosaic_w),
            "mosaic_unpadded_hw_in_patches": (unpadded_h, unpadded_w),
            "newline_tokens_enabled": include_newline_tokens,
            "row_width_with_newline": row_width_with_newline,
            "base_tokens": base_tokens,
            "mosaic_tokens": mosaic_tokens,
            "total_packed_image_tokens": total_packed_image_tokens,
        },
        "tokens": tokens,
    }


# ======================================================================================
# Dataset + prompt building (aligned with your evaluator logic)
# ======================================================================================

def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    ds = load_dataset(dataset_id, split=split)
    try:
        ds.save_to_disk(str(cache_dir))
    except Exception:
        pass
    return ds


def format_mcq_prompt(question: str, options: Dict[str, str], instruction: Optional[str] = None) -> str:
    if instruction is None:
        instruction = "Answer the following multiple-choice question by selecting the correct option."
    prompt = f"{instruction}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for k, v in options.items():
        prompt += f"{k}) {v}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def build_item_from_sample(
    sample: Dict[str, Any],
    *,
    variant: str,
    shuffle_options: bool,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Creates a single (image, prompt, correct_letter, option_meta) item from one dataset sample.
    """
    qid = sample.get("question_id", None)
    if qid is None:
        return None

    question = sample.get("question", "")
    labels = ["A", "B", "C", "D"]

    candidates = [
        {"key": "correct_answer", "text": sample["correct_answer"]["text"]},
        {"key": "misleading_groundable", "text": sample["misleading_groundable"]["text"]},
        {"key": "misleading_ungroundable", "text": sample["misleading_ungroundable"]["text"]},
        {"key": "irrelevant_word", "text": sample["irrelevant_word"]["text"]},
    ]

    if shuffle_options:
        import random
        rng = random.Random(f"{seed}_{qid}")
        rng.shuffle(candidates)

    options = {labels[i]: cand["text"] for i, cand in enumerate(candidates)}
    correct_index = next(i for i, cand in enumerate(candidates) if cand["key"] == "correct_answer")
    correct_letter = labels[correct_index]

    if variant == "notext":
        img_obj = sample["notext"]["image"]
    else:
        if variant not in sample:
            return None
        img_obj = sample[variant]["image"]

    option_meta = {
        "order": [cand["key"] for cand in candidates],
        "label_to_key": {labels[i]: cand["key"] for i, cand in enumerate(candidates)},
    }

    return {
        "question_id": str(qid),
        "image": img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj,
        "question": question,
        "options": options,
        "correct_letter": correct_letter,
        "option_meta": option_meta,
    }


def load_question_id_filter(path: str) -> set:
    ids = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids

def _as_packed_img_tensor(x):
    # Turn get_image_features output into a single tensor (1, N, D)
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
        else:
            x = torch.cat(x, dim=-2 if x[0].ndim == 2 else 1)

    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor from get_image_features, got {type(x)}")

    # Accept (N, D) -> add batch dim
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, N, D)

    if x.ndim != 3:
        raise ValueError(f"Expected (1, N, D), got {tuple(x.shape)}")

    return x


# ======================================================================================
# Token-space IG (supports both modes)
# ======================================================================================

def _image_placeholder_positions(model, input_ids: torch.Tensor) -> torch.Tensor:
    image_token_id = model.config.image_token_id
    return (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]  # (N_img,)


def _token_id_for_answer_letter(processor, letter: str) -> Tuple[int, str]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise ValueError("Processor has no tokenizer; cannot map answer letter to token id.")
    for s in [f" {letter}", letter]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s
    raise ValueError(f"Answer letter {letter!r} does not map to a single token (tried ' {letter}' and '{letter}').")


def _extract_answer_letter(text: str) -> str:
    import re
    t = text.strip().upper()
    patterns = [
        r'ANSWER[:\s]+([ABCD])\b',
        r'^\s*([ABCD])\s*$',
        r'^([ABCD])\b',
        r'\b([ABCD])\s*$',
        r'\b([ABCD])\b',
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return "UNKNOWN"


@torch.no_grad()
def predict_answer_letter_llava_next(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    image: Image.Image,
    mcq_prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 5,
) -> str:
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": mcq_prompt_text}, {"type": "image"}]},
    ]
    formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted, return_tensors="pt").to(device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=False,
    )
    decoded = processor.decode(gen[0], skip_special_tokens=True)
    return _extract_answer_letter(decoded)


def token_space_ig(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    image: Image.Image,
    mcq_prompt_text: str,
    answer_letter: str,
    *,
    mode: str,
    steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Returns:
      token_scores: (N_img_tokens,) float32
      meta:         dict
      inputs:       dict of model inputs (for mapping: pixel_values, image_sizes, input_ids)
    """
    model.eval()

    if mode == "teacher_forced":
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": mcq_prompt_text}, {"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_letter}]},
        ]
        formatted = processor.apply_chat_template(conversation, add_generation_prompt=False)
        inputs = processor(images=image, text=formatted, return_tensors="pt")
        logit_pos = -2  # predicts last token
        # robust target: last token actually present
        target_token_id = int(inputs["input_ids"][0, -1].item())
        target_str = "<teacher_forced_last_token>"
    elif mode == "prefill_next_token":
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": mcq_prompt_text}, {"type": "image"}]},
        ]
        formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=formatted, return_tensors="pt")
        logit_pos = -1  # predicts next token at generation start
        target_token_id, target_str = _token_id_for_answer_letter(processor, answer_letter)
    else:
        raise ValueError("mode must be 'teacher_forced' or 'prefill_next_token'")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)

    # 1) Get packed image token embeddings (robust handling)
    if not hasattr(model, "get_image_features"):
        raise AttributeError(
            "Model has no get_image_features(). Add a projector-hook fallback for your model class."
        )

    with torch.no_grad():
        raw_img_feats = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=inputs.get("image_sizes", None),
        )

    # Normalize into a single tensor img_embeds of shape (B, N_img, D)
    img_embeds = None

    # Case 1: direct tensor
    if isinstance(raw_img_feats, torch.Tensor):
        img_embeds = raw_img_feats

    # Case 2: list/tuple
    elif isinstance(raw_img_feats, (list, tuple)):
        tensor_items = [x for x in raw_img_feats if isinstance(x, torch.Tensor)]
        if len(tensor_items) == 1:
            img_embeds = tensor_items[0]
        elif len(tensor_items) > 1:
            # concatenate along token dimension
            img_embeds = torch.cat(tensor_items, dim=1)
        else:
            raise RuntimeError("get_image_features returned list/tuple without tensors.")

    # Case 3: dict
    elif isinstance(raw_img_feats, dict):
        for k in ("image_embeds", "image_features", "embeds", "features"):
            if k in raw_img_feats and isinstance(raw_img_feats[k], torch.Tensor):
                img_embeds = raw_img_feats[k]
                break

    if img_embeds is None:
        raise RuntimeError(
            f"Could not parse output of get_image_features(). Got type={type(raw_img_feats)}"
        )

    img_embeds = img_embeds.to(device)

    # Ensure shape (B, N_img, D)
    if img_embeds.dim() == 2:
        img_embeds = img_embeds.unsqueeze(0)
    elif img_embeds.dim() != 3:
        raise RuntimeError(f"Unexpected img_embeds shape: {tuple(img_embeds.shape)}")

    _, N_img, D = img_embeds.shape

    # 2) Find where placeholders are, and build full embeds
    img_positions = _image_placeholder_positions(model, input_ids)
    if int(img_positions.numel()) != int(N_img):
        raise RuntimeError(f"placeholder count {img_positions.numel()} != packed image tokens {N_img}")

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        full_embeds = embed_layer(input_ids)  # (1, S, D)

    blank = Image.new("RGB", image.size, (127,127,127))
    blank_inputs = processor(images=blank, text=formatted, return_tensors="pt")
    blank_inputs = {k:v.to(device) for k,v in blank_inputs.items()}

    with torch.no_grad():
        baseline = model.get_image_features(
            pixel_values=blank_inputs["pixel_values"],
            image_sizes=blank_inputs.get("image_sizes", None),
        )
    baseline = _as_packed_img_tensor(baseline)   # ✅ make sure it's a Tensor

    if baseline.shape[1] != img_embeds.shape[1]:
        raise RuntimeError(
            f"Baseline N_img={baseline.shape[1]} != img_embeds N_img={img_embeds.shape[1]} "
            "(blank image produced different tiling / packing)"
        )

    def forward_func(img_embeds_var: torch.Tensor) -> torch.Tensor:
        inputs_embeds = full_embeds.clone()
        inputs_embeds[:, img_positions, :] = img_embeds_var

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            # pixel_values=inputs["pixel_values"],
            # image_sizes=inputs.get("image_sizes", None),
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits  # (1, S, V)
        logp = log_softmax(logits[:, logit_pos, :], dim=-1)
        return logp[:, target_token_id]  # (1,)

    ig = IntegratedGradients(forward_func)

    attr = ig.attribute(
        img_embeds,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
    )  # (1, N_img, D)

        # ---------------- SANITY CHECKS ----------------
    with torch.no_grad():
        fx = forward_func(img_embeds).item()
        f0 = forward_func(baseline).item()
        delta = fx - f0

    # signed total attribution (over all image tokens + dims)
    total_attr = attr.sum().item()

    rel_err = abs(total_attr - delta) / (abs(delta) + 1e-8)



    if not np.isfinite(rel_err) or rel_err > 0.15:
        print(
            f"[SANITY] IG completeness warning | "
            f"delta={delta:.6f}, total_attr={total_attr:.6f}, rel_err={rel_err:.3f} | "
        )

    token_scores = attr[0].sum(dim=-1).detach().cpu().numpy().astype(np.float32)
    meta = {
        "mode": mode,
        "answer_letter": answer_letter,
        "target_token_id": int(target_token_id),
        "target_str": target_str,
        "seq_len": int(input_ids.shape[1]),
        "logit_pos": int(logit_pos if logit_pos >= 0 else (input_ids.shape[1] + logit_pos)),
        "n_img_tokens": int(N_img),
        "embed_dim": int(D),
        "steps": int(steps),
    }

    # meta.update({
    #     "sanity_fx": fx,
    #     "sanity_f0": f0,
    #     "sanity_delta": delta,
    #     "sanity_total_attr": total_attr,
    #     "sanity_rel_err": rel_err,
    # })

    print(f"sanity_fx = {fx:.6f}, sanity_f0 = {f0:.6f}, sanity_delta = {delta:.6f}, sanity_total_attr = {total_attr:.6f}, sanity_rel_err = {rel_err:.3f}")

    return token_scores, meta, inputs


# ======================================================================================
# Visualization: draw top-K token bboxes onto original image
# ======================================================================================

def draw_topk_token_boxes(
    img: Image.Image,
    mapping_tokens: List[Dict[str, Any]],
    token_scores: np.ndarray,
    out_path: str,
    *,
    top_k: int = 50,
    kinds: Tuple[str, ...] = ("mosaic_patch",),
    score_power: float = 0.5,
):
    """
    Draw top-K bboxes by token_scores for selected kinds.

    - token_scores is indexed by packed image token_idx (0..N_img-1).
    - mapping_tokens is the output tokens list from build_packed_image_token_map_for_one_image.
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # collect candidates with bbox
    candidates = []
    for t in mapping_tokens:
        idx = t["token_idx"]
        if idx < 0 or idx >= len(token_scores):
            continue
        if t.get("kind") not in kinds:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue
        candidates.append((idx, float(token_scores[idx]), bb, t.get("kind")))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:top_k]

    if not candidates:
        im.save(out_path)
        return

    # Normalize for optional label/alpha decisions (simple)
    scores = np.array([c[1] for c in candidates], dtype=np.float32)
    smax = float(scores.max()) if scores.size else 1.0
    smin = float(scores.min()) if scores.size else 0.0
    denom = (smax - smin + 1e-8)

    for idx, s, bb, kind in candidates:
        y0, x0, y1, x1 = bb
        # simple thickness scaling
        norm = (s - smin) / denom
        width = int(1 + 4 * (norm ** score_power))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=width)
        if font is not None:
            draw.text((x0 + 1, y0 + 1), f"{idx}:{s:.3g}", fill=(255, 0, 0), font=font)

    im.save(out_path)


# ======================================================================================
## Visualization: overlay upsampled token scores onto image (alternative to boxes)
# ======================================================================================



def robust_normalize(x, clip_percentiles=(5, 95), signed=False, eps=1e-8):
    """
    Normalize for visualization.
    - If signed=True: symmetric scaling around 0 using percentile(|x|).
    - Else: scale to [0,1] after clipping.
    """
    x = x.astype(np.float32)

    if signed:
        a = np.abs(x)
        hi = np.percentile(a, clip_percentiles[1])
        hi = max(hi, eps)
        x = np.clip(x, -hi, hi) / hi  # now in [-1,1]
        return x

    lo = np.percentile(x, clip_percentiles[0])
    hi = np.percentile(x, clip_percentiles[1])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def token_scores_to_grids(mapping_tokens, token_scores, summary):
    """
    Convert per-image-token scores (length = total_packed_image_tokens) into:
      - base_grid:  24x24
      - mosaic_grid: (unpadded_h x unpadded_w)
    token_scores: np.ndarray shape (N_tokens,)
    """
    patches_per_side = summary["patches_per_side"]  # usually 24
    mosaic_h, mosaic_w = summary["mosaic_unpadded_hw_in_patches"]
    base_grid = np.zeros((patches_per_side, patches_per_side), dtype=np.float32)
    mosaic_grid = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

    # Fill grids
    for t in mapping_tokens:
        idx = t["token_idx"]
        s = float(token_scores[idx])

        if t["kind"] == "base_patch":
            r, c = t["row"], t["col"]
            base_grid[r, c] += s

        elif t["kind"] == "mosaic_patch":
            r, c = t["row"], t["col"]
            mosaic_grid[r, c] += s

        # newline/base_cls ignored

    return base_grid, mosaic_grid


def overlay_grid_on_image(img, grid, out_path, title, signed=False, cmap=None, alpha=0.55):
    """
    Upsample a token grid to image size and overlay on the image.
    """
    H, W = img.height, img.width
    gh, gw = grid.shape

    # Upsample grid to image resolution
    scale_h = H / gh
    scale_w = W / gw
    grid_up = zoom(grid, (scale_h, scale_w), order=1)  # smooth-ish

    # Normalize for viz
    grid_norm = robust_normalize(grid_up, signed=signed)

    plt.figure(figsize=(10, 4))
    plt.imshow(img)

    if signed:
        # diverging: values in [-1,1]
        if cmap is None:
            cmap = "RdBu_r"
        plt.imshow(grid_norm, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)
    else:
        # magnitude: values in [0,1]
        if cmap is None:
            cmap = "jet"
        plt.imshow(grid_norm, alpha=alpha, cmap=cmap, vmin=0, vmax=1)

    plt.axis("off")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def get_token_indices_in_bbox(mapping_tokens: List[Dict[str, Any]], bbox: Tuple[float,float,float,float]):
    """
    bbox = (y0, x0, y1, x1) in image pixel coords (same coordinate space as mapping token bboxes).
    Returns token indices belonging to patches whose center is inside bbox.
    """
    y0, x0, y1, x1 = bbox
    inds = []
    for t in mapping_tokens:
        bb = t.get("bbox")
        if bb is None:
            continue
        ty0, tx0, ty1, tx1 = bb
        cy = 0.5 * (ty0 + ty1)
        cx = 0.5 * (tx0 + tx1)
        if (cx >= x0 and cx <= x1 and cy >= y0 and cy <= y1):
            inds.append(int(t["token_idx"]))
    return inds


def get_token_indices_in_mask(mapping_tokens: List[Dict[str,Any]], mask: np.ndarray):
    """
    mask: boolean 2D array of shape (H, W) True indicates ROI.
    We test the patch center pixel location mapped to mask coords (mask is image pixel coords).
    """
    H, W = mask.shape
    inds = []
    for t in mapping_tokens:
        bb = t.get("bbox")
        if bb is None:
            continue
        ty0, tx0, ty1, tx1 = bb
        cy = int(round(0.5 * (ty0 + ty1)))
        cx = int(round(0.5 * (tx0 + tx1)))
        # clamp
        cy = max(0, min(H - 1, cy))
        cx = max(0, min(W - 1, cx))
        if mask[cy, cx]:
            inds.append(int(t["token_idx"]))
    return inds


def compute_region_scores(token_scores: np.ndarray, mapping_tokens: List[Dict[str,Any]], region_token_inds: List[int]):
    """
    returns (region_sum, region_mean, region_share)
    """
    total = float(token_scores.sum())
    if total == 0:
        total = 1e-12
    region_sum = float(token_scores[np.array(region_token_inds)].sum()) if region_token_inds else 0.0
    region_mean = float(token_scores[np.array(region_token_inds)].mean()) if region_token_inds else 0.0
    share = region_sum / total
    return {"sum": region_sum, "mean": region_mean, "share": share, "n_tokens": len(region_token_inds)}


def permutation_test_region(token_scores: np.ndarray, region_token_inds: List[int], n_perm: int = 500):
    """
    Permute token_scores many times and compute region_sum under random assignment
    (keeps token positions but shuffles values).
    Returns (real_sum, mean_perm, std_perm, z, p_value)
    """
    if len(region_token_inds) == 0:
        return None
    real = float(token_scores[np.array(region_token_inds)].sum())
    perm_sums = []
    N = token_scores.shape[0]
    arr = token_scores.copy()
    for _ in range(n_perm):
        np.random.shuffle(arr)
        perm_sums.append(float(arr[np.array(region_token_inds)].sum()))
    perm_sums = np.array(perm_sums, dtype=np.float32)
    mean_p = float(perm_sums.mean())
    std_p = float(perm_sums.std(ddof=1)) if perm_sums.size > 1 else 1e-12
    z = (real - mean_p) / (std_p + 1e-12)
    # two-sided p-value
    pval = float((np.abs(perm_sums - mean_p) >= np.abs(real - mean_p)).sum() / max(1, perm_sums.size))
    return {"real": real, "perm_mean": mean_p, "perm_std": std_p, "z": z, "pval": pval}

# ======================================================================================
# Main driver
# ======================================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    p.add_argument("--hf_cache_dir", type=str, default="./hf_dataset_GUIC")
    p.add_argument("--split", type=str, default="test")

    p.add_argument("--ids_file", type=str, default="../inference/no_overlap_question_ids.txt", help="File with question_ids to keep (one per line).")
    p.add_argument("--variant", type=str, default="notext",
                   choices=["notext", "correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"])
    p.add_argument("--shuffle_options", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mode", type=str, default="prefill_next_token",
                   choices=["teacher_forced", "prefill_next_token"])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--kinds", type=str, default="mosaic_patch,base_patch",
                   help="Comma-separated kinds to draw: mosaic_patch,base_patch")
    p.add_argument("--out_dir", type=str, default="./ig_token_outputs")
    p.add_argument("--max_samples", type=int, default=0, help="0 = no limit")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--viz_signed", action="store_true", help="Use signed values (recommended).")
    p.add_argument("--viz_clamp", type=float, default=0.0,
                help="If >0, clip signed grid to [-viz_clamp, +viz_clamp] before normalize.")
    p.add_argument("--viz_diverging", type=str, default="RdBu_r",
                help="Matplotlib diverging cmap for signed overlays.")
    p.add_argument("--save_grids", action="store_true",
                help="Also save base/mosaic grids in the .npz for later analysis.")
    p.add_argument("--debug_permutation", action="store_true")
    p.add_argument("--block_overlay", action="store_true", help="Use block (nearest) overlay instead of smooth.")
    p.add_argument("--predicted", action="store_true", help="if you want to compute IG for the predicted answer token instead of the correct answer token (prefill_next_token mode only)")
    p.add_argument("--region_bbox", type=str, default="", help='ROI bbox as "y0,x0,y1,x1" in PIXELS for region test.')
    p.add_argument("--start", type=int, default=0, help='Start index for processing samples.')
    p.add_argument("--end", type=int, default=500, help='End index (exclusive) for processing samples. 0 = no limit.')

    args = p.parse_args()

    def _parse_bbox(s: str):
        if not s:
            return None
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError('--region_bbox must be "y0,x0,y1,x1"')
        y0, x0, y1, x1 = parts
        return (y0, x0, y1, x1)

    roi_bbox = _parse_bbox(args.region_bbox)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load model + processor
    print(f"Loading model {args.model_id} on {device} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)


    model.eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    # Load dataset
    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)

    # Load filter IDs
    keep_ids = load_question_id_filter(args.ids_file)
    print(f"Loaded {len(keep_ids)} question_ids from {args.ids_file}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    kinds = tuple([k.strip() for k in args.kinds.split(",") if k.strip()])

    processed = 0

    # Read all subdirectories in output root
    existing_subdirs = []
    if out_root.exists():
        existing_subdirs = [d.name for d in out_root.iterdir() if d.is_dir()]
    print(f"Found {len(existing_subdirs)} existing subdirectories in {out_root}: {existing_subdirs}")

    # Iterate dataset and pick matching qids
    for i in tqdm(range(len(ds)), desc="Scanning dataset"):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break
        sample = ds[i]
        qid = sample.get("question_id", None)
        if qid is None:
            continue
        qid = str(qid)
        if qid not in keep_ids:
            continue
        if qid in existing_subdirs:
            print(f"[{qid}] Output already exists, skipping.")
            continue

        item = build_item_from_sample(
            sample,
            variant=args.variant,
            shuffle_options=args.shuffle_options,
            seed=args.seed,
        )
        if item is None:
            continue

        image: Image.Image = item["image"]
        mcq_prompt_text = format_mcq_prompt(item["question"], item["options"])

        # 1) Predict model answer letter (A/B/C/D)
        pred_letter = predict_answer_letter_llava_next(
            model=model,
            processor=processor,
            image=image,
            mcq_prompt_text=mcq_prompt_text,
            device=device,
            max_new_tokens=5,
        )
        if pred_letter not in ("A", "B", "C", "D"):
            # still save something but skip IG (target token unclear)
            print(f"[{qid}] predicted answer not a single letter: {pred_letter!r} -> skipping IG")
            continue

        target_token = pred_letter if args.predicted else item["correct_letter"]
        # 2) Compute token-space IG for the chosen mode
        token_scores, meta, inputs = token_space_ig(
            model=model,
            processor=processor,
            image=image,
            mcq_prompt_text=mcq_prompt_text,
            answer_letter=target_token,
            mode=args.mode,
            steps=args.steps,
            device=device,
        )

        # 3) Build packed token → bbox mapping
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is None:
            raise ValueError("Processor did not return image_sizes; mapping requires it.")

        # pixel_values: (B,V,3,336,336)
        V = int(pixel_values.shape[1]) if pixel_values.ndim == 5 else 1
        orig_h = int(image_sizes[0, 0].item())
        orig_w = int(image_sizes[0, 1].item())

        mapping = build_packed_image_token_map_for_one_image(
            model=model,
            original_size_hw=(orig_h, orig_w),
            num_views_provided=V,
            include_newline_tokens=True,
        )

        ## Sanity check: token_scores length should match mapping summary
        # assert token_scores.shape[0] == mapping["summary"]["total_packed_image_tokens"]
        if token_scores.shape[0] != mapping["summary"]["total_packed_image_tokens"]:
            print(f"[{qid}] WARNING: token_scores N={token_scores.shape[0]} != mapping expected={mapping['summary']['total_packed_image_tokens']}")
            # Save mismatched qid to file
            mismatch_log = Path(args.out_dir) / "mismatched_qids.txt"

            # Ensure directory exists
            mismatch_log.parent.mkdir(parents=True, exist_ok=True)

            with mismatch_log.open("a") as f:
                f.write(f"{qid}\n")

        mapping_tokens = mapping["tokens"]
        expected = int(mapping["summary"]["total_packed_image_tokens"])

        if token_scores.shape[0] != expected:
            print(f"[{qid}] WARNING: token_scores N={token_scores.shape[0]} != mapping expected={expected}")

        # 4) Save + visualize
        out_dir = out_root / args.variant / qid
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / f"ig_{args.mode}.npz"
        np.savez_compressed(
            str(npz_path),
            token_scores=token_scores,
            meta=json.dumps(meta),
            mapping_summary=json.dumps(mapping["summary"]),
        )

        # overlay_path = out_dir / f"overlay_{args.mode}.png"
        # ---- 4) Save + visualize (SIGNED dense overlays: base + mosaic) ----
        out_dir = out_root / args.variant / qid
        out_dir.mkdir(parents=True, exist_ok=True)

        # =======================
        # Region importance test (optional)
        # =======================
        if roi_bbox is not None:
            roi_inds = get_token_indices_in_bbox(mapping_tokens=mapping_tokens, bbox=roi_bbox)

            region_stats = compute_region_scores(
                token_scores=token_scores,
                mapping_tokens=mapping_tokens,
                region_token_inds=roi_inds,
            )

            perm_res = permutation_test_region(
                token_scores=token_scores,
                region_token_inds=roi_inds,
                n_perm=500,
            )

            print(f"[{qid}] ROI bbox={roi_bbox} -> n={region_stats['n_tokens']} "
                f"share={region_stats['share']:.4f} sum={region_stats['sum']:.4g} "
                f"z={perm_res['z']:.3f} p={perm_res['pval']:.3f}")

            with open(out_dir / "region_stats.json", "w") as f:
                json.dump(
                    {"roi_bbox": roi_bbox, "roi_inds": roi_inds, "region_stats": region_stats, "perm": perm_res},
                    f,
                    indent=2,
                )

        # Convert token scores -> base_grid + mosaic_grid (SIGNED)
        base_grid, mosaic_grid = token_scores_to_grids(
            mapping_tokens=mapping_tokens,
            token_scores=token_scores,
            summary=mapping["summary"],
        )

        # =======================
        # DEBUG: permutation test
        # =======================
        if args.debug_permutation:  # change to args.debug_permutation if you want CLI flag
            perm = np.random.permutation(token_scores.shape[0])
            token_scores_perm = token_scores[perm]

            base_grid_perm, mosaic_grid_perm = token_scores_to_grids(
                mapping_tokens=mapping_tokens,
                token_scores=token_scores_perm,
                summary=mapping["summary"],
            )

            base_grid_perm_norm = robust_normalize(base_grid_perm, signed=True)
            mosaic_grid_perm_norm = robust_normalize(mosaic_grid_perm, signed=True)


            if args.debug_permutation:
                overlay_grid_block_on_image(
                    img=image,
                    grid=base_grid_perm,
                    out_path=str(out_dir / f"overlay_base_PERM_{args.mode}.png"),
                    title=f"{qid} | PERM | BASE",
                    signed=True,
                    show_top_bottom_k=args.top_k,        # use your existing --top_k
                    clip_percentiles=(5, 95),
                )

                overlay_grid_block_on_image(
                    img=image,
                    grid=mosaic_grid_perm,
                    out_path=str(out_dir / f"overlay_mosaic_PERM_{args.mode}.png"),
                    title=f"{qid} | PERM | MOSAIC",
                    signed=True,
                    show_top_bottom_k=args.top_k,
                    clip_percentiles=(5, 95),
                )

        # Optional clipping (you said you don't want it; keep args.viz_clamp=0.0)
        if args.viz_clamp and args.viz_clamp > 0:
            base_grid = np.clip(base_grid, -args.viz_clamp, args.viz_clamp)
            mosaic_grid = np.clip(mosaic_grid, -args.viz_clamp, args.viz_clamp)

        # Normalize for viz (signed)
        base_grid_norm = robust_normalize(base_grid, signed=True)
        mosaic_grid_norm = robust_normalize(mosaic_grid, signed=True)

        # Save NPZ
        npz_path = out_dir / f"ig_{args.mode}.npz"
        save_dict = {
            "token_scores": token_scores,
            "meta": json.dumps(meta),
            "mapping_summary": json.dumps(mapping["summary"]),
            "mapping_tokens": json.dumps(mapping_tokens),   # <-- ADD THIS
        }
        if args.save_grids:
            save_dict["base_grid_signed"] = base_grid
            save_dict["mosaic_grid_signed"] = mosaic_grid
        np.savez_compressed(str(npz_path), **save_dict)

        # Two overlays
        base_overlay_path = out_dir / f"overlay_base_{args.mode}.png"
        mosaic_overlay_path = out_dir / f"overlay_mosaic_{args.mode}.png"

        if args.block_overlay:
            overlay_grid_block_on_image(
                img=image,
                grid=base_grid,
                out_path=str(base_overlay_path),
                cmap=args.viz_diverging,
                title=f"{qid} | {args.variant} | {args.mode} | BASE (signed)",
                signed=True,
                show_top_bottom_k=args.top_k,
                clip_percentiles=(5, 95),
            )

            overlay_grid_block_on_image(
                img=image,
                grid=mosaic_grid,
                out_path=str(mosaic_overlay_path),
                cmap=args.viz_diverging,
                title=f"{qid} | {args.variant} | {args.mode} | MOSAIC (signed)",
                signed=True,
                show_top_bottom_k=args.top_k,
                clip_percentiles=(5, 95),
            )
        else:
            overlay_grid_on_image(
                img=image,
                grid=base_grid_norm,
                out_path=str(base_overlay_path),
                cmap=args.viz_diverging,
                title=f"{qid} | {args.variant} | {args.mode} | BASE (signed)",
                signed=True,
            )

            overlay_grid_on_image(
                img=image,
                grid=mosaic_grid_norm,
                out_path=str(mosaic_overlay_path),
                cmap=args.viz_diverging,
                title=f"{qid} | {args.variant} | {args.mode} | MOSAIC (signed)",
                signed=True,
            )

        # Save run info
        info = {
            "question_id": qid,
            "variant": args.variant,
            "mode": args.mode,
            "predicted_letter": pred_letter,
            "correct_letter": item["correct_letter"],
            "is_correct": bool(pred_letter == item["correct_letter"]),
            "npz_path": str(npz_path),
            "overlay_base_path": str(base_overlay_path),
            "overlay_mosaic_path": str(mosaic_overlay_path),
            "meta": meta,
        }
        with open(out_dir / "run_info.json", "w") as f:
            json.dump(info, f, indent=2)

        processed += 1
        if args.max_samples and processed >= args.max_samples:
            break

    print(f"\nDone. Processed {processed} samples.")
    print(f"Outputs in: {out_root}")


if __name__ == "__main__":
    main()