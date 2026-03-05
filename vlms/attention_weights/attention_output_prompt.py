# cache_generated_token_attn_with_mapping_npz_chattemplate.py
"""
Generation-step attention caching for Llava-NeXT, consistent with your evaluator/IG setup:
- Uses processor.apply_chat_template(conversation, add_generation_prompt=True)
- Extracts per-layer head-averaged attention from the generated token -> prompt tokens only
- Builds full packed image-token mapping (token_idx -> kind/row/col/bbox) like your IG script
- Saves everything into ONE compressed NPZ per sample, IG-style:
    - attn (L, S_prompt) float16
    - meta (json string)
    - packed_mapping_summary (json string)
    - packed_mapping_tokens (json string; full mapping list)
    - image_placeholder_positions (int32 array)
    - prompt_input_ids (int32 array)
    - attention_mask (int32 array, if present)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List, Any
import matplotlib.pyplot as plt

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution
from scipy.ndimage import zoom


# -----------------------------
# Dataset utils
# -----------------------------
def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = local_cache_root / sanitize_repo_id(dataset_id)

    if cache_dir.exists():
        print(f"Loading dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"Downloading dataset '{dataset_id}' from Hugging Face...")
    ds = load_dataset(dataset_id, split=split)
    try:
        ds.save_to_disk(str(cache_dir))
        print(f"Saved dataset to cache: {cache_dir}")
    except Exception as e:
        print(f"Warning: failed to save dataset to disk: {e}")
    return ds


def load_qid_whitelist(path: str) -> Set[str]:
    qids = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.add(s)
    return qids


# -----------------------------
# Prompt formatting (same content as your evaluator)
# -----------------------------
def format_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    instruction = "Answer the following multiple-choice question by selecting the correct option."
    prompt = f"{instruction}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for k in ["A", "B", "C", "D"]:
        if k in options:
            prompt += f"{k}) {options[k]}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def build_options_from_sample(sample: Dict, shuffle: bool = False, seed: int = 0) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
    """
    Matches your evaluator logic: optional deterministic shuffle (qid-based).
    Returns:
      options: dict A/B/C/D -> text
      correct_letter: which label corresponds to correct_answer
      option_meta: order and label_to_key for auditing
    """
    labels = ["A", "B", "C", "D"]
    candidates = [
        {"key": "correct_answer", "text": sample["correct_answer"]["text"]},
        {"key": "misleading_groundable", "text": sample["misleading_groundable"]["text"]},
        {"key": "misleading_ungroundable", "text": sample["misleading_ungroundable"]["text"]},
        {"key": "irrelevant_word", "text": sample["irrelevant_word"]["text"]},
    ]

    qid = str(sample.get("question_id", ""))
    if shuffle:
        import random
        rng = random.Random(f"{seed}_{qid}")
        rng.shuffle(candidates)

    options = {labels[i]: candidates[i]["text"] for i in range(4)}
    correct_index = next(i for i, c in enumerate(candidates) if c["key"] == "correct_answer")
    correct_letter = labels[correct_index]

    option_meta = {
        "order": [c["key"] for c in candidates],
        "label_to_key": {labels[i]: candidates[i]["key"] for i in range(4)},
        "shuffle": bool(shuffle),
        "seed": int(seed),
    }
    return options, correct_letter, option_meta


# -----------------------------
# Image placeholder positions
# -----------------------------
def get_image_placeholder_positions(model, input_ids: torch.Tensor) -> List[int]:
    image_token_id = model.config.image_token_id
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return pos.tolist()


# -----------------------------
# Packed image-token mapping logic (full mapping, same as IG script)
# -----------------------------
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
        scale = current_w / original_w
        new_h = int(round(original_h * scale, 7))
        pad = (current_h - new_h) // 2
        return {"mode": "slice_h", "scale": scale, "pad": pad, "new_h": new_h, "new_w": current_w}
    else:
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
    scale = (current_w / original_w) if (orig_ar > curr_ar) else (current_h / original_h)

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
    num_views_provided: int,                # pixel_values.shape[1] if anyres
    include_newline_tokens: bool = True,
) -> Dict[str, Any]:
    cfg = model.config
    vision_cfg = cfg.vision_config

    tile_size = int(vision_cfg.image_size)
    patch_size = int(vision_cfg.patch_size)
    patches_per_side = tile_size // patch_size
    per_tile_patch_tokens = patches_per_side * patches_per_side

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

    # Mosaic + newline tokens
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

    return {"summary": {
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
            "tokens": tokens}


# -----------------------------
# Generation-step attention extraction
# -----------------------------
@torch.no_grad()
def generate_get_head_avg_attn_to_prompt(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    S_prompt = int(inputs["input_ids"].shape[1])

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
    )

    print("attention_mask:", inputs["attention_mask"].shape,
      inputs["attention_mask"].dtype,
      inputs["attention_mask"].min().item(),
      inputs["attention_mask"].max().item(),
      inputs["attention_mask"].sum().item())

    print("input_ids shape:", inputs["input_ids"].shape)
    print("num masked tokens:", (inputs["attention_mask"][0] == 0).sum().item())

    sequences = gen.sequences  # prompt + generated
    gen_token_id = int(sequences[0, -1].item())
    tok = getattr(processor, "tokenizer", None)
    gen_token_text = tok.decode([gen_token_id], skip_special_tokens=False) if tok is not None else str(gen_token_id)

    attentions = gen.attentions
    if attentions is None or len(attentions) < 1:
        raise RuntimeError("No attentions returned from generate().")

    step_attn = attentions[-1]  # last step; tuple length L

    layer_vecs = []
    S_so_far_ref: Optional[int] = None
    for layer_attn in step_attn:
        # expected (B, H, 1, S_so_far)
        if layer_attn is None:
            raise RuntimeError("Found None layer attention.")
        if layer_attn.ndim != 4:
            raise RuntimeError(f"Unexpected attention shape {tuple(layer_attn.shape)}; expected (B,H,1,S).")

        S_so_far = int(layer_attn.shape[-1])
        if S_so_far_ref is None:
            S_so_far_ref = S_so_far

        v = layer_attn[0].mean(dim=0)[0]  # (S_so_far,)
        layer_vecs.append(v[:S_prompt])   # prompt-only slice

    A = torch.stack(layer_vecs, dim=0)  # (L, S_prompt)
    attn_np = A.to(torch.float16).cpu().numpy()

    info = {
        "num_layers": int(A.shape[0]),
        "prompt_seq_len": int(S_prompt),
        "attn_seq_len_so_far": int(S_so_far_ref) if S_so_far_ref is not None else None,
        "generated_token_id": int(gen_token_id),
        "generated_token_text": gen_token_text,
        "max_new_tokens": int(max_new_tokens),
    }
    return attn_np, info

# -----------------------------
# Generation-step attention extraction (manual 2-step version)
# -----------------------------
@torch.no_grad()
def next_token_attn_to_prompt(model, processor, inputs):
    S_prompt = int(inputs["input_ids"].shape[1])

    # 1) Prefill pass (prompt)
    out0 = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = out0.past_key_values

    # Greedy next token (same as do_sample=False)
    next_id = out0.logits[:, -1].argmax(dim=-1, keepdim=True)

    # 2) One decode step with attentions
    attn_mask2 = torch.cat(
        [inputs["attention_mask"], torch.ones_like(next_id)], dim=1
    )

    out1 = model(
        input_ids=next_id,
        attention_mask=attn_mask2,
        past_key_values=past,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

    step_attn = out1.attentions  # tuple(L) of (B,H,1,S_so_far)

    layer_vecs = []
    S_so_far_ref = None

    for layer_attn in step_attn:
        S_so_far = int(layer_attn.shape[-1])
        if S_so_far_ref is None:
            S_so_far_ref = S_so_far

        v = layer_attn[0].mean(dim=0)[0]   # (S_so_far,)
        layer_vecs.append(v[:S_prompt])    # prompt-only slice

    A = torch.stack(layer_vecs, dim=0)  # (L, S_prompt)
    attn_np = A.to(torch.float32).cpu().numpy()

    gen_token_id = int(next_id[0, 0].item())
    tok = getattr(processor, "tokenizer", None)
    gen_token_text = tok.decode([gen_token_id], skip_special_tokens=False) if tok else str(gen_token_id)

    info = {
        "num_layers": int(A.shape[0]),
        "prompt_seq_len": int(S_prompt),
        "attn_seq_len_so_far": int(S_so_far_ref),
        "generated_token_id": gen_token_id,
        "generated_token_text": gen_token_text,
        "max_new_tokens": 1,
    }

    return attn_np, info
# -----------------------------
# Save NPZ (IG style)
# -----------------------------
def save_npz_sample(
    out_dir: str,
    qid: str,
    variant: str,
    *,
    attn_np: np.ndarray,
    meta: Dict[str, Any],
    mapping: Dict[str, Any],
    image_placeholder_positions: List[int],
    prompt_input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
    mode: str = "gen_token",
):
    sample_dir = Path(out_dir) / variant / qid
    sample_dir.mkdir(parents=True, exist_ok=True)

    npz_path = sample_dir / f"gen_attn_{mode}.npz"

    save_dict: Dict[str, Any] = {
        "attn": attn_np,
        "meta": json.dumps(meta),
        "packed_mapping_summary": json.dumps(mapping["summary"]),
        "packed_mapping_tokens": json.dumps(mapping["tokens"]),
        "image_placeholder_positions": np.asarray(image_placeholder_positions, dtype=np.int32),
        "prompt_input_ids": prompt_input_ids.astype(np.int32),
    }
    if attention_mask is not None:
        save_dict["attention_mask"] = attention_mask.astype(np.int32)

    np.savez_compressed(str(npz_path), **save_dict)

def run_sanity_checks(
    *,
    qid: str,
    attn_np: np.ndarray,                 # (L, S_prompt)
    image_placeholder_positions: List[int],
    mapping: Dict[str, Any],
    meta: Dict[str, Any],
):
    # 1) finite
    if not np.isfinite(attn_np).all():
        raise RuntimeError(f"[{qid}] attn contains NaN/Inf")

    L, S = attn_np.shape
    # 2) sums ~1 for max_new_tokens=1
    max_new_tokens = int(meta.get("max_new_tokens", 1))
    if max_new_tokens == 1:
        sums = attn_np.sum(axis=1)  # per layer
        max_err = float(np.max(np.abs(sums - 1.0)))
        print(f"[{qid}] sanity: per-layer sum(attn_prompt) max|err|={max_err:.6e}")
        if max_err > 5e-3:
            print(f"[{qid}] attention row sums deviate too much from 1.0 (max_err={max_err})")

    # 3) mapping length match
    expected = int(mapping["summary"]["total_packed_image_tokens"])
    if len(image_placeholder_positions) != expected:
        raise RuntimeError(
            f"[{qid}] mapping mismatch: len(img_pos)={len(image_placeholder_positions)} vs expected={expected}"
        )

    # 4) image mass stats
    img_pos = np.asarray(image_placeholder_positions, dtype=np.int64)
    img_mass = attn_np[:, img_pos].sum(axis=1)  # (L,)
    print(f"[{qid}] sanity: image-attn mass per layer: "
          f"min={img_mass.min():.4f}, mean={img_mass.mean():.4f}, max={img_mass.max():.4f}")

    # 5) bbox bounds check (sample a few)
    H, W = mapping["summary"]["original_size_hw"]
    bad = 0
    checked = 0
    for t in mapping["tokens"]:
        bb = t.get("bbox")
        if bb is None:
            continue
        y0, x0, y1, x1 = bb
        checked += 1
        if not (0.0 <= x0 <= W and 0.0 <= x1 <= W and 0.0 <= y0 <= H and 0.0 <= y1 <= H and x1 >= x0 and y1 >= y0):
            bad += 1
            if bad <= 5:
                print(f"[{qid}] sanity: bad bbox token_idx={t.get('token_idx')} bbox={bb} H,W={H,W}")
    print(f"[{qid}] sanity: bbox checked={checked}, bad={bad}")
    if bad > 0:
        raise RuntimeError(f"[{qid}] found {bad} invalid bboxes (see prints above)")
    

## plotting helpers
def attn_to_image_token_scores(attn_np: np.ndarray, img_pos: List[int], how: str = "last") -> np.ndarray:
    """
    attn_np: (L, S_prompt) attention from generated token -> prompt tokens
    img_pos: list of prompt indices that correspond to packed image tokens (len = N_img)
    how: "last" | "mean" | "layer:<idx>"  (idx can be negative)
    returns: token_scores (N_img,) float32
    """
    if how == "last":
        v = attn_np[-1]
    elif how == "mean":
        v = attn_np.mean(axis=0)
    elif how.startswith("layer:"):
        idx = int(how.split(":", 1)[1])
        v = attn_np[idx]
    else:
        raise ValueError(f"Unknown how={how}")

    img_pos = np.asarray(img_pos, dtype=np.int64)
    return v[img_pos].astype(np.float32)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="attn_cache_gen")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_new_tokens", type=int, default=1)

    # keep consistent with IG/evaluator
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    

    # mapping
    parser.add_argument("--include_newline_tokens", action="store_true")
    parser.add_argument("--debug_first_only", action="store_true",
                    help="If set, process and save only the first matching sample.")
    parser.add_argument("--sanity_checks", action="store_true",
                    help="Run sanity checks and print debug stats for the sample(s).")
    parser.add_argument("--start", type=int, default=0, help='Start index for processing samples.')
    parser.add_argument("--end", type=int, default=500, help='End index (exclusive) for processing samples. 0 = no limit.')

    args = parser.parse_args()

    # import transformers
    # print(transformers.__version__)
    # import torch 
    # print(torch.__version__)

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device_t = torch.device(device)

    print(f"Loading model: {args.model_id} on {device}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # IMPORTANT: pass at load time
    ).to(device_t)
    
    
    # Force eager attention so output_attentions works
    # if hasattr(model, "set_attn_implementation"):
    #     model.set_attn_implementation("eager")
    # else:
    #     # fallback for older versions: set in config (works for many models)
    #     try:
    #         model.config.attn_implementation = "eager"
    #     except Exception:
    #         pass

    model.eval()

    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    # Read all subdirectories in output root

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    existing_subdirs = []
    if out_root.exists():
        existing_subdirs = [d.name for d in out_root.iterdir() if d.is_dir()]
    print(f"Found {len(existing_subdirs)} existing subdirectories in {out_root}: {existing_subdirs}")

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching gen-token attn (chat template) [{args.variant}]"):
        sample = ds[i]
        qid = str(sample.get("question_id", f"unknown_{i}"))
        if qid not in qids:
            continue
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        if qid in existing_subdirs:
            print(f"[{qid}] Output already exists, skipping.")
            continue
        # choose image variant
        if args.variant == "notext":
            img = sample["notext"]["image"]
        else:
            if args.variant not in sample:
                continue
            img = sample[args.variant]["image"]
        img = img.convert("RGB")

        # build prompt (same content as evaluator)
        options, correct_letter, option_meta = build_options_from_sample(
            sample,
            shuffle=args.shuffle_options,
            seed=args.seed
        )
        prompt_text = format_mcq_prompt(sample.get("question", ""), options)

        # IMPORTANT: evaluator-style chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, topkenize = False)

        inputs = processor(images=img, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)

        # generation-step attention
        # attn_np, gen_info = generate_get_head_avg_attn_to_prompt(
        #     model, processor, inputs, max_new_tokens=args.max_new_tokens
        # )
        attn_np, gen_info = next_token_attn_to_prompt(
            model, processor, inputs
        )

        # print(attn_np, gen_info)

        # placeholders
        input_ids = inputs["input_ids"]
        img_pos = get_image_placeholder_positions(model, input_ids)

        # mapping prerequisites
        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is None:
            raise RuntimeError("Processor did not return image_sizes; packed mapping needs original H,W.")
        orig_h = int(image_sizes[0, 0].item())
        orig_w = int(image_sizes[0, 1].item())

        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            raise RuntimeError("Processor did not return pixel_values; packed mapping needs num_views_provided.")
        num_views = int(pixel_values.shape[1]) if pixel_values.ndim == 5 else 1

        packed_mapping = build_packed_image_token_map_for_one_image(
            model=model,
            original_size_hw=(orig_h, orig_w),
            num_views_provided=num_views,
            include_newline_tokens=bool(args.include_newline_tokens),
        )

        # ---- Debug visualization: overlay attention on image-token grids ----
        # mapping_tokens = packed_mapping["tokens"]
        # summary = packed_mapping["summary"]

        # choose which layer aggregation to visualize
        # options: "last", "mean", "layer:0", "layer:-1", "layer:15", etc.
        # num_layers = attn_np.shape[0]

        # for layer_idx in range(num_layers):
        #     token_scores = attn_np[layer_idx, img_pos].astype(np.float32)

        #     # Optional but recommended: normalize within image tokens
        #     token_scores = token_scores / (token_scores.sum() + 1e-12)

        #     base_grid, mosaic_grid = token_scores_to_grids(
        #         mapping_tokens=mapping_tokens,
        #         token_scores=token_scores,
        #         summary=summary,
        #     )
        #     # Save overlays
        #     viz_dir = Path(args.out_dir) / args.variant / qid
        #     viz_dir.mkdir(parents=True, exist_ok=True)

        #     overlay_base_path = viz_dir / f"attn_overlay_base_layer{layer_idx:02d}.png"
        #     overlay_mosaic_path = viz_dir / f"attn_overlay_mosaic_layer{layer_idx:02d}.png"

        #     overlay_grid_block_on_image(
        #         img=img,
        #         grid=base_grid,
        #         out_path=str(overlay_base_path),
        #         title=f"{qid} | layer {layer_idx} | BASE",
        #         signed=False,
        #         cmap="jet",
        #         alpha=0.55,
        #         show_top_bottom_k=200,
        #         clip_percentiles=(5, 99),
        #     )

        #     overlay_grid_block_on_image(
        #         img=img,
        #         grid=mosaic_grid,
        #         out_path=str(overlay_mosaic_path),
        #         title=f"{qid} | layer {layer_idx} | MOSAIC",
        #         signed=False,
        #         cmap="jet",
        #         alpha=0.55,
        #         show_top_bottom_k=400,
        #         clip_percentiles=(5, 99),
        #     )
        expected_img_tokens = int(packed_mapping["summary"]["total_packed_image_tokens"])
        if len(img_pos) != expected_img_tokens:
            print(
                f"[{qid}] Packed mapping mismatch: "
                f"len(image_placeholder_positions)={len(img_pos)} != expected={expected_img_tokens}. "
                "This means attention->image-token alignment is unsafe."
            )
            continue

        # meta (small; mapping stored separately as json strings in npz)
        meta = {
            "question_id": qid,
            "variant": args.variant,
            "prompt_seq_len": int(input_ids.shape[1]),
            "image_placeholder_count": len(img_pos),
            "correct_letter": correct_letter,
            "option_meta": option_meta,
            "original_image_size_hw": [orig_h, orig_w],
            "num_views_provided": num_views,
            "include_newline_tokens": bool(args.include_newline_tokens),
            **gen_info,
        }

        prompt_input_ids = input_ids[0].detach().cpu().numpy()
        attention_mask = inputs["attention_mask"][0].detach().cpu().numpy() if "attention_mask" in inputs else None

        save_npz_sample(
            args.out_dir,
            qid,
            args.variant,
            attn_np=attn_np,
            meta=meta,
            mapping=packed_mapping,
            image_placeholder_positions=img_pos,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            mode="gen_token",
        )
        if args.sanity_checks:
            run_sanity_checks(
                qid=qid,
                attn_np=attn_np,
                image_placeholder_positions=img_pos,
                mapping=packed_mapping,
                meta=meta,
            )
        if args.debug_first_only:
            print(f"[debug_first_only] Saved first sample qid={qid}. Exiting.")
            break

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Cached {kept} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()