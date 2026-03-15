#!/usr/bin/env python3
"""
Token-space Integrated Gradients for Qwen2-VL / Qwen2.5-VL + token-to-patch visualization.

What this script does
1) Loads a HF dataset sample-by-sample (optionally filtered by question_id list).
2) Builds an MCQ prompt (same style as your evaluator).
3) Runs the model once to get a predicted answer letter (A/B/C/D).
4) Computes Integrated Gradients over the Qwen image-token embeddings.
5) Maps each <|image_pad|> token back to its merged patch region.
6) Saves overlays and attribution arrays.

Outputs per sample:
- out_dir/{variant}/{qid}/
    - ig_{mode}.npz
    - overlay_grid_{mode}.png
    - run_info.json

Notes
- This is for Qwen/Qwen2-VL-* and Qwen/Qwen2.5-VL-* style models.
- Mapping is based on image_grid_thw + patch_size + merge_size.
- One image token corresponds to one merged patch block on the resized grid.

Run example:
python ig_token_space_qwen.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --hf_dataset AHAAM/GUIC \
  --ids_file no_overlap_question_ids.txt \
  --variant notext \
  --mode prefill_next_token \
  --out_dir ./ig_qwen_outputs \
  --max_samples 50
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import log_softmax
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from captum.attr import IntegratedGradients

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ======================================================================================
# Dataset helpers
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


def load_question_id_filter(path: str) -> set:
    ids = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


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

    return {
        "question_id": str(qid),
        "image": img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj,
        "question": question,
        "options": options,
        "correct_letter": correct_letter,
    }


# ======================================================================================
# Qwen input helpers
# ======================================================================================

def build_qwen_messages(image: Image.Image, prompt_text: str, assistant_text: Optional[str] = None):
    if assistant_text is None:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]


def prepare_qwen_inputs(processor, image: Image.Image, prompt_text: str, assistant_text: Optional[str] = None):
    messages = build_qwen_messages(image=image, prompt_text=prompt_text, assistant_text=assistant_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=(assistant_text is None))
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    return inputs, text, messages


# ======================================================================================
# Token helpers
# ======================================================================================

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


def _token_id_for_answer_letter(tokenizer, letter: str) -> Tuple[int, str]:
    for s in [f" {letter}", letter]:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s
    raise ValueError(f"Answer letter {letter!r} does not map to a single token.")


def qwen_image_token_positions(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    return (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]


def qwen_vision_span_positions(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    ids = []
    for tok in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        ids.append(tok_id)
    mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    for tok_id in ids:
        mask |= (input_ids[0] == tok_id)
    return mask.nonzero(as_tuple=True)[0]


# ======================================================================================
# Qwen mapping
# ======================================================================================

def build_qwen_image_token_map_for_one_image(
    *,
    image: Image.Image,
    image_grid_thw: Tuple[int, int, int],
    patch_size: int,
    merge_size: int,
) -> Dict[str, Any]:
    orig_w, orig_h = image.size
    T, H, W = image_grid_thw

    if T != 1:
        raise ValueError(f"This mapper assumes a still image with T=1, got T={T}")

    merged_h = H // merge_size
    merged_w = W // merge_size
    num_image_tokens = merged_h * merged_w

    resized_w = W * patch_size
    resized_h = H * patch_size

    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    tokens = []
    token_idx = 0

    for row in range(merged_h):
        for col in range(merged_w):
            patch_row_start = row * merge_size
            patch_row_end = (row + 1) * merge_size
            patch_col_start = col * merge_size
            patch_col_end = (col + 1) * merge_size

            x0_r = patch_col_start * patch_size
            y0_r = patch_row_start * patch_size
            x1_r = patch_col_end * patch_size
            y1_r = patch_row_end * patch_size

            x0 = x0_r * scale_x
            y0 = y0_r * scale_y
            x1 = x1_r * scale_x
            y1 = y1_r * scale_y

            tokens.append({
                "token_idx": token_idx,
                "kind": "merged_patch",
                "row": row,
                "col": col,
                "patch_rows": (patch_row_start, patch_row_end - 1),
                "patch_cols": (patch_col_start, patch_col_end - 1),
                "resized_box_xyxy": (x0_r, y0_r, x1_r, y1_r),
                "bbox": (y0, x0, y1, x1),  # keep same bbox convention as your old script
            })
            token_idx += 1

    return {
        "summary": {
            "original_size_hw": (orig_h, orig_w),
            "resized_size_hw": (resized_h, resized_w),
            "image_grid_thw": (T, H, W),
            "patch_size": patch_size,
            "merge_size": merge_size,
            "merged_grid_hw": (merged_h, merged_w),
            "num_image_tokens": num_image_tokens,
        },
        "tokens": tokens,
    }


# ======================================================================================
# Visualization
# ======================================================================================

def robust_normalize(x, clip_percentiles=(5, 95), signed=False, eps=1e-8):
    x = x.astype(np.float32)
    if signed:
        a = np.abs(x)
        hi = np.percentile(a, clip_percentiles[1])
        hi = max(hi, eps)
        x = np.clip(x, -hi, hi) / hi
        return x

    lo = np.percentile(x, clip_percentiles[0])
    hi = np.percentile(x, clip_percentiles[1])
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def token_scores_to_qwen_grid(mapping_tokens, token_scores, summary):
    gh, gw = summary["merged_grid_hw"]
    grid = np.zeros((gh, gw), dtype=np.float32)
    for t in mapping_tokens:
        idx = t["token_idx"]
        s = float(token_scores[idx])
        r, c = t["row"], t["col"]
        grid[r, c] += s
    return grid


def overlay_grid_on_image(img, grid, out_path, title="", signed=True, alpha=0.55, cmap="RdBu_r"):
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    H, W = img.height, img.width
    gh, gw = grid.shape

    scale_h = H / gh
    scale_w = W / gw

    # smooth interpolation like Llava script
    grid_up = zoom(grid, (scale_h, scale_w), order=1)

    grid_norm = robust_normalize(grid_up, signed=signed)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.imshow(img)

    if signed:
        mappable = ax.imshow(grid_norm, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)
    else:
        mappable = ax.imshow(grid_norm, alpha=alpha, cmap="jet", vmin=0, vmax=1)

    ax.axis("off")
    ax.set_title(title)

    fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def overlay_grid_block_on_image(
    img,
    grid,
    out_path,
    title="",
    signed=False,
    cmap=None,
    alpha=0.55,
    show_top_bottom_k: int = 0,
    clip_percentiles=(5, 95),
):
    H, W = img.height, img.width
    gh, gw = grid.shape

    if show_top_bottom_k and show_top_bottom_k > 0:
        flat = grid.reshape(-1)
        lo = np.percentile(flat, clip_percentiles[0])
        hi = np.percentile(flat, clip_percentiles[1])
        in_band = (flat >= lo) & (flat <= hi) & np.isfinite(flat)
        idx_band = np.nonzero(in_band)[0]

        if idx_band.size > 0:
            vals_band = flat[idx_band]
            k_eff = min(show_top_bottom_k, idx_band.size)
            bot_rel = np.argpartition(vals_band, k_eff - 1)[:k_eff]
            top_rel = np.argpartition(vals_band, -(k_eff))[-k_eff:]
            keep_idx = np.unique(np.concatenate([idx_band[bot_rel], idx_band[top_rel]]))

            mask = np.zeros_like(flat, dtype=bool)
            mask[keep_idx] = True
            mask = mask.reshape(grid.shape)

            grid_kept = np.zeros_like(grid, dtype=np.float32)
            grid_kept[mask] = grid[mask]
        else:
            mask = np.zeros_like(grid, dtype=bool)
            grid_kept = np.zeros_like(grid, dtype=np.float32)
    else:
        grid_kept = grid
        mask = np.ones_like(grid_kept, dtype=bool)

    scale_h = H / gh
    scale_w = W / gw
    grid_up = zoom(grid_kept, (scale_h, scale_w), order=0)
    mask_up = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5

    grid_norm = robust_normalize(grid_up, signed=signed)
    grid_vis = np.ma.array(grid_norm, mask=~mask_up)

    fig, ax = plt.subplots(figsize=(10, 6))
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


def draw_topk_token_boxes(
    img: Image.Image,
    mapping_tokens: List[Dict[str, Any]],
    token_scores: np.ndarray,
    out_path: str,
    *,
    top_k: int = 50,
    score_power: float = 0.5,
):
    im = img.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    candidates = []
    for t in mapping_tokens:
        idx = t["token_idx"]
        bb = t.get("bbox")
        if bb is None or idx >= len(token_scores):
            continue
        candidates.append((idx, float(token_scores[idx]), bb))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:top_k]

    if not candidates:
        im.save(out_path)
        return

    scores = np.array([c[1] for c in candidates], dtype=np.float32)
    smax = float(scores.max()) if scores.size else 1.0
    smin = float(scores.min()) if scores.size else 0.0
    denom = smax - smin + 1e-8

    for idx, s, bb in candidates:
        y0, x0, y1, x1 = bb
        norm = (s - smin) / denom
        width = int(1 + 4 * (norm ** score_power))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=width)
        if font is not None:
            draw.text((x0 + 1, y0 + 1), f"{idx}:{s:.3g}", fill=(255, 0, 0), font=font)

    im.save(out_path)


# ======================================================================================
# Prediction
# ======================================================================================

@torch.no_grad()
def predict_answer_letter_qwen(
    model,
    processor,
    image: Image.Image,
    mcq_prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 5,
) -> str:
    inputs, _, _ = prepare_qwen_inputs(processor, image=image, prompt_text=mcq_prompt_text, assistant_text=None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=False,
    )
    decoded = processor.batch_decode(
        gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _extract_answer_letter(decoded)


# ======================================================================================
# Image embedding extraction for Qwen
# ======================================================================================

def get_qwen_image_embeds(model, inputs):
    """
    Attempts to obtain the image token embeddings inserted into the LLM for Qwen.

    This part is the most model-sensitive section.
    """
    if hasattr(model, "get_image_features"):
        raw = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs.get("image_grid_thw", None),
        )
    else:
        raise AttributeError("This Qwen model class has no get_image_features().")

    if isinstance(raw, torch.Tensor):
        img_embeds = raw
    elif isinstance(raw, (list, tuple)):
        tensor_items = [x for x in raw if isinstance(x, torch.Tensor)]
        if len(tensor_items) == 1:
            img_embeds = tensor_items[0]
        elif len(tensor_items) > 1:
            img_embeds = torch.cat(tensor_items, dim=1)
        else:
            raise RuntimeError("get_image_features returned list/tuple without tensors.")
    elif isinstance(raw, dict):
        img_embeds = None
        for k in ("image_embeds", "image_features", "embeds", "features"):
            if k in raw and isinstance(raw[k], torch.Tensor):
                img_embeds = raw[k]
                break
        if img_embeds is None:
            raise RuntimeError("Could not parse dict output of get_image_features().")
    else:
        raise RuntimeError(f"Unexpected get_image_features output type: {type(raw)}")

    if img_embeds.dim() == 2:
        img_embeds = img_embeds.unsqueeze(0)
    elif img_embeds.dim() != 3:
        raise RuntimeError(f"Unexpected img_embeds shape: {tuple(img_embeds.shape)}")

    return img_embeds


# ======================================================================================
# Integrated Gradients
# ======================================================================================

def token_space_ig_qwen(
    model,
    processor,
    image: Image.Image,
    mcq_prompt_text: str,
    answer_letter: str,
    *,
    mode: str,
    steps: int,
    device: torch.device,
):
    model.eval()

    if mode == "teacher_forced":
        inputs, _, _ = prepare_qwen_inputs(
            processor, image=image, prompt_text=mcq_prompt_text, assistant_text=answer_letter
        )
        logit_pos = -2
        target_token_id = int(inputs["input_ids"][0, -1].item())
        target_str = "<teacher_forced_last_token>"
    elif mode == "prefill_next_token":
        inputs, _, _ = prepare_qwen_inputs(
            processor, image=image, prompt_text=mcq_prompt_text, assistant_text=None
        )
        logit_pos = -1
        target_token_id, target_str = _token_id_for_answer_letter(processor.tokenizer, answer_letter)
    else:
        raise ValueError("mode must be 'teacher_forced' or 'prefill_next_token'")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)

    with torch.no_grad():
        img_embeds = get_qwen_image_embeds(model, inputs).to(device)

    _, N_img, D = img_embeds.shape

    img_positions = qwen_image_token_positions(input_ids, processor.tokenizer)
    if int(img_positions.numel()) != int(N_img):
        raise RuntimeError(
            f"placeholder count {img_positions.numel()} != image embeds {N_img}"
        )

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        full_embeds = embed_layer(input_ids)

    blank = Image.new("RGB", image.size, (127, 127, 127))
    blank_inputs, _, _ = prepare_qwen_inputs(
        processor, image=blank, prompt_text=mcq_prompt_text, assistant_text=(answer_letter if mode == "teacher_forced" else None)
    )
    blank_inputs = {k: v.to(device) for k, v in blank_inputs.items()}

    with torch.no_grad():
        baseline = get_qwen_image_embeds(model, blank_inputs).to(device)

    if baseline.shape != img_embeds.shape:
        raise RuntimeError(
            f"Baseline shape {tuple(baseline.shape)} != img_embeds shape {tuple(img_embeds.shape)}"
        )

    def forward_func(img_embeds_var: torch.Tensor) -> torch.Tensor:
        inputs_embeds = full_embeds.clone()
        inputs_embeds[:, img_positions, :] = img_embeds_var

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits
        logp = log_softmax(logits[:, logit_pos, :], dim=-1)
        return logp[:, target_token_id]

    ig = IntegratedGradients(forward_func)
    attr = ig.attribute(
        img_embeds,
        baselines=baseline,
        n_steps=steps,
        internal_batch_size=1,
    )

    with torch.no_grad():
        fx = forward_func(img_embeds).item()
        f0 = forward_func(baseline).item()
        delta = fx - f0

    total_attr = attr.sum().item()
    rel_err = abs(total_attr - delta) / (abs(delta) + 1e-8)
    if not np.isfinite(rel_err) or rel_err > 0.15:
        print(
            f"[SANITY] IG completeness warning | "
            f"delta={delta:.6f}, total_attr={total_attr:.6f}, rel_err={rel_err:.3f}"
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
        "sanity_fx": fx,
        "sanity_f0": f0,
        "sanity_delta": delta,
        "sanity_total_attr": total_attr,
        "sanity_rel_err": rel_err,
    }

    return token_scores, meta, inputs


# ======================================================================================
# Main
# ======================================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    p.add_argument("--hf_cache_dir", type=str, default="./hf_dataset_GUIC")
    p.add_argument("--split", type=str, default="test")

    p.add_argument("--ids_file", type=str, default="../inference/no_overlap_question_ids.txt", help="Optional file with question_ids to keep.")
    p.add_argument("--variant", type=str, default="misleading_groundable",
                   choices=["notext", "correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"])
    p.add_argument("--shuffle_options", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mode", type=str, default="prefill_next_token",
                   choices=["teacher_forced", "prefill_next_token"])
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="./ig_qwen_outputs")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--predicted", action="store_true", help="Use predicted answer instead of correct answer.")
    p.add_argument("--block_overlay", action="store_true")
    p.add_argument("--save_grids", action="store_true")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading model {args.model_id} on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)

    keep_ids = None
    if args.ids_file:
        keep_ids = load_question_id_filter(args.ids_file)
        print(f"Loaded {len(keep_ids)} question_ids from {args.ids_file}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0

    for i in tqdm(range(len(ds)), desc="Scanning dataset"):
        if i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        sample = ds[i]
        qid = sample.get("question_id", None)
        if qid is None:
            continue
        qid = str(qid)

        if keep_ids is not None and qid not in keep_ids:
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

        pred_letter = predict_answer_letter_qwen(
            model=model,
            processor=processor,
            image=image,
            mcq_prompt_text=mcq_prompt_text,
            device=device,
            max_new_tokens=5,
        )
        if pred_letter not in ("A", "B", "C", "D"):
            print(f"[{qid}] predicted answer not a single letter: {pred_letter!r} -> skipping IG")
            continue

        target_letter = pred_letter if args.predicted else item["correct_letter"]

        token_scores, meta, inputs = token_space_ig_qwen(
            model=model,
            processor=processor,
            image=image,
            mcq_prompt_text=mcq_prompt_text,
            answer_letter=target_letter,
            mode=args.mode,
            steps=args.steps,
            device=device,
        )

        grid = inputs["image_grid_thw"][0].tolist()
        patch_size = processor.image_processor.patch_size
        merge_size = processor.image_processor.merge_size

        mapping = build_qwen_image_token_map_for_one_image(
            image=image,
            image_grid_thw=tuple(grid),
            patch_size=patch_size,
            merge_size=merge_size,
        )

        expected = int(mapping["summary"]["num_image_tokens"])
        if token_scores.shape[0] != expected:
            print(f"[{qid}] WARNING: token_scores N={token_scores.shape[0]} != mapping expected={expected}")
            continue

        mapping_tokens = mapping["tokens"]
        summary = mapping["summary"]

        qwen_grid = token_scores_to_qwen_grid(
            mapping_tokens=mapping_tokens,
            token_scores=token_scores,
            summary=summary,
        )

        out_dir = out_root / args.variant / qid
        out_dir.mkdir(parents=True, exist_ok=True)

        npz_path = out_dir / f"ig_{args.mode}.npz"
        save_dict = {
            "token_scores": token_scores,
            "meta": json.dumps(meta),
            "mapping_summary": json.dumps(summary),
            "mapping_tokens": json.dumps(mapping_tokens),
        }
        if args.save_grids:
            save_dict["qwen_grid_signed"] = qwen_grid
        np.savez_compressed(str(npz_path), **save_dict)

        if args.block_overlay:
            overlay_grid_on_image(
                img=image,
                grid=qwen_grid,
                out_path=str(out_dir / f"overlay_grid_{args.mode}.png"),
                title=f"{qid} | {args.variant} | {args.mode} | Qwen merged grid",
                signed=True,
                # show_top_bottom_k=args.top_k,
                # clip_percentiles=(5, 95),
            )
        else:
            draw_topk_token_boxes(
                img=image,
                mapping_tokens=mapping_tokens,
                token_scores=token_scores,
                out_path=str(out_dir / f"overlay_boxes_{args.mode}.png"),
                top_k=args.top_k,
            )

        info = {
            "question_id": qid,
            "variant": args.variant,
            "mode": args.mode,
            "predicted_letter": pred_letter,
            "correct_letter": item["correct_letter"],
            "is_correct": bool(pred_letter == item["correct_letter"]),
            "npz_path": str(npz_path),
            "meta": meta,
            "mapping_summary": summary,
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