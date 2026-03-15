#!/usr/bin/env python3
"""
Prefill-step full prompt self-attention caching (L,S,S) + token mappings.

Saves:
- attn_LSS: (L,S,S) float16 full prompt self-attention (head-averaged)
- token_map: per-position label of what each prompt token "is"
- image mappings:
    base/mosaic patch token indices in packed-image space + bbox membership masks:
      - in_text_bbox
      - in_corr_bbox
      - in_mis_bbox
- question mappings:
    token spans for each option:
      - correct_option
      - misleading_groundable_option
      - misleading_ungroundable_option
      - irrelevant_word_option
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List, Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution

# -----------------------------
# Dataset utils (same as yours)
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
# Prompt with delimiters
# -----------------------------
def format_mcq_prompt_with_markers(question: str, options: Dict[str, str]) -> str:
    # Delimit each option so we can locate exact token spans post-tokenization
    instruction = "Answer the following multiple-choice question by selecting the correct option."
    prompt = f"{instruction}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for k in ["A", "B", "C", "D"]:
        if k in options:
            prompt += f"{k}) <<{k}>> {options[k]} <</{k}>>\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt

def build_options_from_sample(sample: Dict, shuffle: bool = False, seed: int = 0) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
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
# Token span location (robust)
# -----------------------------
def find_subsequence(haystack: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) of first occurrence of needle in haystack, else None."""
    if len(needle) == 0:
        return None
    H, N = len(haystack), len(needle)
    for i in range(H - N + 1):
        if haystack[i:i+N] == needle:
            return (i, i+N)
    return None

def find_between_markers(
    tokenizer,
    input_ids: List[int],
    start_marker: str,
    end_marker: str,
) -> Optional[Tuple[int, int]]:
    """
    Find token span strictly between start_marker and end_marker (exclusive).
    We find the marker token sequences in input_ids, then return interior span.
    """
    start_ids = tokenizer.encode(start_marker, add_special_tokens=False)
    end_ids   = tokenizer.encode(end_marker, add_special_tokens=False)

    s = find_subsequence(input_ids, start_ids)
    if s is None:
        return None
    # search end after start
    tail = input_ids[s[1]:]
    e_rel = find_subsequence(tail, end_ids)
    if e_rel is None:
        return None
    e = (s[1] + e_rel[0], s[1] + e_rel[1])  # start/end of end_marker
    inner_start = s[1]
    inner_end = e[0]
    if inner_end <= inner_start:
        return None
    return (inner_start, inner_end)


# -----------------------------
# Image placeholder positions
# -----------------------------
def get_image_placeholder_positions(model, input_ids: torch.Tensor) -> List[int]:
    image_token_id = model.config.image_token_id
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return pos.tolist()


# -----------------------------
# Packed mapping (yours, unchanged)
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
    original_size_hw: Tuple[int, int],
    num_views_provided: int,
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
        "tokens": tokens
    }


# -----------------------------
# GUIC bbox helpers (for region membership)
# -----------------------------
def guic_text_bbox_to_yxyx(b: List[float]) -> Tuple[float, float, float, float]:
    # [x1,y1,x2,y2] -> (y0,x0,y1,x1)
    x1, y1, x2, y2 = map(float, b)
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return (y1, x1, y2, x2)

def guic_obj_bbox_to_yxyx(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, (x, y, w, h))
    y0, x0, y1, x1 = (y, x, y + h, x + w)
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return (y0, x0, y1, x1)

def get_guic_bboxes(sample: dict, variant: str) -> Tuple[Optional[Tuple[float,float,float,float]],
                                                         Optional[Tuple[float,float,float,float]],
                                                         Optional[Tuple[float,float,float,float]]]:
    bbox_text = None
    if variant != "notext":
        if variant in sample and "bbox" in sample[variant]:
            bbox_text = guic_text_bbox_to_yxyx(sample[variant]["bbox"])
    else:
        # no text overlay in notext; keep None
        bbox_text = None

    bbox_corr = None
    if "correct_answer" in sample and all(k in sample["correct_answer"] for k in ("x","y","w","h")):
        bbox_corr = guic_obj_bbox_to_yxyx(sample["correct_answer"]["x"], sample["correct_answer"]["y"],
                                          sample["correct_answer"]["w"], sample["correct_answer"]["h"])

    bbox_mis = None
    if "misleading_groundable" in sample and all(k in sample["misleading_groundable"] for k in ("x","y","w","h")):
        bbox_mis = guic_obj_bbox_to_yxyx(sample["misleading_groundable"]["x"], sample["misleading_groundable"]["y"],
                                         sample["misleading_groundable"]["w"], sample["misleading_groundable"]["h"])
    return bbox_text, bbox_corr, bbox_mis

def bbox_intersection_area(bb1, bb2) -> float:
    y0 = max(bb1[0], bb2[0]); x0 = max(bb1[1], bb2[1])
    y1 = min(bb1[2], bb2[2]); x1 = min(bb1[3], bb2[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return (y1-y0)*(x1-x0)

def bbox_area(bb) -> float:
    return max(0.0, bb[2]-bb[0]) * max(0.0, bb[3]-bb[1])

def token_in_region(mapping_token: dict, region_bbox, assign: str, thr: float) -> bool:
    bb = mapping_token.get("bbox")
    if bb is None or region_bbox is None:
        return False
    bb = tuple(map(float, bb))
    if assign == "center":
        cy = 0.5*(bb[0]+bb[2]); cx = 0.5*(bb[1]+bb[3])
        return (region_bbox[1] <= cx <= region_bbox[3]) and (region_bbox[0] <= cy <= region_bbox[2])
    elif assign == "overlap":
        a = bbox_area(bb)
        if a <= 0:
            return False
        frac = bbox_intersection_area(bb, region_bbox) / a
        return frac >= thr
    else:
        raise ValueError("assign must be center|overlap")


# -----------------------------
# Prefill self-attn extraction (same, but we KEEP full)
# -----------------------------
@torch.no_grad()
def prefill_head_avg_attn_prompt_self(model, inputs) -> np.ndarray:
    out = model(**inputs, use_cache=False, output_attentions=True, return_dict=True)
    attns = out.attentions  # tuple(L) of (B,H,S,S)
    if attns is None or len(attns) == 0:
        raise RuntimeError("No attentions returned. Ensure attn_implementation='eager' at model load.")
    layer_mats = []
    for layer_attn in attns:
        A = layer_attn[0].mean(dim=0)  # (S,S)
        layer_mats.append(A)
    A = torch.stack(layer_mats, dim=0)  # (L,S,S)
    return A.to(torch.float16).cpu().numpy()


def split_image_token_indices(mapping_tokens: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    base_idx: List[int] = []
    mosaic_idx: List[int] = []
    for t in mapping_tokens:
        k = t["kind"]
        i = int(t["token_idx"])
        if k == "base_patch":
            base_idx.append(i)
        elif k == "mosaic_patch":
            mosaic_idx.append(i)
    return np.asarray(base_idx, dtype=np.int64), np.asarray(mosaic_idx, dtype=np.int64)


# -----------------------------
# Build full token mapping for prompt positions [0..S-1]
# -----------------------------
def build_prompt_token_map(
    *,
    tokenizer,
    input_ids_1d: List[int],
    img_pos: List[int],
    packed_mapping_tokens: List[Dict[str, Any]],
    sample: dict,
    variant: str,
    option_meta: dict,
    correct_letter: str,
    assign: str,
    thr: float,
) -> Dict[str, Any]:
    """
    Returns:
      - token_type: int8 array length S
      - token_type_legend
      - option_spans: dict of spans
      - image_region_masks: dict of boolean masks over prompt positions for each region & base/mosaic
      - packed_mapping_tokens: saved separately too
    """
    S = len(input_ids_1d)
    img_pos = list(img_pos)
    img_pos_set = set(img_pos)

    # token_type codes (extend as needed)
    # 0=other_text, 1=image_token, 2=option_correct, 3=option_mg, 4=option_mu, 5=option_irrel
    token_type = np.zeros((S,), dtype=np.int8)

    # Mark image token positions
    for p in img_pos:
        if 0 <= p < S:
            token_type[p] = 1

    # Find option spans using markers
    # We expect prompt contains: <<A>> ... <</A>> etc
    spans = {}
    for L in ["A", "B", "C", "D"]:
        sp = find_between_markers(tokenizer, input_ids_1d, f"<<{L}>>", f"<</{L}>>")
        spans[L] = sp

    # Map label->key and compute which label corresponds to which option type
    # option_meta["label_to_key"]: { "A": "correct_answer" | ... }
    label_to_key = option_meta.get("label_to_key", {})
    # Determine which letter corresponds to each semantic option type
    key_to_label = {v: k for k, v in label_to_key.items()}

    label_correct = key_to_label.get("correct_answer", "")
    label_mg = key_to_label.get("misleading_groundable", "")
    label_mu = key_to_label.get("misleading_ungroundable", "")
    label_ir = key_to_label.get("irrelevant_word", "")

    option_spans_named = {
        "correct": spans.get(label_correct),
        "misleading_groundable": spans.get(label_mg),
        "misleading_ungroundable": spans.get(label_mu),
        "irrelevant_word": spans.get(label_ir),
        "by_letter": spans,
        "correct_letter": correct_letter,
        "label_to_key": label_to_key,
    }

    # Apply token_type codes to option spans
    def paint(span, code):
        if span is None:
            return
        a, b = span
        a = max(0, a); b = min(S, b)
        if b > a:
            token_type[a:b] = code

    paint(option_spans_named["correct"], 2)
    paint(option_spans_named["misleading_groundable"], 3)
    paint(option_spans_named["misleading_ungroundable"], 4)
    paint(option_spans_named["irrelevant_word"], 5)

    # Image region membership masks over prompt positions
    bbox_text, bbox_corr, bbox_mis = get_guic_bboxes(sample, variant)

    # Build masks over PACKED image-token space [0..N-1] first
    N = len(img_pos)
    in_text = np.zeros((N,), dtype=bool)
    in_corr = np.zeros((N,), dtype=bool)
    in_mis  = np.zeros((N,), dtype=bool)
    kind_is_base = np.zeros((N,), dtype=bool)
    kind_is_mosaic = np.zeros((N,), dtype=bool)

    # packed_mapping_tokens[j] corresponds to packed image token j (0..N-1)
    for j, t in enumerate(packed_mapping_tokens):
        k = t.get("kind", "")
        if k == "base_patch":
            kind_is_base[j] = True
        elif k == "mosaic_patch":
            kind_is_mosaic[j] = True

        if token_in_region(t, bbox_text, assign, thr):
            in_text[j] = True
        if token_in_region(t, bbox_corr, assign, thr):
            in_corr[j] = True
        if token_in_region(t, bbox_mis, assign, thr):
            in_mis[j] = True

    # Convert packed-image masks -> prompt-position masks (length S)
    def packed_mask_to_prompt_mask(packed_mask: np.ndarray) -> np.ndarray:
        pm = np.zeros((S,), dtype=bool)
        for j in range(N):
            if packed_mask[j]:
                pos = img_pos[j]
                if 0 <= pos < S:
                    pm[pos] = True
        return pm

    image_region_masks = {
        # base
        "base_text":  packed_mask_to_prompt_mask(kind_is_base & in_text),
        "base_corr":  packed_mask_to_prompt_mask(kind_is_base & in_corr),
        "base_mis":   packed_mask_to_prompt_mask(kind_is_base & in_mis),
        # mosaic
        "mosaic_text": packed_mask_to_prompt_mask(kind_is_mosaic & in_text),
        "mosaic_corr": packed_mask_to_prompt_mask(kind_is_mosaic & in_corr),
        "mosaic_mis":  packed_mask_to_prompt_mask(kind_is_mosaic & in_mis),
        # also useful:
        "img_text_all": packed_mask_to_prompt_mask(in_text),
        "img_corr_all": packed_mask_to_prompt_mask(in_corr),
        "img_mis_all":  packed_mask_to_prompt_mask(in_mis),
        "img_base_all": packed_mask_to_prompt_mask(kind_is_base),
        "img_mosaic_all": packed_mask_to_prompt_mask(kind_is_mosaic),
    }

    token_type_legend = {
        "0": "other_text",
        "1": "image_token",
        "2": "option_correct",
        "3": "option_misleading_groundable",
        "4": "option_misleading_ungroundable",
        "5": "option_irrelevant_word",
    }

    return {
        "token_type": token_type,
        "token_type_legend": token_type_legend,
        "option_spans": option_spans_named,
        "image_region_masks": {k: v.astype(np.uint8) for k, v in image_region_masks.items()},
    }


# -----------------------------
# Save NPZ (full attention + maps)
# -----------------------------
def save_npz_sample(
    out_dir: str,
    qid: str,
    variant: str,
    *,
    meta: Dict[str, Any],
    packed_mapping: Dict[str, Any],
    image_placeholder_positions: List[int],
    prompt_input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
    attn_LSS: np.ndarray,
    token_type: np.ndarray,
    token_type_legend: Dict[str, str],
    option_spans: Dict[str, Any],
    image_region_masks: Dict[str, np.ndarray],
    mode: str = "prefill_full_attn",
):
    sample_dir = Path(out_dir) / variant / qid
    sample_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sample_dir / f"{mode}.npz"

    save_dict: Dict[str, Any] = {
        "meta": json.dumps(meta),
        "packed_mapping_summary": json.dumps(packed_mapping["summary"]),
        "packed_mapping_tokens": json.dumps(packed_mapping["tokens"]),
        "image_placeholder_positions": np.asarray(image_placeholder_positions, dtype=np.int32),
        "prompt_input_ids": prompt_input_ids.astype(np.int32),
        "token_type": token_type.astype(np.int8),
        "token_type_legend": json.dumps(token_type_legend),
        "option_spans": json.dumps(option_spans),
        # full prompt self-attn
        "attn_LSS": attn_LSS.astype(np.float16),
    }

    # region masks as uint8 arrays of length S
    for k, v in image_region_masks.items():
        save_dict[f"mask_{k}"] = v.astype(np.uint8)

    if attention_mask is not None:
        save_dict["attention_mask"] = attention_mask.astype(np.int32)

    np.savez_compressed(str(npz_path), **save_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="attn_cache_prefill_full")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # prompt consistency
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # mapping
    parser.add_argument("--include_newline_tokens", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="End index (exclusive). 0 = no limit.")

    # bbox->token assignment
    parser.add_argument("--assign", choices=["center", "overlap"], default="center")
    parser.add_argument("--thr", type=float, default=0.25)

    args = parser.parse_args()

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
        attn_implementation="eager",
    ).to(device_t)
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching full prompt attn [{args.variant}]"):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        sample = ds[i]
        qid = str(sample.get("question_id", f"unknown_{i}"))
        if qid not in qids:
            continue

        # choose image variant
        if args.variant == "notext":
            img = sample["notext"]["image"]
        else:
            if args.variant not in sample:
                continue
            img = sample[args.variant]["image"]
        img = img.convert("RGB")

        # build prompt with markers
        options, correct_letter, option_meta = build_options_from_sample(
            sample, shuffle=args.shuffle_options, seed=args.seed
        )
        prompt_text = format_mcq_prompt_with_markers(sample.get("question", ""), options)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(images=img, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)

        input_ids = inputs["input_ids"]
        S = int(input_ids.shape[1])

        # image placeholder positions in prompt
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

        expected_img_tokens = int(packed_mapping["summary"]["total_packed_image_tokens"])
        if len(img_pos) != expected_img_tokens:
            print(f"[{qid}] Packed mapping mismatch: len(img_pos)={len(img_pos)} != expected={expected_img_tokens}. Skipping.")
            continue

        # full prompt self-attn (L,S,S)
        attn_LSS = prefill_head_avg_attn_prompt_self(model, inputs)

        # prompt token map
        input_ids_1d = input_ids[0].detach().cpu().tolist()
        token_map = build_prompt_token_map(
            tokenizer=tokenizer,
            input_ids_1d=input_ids_1d,
            img_pos=img_pos,
            packed_mapping_tokens=packed_mapping["tokens"],
            sample=sample,
            variant=args.variant,
            option_meta=option_meta,
            correct_letter=correct_letter,
            assign=args.assign,
            thr=args.thr,
        )

        meta = {
            "question_id": qid,
            "variant": args.variant,
            "prompt_seq_len": int(S),
            "image_placeholder_count": len(img_pos),
            "correct_letter": correct_letter,
            "option_meta": option_meta,
            "original_image_size_hw": [orig_h, orig_w],
            "num_views_provided": num_views,
            "include_newline_tokens": bool(args.include_newline_tokens),
            "num_layers": int(attn_LSS.shape[0]),
        }

        prompt_input_ids = input_ids[0].detach().cpu().numpy()
        attention_mask = inputs["attention_mask"][0].detach().cpu().numpy() if "attention_mask" in inputs else None

        save_npz_sample(
            args.out_dir,
            qid,
            args.variant,
            meta=meta,
            packed_mapping=packed_mapping,
            image_placeholder_positions=img_pos,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            attn_LSS=attn_LSS,
            token_type=token_map["token_type"],
            token_type_legend=token_map["token_type_legend"],
            option_spans=token_map["option_spans"],
            image_region_masks=token_map["image_region_masks"],
            mode="prefill_full_attn",
        )

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Cached {kept} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()