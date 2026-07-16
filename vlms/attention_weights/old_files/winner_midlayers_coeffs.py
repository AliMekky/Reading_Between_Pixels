#!/usr/bin/env python3
"""
Compute per-layer region "winner" using the SAME metric as your plot:
density_ratio(region/view) = (avg attn per region token) / (avg attn per view token)

Supports:
  --view base     (denominator = base_patch tokens; region tokens restricted to base_patch)
  --view mosaic   (denominator = mosaic_patch tokens; region tokens restricted to mosaic_patch)

Assumes your cached NPZ structure:
  - attn: (L, S_prompt)
  - packed_mapping_tokens: json list of packed tokens with kind/row/col/bbox/token_idx
  - image_placeholder_positions: list mapping packed token_idx -> prompt position
  - meta: json string (optional; used for qid/variant)

You must provide a region-token source:
  A) If you already have per-sample region token lists saved somewhere, plug them in.
  B) Otherwise you must define how to build region tokens from dataset bboxes.
     This script includes a placeholder hook `get_region_bboxes_from_dataset(...)`.
"""

import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from datasets import load_from_disk, load_dataset


# -----------------------------
# Helpers: mapping / token sets
# -----------------------------
def packed_token_indices(mapping_tokens: List[Dict[str, Any]], kind_set: set) -> np.ndarray:
    return np.asarray(
        [int(t["token_idx"]) for t in mapping_tokens if t.get("kind") in kind_set],
        dtype=np.int64
    )

def tokenidx_to_promptpos(img_pos: np.ndarray, token_idx_arr: np.ndarray) -> np.ndarray:
    # token_idx indexes into packed image tokens; img_pos[token_idx] gives prompt position
    token_idx_arr = token_idx_arr[(token_idx_arr >= 0) & (token_idx_arr < len(img_pos))]
    return img_pos[token_idx_arr]

def mean_attn_on_tokens(attn_layer_vec: np.ndarray, prompt_positions: np.ndarray) -> float:
    if prompt_positions.size == 0:
        return 0.0
    return float(attn_layer_vec[prompt_positions].mean())

def density_ratio(attn_layer_vec: np.ndarray,
                  img_pos: np.ndarray,
                  view_token_idx: np.ndarray,
                  region_token_idx: np.ndarray) -> float:
    """
    density_ratio(region/view) at a single layer.
    view_token_idx: packed token_idx for ALL tokens in the chosen view (base_patch or mosaic_patch)
    region_token_idx: packed token_idx for region tokens restricted to same view
    """
    view_pp = tokenidx_to_promptpos(img_pos, view_token_idx)
    reg_pp  = tokenidx_to_promptpos(img_pos, region_token_idx)

    view_mean = mean_attn_on_tokens(attn_layer_vec, view_pp)
    reg_mean  = mean_attn_on_tokens(attn_layer_vec, reg_pp)

    if view_mean <= 1e-12:
        return 0.0
    return reg_mean / view_mean

def restrict_to_view(region_token_idx: np.ndarray, view_token_idx: np.ndarray) -> np.ndarray:
    view_set = set(view_token_idx.tolist())
    return np.asarray([t for t in region_token_idx.tolist() if t in view_set], dtype=np.int64)


# -----------------------------
# Region token building (bbox -> token_idx)
# -----------------------------
def bbox_contains_point(bbox_yxyx: Tuple[float, float, float, float], y: float, x: float) -> bool:
    y0, x0, y1, x1 = bbox_yxyx
    return (y >= y0) and (y <= y1) and (x >= x0) and (x <= x1)

def token_center_from_bbox(bb: Tuple[float, float, float, float]) -> Tuple[float, float]:
    y0, x0, y1, x1 = bb
    return (0.5 * (y0 + y1), 0.5 * (x0 + x1))

def tokens_in_bbox(mapping_tokens: List[Dict[str, Any]],
                   bbox_yxyx: Tuple[float, float, float, float],
                   allowed_kinds: set) -> np.ndarray:
    """
    Select packed token_idx whose token bbox center lies inside bbox_yxyx.
    allowed_kinds should be {"base_patch"} or {"mosaic_patch"} or both.
    """
    out = []
    for t in mapping_tokens:
        if t.get("kind") not in allowed_kinds:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue
        cy, cx = token_center_from_bbox(tuple(bb))
        if bbox_contains_point(bbox_yxyx, cy, cx):
            out.append(int(t["token_idx"]))
    return np.asarray(out, dtype=np.int64)

def to_yxyx_from_xywh(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    # dataset object bbox is (x,y,w,h)
    return (float(y), float(x), float(y + h), float(x + w))

def get_region_bboxes_from_dataset(sample: Dict[str, Any],
                                  variant: str) -> Tuple[Tuple[float,float,float,float],
                                                         Optional[Tuple[float,float,float,float]],
                                                         Optional[Tuple[float,float,float,float]]]:
    """
    Returns:
      text_bbox_yxyx, corr_bbox_yxyx (or None), mis_bbox_yxyx (or None)

    GUIC schema:
      - text bbox: sample[variant]["bbox"] as [x1,y1,x2,y2] OR [y1,x1,y2,x2]?
        In your description it is [x1,y1,x2,y2]. We'll convert to yxyx here.
      - corr object bbox exists in correct_answer: x,y,w,h
      - mis object bbox exists in misleading_groundable: x,y,w,h
    """
    # text bbox: [x1,y1,x2,y2] -> yxyx
    if variant == "notext":
        xb = sample["correct_answer"]["bbox"]
    else:
        xb = sample[variant]["bbox"]
    x1, y1, x2, y2 = map(float, xb)
    text_bbox = (y1, x1, y2, x2)

    corr_bbox = None
    mis_bbox = None

    # correct object region uses correct_answer object bbox regardless of current variant
    if "correct_answer" in sample and all(k in sample["correct_answer"] for k in ["x","y","w","h"]):
        corr_bbox = to_yxyx_from_xywh(sample["correct_answer"]["x"],
                                      sample["correct_answer"]["y"],
                                      sample["correct_answer"]["w"],
                                      sample["correct_answer"]["h"])

    # misleading object region uses misleading_groundable object bbox (only if exists)
    if "misleading_groundable" in sample and all(k in sample["misleading_groundable"] for k in ["x","y","w","h"]):
        mis_bbox = to_yxyx_from_xywh(sample["misleading_groundable"]["x"],
                                     sample["misleading_groundable"]["y"],
                                     sample["misleading_groundable"]["w"],
                                     sample["misleading_groundable"]["h"])

    return text_bbox, corr_bbox, mis_bbox


# -----------------------------
# Winner computation
# -----------------------------
def layer_winner(attn: np.ndarray,
                 img_pos: np.ndarray,
                 mapping_tokens: List[Dict[str, Any]],
                 view: str,
                 text_tok: np.ndarray,
                 corr_tok: np.ndarray,
                 mis_tok: np.ndarray,
                 layer_idx: int) -> Tuple[str, Dict[str, float]]:
    if view == "base":
        view_token_idx = packed_token_indices(mapping_tokens, {"base_patch"})
        allowed_kinds = {"base_patch"}
    elif view == "mosaic":
        view_token_idx = packed_token_indices(mapping_tokens, {"mosaic_patch"})
        allowed_kinds = {"mosaic_patch"}
    else:
        raise ValueError(f"Unknown view={view}")

    # Restrict region tokens to view
    text_tok_v = restrict_to_view(text_tok, view_token_idx)
    corr_tok_v = restrict_to_view(corr_tok, view_token_idx)
    mis_tok_v  = restrict_to_view(mis_tok,  view_token_idx)

    v = attn[layer_idx]  # (S_prompt,)
    ratios = {
        "text": density_ratio(v, img_pos, view_token_idx, text_tok_v),
        "corr": density_ratio(v, img_pos, view_token_idx, corr_tok_v),
        "mis":  density_ratio(v, img_pos, view_token_idx, mis_tok_v),
    }
    winner = max(ratios.items(), key=lambda kv: kv[1])[0]
    return winner, ratios


# -----------------------------
# IO
# -----------------------------
def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    z = np.load(npz_path, allow_pickle=True)
    attn = z["attn"].astype(np.float32)  # (L, S_prompt)
    img_pos = z["image_placeholder_positions"].astype(np.int64)  # length N_img_tokens
    mapping_tokens = json.loads(str(z["packed_mapping_tokens"]))
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return attn, img_pos, mapping_tokens, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_root", type=str, default="llava-next_attentions",
                    help="Root like attn_cache_gen (contains variant/qid/gen_attn_gen_token.npz)")
    ap.add_argument("--variant", type=str, default="misleading_ungroundable",
                    choices=["correct_answer","misleading_groundable","misleading_ungroundable","irrelevant_word","notext"])
    ap.add_argument("--view", type=str, default="base", choices=["base","mosaic"],
                    help="Which view to compute winners on: base_patch or mosaic_patch")
    ap.add_argument("--layers", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                    help="Comma-separated layer indices (0-based) to compute winners for")
    ap.add_argument("--hf_dataset", type=str, default="../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC",
                    help="HF dataset id OR a path if using --dataset_from_disk")
    ap.add_argument("--dataset_from_disk", action="store_true",
                    help="If set, treat --hf_dataset as a path for load_from_disk")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt",
                    help="Optional file with allowed qids (one per line). If omitted, uses all found npzs.")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="winners_by_layer.csv")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    # load dataset for bboxes
    if args.dataset_from_disk:
        ds = load_from_disk(args.hf_dataset)
    else:
        ds = load_dataset(args.hf_dataset, split=args.split)

    # optional whitelist
    whitelist = None
    if args.qid_file:
        whitelist = set(Path(args.qid_file).read_text().splitlines())

    attn_root = Path(args.attn_root)
    variant_dir = attn_root / args.variant / args.variant
    if not variant_dir.exists():
        raise FileNotFoundError(f"Variant dir not found: {variant_dir}")

    # index dataset by qid for fast lookup
    qid_to_sample = {}
    for s in ds:
        qid_to_sample[str(s["question_id"])] = s

    rows = []
    kept = 0

    for qid_dir in sorted([d for d in variant_dir.iterdir() if d.is_dir()]):
        qid = qid_dir.name
        if whitelist is not None and qid not in whitelist:
            continue
        if qid not in qid_to_sample:
            continue

        npz_path = qid_dir / "gen_attn_gen_token.npz"
        if not npz_path.exists():
            # fallback name from your script
            npz_path = qid_dir / "gen_attn_gen_token.npz"
        if not npz_path.exists():
            continue

        attn, img_pos, mapping_tokens, meta = load_npz(npz_path)

        sample = qid_to_sample[qid]
        text_bbox, corr_bbox, mis_bbox = get_region_bboxes_from_dataset(sample, args.variant)

        # Build region tokens using the SAME view restriction logic as the plot:
        # start by selecting tokens inside bboxes, allowing BOTH kinds, then restrict by view later.
        text_tok = tokens_in_bbox(mapping_tokens, text_bbox, {"base_patch","mosaic_patch"})

        if corr_bbox is None:
            corr_tok = np.asarray([], dtype=np.int64)
        else:
            corr_tok = tokens_in_bbox(mapping_tokens, corr_bbox, {"base_patch","mosaic_patch"})

        if mis_bbox is None:
            mis_tok = np.asarray([], dtype=np.int64)
        else:
            mis_tok = tokens_in_bbox(mapping_tokens, mis_bbox, {"base_patch","mosaic_patch"})

        for L in layers:
            if L < 0 or L >= attn.shape[0]:
                continue
            winner, ratios = layer_winner(
                attn=attn,
                img_pos=img_pos,
                mapping_tokens=mapping_tokens,
                view=args.view,
                text_tok=text_tok,
                corr_tok=corr_tok,
                mis_tok=mis_tok,
                layer_idx=L
            )
            rows.append({
                "question_id": qid,
                "variant": args.variant,
                "view": args.view,
                "layer": L,
                "winner": winner,
                "ratio_text": ratios["text"],
                "ratio_corr": ratios["corr"],
                "ratio_mis": ratios["mis"],
                "generated_token_text": meta.get("generated_token_text", ""),
                "generated_token_id": meta.get("generated_token_id", -1),
                "correct_letter": meta.get("correct_letter", ""),
            })

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    # save csv
    import csv
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()