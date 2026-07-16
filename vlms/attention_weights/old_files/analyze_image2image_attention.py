# analyze_prefill_img_self_attn_regions.py

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk


# ----------------------------
# BBox conversions (GUIC -> yxyx)
# ----------------------------
def guic_text_bbox_to_yxyx(b):  # [x1,y1,x2,y2]
    x1, y1, x2, y2 = b
    return (float(y1), float(x1), float(y2), float(x2))


def guic_obj_bbox_to_yxyx(x, y, w, h):  # (x,y,w,h)
    return (float(y), float(x), float(y + h), float(x + w))


# ----------------------------
# Load NPZ and mapping
# ----------------------------
def load_npz(npz_path: Path) -> Dict[str, Any]:
    z = np.load(str(npz_path), allow_pickle=True)

    meta = json.loads(str(z["meta"]))
    summary = json.loads(str(z["packed_mapping_summary"]))
    tokens = json.loads(str(z["packed_mapping_tokens"]))

    base_idx = z["base_token_idx"].astype(np.int64)      # packed token_idx list for base_patch tokens
    mosaic_idx = z["mosaic_token_idx"].astype(np.int64)  # packed token_idx list for mosaic_patch tokens

    base_attn = z["base_attn_LBB"].astype(np.float32)    # (L,B,B)
    mosaic_attn = z["mosaic_attn_LMM"].astype(np.float32)# (L,M,M)
    img_attn = z["img_attn_LNN"].astype(np.float32) if "img_attn_LNN" in z else None  # (L,N,N) or None

    out = {
        "meta": meta,
        "summary": summary,
        "tokens": tokens,
        "base_idx": base_idx,
        "mosaic_idx": mosaic_idx,
        "base_attn": base_attn,
        "mosaic_attn": mosaic_attn,
        "img_attn": img_attn,
    }
    return out


# ----------------------------
# Token selection by overlap with region bbox
# ----------------------------
def token_region_indices(
    mapping_tokens: List[Dict[str, Any]],
    region_yxyx: Tuple[float, float, float, float],
    *,
    kind: str,              # "base_patch" or "mosaic_patch"
    min_cover: float = 0.1, # intersection / token_area
) -> np.ndarray:
    ry0, rx0, ry1, rx1 = region_yxyx
    out: List[int] = []

    for t in mapping_tokens:
        if t.get("kind") != kind:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue

        ty0, tx0, ty1, tx1 = map(float, bb)

        # intersection
        iy0, ix0 = max(ry0, ty0), max(rx0, tx0)
        iy1, ix1 = min(ry1, ty1), min(rx1, tx1)
        ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
        inter = ih * iw

        token_area = max(0.0, ty1 - ty0) * max(0.0, tx1 - tx0) + 1e-12
        cover = inter / token_area

        if cover >= min_cover:
            out.append(int(t["token_idx"]))  # packed token_idx (0..N-1)

    return np.asarray(out, dtype=np.int64)


def packed_to_local(packed_idx: np.ndarray, packed_list: np.ndarray) -> np.ndarray:
    """
    Convert packed token indices (0..N-1) to local indices within base_attn (0..B-1)
    or mosaic_attn (0..M-1), given packed_list = base_token_idx or mosaic_token_idx.
    """
    pos = {int(v): i for i, v in enumerate(packed_list.tolist())}
    local = [pos[int(x)] for x in packed_idx.tolist() if int(x) in pos]
    return np.asarray(local, dtype=np.int64)


# ----------------------------
# Attention metrics
# ----------------------------
def mass_region_to_region_L(A_LNN: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """
    Sum over src x dst for each layer.
    A_LNN: (L,N,N) or (L,B,B) or (L,M,M) depending on context.
    """
    L = A_LNN.shape[0]
    if src_idx.size == 0 or dst_idx.size == 0:
        return np.zeros((L,), dtype=np.float32)
    sub = A_LNN[:, src_idx][:, :, dst_idx]  # (L,|src|,|dst|)
    return sub.sum(axis=(1, 2)).astype(np.float32)


def normalized_outflow_L(A_LNN: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray) -> np.ndarray:
    """
    mass(src->dst) / mass(src->all_image_tokens_in_matrix)
    For A_LNN restricted to image tokens, "all_image_tokens_in_matrix" is just all columns in A_LNN.
    """
    L = A_LNN.shape[0]
    if src_idx.size == 0:
        return np.zeros((L,), dtype=np.float32)
    out_total = A_LNN[:, src_idx, :].sum(axis=(1, 2)) + 1e-12
    out_dst = mass_region_to_region_L(A_LNN, src_idx, dst_idx)
    return (out_dst / out_total).astype(np.float32)


def safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (a / (b + eps)).astype(np.float32)


# ----------------------------
# Main analysis per sample
# ----------------------------
def compute_sample_metrics(
    *,
    qid: str,
    variant: str,
    npz: Dict[str, Any],
    guic_text_bbox_xyxy: List[float],  # [x1,y1,x2,y2] for THIS variant
    guic_correct_obj_xywh: Tuple[float, float, float, float],     # (x,y,w,h) from correct_answer
    guic_mislead_obj_xywh: Tuple[float, float, float, float],     # (x,y,w,h) from misleading_groundable
    min_cover: float,
) -> pd.DataFrame:
    tokens = npz["tokens"]
    base_idx_list = npz["base_idx"]
    mosaic_idx_list = npz["mosaic_idx"]

    base_attn = npz["base_attn"]      # (L,B,B)
    mosaic_attn = npz["mosaic_attn"]  # (L,M,M)
    img_attn = npz["img_attn"]        # (L,N,N) or None

    L = int(base_attn.shape[0])

    # Regions in yxyx
    text_yxyx = guic_text_bbox_to_yxyx(guic_text_bbox_xyxy)
    corr_yxyx = guic_obj_bbox_to_yxyx(*guic_correct_obj_xywh)
    misl_yxyx = guic_obj_bbox_to_yxyx(*guic_mislead_obj_xywh)

    # Packed token indices by region (base/mosaic separately)
    text_base_p = token_region_indices(tokens, text_yxyx, kind="base_patch", min_cover=min_cover)
    corr_base_p = token_region_indices(tokens, corr_yxyx, kind="base_patch", min_cover=min_cover)
    misl_base_p = token_region_indices(tokens, misl_yxyx, kind="base_patch", min_cover=min_cover)

    text_mos_p = token_region_indices(tokens, text_yxyx, kind="mosaic_patch", min_cover=min_cover)
    corr_mos_p = token_region_indices(tokens, corr_yxyx, kind="mosaic_patch", min_cover=min_cover)
    misl_mos_p = token_region_indices(tokens, misl_yxyx, kind="mosaic_patch", min_cover=min_cover)

    # Convert to local indices for base/mosaic matrices
    text_base = packed_to_local(text_base_p, base_idx_list)
    corr_base = packed_to_local(corr_base_p, base_idx_list)
    misl_base = packed_to_local(misl_base_p, base_idx_list)

    text_mos = packed_to_local(text_mos_p, mosaic_idx_list)
    corr_mos = packed_to_local(corr_mos_p, mosaic_idx_list)
    misl_mos = packed_to_local(misl_mos_p, mosaic_idx_list)

    rows = []

    def add_block(block_name: str, A: np.ndarray, src_text, src_corr, src_misl, dst_text, dst_corr, dst_misl):
        # raw masses
        t2c = mass_region_to_region_L(A, src_text, dst_corr)
        t2m = mass_region_to_region_L(A, src_text, dst_misl)
        t2t = mass_region_to_region_L(A, src_text, dst_text)

        c2t = mass_region_to_region_L(A, src_corr, dst_text)
        m2t = mass_region_to_region_L(A, src_misl, dst_text)

        c2c = mass_region_to_region_L(A, src_corr, dst_corr)
        m2m = mass_region_to_region_L(A, src_misl, dst_misl)

        # normalized outflow fractions
        nt2c = normalized_outflow_L(A, src_text, dst_corr)
        nt2m = normalized_outflow_L(A, src_text, dst_misl)
        nt2t = normalized_outflow_L(A, src_text, dst_text)

        # comparisons
        ratio_t_m_over_c = safe_ratio(t2m, t2c)
        ratio_nt_m_over_c = safe_ratio(nt2m, nt2c)

        for l in range(A.shape[0]):
            rows.append({
                "qid": qid,
                "variant": variant,
                "block": block_name,
                "layer": l,

                "raw_text_to_correct": float(t2c[l]),
                "raw_text_to_mislead": float(t2m[l]),
                "raw_text_to_text": float(t2t[l]),
                "raw_correct_to_text": float(c2t[l]),
                "raw_mislead_to_text": float(m2t[l]),
                "raw_correct_to_correct": float(c2c[l]),
                "raw_mislead_to_mislead": float(m2m[l]),

                "norm_text_to_correct": float(nt2c[l]),
                "norm_text_to_mislead": float(nt2m[l]),
                "norm_text_to_text": float(nt2t[l]),

                "ratio_raw_text_mislead_over_correct": float(ratio_t_m_over_c[l]),
                "ratio_norm_text_mislead_over_correct": float(ratio_nt_m_over_c[l]),

                "n_text_tokens": int(src_text.size),
                "n_correct_tokens": int(src_corr.size),
                "n_mislead_tokens": int(src_misl.size),
            })

    # Base-only block
    add_block("base", base_attn, text_base, corr_base, misl_base, text_base, corr_base, misl_base)

    # Mosaic-only block
    add_block("mosaic", mosaic_attn, text_mos, corr_mos, misl_mos, text_mos, corr_mos, misl_mos)

    # Full packed image-token block, if present (enables cross base<->mosaic effects)
    if img_attn is not None:
        # Define “all tokens for each region” in packed space (N indexing)
        text_all = np.unique(np.concatenate([text_base_p, text_mos_p])).astype(np.int64)
        corr_all = np.unique(np.concatenate([corr_base_p, corr_mos_p])).astype(np.int64)
        misl_all = np.unique(np.concatenate([misl_base_p, misl_mos_p])).astype(np.int64)

        # Also compute cross (text_base -> text_mosaic) style interactions if you want later
        add_block("full", img_attn, text_all, corr_all, misl_all, text_all, corr_all, misl_all)

    return pd.DataFrame(rows)


# ----------------------------
# Aggregation
# ----------------------------
def aggregate_per_layer(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate numeric columns except identifiers
    id_cols = {"qid", "variant", "block", "layer"}
    value_cols = [c for c in df.columns if c not in id_cols and df[c].dtype != object]

    agg = (
        df.groupby(["variant", "block", "layer"], as_index=False)[value_cols]
          .agg(["mean", "std", "count"])
    )

    # flatten columns
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    # rename group columns back
    agg = agg.rename(columns={
        "variant_": "variant",
        "block_": "block",
        "layer_": "layer",
    })

    return agg


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_root", type=str, default="./llava-next_image2image_attentions",
                    help="Root dir where NPZs are saved: npz_root/variant/qid/prefill_img_self.npz")
    ap.add_argument("--variant", type=str, default="misleading_ungroundable",
                    choices=["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word", "notext"],
                    help="Which image variant directory to analyze (must match how you saved NPZs).")
    ap.add_argument("--out_dir", type=str, default="./output_image2image", help="Where to write CSV outputs.")
    ap.add_argument("--min_cover", type=float, default=0.1,
                    help="Token included in region if (intersection/token_area) >= min_cover")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    npz_root = Path(args.npz_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GUIC
    # ds = load_dataset("AHAAM/GUIC", split="test")
    ds = load_from_disk("../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC")  # if you have a local copy with faster loading

    per_sample_rows = []

    n_done = 0
    for i in range(len(ds)):
        if i < args.start:
            continue
        if args.max_samples > 0 and n_done >= args.max_samples:
            break

        ex = ds[i]
        qid = str(ex.get("question_id", f"unknown_{i}"))

        npz_path = npz_root / args.variant / args.variant / qid / "prefill_img_self.npz"
        if not npz_path.exists():
            continue

        # text bbox for the chosen analyzed variant
        if args.variant == "notext":
            # "notext" has no overlay bbox in some datasets; GUIC still provides "notext" entry but not always "bbox".
            # If missing, skip.
            if "notext" not in ex or "bbox" not in ex["notext"]:
                continue
            text_bbox = ex["notext"]["bbox"]
        else:
            text_bbox = ex[args.variant]["bbox"]  # [x1,y1,x2,y2]

        # object bboxes come from fixed fields in correct_answer and misleading_groundable
        # (present in dataset according to card)
        corr_xywh = (ex["correct_answer"]["x"], ex["correct_answer"]["y"], ex["correct_answer"]["w"], ex["correct_answer"]["h"])
        misl_xywh = (ex["misleading_groundable"]["x"], ex["misleading_groundable"]["y"], ex["misleading_groundable"]["w"], ex["misleading_groundable"]["h"])

        npz = load_npz(npz_path)

        df_sample = compute_sample_metrics(
            qid=qid,
            variant=args.variant,
            npz=npz,
            guic_text_bbox_xyxy=text_bbox,
            guic_correct_obj_xywh=corr_xywh,
            guic_mislead_obj_xywh=misl_xywh,
            min_cover=args.min_cover,
        )
        per_sample_rows.append(df_sample)

        n_done += 1

    if len(per_sample_rows) == 0:
        raise RuntimeError("No samples processed. Check --npz_root and --variant match your saved outputs.")

    df = pd.concat(per_sample_rows, ignore_index=True)
    df.to_csv(out_dir / "per_sample_metrics.csv", index=False)

    df_agg = aggregate_per_layer(df)
    df_agg.to_csv(out_dir / "per_layer_aggregate.csv", index=False)

    print(f"Wrote: {out_dir / 'per_sample_metrics.csv'}")
    print(f"Wrote: {out_dir / 'per_layer_aggregate.csv'}")
    print(f"Samples processed: {n_done}")


if __name__ == "__main__":
    main()