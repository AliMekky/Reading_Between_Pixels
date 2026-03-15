#!/usr/bin/env python3
# aggregate_plot_guic_base_mosaic_mean_of_ratios.py

"""
Dataset-level GUIC analysis on a whitelist of non-overlapping question_ids.

NON-SKIPPING behavior (requested):
- If NPZ missing OR GUIC sample missing OR layer-count mismatch OR mapping mismatch:
  -> DO NOT drop the qid from aggregation.
  -> Instead, append all-NaN curves for that qid (so it contributes 0 to nanmean).
  -> Log the qid into the report under the appropriate bucket.

Mapping mismatch handling (requested "same as IG"):
- If len(image_placeholder_positions) != number of packed mapping tokens:
  -> do NOT skip
  -> use a SAFE truncation to the common prefix K = min(len(img_pos), max_token_idx+1)
  -> filter mapping_tokens to token_idx < K and truncate img_pos to length K
  -> compute with that truncated alignment
  -> record qid in report["qids_mapping_mismatch_truncated"]

Metrics:
- For each sample, per layer:
  base_ratio[l]   = mean_attn(region ∩ base)   / mean_attn(base)
  mosaic_ratio[l] = mean_attn(region ∩ mosaic) / mean_attn(mosaic)

- Validity gating (NaN) uses the ORIGINAL denom mass:
    gate_base[l]   = mass(base)
    gate_mosaic[l] = mass(mosaic)
  invalid if gate < min_denom (matches earlier behavior).

Outputs:
  out_dir/<variant>/
    base_conditional_mean_of_ratios.png
    mosaic_conditional_mean_of_ratios.png
    stream_usage.png
    report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


# -----------------------------
# IO helpers
# -----------------------------
def load_qid_whitelist(path: str) -> List[str]:
    qids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.append(s)
    return qids


def load_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    attn = z["attn"].astype(np.float32)  # (L, S_prompt)
    img_pos = z["image_placeholder_positions"].astype(np.int64).tolist()
    mapping_tokens = json.loads(str(z["packed_mapping_tokens"]))
    meta = json.loads(str(z["meta"]))
    attention_mask = z["attention_mask"].astype(np.int64) if "attention_mask" in z else None
    return attn, img_pos, mapping_tokens, meta, attention_mask


# -----------------------------
# GUIC bbox conversions
# -----------------------------
def guic_text_bbox_to_yxyx(text_bbox_xyxy: List[float]) -> Tuple[float, float, float, float]:
    # GUIC text bbox: [x1, y1, x2, y2] -> (y0,x0,y1,x1)
    x1, y1, x2, y2 = text_bbox_xyxy
    return (float(y1), float(x1), float(y2), float(x2))


def guic_object_bbox_xywh_to_yxyx(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    # x,y,w,h -> (y0,x0,y1,x1)
    x0, y0 = float(x), float(y)
    return (y0, x0, y0 + float(h), x0 + float(w))


# -----------------------------
# Geometry
# -----------------------------
def bbox_area(bb: Tuple[float, float, float, float]) -> float:
    y0, x0, y1, x1 = bb
    return float(max(0.0, y1 - y0) * max(0.0, x1 - x0))


def intersect_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    y0 = max(ay0, by0)
    x0 = max(ax0, bx0)
    y1 = min(ay1, by1)
    x1 = min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float((y1 - y0) * (x1 - x0))


# -----------------------------
# Token selection
# -----------------------------
def _max_token_idx(mapping_tokens: List[Dict[str, Any]]) -> int:
    m = -1
    for t in mapping_tokens:
        try:
            m = max(m, int(t.get("token_idx", -1)))
        except Exception:
            pass
    return m


def safe_align_img_pos_and_mapping(
    img_pos: List[int],
    mapping_tokens: List[Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]], bool, Dict[str, Any]]:
    """
    Returns:
      img_pos_aligned
      mapping_tokens_aligned
      did_truncate (bool)
      info (dict)
    """
    max_idx = _max_token_idx(mapping_tokens)
    expected_by_mapping = max_idx + 1 if max_idx >= 0 else 0
    K = min(len(img_pos), expected_by_mapping) if expected_by_mapping > 0 else 0

    did_truncate = False
    if expected_by_mapping != len(img_pos):
        did_truncate = True

    if K <= 0:
        # Nothing usable; keep empty and caller will produce NaNs.
        return [], [], did_truncate, {
            "len_img_pos": len(img_pos),
            "expected_by_mapping": expected_by_mapping,
            "K": K,
        }

    img_pos2 = img_pos[:K]
    mapping2 = [t for t in mapping_tokens if 0 <= int(t.get("token_idx", -1)) < K]
    # Note: mapping2 might not cover all [0..K-1] if tokens are missing; that's still the safest.

    return img_pos2, mapping2, did_truncate, {
        "len_img_pos": len(img_pos),
        "expected_by_mapping": expected_by_mapping,
        "K": K,
        "len_mapping_filtered": len(mapping2),
    }


def stream_prompt_indices(mapping_tokens: List[Dict[str, Any]], img_pos: List[int], stream: str) -> List[int]:
    if stream == "base":
        kinds = {"base_patch"}
    elif stream == "mosaic":
        kinds = {"mosaic_patch"}
    else:
        raise ValueError("stream must be 'base' or 'mosaic'")

    out = []
    for t in mapping_tokens:
        if t.get("kind") not in kinds:
            continue
        packed_idx = int(t["token_idx"])
        if packed_idx < 0 or packed_idx >= len(img_pos):
            continue
        out.append(int(img_pos[packed_idx]))
    return sorted(set(out))


def region_prompt_indices_from_bbox(
    mapping_tokens: List[Dict[str, Any]],
    img_pos: List[int],
    region_bbox_yxyx: Tuple[float, float, float, float],
    *,
    stream: str,
    min_overlap_frac: float,
) -> List[int]:
    if stream == "base":
        kinds = {"base_patch"}
    elif stream == "mosaic":
        kinds = {"mosaic_patch"}
    else:
        raise ValueError("stream must be 'base' or 'mosaic'")

    out = []
    for t in mapping_tokens:
        if t.get("kind") not in kinds:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue
        bb = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))  # (y0,x0,y1,x1)

        inter = intersect_area(bb, region_bbox_yxyx)
        frac = inter / (bbox_area(bb) + 1e-12)  # overlap fraction of PATCH area
        if frac >= min_overlap_frac:
            packed_idx = int(t["token_idx"])
            if packed_idx < 0 or packed_idx >= len(img_pos):
                continue
            out.append(int(img_pos[packed_idx]))
    return sorted(set(out))


def text_prompt_indices(S_prompt: int, img_pos: List[int], attention_mask: Optional[np.ndarray]) -> List[int]:
    img_set = set(img_pos)
    out = []
    for i in range(S_prompt):
        if i in img_set:
            continue
        if attention_mask is not None and int(attention_mask[i]) == 0:
            continue
        out.append(i)
    return out


# -----------------------------
# Aggregation helpers
# -----------------------------
def mass_curve(attn: np.ndarray, idxs: List[int]) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros((attn.shape[0],), dtype=np.float32)
    return attn[:, idxs].sum(axis=1)


def density_curve(attn: np.ndarray, idxs: List[int]) -> np.ndarray:
    denom = float(max(1, len(idxs)))
    return mass_curve(attn, idxs) / denom


def ratio_curve_with_gate(
    numer: np.ndarray,
    denom: np.ndarray,
    gate: np.ndarray,
    *,
    min_gate: float,
    eps: float = 1e-12,
) -> np.ndarray:
    out = np.full_like(numer, np.nan, dtype=np.float32)
    good = gate >= float(min_gate)
    out[good] = numer[good] / (denom[good] + float(eps))
    return out


def mean_and_ci(curves: np.ndarray, ci: float = 95.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(curves, axis=0)
    lo = np.nanpercentile(curves, (100.0 - ci) / 2.0, axis=0)
    hi = np.nanpercentile(curves, 100.0 - (100.0 - ci) / 2.0, axis=0)
    n_eff = np.sum(np.isfinite(curves), axis=0)
    return mean, lo, hi, n_eff.astype(np.int32)


# -----------------------------
# Plotting
# -----------------------------
def plot_mean_ci_lines(
    x: np.ndarray,
    series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
):
    plt.figure()
    for name, (m, lo, hi) in series.items():
        plt.plot(x, m, label=name)
        plt.fill_between(x, lo, hi, alpha=0.2)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_lines(
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
):
    plt.figure()
    for name, y in series.items():
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Region builder from GUIC
# -----------------------------
def build_regions_from_guic(sample: Dict[str, Any], variant: str) -> Dict[str, Tuple[float, float, float, float]]:
    regions: Dict[str, Tuple[float, float, float, float]] = {}

    if variant == "notext":
        text_bbox_xyxy = sample["correct_answer"]["bbox"]  # fallback
    else:
        text_bbox_xyxy = sample[variant]["bbox"]
    regions["text region"] = guic_text_bbox_to_yxyx(text_bbox_xyxy)

    ca = sample.get("correct_answer", None)
    if ca is not None and all(k in ca for k in ["x", "y", "w", "h"]):
        regions["correct object region"] = guic_object_bbox_xywh_to_yxyx(ca["x"], ca["y"], ca["w"], ca["h"])

    mg = sample.get("misleading_groundable", None)
    if mg is not None and all(k in mg for k in ["x", "y", "w", "h"]):
        regions["misleading object region"] = guic_object_bbox_xywh_to_yxyx(mg["x"], mg["y"], mg["w"], mg["h"])

    return regions


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz_root",
        type=str,
        default="/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/attention_weights/llava-next_attentions",
        help="Root dir containing <variant>/<qid>/gen_attn_gen_token.npz",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="notext",
        choices=["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word", "notext"],
    )
    ap.add_argument(
        "--qid_file",
        type=str,
        default="../inference/no_overlap_question_ids.txt",
        help="Path to whitelist of non-overlapping question_ids (one per line).",
    )
    ap.add_argument("--out_dir", type=str, default="./plots_normalized/")
    ap.add_argument("--min_overlap_frac", type=float, default=0.25)
    ap.add_argument("--min_denom", type=float, default=1e-4)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    whitelist = load_qid_whitelist(args.qid_file)

    # Load GUIC and index by qid
    ds = load_dataset("AHAAM/GUIC", split="test")
    by_qid = {str(ex["question_id"]): ex for ex in ds}

    legend_names = ["text region", "correct object region", "misleading object region"]

    # --- Pass 0: find L_ref (first available NPZ) ---
    L_ref: Optional[int] = None
    S_ref: Optional[int] = None
    first_npz_qid: Optional[str] = None
    for qid in whitelist:
        npz_path = Path(args.npz_root) / args.variant / args.variant / qid / "gen_attn_gen_token.npz"
        if npz_path.exists():
            attn, img_pos, mapping_tokens, meta, attention_mask = load_npz(str(npz_path))
            L_ref, S_ref = attn.shape
            first_npz_qid = qid
            break
    if L_ref is None:
        raise RuntimeError("Could not find ANY NPZ to establish L_ref. Check --npz_root / --variant paths.")

    # Collect per-sample curves (keep alignment with whitelist; NaNs for unusable samples)
    base_region_curves: Dict[str, List[np.ndarray]] = {k: [] for k in legend_names}
    mosaic_region_curves: Dict[str, List[np.ndarray]] = {k: [] for k in legend_names}

    base_mass_curves: List[np.ndarray] = []
    mosaic_mass_curves: List[np.ndarray] = []
    image_mass_curves: List[np.ndarray] = []
    text_prompt_mass_curves: List[np.ndarray] = []

    token_counts = {
        "base_tokens": [],
        "mosaic_tokens": [],
        "text_prompt_tokens": [],
        "region_tokens_base": {k: [] for k in legend_names},
        "region_tokens_mosaic": {k: [] for k in legend_names},
    }

    # logging buckets (non-skipping)
    qids_missing_npz: List[str] = []
    qids_missing_guic: List[str] = []
    qids_layer_mismatch: List[str] = []
    qids_mapping_mismatch_truncated: List[str] = []
    qids_no_usable_alignment: List[str] = []

    used_qids: List[str] = []

    def _nan_curve() -> np.ndarray:
        return np.full((L_ref,), np.nan, dtype=np.float32)

    def _zero_curve() -> np.ndarray:
        return np.zeros((L_ref,), dtype=np.float32)

    for qid in whitelist:
        used_qids.append(qid)

        # NPZ path
        npz_path = Path(args.npz_root) / args.variant / args.variant / qid / "gen_attn_gen_token.npz"
        if not npz_path.exists():
            qids_missing_npz.append(qid)
            # append NaNs/zeros so we don't drop the qid
            for k in legend_names:
                base_region_curves[k].append(_nan_curve())
                mosaic_region_curves[k].append(_nan_curve())
                token_counts["region_tokens_base"][k].append(0)
                token_counts["region_tokens_mosaic"][k].append(0)
            base_mass_curves.append(_zero_curve())
            mosaic_mass_curves.append(_zero_curve())
            image_mass_curves.append(_zero_curve())
            text_prompt_mass_curves.append(_zero_curve())
            token_counts["base_tokens"].append(0)
            token_counts["mosaic_tokens"].append(0)
            token_counts["text_prompt_tokens"].append(0)
            continue

        # GUIC sample
        if qid not in by_qid:
            qids_missing_guic.append(qid)
            # append NaNs/zeros so we don't drop the qid
            for k in legend_names:
                base_region_curves[k].append(_nan_curve())
                mosaic_region_curves[k].append(_nan_curve())
                token_counts["region_tokens_base"][k].append(0)
                token_counts["region_tokens_mosaic"][k].append(0)
            base_mass_curves.append(_zero_curve())
            mosaic_mass_curves.append(_zero_curve())
            image_mass_curves.append(_zero_curve())
            text_prompt_mass_curves.append(_zero_curve())
            token_counts["base_tokens"].append(0)
            token_counts["mosaic_tokens"].append(0)
            token_counts["text_prompt_tokens"].append(0)
            continue

        attn, img_pos, mapping_tokens, meta, attention_mask = load_npz(str(npz_path))
        L, S_prompt = attn.shape

        if L != L_ref:
            qids_layer_mismatch.append(qid)
            # append NaNs/zeros so we don't drop the qid
            for k in legend_names:
                base_region_curves[k].append(_nan_curve())
                mosaic_region_curves[k].append(_nan_curve())
                token_counts["region_tokens_base"][k].append(0)
                token_counts["region_tokens_mosaic"][k].append(0)
            base_mass_curves.append(_zero_curve())
            mosaic_mass_curves.append(_zero_curve())
            image_mass_curves.append(_zero_curve())
            text_prompt_mass_curves.append(_zero_curve())
            token_counts["base_tokens"].append(0)
            token_counts["mosaic_tokens"].append(0)
            token_counts["text_prompt_tokens"].append(0)
            continue

        # safe align (non-skipping)
        img_pos2, mapping2, did_truncate, align_info = safe_align_img_pos_and_mapping(img_pos, mapping_tokens)
        if did_truncate:
            qids_mapping_mismatch_truncated.append(qid)

        if len(img_pos2) == 0 or len(mapping2) == 0:
            qids_no_usable_alignment.append(qid)
            # append NaNs/zeros so we don't drop the qid
            for k in legend_names:
                base_region_curves[k].append(_nan_curve())
                mosaic_region_curves[k].append(_nan_curve())
                token_counts["region_tokens_base"][k].append(0)
                token_counts["region_tokens_mosaic"][k].append(0)
            base_mass_curves.append(_zero_curve())
            mosaic_mass_curves.append(_zero_curve())
            image_mass_curves.append(_zero_curve())
            text_prompt_mass_curves.append(_zero_curve())
            token_counts["base_tokens"].append(0)
            token_counts["mosaic_tokens"].append(0)
            token_counts["text_prompt_tokens"].append(0)
            continue

        sample = by_qid[qid]
        regions = build_regions_from_guic(sample, args.variant)

        # Stream indices
        base_idxs = stream_prompt_indices(mapping2, img_pos2, "base")
        mosaic_idxs = stream_prompt_indices(mapping2, img_pos2, "mosaic")
        text_idxs = text_prompt_indices(S_prompt, img_pos2, attention_mask)

        # Mass curves (unconditional)
        p_base = mass_curve(attn, base_idxs)
        p_mosaic = mass_curve(attn, mosaic_idxs)

        base_mass_curves.append(p_base)
        mosaic_mass_curves.append(p_mosaic)
        image_mass_curves.append(p_base + p_mosaic)
        text_prompt_mass_curves.append(mass_curve(attn, text_idxs))

        token_counts["base_tokens"].append(len(base_idxs))
        token_counts["mosaic_tokens"].append(len(mosaic_idxs))
        token_counts["text_prompt_tokens"].append(len(text_idxs))

        # density denominators
        d_base = density_curve(attn, base_idxs)
        d_mosaic = density_curve(attn, mosaic_idxs)

        for legend_name in legend_names:
            if legend_name not in regions:
                base_region_curves[legend_name].append(_nan_curve())
                mosaic_region_curves[legend_name].append(_nan_curve())
                token_counts["region_tokens_base"][legend_name].append(0)
                token_counts["region_tokens_mosaic"][legend_name].append(0)
                continue

            bb = regions[legend_name]

            r_base = region_prompt_indices_from_bbox(
                mapping2, img_pos2, bb, stream="base", min_overlap_frac=args.min_overlap_frac
            )
            r_mosaic = region_prompt_indices_from_bbox(
                mapping2, img_pos2, bb, stream="mosaic", min_overlap_frac=args.min_overlap_frac
            )

            token_counts["region_tokens_base"][legend_name].append(len(r_base))
            token_counts["region_tokens_mosaic"][legend_name].append(len(r_mosaic))

            d_r_base = density_curve(attn, r_base)
            d_r_mosaic = density_curve(attn, r_mosaic)

            base_region_curves[legend_name].append(
                ratio_curve_with_gate(d_r_base, d_base, p_base, min_gate=args.min_denom)
            )
            mosaic_region_curves[legend_name].append(
                ratio_curve_with_gate(d_r_mosaic, d_mosaic, p_mosaic, min_gate=args.min_denom)
            )

        if args.max_samples and len(used_qids) >= args.max_samples:
            break

    # Stack arrays
    N_total = len(used_qids)
    layers = np.arange(L_ref, dtype=np.int32)

    def stack_curves(curve_list: List[np.ndarray]) -> np.ndarray:
        return np.stack(curve_list, axis=0).astype(np.float32)

    base_series = {}
    mosaic_series = {}
    n_eff_base = {}
    n_eff_mosaic = {}

    for legend_name in legend_names:
        arr_b = stack_curves(base_region_curves[legend_name])    # (N,L)
        arr_m = stack_curves(mosaic_region_curves[legend_name])  # (N,L)

        mean_b, lo_b, hi_b, ne_b = mean_and_ci(arr_b, ci=args.ci)
        mean_m, lo_m, hi_m, ne_m = mean_and_ci(arr_m, ci=args.ci)

        base_series[legend_name] = (mean_b, lo_b, hi_b)
        mosaic_series[legend_name] = (mean_m, lo_m, hi_m)

        n_eff_base[legend_name] = ne_b.tolist()
        n_eff_mosaic[legend_name] = ne_m.tolist()

    base_mass = stack_curves(base_mass_curves)
    mosaic_mass = stack_curves(mosaic_mass_curves)
    image_mass = stack_curves(image_mass_curves)
    text_mass = stack_curves(text_prompt_mass_curves)

    base_mass_mean = np.nanmean(base_mass, axis=0)
    mosaic_mass_mean = np.nanmean(mosaic_mass, axis=0)
    image_mass_mean = np.nanmean(image_mass, axis=0)
    text_mass_mean = np.nanmean(text_mass, axis=0)

    # Plots
    plot_mean_ci_lines(
        layers,
        base_series,
        title=f"{args.variant} | base: mean of per-sample density_ratio(region/base) (N_total={N_total})",
        ylabel="(avg attn per region token) / (avg attn per base token)",
        out_path=out_dir / "base_conditional_mean_of_ratios.png",
        ylim=(0.0, 10.0),
    )

    plot_mean_ci_lines(
        layers,
        mosaic_series,
        title=f"{args.variant} | mosaic: mean of per-sample density_ratio(region/mosaic) (N_total={N_total})",
        ylabel="(avg attn per region token) / (avg attn per mosaic token)",
        out_path=out_dir / "mosaic_conditional_mean_of_ratios.png",
        ylim=(0.0, 10.0),
    )

    plot_mean_lines(
        layers,
        {
            "mean p(base)": base_mass_mean,
            "mean p(mosaic)": mosaic_mass_mean,
            "mean p(image)": image_mass_mean,
            "mean p(text_prompt)": text_mass_mean,
        },
        title=f"{args.variant} | stream usage means (N_total={N_total})",
        ylabel="unconditional attention mass (mean across samples)",
        out_path=out_dir / "stream_usage.png",
        ylim=(0.0, 1.0),
    )

    # Report
    report = {
        "variant": args.variant,
        "npz_root": str(Path(args.npz_root).resolve()),
        "qid_file": str(Path(args.qid_file).resolve()),
        "N_total_whitelist_used_for_arrays": N_total,
        "L_ref": int(L_ref),
        "S_ref_from_first_npz": int(S_ref) if S_ref is not None else None,
        "first_npz_qid_used_for_ref": first_npz_qid,
        "settings": {
            "min_overlap_frac": float(args.min_overlap_frac),
            "min_denom": float(args.min_denom),
            "ci": float(args.ci),
            "aggregation": "mean of per-sample ratios (nanmean), CI from nanpercentile; missing/invalid qids contribute NaNs (non-skipping)",
            "region_metric": "density_ratio = (avg attn per region token) / (avg attn per stream token)",
            "gating": "gate uses stream attention MASS (p_base/p_mosaic) and NaNs when < min_denom",
            "mapping_mismatch_policy": "truncate to common prefix K and filter mapping_tokens to token_idx<K",
        },
        "non_skipping_buckets": {
            "n_missing_npz": len(qids_missing_npz),
            "n_missing_guic": len(qids_missing_guic),
            "n_layer_mismatch": len(qids_layer_mismatch),
            "n_mapping_mismatch_truncated": len(qids_mapping_mismatch_truncated),
            "n_no_usable_alignment": len(qids_no_usable_alignment),
            "qids_missing_npz_first_50": qids_missing_npz[:50],
            "qids_missing_guic_first_50": qids_missing_guic[:50],
            "qids_layer_mismatch_first_50": qids_layer_mismatch[:50],
            "qids_mapping_mismatch_truncated_first_50": qids_mapping_mismatch_truncated[:50],
            "qids_no_usable_alignment_first_50": qids_no_usable_alignment[:50],
        },
        "token_count_stats": {
            "base_tokens": {
                "mean": float(np.mean(token_counts["base_tokens"])) if token_counts["base_tokens"] else 0.0,
                "min": int(np.min(token_counts["base_tokens"])) if token_counts["base_tokens"] else 0,
                "max": int(np.max(token_counts["base_tokens"])) if token_counts["base_tokens"] else 0,
            },
            "mosaic_tokens": {
                "mean": float(np.mean(token_counts["mosaic_tokens"])) if token_counts["mosaic_tokens"] else 0.0,
                "min": int(np.min(token_counts["mosaic_tokens"])) if token_counts["mosaic_tokens"] else 0,
                "max": int(np.max(token_counts["mosaic_tokens"])) if token_counts["mosaic_tokens"] else 0,
            },
            "text_prompt_tokens": {
                "mean": float(np.mean(token_counts["text_prompt_tokens"])) if token_counts["text_prompt_tokens"] else 0.0,
                "min": int(np.min(token_counts["text_prompt_tokens"])) if token_counts["text_prompt_tokens"] else 0,
                "max": int(np.max(token_counts["text_prompt_tokens"])) if token_counts["text_prompt_tokens"] else 0,
            },
        },
        "n_effective_per_layer": {
            "base": n_eff_base,
            "mosaic": n_eff_mosaic,
        },
        "outputs": {
            "base_conditional_mean_of_ratios": "base_conditional_mean_of_ratios.png",
            "mosaic_conditional_mean_of_ratios": "mosaic_conditional_mean_of_ratios.png",
            "stream_usage": "stream_usage.png",
            "report": "report.json",
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()