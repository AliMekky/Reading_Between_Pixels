#!/usr/bin/env python3
# aggregate_plot_guic_base_mosaic_mean_of_ratios.py
"""
Dataset-level GUIC analysis on a whitelist of non-overlapping question_ids.

For each qid in whitelist:
  - Load NPZ: attn (L,S_prompt), image_placeholder_positions, packed_mapping_tokens
  - Load GUIC sample to get region bboxes:
      - text_region: sample[variant]["bbox"] in [x1,y1,x2,y2]
      - correct_object_region: sample["correct_answer"]["x,y,w,h"]
      - misleading_object_region: sample["misleading_groundable"]["x,y,w,h"]
  - Map image patch tokens to prompt indices via mapping + image_placeholder_positions
  - Compute, per layer l, per sample s:
      base_ratio_s[l]   = mass_s(region ∩ base)   / mass_s(base)
      mosaic_ratio_s[l] = mass_s(region ∩ mosaic) / mass_s(mosaic)
    (This is the "mean of ratios" approach: compute ratios per sample, then average)

Outputs:
  out_dir/<variant>/
    base_conditional_mean_of_ratios.png
    mosaic_conditional_mean_of_ratios.png
    stream_usage.png                 (mean p(base), mean p(mosaic), mean p(image), mean p(text_prompt))
    report.json                      (counts + settings)

Legends on conditional plots:
  - text region
  - correct object region
  - misleading object region

Notes:
- By default, conditionals are NaN for a sample+layer when denom < min_denom (e.g., model barely attends to base/mosaic).
  Aggregation uses nanmean / nanpercentile so those layers are averaged over valid samples only.
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
        frac = inter / (bbox_area(bb) + 1e-12)
        if frac >= min_overlap_frac:
            packed_idx = int(t["token_idx"])
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


def ratio_curve(numer: np.ndarray, denom: np.ndarray, *, min_denom: float, eps: float = 1e-12) -> np.ndarray:
    """
    Per-sample ratio across layers. Invalid layers (denom < min_denom) -> NaN.
    """
    out = np.full_like(numer, np.nan, dtype=np.float32)
    good = denom >= float(min_denom)
    out[good] = numer[good] / (denom[good] + float(eps))
    return out


def mean_and_ci(curves: np.ndarray, ci: float = 95.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    curves: (N, L) possibly containing NaNs
    returns:
      mean (L,), lo (L,), hi (L,), n_eff (L,)  [effective non-NaN count per layer]
    """
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
    """
    series: name -> (mean, lo, hi)
    """
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

    # overlaid text bbox for this variant
    if variant == "notext":
        text_bbox_xyxy = sample["correct_answer"]["bbox"]  # fallback
    else:
        text_bbox_xyxy = sample[variant]["bbox"]
    regions["text region"] = guic_text_bbox_to_yxyx(text_bbox_xyxy)

    # correct object bbox (always exists in dataset definition)
    ca = sample.get("correct_answer", None)
    if ca is not None and all(k in ca for k in ["x", "y", "w", "h"]):
        regions["correct object region"] = guic_object_bbox_xywh_to_yxyx(ca["x"], ca["y"], ca["w"], ca["h"])

    # misleading grounded object bbox (only meaningful if the dataset provides it; it should)
    mg = sample.get("misleading_groundable", None)
    if mg is not None and all(k in mg for k in ["x", "y", "w", "h"]):
        regions["misleading object region"] = guic_object_bbox_xywh_to_yxyx(mg["x"], mg["y"], mg["w"], mg["h"])

    return regions


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_root", type=str, default='/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/attention_weights/llava-next_attentions',
                    help="Root dir containing <variant>/<qid>/gen_attn_gen_token.npz")
    ap.add_argument("--variant", type=str, default="irrelevant_word",
                    choices=["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word", "notext"])
    ap.add_argument("--qid_file", type=str, default ="../inference/no_overlap_question_ids.txt",
                    help="Path to whitelist of non-overlapping question_ids (one per line).")
    ap.add_argument("--out_dir", type=str, default="./plots/")
    ap.add_argument("--min_overlap_frac", type=float, default=0.25)
    ap.add_argument("--min_denom", type=float, default=1e-4)
    ap.add_argument("--ci", type=float, default=95.0, help="Percent CI band from percentiles (default 95).")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    whitelist = load_qid_whitelist(args.qid_file)
    whitelist_set = set(whitelist)

    # Load GUIC and index by qid
    ds = load_dataset("AHAAM/GUIC", split="test")
    by_qid = {str(ex["question_id"]): ex for ex in ds}

    # Collect per-sample curves
    # We will build arrays of shape (N, L) for each legend region, separately for base and mosaic.
    base_region_curves: Dict[str, List[np.ndarray]] = {
        "text region": [],
        "correct object region": [],
        "misleading object region": [],
    }
    mosaic_region_curves: Dict[str, List[np.ndarray]] = {
        "text region": [],
        "correct object region": [],
        "misleading object region": [],
    }

    # Stream usage curves per sample (unconditional masses)
    base_mass_curves: List[np.ndarray] = []
    mosaic_mass_curves: List[np.ndarray] = []
    image_mass_curves: List[np.ndarray] = []
    text_prompt_mass_curves: List[np.ndarray] = []

    # Track token counts for reporting
    token_counts = {
        "base_tokens": [],
        "mosaic_tokens": [],
        "text_prompt_tokens": [],
        "region_tokens_base": {k: [] for k in base_region_curves.keys()},
        "region_tokens_mosaic": {k: [] for k in mosaic_region_curves.keys()},
    }

    used_qids: List[str] = []
    L_ref: Optional[int] = None

    for qid in whitelist:
        if qid not in by_qid:
            continue

        npz_path = Path(args.npz_root) / args.variant / args.variant / qid / "gen_attn_gen_token.npz"
        # print(npz_path)
        if not npz_path.exists():
            print(f"[WARN] NPZ not found for qid {qid} at expected path {npz_path}. Skipping.")
            continue

        attn, img_pos, mapping_tokens, meta, attention_mask = load_npz(str(npz_path))
        L, S_prompt = attn.shape

        # --- SANITY CHECK: mapping alignment ---
        expected_img_tokens = len(mapping_tokens)

        if len(img_pos) != expected_img_tokens:
            # Skip sample if misaligned
            print(f"[WARN] {qid}: mapping mismatch "
                f"(len(image_placeholder_positions)={len(img_pos)} "
                f"!= len(mapping_tokens)={expected_img_tokens}). Skipping.")
            continue

        if L_ref is None:
            L_ref = L
        elif L != L_ref:
            # Mixed layer counts would complicate aggregation. Skip mismatched models.
            continue

        sample = by_qid[qid]
        regions = build_regions_from_guic(sample, args.variant)

        # Stream token sets + masses
        base_idxs = stream_prompt_indices(mapping_tokens, img_pos, "base")
        mosaic_idxs = stream_prompt_indices(mapping_tokens, img_pos, "mosaic")
        image_idxs = sorted(set(base_idxs).union(mosaic_idxs))
        text_idxs = text_prompt_indices(S_prompt, img_pos, attention_mask)

        p_base = mass_curve(attn, base_idxs)
        p_mosaic = mass_curve(attn, mosaic_idxs)

        base_mass_curves.append(p_base)
        mosaic_mass_curves.append(p_mosaic)
        image_mass_curves.append(p_base + p_mosaic)
        text_prompt_mass_curves.append(mass_curve(attn, text_idxs))

        token_counts["base_tokens"].append(len(base_idxs))
        token_counts["mosaic_tokens"].append(len(mosaic_idxs))
        token_counts["text_prompt_tokens"].append(len(text_idxs))

        # Per-region, per-stream ratios (mean of ratios later)
        for legend_name in ["text region", "correct object region", "misleading object region"]:
            if legend_name not in regions:
                # If region missing (should not happen for first + last; but be robust)
                # Add all-NaN curve so sample count stays aligned
                base_region_curves[legend_name].append(np.full((L,), np.nan, dtype=np.float32))
                mosaic_region_curves[legend_name].append(np.full((L,), np.nan, dtype=np.float32))
                token_counts["region_tokens_base"][legend_name].append(0)
                token_counts["region_tokens_mosaic"][legend_name].append(0)
                continue

            bb = regions[legend_name]

            r_base = region_prompt_indices_from_bbox(
                mapping_tokens, img_pos, bb, stream="base", min_overlap_frac=args.min_overlap_frac
            )
            r_mosaic = region_prompt_indices_from_bbox(
                mapping_tokens, img_pos, bb, stream="mosaic", min_overlap_frac=args.min_overlap_frac
            )

            token_counts["region_tokens_base"][legend_name].append(len(r_base))
            token_counts["region_tokens_mosaic"][legend_name].append(len(r_mosaic))

            m_base = mass_curve(attn, r_base)
            m_mosaic = mass_curve(attn, r_mosaic)

            base_region_curves[legend_name].append(
                ratio_curve(m_base, p_base, min_denom=args.min_denom)
            )
            mosaic_region_curves[legend_name].append(
                ratio_curve(m_mosaic, p_mosaic, min_denom=args.min_denom)
            )

        used_qids.append(qid)

        if args.max_samples and len(used_qids) >= args.max_samples:
            break

    if L_ref is None or len(used_qids) == 0:
        raise RuntimeError("No samples found/loaded. Check npz_root/variant/qid_file paths.")

    # Stack to arrays: (N, L)
    N = len(used_qids)
    layers = np.arange(L_ref, dtype=np.int32)

    def stack_curves(curve_list: List[np.ndarray]) -> np.ndarray:
        return np.stack(curve_list, axis=0).astype(np.float32)

    # Mean+CI for base conditional curves (mean of ratios)
    base_series = {}
    mosaic_series = {}

    n_eff_base = {}
    n_eff_mosaic = {}

    for legend_name in ["text region", "correct object region", "misleading object region"]:
        arr_b = stack_curves(base_region_curves[legend_name])    # (N,L)
        arr_m = stack_curves(mosaic_region_curves[legend_name])  # (N,L)

        mean_b, lo_b, hi_b, ne_b = mean_and_ci(arr_b, ci=args.ci)
        mean_m, lo_m, hi_m, ne_m = mean_and_ci(arr_m, ci=args.ci)

        base_series[legend_name] = (mean_b, lo_b, hi_b)
        mosaic_series[legend_name] = (mean_m, lo_m, hi_m)

        n_eff_base[legend_name] = ne_b.tolist()
        n_eff_mosaic[legend_name] = ne_m.tolist()

    # Stream usage means (unconditional)
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
        title=f"{args.variant} | base: mean of per-sample p(region | base) (N={N})",
        ylabel="p(region | base)  (mean of ratios)",
        out_path=out_dir / "base_conditional_mean_of_ratios.png",
        ylim=(0.0, 1.0),
    )

    plot_mean_ci_lines(
        layers,
        mosaic_series,
        title=f"{args.variant} | mosaic: mean of per-sample p(region | mosaic) (N={N})",
        ylabel="p(region | mosaic)  (mean of ratios)",
        out_path=out_dir / "mosaic_conditional_mean_of_ratios.png",
        ylim=(0.0, 1.0),
    )

    plot_mean_lines(
        layers,
        {
            "mean p(base)": base_mass_mean,
            "mean p(mosaic)": mosaic_mass_mean,
            "mean p(image)": image_mass_mean,
            "mean p(text_prompt)": text_mass_mean,
        },
        title=f"{args.variant} | stream usage means (N={N})",
        ylabel="unconditional attention mass (mean across samples)",
        out_path=out_dir / "stream_usage.png",
        ylim=(0.0, 1.0),
    )

    # Report
    report = {
        "variant": args.variant,
        "npz_root": str(Path(args.npz_root).resolve()),
        "qid_file": str(Path(args.qid_file).resolve()),
        "N_used": N,
        "used_qids_first_20": used_qids[:20],
        "settings": {
            "min_overlap_frac": float(args.min_overlap_frac),
            "min_denom": float(args.min_denom),
            "ci": float(args.ci),
            "aggregation": "mean of per-sample ratios (nanmean), CI from nanpercentile",
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
            "base": n_eff_base,      # how many samples had denom>=min_denom at each layer
            "mosaic": n_eff_mosaic,
        },
        "outputs": {
            "base_conditional_mean_of_ratios": "base_conditional_mean_of_ratios.png",
            "mosaic_conditional_mean_of_ratios": "mosaic_conditional_mean_of_ratios.png",
            "stream_usage": "stream_usage.png",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
