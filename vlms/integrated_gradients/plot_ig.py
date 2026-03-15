#!/usr/bin/env python3
"""
Enhanced GUIC IG Plotter — Publication / Presentation Quality
============================================================
Improvements over baseline:
  1. Cleaner figure structure with explicit section dividers
  2. Per-variant column headers styled by category
     (notext=grey, correct=green, misleading*=red, irrelevant=orange)
  3. Shared diverging colorbar with explicit tick labels (negative / zero / positive)
  4. Legend panel summarising box colours + score interpretation
  5. Row labels (BASE / MOSAIC) on the left axis
  6. Subtle inter-row separator line
  7. Question / answer block rendered as a clean text-panel above the grid
  8. Per-cell score statistics (mean |IG| on image, mean |IG| on text region)
     printed as tiny annotations at the bottom of each cell
  9. Optional model-response annotation per variant (pass --responses JSON)
 10. ACL-friendly white background, black spines, consistent font sizes
"""

import json
import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from datasets import load_dataset

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
})

VARIANTS_ORDER = [
    "notext",
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
]

VARIANT_COLORS = {
    "notext":                  "#888888",
    "correct_answer":          "#1a7d3a",
    "misleading_groundable":   "#c0392b",
    "misleading_ungroundable": "#922b21",
    "irrelevant_word":         "#d97b00",
}

VARIANT_LABELS = {
    "notext":                  "No Text",
    "correct_answer":          "Correct Answer",
    "misleading_groundable":   "Misleading\n(Groundable)",
    "misleading_ungroundable": "Misleading\n(Ungroundable)",
    "irrelevant_word":         "Irrelevant Word",
}

ROW_LABELS = ["BASE\nEncoding", "MOSAIC\nEncoding"]


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_ds(dataset_id: str, split: str = "test"):
    return load_dataset(dataset_id, split=split)


def find_sample_by_qid(ds, ques_id: str):
    for i in range(len(ds)):
        if str(ds[i].get("question_id", "")) == str(ques_id):
            return ds[i]
    return None


def load_variant_image(sample: Dict[str, Any], variant: str) -> Image.Image:
    img_obj = sample["notext"]["image"] if variant == "notext" else sample[variant]["image"]
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    return Image.open(img_obj).convert("RGB")


def get_question_and_options(sample: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
    q = str(sample.get("question", "")).strip()
    opts = []
    for k in ["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"]:
        entry = sample.get(k, {})
        if isinstance(entry, dict) and "text" in entry:
            opts.append((k, str(entry["text"]).strip()))
    return q, opts


# ── IG overlay helpers ─────────────────────────────────────────────────────────

def robust_normalize(x, clip_percentiles=(5, 95), signed=False, eps=1e-8):
    x = x.astype(np.float32)
    if signed:
        hi = max(float(np.percentile(np.abs(x), clip_percentiles[1])), eps)
        return (np.clip(x, -hi, hi) / hi).astype(np.float32)
    lo, hi = np.percentile(x, clip_percentiles[0]), np.percentile(x, clip_percentiles[1])
    if float(hi) - float(lo) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((np.clip(x, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def token_scores_to_grids(mapping_tokens, token_scores, summary):
    p = summary["patches_per_side"]
    mh, mw = summary["mosaic_unpadded_hw_in_patches"]
    base_grid = np.zeros((p, p), dtype=np.float32)
    mosaic_grid = np.zeros((mh, mw), dtype=np.float32)
    for t in mapping_tokens:
        idx = int(t["token_idx"])
        if idx < 0 or idx >= token_scores.shape[0]:
            continue
        s = float(token_scores[idx])
        if t["kind"] == "base_patch":
            base_grid[int(t["row"]), int(t["col"])] += s
        elif t["kind"] == "mosaic_patch":
            mosaic_grid[int(t["row"]), int(t["col"])] += s
    return base_grid, mosaic_grid


def make_overlay_norm(img, grid, *, signed, smooth_sigma_grid, clip_percentiles=(5, 95)):
    H, W = img.height, img.width
    gh, gw = grid.shape
    g = grid.astype(np.float32, copy=False)
    if smooth_sigma_grid > 0:
        g = gaussian_filter(g, sigma=float(smooth_sigma_grid), mode="nearest")
    g_up = zoom(g, (H / gh, W / gw), order=1)
    return robust_normalize(g_up, clip_percentiles=clip_percentiles, signed=signed)


def overlay_to_rgba(overlay_norm, *, signed, cmap_name, base_alpha, mask_thr, gamma):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if signed:
        t = (overlay_norm + 1.0) * 0.5
        mag = np.abs(overlay_norm)
    else:
        t = overlay_norm
        mag = overlay_norm
    t = np.clip(t, 0.0, 1.0)
    a = np.clip((mag - mask_thr) / max(1e-8, 1.0 - mask_thr), 0.0, 1.0) ** gamma * base_alpha
    rgba = cmap(t)
    rgba[..., 3] = a
    return rgba


# ── NPZ loader ─────────────────────────────────────────────────────────────────

def load_npz(out_dir: str, variant: str, ques_id: str, mode: str):
    run_dir = Path(out_dir) / variant / variant / str(ques_id)
    npz_path = run_dir / f"ig_{mode}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")
    data = np.load(str(npz_path), allow_pickle=True)
    for k in ["token_scores", "mapping_summary", "mapping_tokens"]:
        if k not in data:
            raise RuntimeError(f"{npz_path} missing key: {k}")
    token_scores = data["token_scores"].astype(np.float32)
    summary = json.loads(str(data["mapping_summary"]))
    mapping_tokens = json.loads(str(data["mapping_tokens"]))
    return token_scores, summary, mapping_tokens


# ── Bbox helpers ───────────────────────────────────────────────────────────────

def get_text_bbox(sample, variant):
    if variant == "notext":
        return None
    entry = sample.get(variant, None)
    if not isinstance(entry, dict):
        return None
    bb = entry.get("bbox", None)
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return tuple(map(float, bb))
    return None


def get_object_bbox_xyxy(entry):
    if not isinstance(entry, dict):
        return None
    if not all(k in entry for k in ["x", "y", "w", "h"]):
        return None
    x, y, w, h = float(entry["x"]), float(entry["y"]), float(entry["w"]), float(entry["h"])
    return (x, y, x + w, y + h)


def get_global_object_boxes(sample):
    return (
        get_object_bbox_xyxy(sample.get("correct_answer", None)),
        get_object_bbox_xyxy(sample.get("misleading_groundable", None)),
    )


def draw_boxes(ax, *, correct_obj, misleading_obj, text_bbox, lw):
    def _rect(ax, xyxy, color, linestyle="-", label=None):
        if xyxy is None:
            return
        x1, y1, x2, y2 = xyxy
        ax.add_patch(patches.Rectangle(
            (x1, y1), max(0., x2 - x1), max(0., y2 - y1),
            fill=False, linewidth=lw, edgecolor=color, linestyle=linestyle,
            label=label,
        ))

    _rect(ax, text_bbox,     "#4a90d9",  "-")
    _rect(ax, correct_obj,   "#27ae60",  "-")
    _rect(ax, misleading_obj, "#e74c3c", "--")


# ── Score stats annotation ─────────────────────────────────────────────────────

def cell_stats(overlay_norm: np.ndarray, text_bbox_xyxy, img_size) -> str:
    """Return a compact stats string for the bottom of a cell."""
    H, W = img_size
    mean_all = float(np.abs(overlay_norm).mean())
    stats = f"mean|IG|={mean_all:.3f}"
    if text_bbox_xyxy is not None:
        x1, y1, x2, y2 = text_bbox_xyxy
        # normalise coords to [0,1] assuming bbox is in pixel coords
        r0 = max(0, int(y1 / H * overlay_norm.shape[0]))
        r1 = min(overlay_norm.shape[0], int(y2 / H * overlay_norm.shape[0]) + 1)
        c0 = max(0, int(x1 / W * overlay_norm.shape[1]))
        c1 = min(overlay_norm.shape[1], int(x2 / W * overlay_norm.shape[1]) + 1)
        region = overlay_norm[r0:r1, c0:c1]
        if region.size > 0:
            stats += f"  |text={float(np.abs(region).mean()):.3f}"
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ques_id", default="13539366", type=str)
    p.add_argument("--out_dir", type=str, default="./ig_plots")
    p.add_argument("--mode", type=str, default="prefill_next_token",
                   choices=["teacher_forced", "prefill_next_token"])
    p.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    p.add_argument("--split", type=str, default="test")

    # overlay
    p.add_argument("--unsigned", action="store_true")
    p.add_argument("--cmap", type=str, default="RdBu_r")
    p.add_argument("--clip_p0", type=float, default=5.0)
    p.add_argument("--clip_p1", type=float, default=95.0)
    p.add_argument("--smooth_sigma_grid", type=float, default=1.0)

    # transparency tuning
    p.add_argument("--base_alpha", type=float, default=0.9)
    p.add_argument("--mask_thr", type=float, default=0.12)
    p.add_argument("--alpha_gamma", type=float, default=1.7)

    # styling
    p.add_argument("--bbox_linewidth", type=float, default=3.0)
    p.add_argument("--wrap_width", type=int, default=140)
    p.add_argument("--show_stats", action="store_true",
                   help="Annotate each cell with mean |IG| statistics")
    p.add_argument("--responses", type=str, default="",
                   help="Path to JSON: {variant: model_response_string} for per-column annotation")

    # output
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    ques_id = str(args.ques_id)
    signed = not args.unsigned
    clip_percentiles = (args.clip_p0, args.clip_p1)
    cmap_name = args.cmap if signed else "hot"

    # optional per-variant model responses
    responses: Dict[str, str] = {}
    if args.responses:
        with open(args.responses) as f:
            responses = json.load(f)

    # ── Load data ──────────────────────────────────────────────────────────────
    ds = load_ds(args.hf_dataset, split=args.split)
    sample = find_sample_by_qid(ds, ques_id)
    if sample is None:
        raise RuntimeError(f"question_id={ques_id} not found in {args.hf_dataset}/{args.split}")

    question, options = get_question_and_options(sample)
    correct_obj_xyxy, misleading_obj_xyxy = get_global_object_boxes(sample)

    nrows, ncols = 2, len(VARIANTS_ORDER)

    # ── Figure layout ──────────────────────────────────────────────────────────
    # Top section: question panel
    # Main section: 2 × 5 image grid with row labels
    # Bottom section: legend

    fig = plt.figure(figsize=(4.6 * ncols + 1.8, 10.5), facecolor="white")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 7.0, 0.9],
        hspace=0.18,
    )

    # ── Question panel (top) ───────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor("#f8f8f8")
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    for spine in ax_hdr.spines.values():
        spine.set_visible(False)

    # QID tag
    ax_hdr.text(0.0, 0.97, f"QID {ques_id}",
                ha="left", va="top", fontsize=9, color="#888888",
                fontweight="bold", transform=ax_hdr.transAxes)

    # Question
    q_wrapped = textwrap.fill(f"Q: {question}", width=int(args.wrap_width))
    ax_hdr.text(0.0, 0.82, q_wrapped,
                ha="left", va="top", fontsize=11.5, color="#111111",
                fontweight="bold", transform=ax_hdr.transAxes,
                linespacing=1.35)

    # Options
    opt_labels = {
        "correct_answer":          ("✓ Correct",         "#1a7d3a"),
        "misleading_groundable":   ("✗ Mislead (ground.)", "#c0392b"),
        "misleading_ungroundable": ("✗ Mislead (unground.)", "#922b21"),
        "irrelevant_word":         ("~ Irrelevant",       "#d97b00"),
    }
    opt_x = 0.0
    opt_y = 0.30
    for key, text in options:
        label, color = opt_labels.get(key, (key, "#333333"))
        snippet = textwrap.shorten(f'{label}: "{text}"', width=70, placeholder="…")
        ax_hdr.text(opt_x, opt_y, snippet,
                    ha="left", va="top", fontsize=9.5, color=color,
                    transform=ax_hdr.transAxes)
        opt_x += 0.25
        if opt_x > 0.9:
            opt_x = 0.0
            opt_y -= 0.18

    # Mode badge
    mode_label = "Teacher-Forced IG" if args.mode == "teacher_forced" else "Prefill Next-Token IG"
    ax_hdr.text(1.0, 0.97, f"Mode: {mode_label}",
                ha="right", va="top", fontsize=8.5, color="#555555",
                style="italic", transform=ax_hdr.transAxes)

    # Thin separator below header
    ax_hdr.axhline(0.0, color="#cccccc", linewidth=0.8)

    # ── Image grid ────────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols + 1,          # +1 for row label column
        subplot_spec=outer[1],
        wspace=0.04,
        hspace=0.10,
        width_ratios=[0.20] + [1.0] * ncols,
    )

    for r in range(nrows):
        # Row label
        ax_rl = fig.add_subplot(inner[r, 0])
        ax_rl.set_facecolor("white")
        ax_rl.axis("off")
        ax_rl.text(0.5, 0.5, ROW_LABELS[r],
                   ha="center", va="center", fontsize=11, fontweight="bold",
                   color="#222222", transform=ax_rl.transAxes,
                   rotation=90)

    for c, variant in enumerate(VARIANTS_ORDER):
        img = load_variant_image(sample, variant)
        text_bbox = get_text_bbox(sample, variant)

        token_scores, summary, mapping_tokens = load_npz(
            "./llava-next_ig_token_outputs_correct_answer_token", variant, ques_id, args.mode
        )
        base_grid, mosaic_grid = token_scores_to_grids(mapping_tokens, token_scores, summary)

        base_norm = make_overlay_norm(
            img, base_grid, signed=signed,
            smooth_sigma_grid=args.smooth_sigma_grid,
            clip_percentiles=clip_percentiles,
        )
        mosaic_norm = make_overlay_norm(
            img, mosaic_grid, signed=signed,
            smooth_sigma_grid=args.smooth_sigma_grid,
            clip_percentiles=clip_percentiles,
        )

        for r, (norm, grid_label) in enumerate([(base_norm, "BASE"), (mosaic_norm, "MOSAIC")]):
            ax = fig.add_subplot(inner[r, c + 1])
            ax.imshow(img, interpolation="bilinear")

            rgba = overlay_to_rgba(
                norm, signed=signed, cmap_name=cmap_name,
                base_alpha=args.base_alpha,
                mask_thr=args.mask_thr,
                gamma=args.alpha_gamma,
            )
            ax.imshow(rgba, interpolation="bilinear")

            draw_boxes(
                ax,
                correct_obj=correct_obj_xyxy,
                misleading_obj=misleading_obj_xyxy,
                text_bbox=text_bbox,
                lw=args.bbox_linewidth,
            )

            # Top title: variant name (colour coded), only on first row
            if r == 0:
                v_color = VARIANT_COLORS[variant]
                # Coloured top bar
                ax.set_title(
                    VARIANT_LABELS[variant],
                    fontsize=10, fontweight="bold",
                    color="white", pad=4,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=v_color,
                        edgecolor="none",
                    ),
                )

            # Stats annotation at bottom
            if args.show_stats:
                stats_str = cell_stats(norm, text_bbox, (img.height, img.width))
                ax.text(
                    0.01, 0.02, stats_str,
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=6.5, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0),
                )

            # Model response annotation (optional)
            if variant in responses and r == 0:
                resp_text = textwrap.shorten(f'→ "{responses[variant]}"', width=38, placeholder="…")
                ax.text(
                    0.5, -0.04, resp_text,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.0, color=VARIANT_COLORS[variant], style="italic",
                )

            ax.set_aspect("equal")
            ax.axis("off")

    # ── Legend panel (bottom) ─────────────────────────────────────────────────
    ax_leg = fig.add_subplot(outer[2])
    ax_leg.set_facecolor("#f8f8f8")
    ax_leg.axis("off")

    legend_entries = [
        Line2D([0], [0], color="#4a90d9", linewidth=2.0, label="Text region (bbox)"),
        Line2D([0], [0], color="#27ae60", linewidth=2.0, label="Correct-answer object"),
        Line2D([0], [0], color="#e74c3c", linewidth=2.0, linestyle="--",
               label="Misleading object (groundable)"),
    ]

    # Colorbar interpretation entries (fake artists)
    if signed:
        legend_entries += [
            patches.Patch(facecolor=matplotlib.cm.get_cmap(cmap_name)(0.05),
                          edgecolor="none", label="Negative IG (suppresses prediction)"),
            patches.Patch(facecolor=matplotlib.cm.get_cmap(cmap_name)(0.95),
                          edgecolor="none", label="Positive IG (supports prediction)"),
            patches.Patch(facecolor=matplotlib.cm.get_cmap(cmap_name)(0.50),
                          edgecolor="none", alpha=0.0, label="(neutral = transparent)"),
        ]
    else:
        legend_entries += [
            patches.Patch(facecolor=matplotlib.cm.get_cmap(cmap_name)(0.05),
                          edgecolor="none", label="Low attribution"),
            patches.Patch(facecolor=matplotlib.cm.get_cmap(cmap_name)(0.95),
                          edgecolor="none", label="High attribution"),
        ]

    ax_leg.legend(
        handles=legend_entries,
        loc="center",
        ncol=min(len(legend_entries), 6),
        fontsize=8.5,
        frameon=True,
        framealpha=0.0,
        edgecolor="#cccccc",
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    ax_leg.text(
        0.0, 0.95,
        "Legend",
        ha="left", va="top", fontsize=9, fontweight="bold",
        color="#555555", transform=ax_leg.transAxes,
    )

    # ── Shared colorbar (right side) ──────────────────────────────────────────
    sm = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.cm.get_cmap(cmap_name),
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1) if signed
             else matplotlib.colors.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3)

    if signed:
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.set_ticklabels(["−1\n(neg.)", "−0.5", "0\n(neutral)", "+0.5", "+1\n(pos.)"],
                            fontsize=7.5)
        cbar.set_label("Normalised IG score", fontsize=8.5, labelpad=6)
    else:
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=7.5)
        cbar.set_label("Attribution (normalised)", fontsize=8.5, labelpad=6)

    cbar.outline.set_linewidth(0.8)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = (
        Path(args.save_path)
        if args.save_path
        else Path(args.out_dir) / f"grid_{ques_id}_{args.mode}_2x5.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=int(args.dpi), bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()