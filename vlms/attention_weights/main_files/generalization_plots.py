#!/usr/bin/env python3
"""
plot_attention_generalization.py
================================
Generates two attention-based generalization figures for the paper.

Fig A  (2 × 1, single-column ~3.5")  — Qwen-VL, misleading_groundable
  (a) Prediction changed  [fooled:  correct → wrong via misleading text]
  (b) Prediction stable   [robust:  correct → correct despite misleading text]

Fig BC (2 × 4, two-column  ~7.0")   — LLaVA-NeXT, changed vs stable
  Row 0 — prediction *changed*
    (a) Mis. grounded [fooled]     (b) Mis. ungrounded [fooled]
    (c) Correct answer [helped]    (d) Irrelevant word [fooled]
  Row 1 — prediction *stable*
    (e) Mis. grounded [robust]     (f) Mis. ungrounded [robust]
    (g) Correct answer [consistently wrong]  (h) Irrelevant word [robust]

Update FILE PATHS before running.
Outputs: figA_qwen_misleading_groundable.{pdf,png}
         figBC_llava_changed_vs_stable.{pdf,png}
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


# =============================================================================
# FILE PATHS — update these
# =============================================================================
LLAVA_NPZ_ROOT = "/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/attention_weights/llava-next_attentions"
QWEN_NPZ_ROOT  = "/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/attention_weights/qwen-vl_attentions"
HF_DATASET     = "AHAAM/GUIC"
HF_CACHE_DIR   = "../integrated_gradients/hf_dataset_GUIC"
QID_FILE       = "../inference/no_overlap_question_ids.txt"
OUTPUT_DIR     = "./"


# =============================================================================
# Unified rcParams — identical block used across all paper figures.
# Nunito is the primary font; DejaVu Sans is the fallback for glyphs
# (e.g. arrows ↑↓) that Nunito does not cover.
# =============================================================================
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Nunito", "DejaVu Sans"],
    "font.size":          9,
    "axes.titlesize":     9.5,
    "axes.labelsize":     9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8.5,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "figure.dpi":         100,
    "savefig.dpi":        300,
})


# =============================================================================
# ACL layout + style constants
# =============================================================================
ACL_COL_W  = 3.5    # single column width (inches)
ACL_2COL_W = 7.0    # two column width (inches)
ACL_ROW_H  = 2.0    # height per row (inches)
LINE_W     = 1.4    # main line width
ALPHA_CI   = 0.18   # CI band transparency


# =============================================================================
# Shared colour palette (Okabe-Ito, colorblind-safe)
# — identical to every other figure in the paper
# =============================================================================
REGION_COLORS = {
    "text region":              "#0072B2",   # blue
    "correct object region":    "#009E73",   # green
    "misleading object region": "#D55E00",   # orange-red
}
REGION_NAMES = list(REGION_COLORS.keys())

REGION_LABELS = {
    "text region":              "Text region",
    "correct object region":    "Correct object",
    "misleading object region": "Misleading object",
}

# Highlight band colour for the "competition region" span
HIGHLIGHT_COLOR = "#AA99CC"
HIGHLIGHT_ALPHA = 0.25
HIGHLIGHT_LABEL = "Competition Region"


# =============================================================================
# Panel specification
# =============================================================================
@dataclass
class PanelSpec:
    """Fully describes one subplot panel."""
    npz_root:         str
    variant:          str   # e.g. "misleading_groundable"
    subset:           str   # "fooled" | "robust" | "helped" | "consistently_wrong"
    preferred_stream: str   # "mosaic" | "base" | "merged" — first available used
    title:            str   # panel letter + short label, e.g. "(a) Mis. grounded"
    show_ylabel:      bool = False
    show_xlabel:      bool = True


# =============================================================================
# Dataset utilities
# =============================================================================
def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(
    dataset_id: str,
    local_cache_root: str,
    split: str = "test",
) -> Dataset:
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = local_cache_root / sanitize_repo_id(dataset_id)
    if cache_dir.exists():
        return load_from_disk(str(cache_dir))
    ds = load_dataset(dataset_id, split=split)
    try:
        ds.save_to_disk(str(cache_dir))
    except Exception:
        pass
    return ds


def build_qid_index(ds: Dataset) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i in range(len(ds)):
        qid = str(ds[i].get("question_id"))
        idx[qid] = i
        if qid.isdigit():
            idx[str(int(qid))] = i
            idx[qid.zfill(8)] = i
    return idx


def load_qid_list(path: str) -> Optional[List[str]]:
    if not path:
        return None
    qids: List[str] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                qids.append(s)
    return qids


# =============================================================================
# BBox / region utilities
# =============================================================================
def xyxy_to_yxyx(bb) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb
    return (float(y1), float(x1), float(y2), float(x2))


def xywh_to_yxyx(x, y, w, h) -> Tuple[float, float, float, float]:
    return (float(y), float(x), float(y + h), float(x + w))


def get_text_bbox_xyxy(
    sample: dict,
    variant: str,
    fallback_variant: str = "correct_answer",
) -> Optional[list]:
    v = variant if variant != "notext" else fallback_variant
    if v not in sample:
        return None
    d = sample[v]
    for k in ("bbox", "text_bbox", "text_box", "bbox_xyxy"):
        if k in d:
            return d[k]
    if "annotations" in d and isinstance(d["annotations"], dict):
        for k in ("bbox", "text_bbox", "bbox_xyxy"):
            if k in d["annotations"]:
                return d["annotations"][k]
    return None


def build_regions_yxyx(
    sample: dict,
    variant: str,
) -> Dict[str, Tuple[float, float, float, float]]:
    regions: Dict[str, Tuple[float, float, float, float]] = {}

    tb = get_text_bbox_xyxy(sample, variant, fallback_variant="correct_answer")
    if tb is not None:
        regions["text region"] = xyxy_to_yxyx(tb)

    ca = sample.get("correct_answer", {})
    if all(k in ca for k in ("x", "y", "w", "h")):
        regions["correct object region"] = xywh_to_yxyx(
            ca["x"], ca["y"], ca["w"], ca["h"]
        )

    mg = sample.get("misleading_groundable", {})
    if all(k in mg for k in ("x", "y", "w", "h")):
        regions["misleading object region"] = xywh_to_yxyx(
            mg["x"], mg["y"], mg["w"], mg["h"]
        )

    return regions


# =============================================================================
# NPZ loading
# =============================================================================
def find_npz(npz_root: Path, variant: str, qid: str) -> Optional[Path]:
    """
    On-disk layout:
      <npz_root>/<variant>/<variant>/<qid>/gen_attn_gen_token.npz
    """
    base = npz_root / variant / variant
    candidates = [qid, qid.lstrip("0") or "0"]
    if qid.isdigit():
        candidates.append(qid.zfill(8))
    for c in candidates:
        p = base / c / "gen_attn_gen_token.npz"
        if p.exists():
            return p
    return None


def load_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    return {
        "attn":           data["attn"].astype(np.float32),
        "img_pos":        data["image_placeholder_positions"].astype(np.int64),
        "mapping_tokens": json.loads(str(data["packed_mapping_tokens"])),
        "summary":        json.loads(str(data["packed_mapping_summary"])),
        "meta":           json.loads(str(data["meta"])) if "meta" in data else {},
    }


# =============================================================================
# Token → region overlap + density_ratio
# =============================================================================
def overlap_frac_yxyx(a: tuple, b: tuple) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih = max(0.0, iy1 - iy0)
    iw = max(0.0, ix1 - ix0)
    at = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    return float(ih * iw / at) if at > 0 else 0.0


def infer_stream_names(mapping_tokens: List[dict]) -> List[str]:
    kinds = {t.get("kind") for t in mapping_tokens}
    if "merged_patch" in kinds:
        return ["merged"]
    streams = []
    if "base_patch"   in kinds: streams.append("base")
    if "mosaic_patch" in kinds: streams.append("mosaic")
    return streams


def get_stream_token_indices(
    mapping_tokens: List[dict],
) -> Dict[str, np.ndarray]:
    kinds = {t.get("kind") for t in mapping_tokens}
    if "merged_patch" in kinds:
        return {
            "merged": np.asarray(
                [int(t["token_idx"]) for t in mapping_tokens
                 if t.get("kind") == "merged_patch"],
                dtype=np.int64,
            )
        }
    out: Dict[str, np.ndarray] = {}
    base   = [int(t["token_idx"]) for t in mapping_tokens if t.get("kind") == "base_patch"]
    mosaic = [int(t["token_idx"]) for t in mapping_tokens if t.get("kind") == "mosaic_patch"]
    if base:   out["base"]   = np.asarray(base,   dtype=np.int64)
    if mosaic: out["mosaic"] = np.asarray(mosaic, dtype=np.int64)
    return out


def get_region_token_mask(
    mapping_tokens: List[dict],
    region_yxyx: Tuple[float, float, float, float],
    min_overlap_frac: float,
) -> np.ndarray:
    n = max(int(t["token_idx"]) for t in mapping_tokens) + 1
    mask = np.zeros(n, dtype=bool)
    for t in mapping_tokens:
        bb = t.get("bbox")
        if bb is None:
            continue
        if overlap_frac_yxyx(tuple(bb), region_yxyx) >= min_overlap_frac:
            mask[int(t["token_idx"])] = True
    return mask


def extract_density_ratio(
    attn: np.ndarray,
    img_pos: np.ndarray,
    mapping_tokens: List[dict],
    regions_yxyx: Dict[str, Tuple[float, float, float, float]],
    min_overlap_frac: float = 0.25,
    min_denom: float = 1e-4,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns out[stream][region] = (L,) density_ratio with NaN where invalid.

    density_ratio(region/stream) =
        (avg attn per region token within stream)
        / (avg attn per stream token)
    Validity gating: NaN where stream mass < min_denom.
    """
    L = attn.shape[0]
    token_scores  = attn[:, img_pos]
    stream_to_idx = get_stream_token_indices(mapping_tokens)
    out: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in stream_to_idx}

    region_masks = {
        name: get_region_token_mask(mapping_tokens, bb, min_overlap_frac)
        for name, bb in regions_yxyx.items()
    }

    for stream, idx in stream_to_idx.items():
        if idx.size == 0:
            for rname in regions_yxyx:
                out[stream][rname] = np.full(L, np.nan, dtype=np.float32)
            continue

        stream_mass = token_scores[:, idx].sum(axis=1)
        ok          = stream_mass >= min_denom
        stream_den  = stream_mass / float(idx.size)

        for rname, rmask in region_masks.items():
            in_region = rmask[idx]
            if not np.any(in_region):
                out[stream][rname] = np.full(L, np.nan, dtype=np.float32)
                continue
            r_idx      = idx[in_region]
            region_den = token_scores[:, r_idx].sum(axis=1) / float(r_idx.size)
            ratio      = np.full(L, np.nan, dtype=np.float32)
            ratio[ok]  = (region_den[ok] / (stream_den[ok] + 1e-12)).astype(np.float32)
            out[stream][rname] = ratio

    return out


# =============================================================================
# Subset predicates
# =============================================================================
def parse_pred_letter(meta: Dict) -> Optional[str]:
    s = str(meta.get("generated_token_text", ""))
    for ch in s:
        if ch in ("A", "B", "C", "D"):
            return ch
    return None


def pred_key_from_meta(meta: Dict) -> Optional[str]:
    letter = parse_pred_letter(meta)
    if letter is None:
        return None
    return meta.get("option_meta", {}).get("label_to_key", {}).get(letter)


def is_fooled(meta_v: Dict, meta_n: Dict, variant: str) -> bool:
    correct = meta_v.get("correct_letter")
    pv      = parse_pred_letter(meta_v)
    pn      = parse_pred_letter(meta_n)
    pk      = pred_key_from_meta(meta_v)
    if pv is None or pn is None or correct is None:
        return False
    return (pv != correct) and (pn == correct)


def is_robust(meta_v: Dict, meta_n: Dict, variant: str) -> bool:
    correct = meta_v.get("correct_letter")
    pv      = parse_pred_letter(meta_v)
    pn      = parse_pred_letter(meta_n)
    if pv is None or pn is None or correct is None:
        return False
    return (pv == correct) and (pn == correct)


def is_helped(meta_v: Dict, meta_n: Dict) -> bool:
    correct = meta_v.get("correct_letter")
    pv      = parse_pred_letter(meta_v)
    pn      = parse_pred_letter(meta_n)
    if pv is None or pn is None or correct is None:
        return False
    return (pn != correct) and (pv == correct)


def is_consistently_wrong(meta_v: Dict, meta_n: Dict) -> bool:
    """Both notext and variant predictions are wrong."""
    correct = meta_v.get("correct_letter")
    pv      = parse_pred_letter(meta_v)
    pn      = parse_pred_letter(meta_n)
    if pv is None or pn is None or correct is None:
        return False
    return (pv != correct) and (pn != correct)


def resolve_subset_fn(
    subset: str,
    variant: str,
) -> Callable[[Dict, Dict], bool]:
    """Map subset name → predicate(meta_v, meta_n) -> bool."""
    if subset == "fooled":
        return lambda mv, mn: is_fooled(mv, mn, variant)
    if subset == "robust":
        return lambda mv, mn: is_robust(mv, mn, variant)
    if subset == "helped":
        return lambda mv, mn: is_helped(mv, mn)
    if subset == "consistently_wrong":
        return lambda mv, mn: is_consistently_wrong(mv, mn)
    raise ValueError(f"Unknown subset: {subset!r}")


# =============================================================================
# Core curve collector
# =============================================================================
def collect_curves(
    npz_root:         str,
    variant:          str,
    subset:           str,
    preferred_stream: str,
    ds:               Dataset,
    qid_index:        Dict[str, int],
    qids:             List[str],
    min_overlap_frac: float = 0.25,
    min_denom:        float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Iterate qids, compute density_ratio curves, filter to subset.

    Returns
    -------
    layers : np.ndarray        shape (L,)
    series : Dict[region_name → np.ndarray shape (N_selected, L)]
    n_sel  : int               number of selected examples
    """
    npz_root  = Path(npz_root)
    subset_fn = resolve_subset_fn(subset, variant)

    # infer L_ref and stream names from the first valid NPZ pair
    L_ref:   Optional[int]       = None
    streams: Optional[List[str]] = None

    for qid in qids:
        pv = find_npz(npz_root, variant, qid)
        pn = find_npz(npz_root, "notext", qid)
        if pv is None or pn is None:
            continue
        try:
            dv  = load_npz(pv)
            dn  = load_npz(pn)
            L_v = int(dv["attn"].shape[0])
            L_n = int(dn["attn"].shape[0])
            if L_v != L_n:
                continue
            L_ref   = L_v
            streams = infer_stream_names(dv["mapping_tokens"])
            break
        except Exception:
            continue

    if L_ref is None:
        raise RuntimeError(
            f"collect_curves: could not infer L_ref for variant={variant!r}, "
            f"npz_root={npz_root!r}"
        )

    # pick stream: preferred if available, else first
    stream = preferred_stream if preferred_stream in streams else streams[0]

    nan_vec = lambda: np.full(L_ref, np.nan, dtype=np.float32)

    # accumulate per-region rows for selected examples only
    rows: Dict[str, List[np.ndarray]] = {r: [] for r in REGION_NAMES}

    for qid in qids:
        pv = find_npz(npz_root, variant, qid)
        pn = find_npz(npz_root, "notext", qid)
        if pv is None or pn is None:
            continue

        try:
            dv = load_npz(pv)
            dn = load_npz(pn)
        except Exception:
            continue

        if dv["attn"].shape[0] != L_ref or dn["attn"].shape[0] != L_ref:
            continue

        # mapping sanity check
        try:
            if (
                len(dv["img_pos"]) != int(dv["summary"]["total_packed_image_tokens"])
                or len(dn["img_pos"]) != int(dn["summary"]["total_packed_image_tokens"])
            ):
                continue
        except Exception:
            pass

        # dataset lookup — try canonical forms of the qid
        qid_key = qid
        if qid_key not in qid_index:
            if qid_key.isdigit() and str(int(qid_key)) in qid_index:
                qid_key = str(int(qid_key))
            elif qid_key.isdigit() and qid_key.zfill(8) in qid_index:
                qid_key = qid_key.zfill(8)
            else:
                continue

        sample  = ds[qid_index[qid_key]]
        regions = build_regions_yxyx(sample, variant)
        if not any(r in regions for r in REGION_NAMES):
            continue

        if not subset_fn(dv["meta"], dn["meta"]):
            continue

        ratio = extract_density_ratio(
            dv["attn"], dv["img_pos"], dv["mapping_tokens"], regions,
            min_overlap_frac=min_overlap_frac,
            min_denom=min_denom,
        )
        for rname in REGION_NAMES:
            rows[rname].append(ratio.get(stream, {}).get(rname, nan_vec()))

    n_sel = len(rows[REGION_NAMES[0]])

    if n_sel == 0:
        series = {r: np.full((1, L_ref), np.nan, dtype=np.float32)
                  for r in REGION_NAMES}
        return np.arange(L_ref), series, 0

    series = {r: np.vstack(rows[r]) for r in REGION_NAMES}
    return np.arange(L_ref, dtype=int), series, n_sel


# =============================================================================
# Statistics
# =============================================================================
def nanmean_ci(
    x: np.ndarray,
    ci: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    lo   = np.nanpercentile(x, (100 - ci) / 2,       axis=0)
    hi   = np.nanpercentile(x, 100 - (100 - ci) / 2, axis=0)
    return mean, lo, hi


# =============================================================================
# Style helpers
# =============================================================================
def _apply_acl_style(ax: plt.Axes) -> None:
    """
    Axis-level style tweaks that rcParams cannot express:
    ylabel anchor position and spine visibility.
    All font sizes come from rcParams.
    """
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _shared_legend(
    fig: plt.Figure,
    highlight_color: Optional[str] = None,
    highlight_label: Optional[str] = None,
    bbox_y: float = -0.08,
) -> None:
    """
    Framed legend centred below all panels.
    bbox_y controls vertical anchor: more negative = further below the figure.
    Default -0.01 suits single-column figures; pass e.g. -0.06 for the
    wider 2×4 BC figure where constrained_layout needs more breathing room.
    """
    handles: List = [
        plt.Line2D([0], [0], color=c, linewidth=LINE_W, label=REGION_LABELS[r])
        for r, c in REGION_COLORS.items()
    ]
    if highlight_color and highlight_label:
        handles.append(
            mpatches.Patch(
                facecolor=highlight_color,
                alpha=HIGHLIGHT_ALPHA,
                linewidth=0,
                label=highlight_label,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCC",
        bbox_to_anchor=(0.5, bbox_y),
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.2,
    )


# =============================================================================
# Panel drawing
# =============================================================================
def draw_curves(
    ax:                  plt.Axes,
    layers:              np.ndarray,
    series:              Dict[str, np.ndarray],
    n_sel:               int,
    title:               str   = "",
    show_ylabel:         bool  = False,
    show_xlabel:         bool  = False,
    ci:                  float = 95.0,
    highlight_spans:     Optional[List[Tuple[int, int]]] = None,
    highlight_color:     str   = HIGHLIGHT_COLOR,
    highlight_alpha:     float = HIGHLIGHT_ALPHA,
    highlight_label:     Optional[str] = None,
    _highlight_labeled:  Optional[set] = None,
) -> None:
    """
    Draw density-ratio curves with CI bands onto ax.

    Title format:  "(a) Prediction changed (n=31)"  — single line, left-aligned,
    matching the title style of plot_ig_figures_column.py.
    Font size comes from rcParams axes.titlesize (9.5).
    """
    # highlight bands first — lines rendered on top
    if highlight_spans:
        for j, (x0, x1) in enumerate(highlight_spans):
            already = _highlight_labeled is not None and len(_highlight_labeled) > 0
            lbl = None if (already or j > 0) else highlight_label
            if lbl and _highlight_labeled is not None:
                _highlight_labeled.add(lbl)
            ax.axvspan(
                x0, x1,
                color=highlight_color, alpha=highlight_alpha,
                linewidth=0, label=lbl,
            )

    for rname, mat in series.items():
        m, lo, hi = nanmean_ci(mat, ci=ci)
        color = REGION_COLORS[rname]
        ax.plot(layers, m, label=rname, color=color, linewidth=LINE_W)
        # ax.fill_between(layers, lo, hi, color=color, alpha=ALPHA_CI)

    # Title: "(a) Panel label (n=31)" on a single line, bold, left-aligned.
    # No explicit fontsize — uses rcParams axes.titlesize = 9.5, same as
    # the "(a) Harmful text caused the flip" titles in plot_ig_figures_column.py.
    title_text = f"{title}" if n_sel > 0 else title
    ax.set_title(title_text, fontweight="bold", loc="left", pad=6)

    # Axis labels — fontsize from rcParams axes.labelsize = 9
    if show_ylabel:
        ax.set_ylabel("Attention ratio")
    if show_xlabel:
        ax.set_xlabel("Layer")

    _apply_acl_style(ax)


# =============================================================================
# Dataset bootstrap
# =============================================================================
def load_shared_data() -> Tuple[Dataset, Dict[str, int], List[str]]:
    ds = get_or_download_hf_dataset(HF_DATASET, HF_CACHE_DIR, split="test")
    if isinstance(ds, DatasetDict):
        ds = ds["test"]
    qid_index = build_qid_index(ds)
    qids      = load_qid_list(QID_FILE) or []
    return ds, qid_index, qids


# =============================================================================
# Save helper
# =============================================================================
def _save(fig: plt.Figure, output_dir: str, name: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # dpi = 300 from rcParams savefig.dpi; no explicit override needed
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {name}.pdf  +  {name}.png")


# =============================================================================
# Fig A — Qwen, misleading_groundable, fooled vs robust  (2 × 1)
# =============================================================================
def make_fig_A(
    ds:              Dataset,
    qid_index:       Dict[str, int],
    qids:            List[str],
    output_dir:      str                     = OUTPUT_DIR,
    ci:              float                   = 95.0,
    highlight_spans: List[Tuple[int, int]]   = [(11, 16)],
    highlight_color: str                     = HIGHLIGHT_COLOR,
    highlight_label: str                     = HIGHLIGHT_LABEL,
) -> None:
    panels = [
        PanelSpec(
            npz_root=LLAVA_NPZ_ROOT, variant="misleading_groundable",
            subset="fooled",  preferred_stream="merged",
            title="(a) Correct → Wrong",
            show_ylabel=True, show_xlabel=False,
        ),
        PanelSpec(
            npz_root=LLAVA_NPZ_ROOT, variant="misleading_groundable",
            subset="robust",  preferred_stream="merged",
            title="(b) Correct Prediction Unchanged",
            show_ylabel=True, show_xlabel=True,
        ),
    ]

    # fig, axes = plt.subplots(
    #     nrows=2, ncols=1,
    #     figsize=(ACL_COL_W, ACL_ROW_H * 2),
    #     sharex=True,
    #     facecolor="white",
    #     gridspec_kw={"hspace": 0.22},
    # )
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(ACL_COL_W, ACL_ROW_H * 2.3), facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, top=0.92, bottom=0.25, left=0.15, right=0.97)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    axes[1].sharex(axes[0])

    labeled: set = set()
    for ax, spec in zip(axes, panels):
        layers, series, n_sel = collect_curves(
            spec.npz_root, spec.variant, spec.subset,
            spec.preferred_stream, ds, qid_index, qids,
        )
        draw_curves(
            ax, layers, series, n_sel,
            title=spec.title,
            show_ylabel=spec.show_ylabel,
            show_xlabel=False,   # set manually below (sharex)
            ci=ci,
            highlight_spans=highlight_spans,
            highlight_color=highlight_color,
            highlight_label=highlight_label,
            _highlight_labeled=labeled,
        )

    # x-axis label on bottom panel only; top panel ticks suppressed by sharex
    axes[-1].set_xlabel("Layer")
    plt.setp(axes[0].get_xticklabels(), visible=False)

    # gap_y = (axes[0].get_position().y0 + axes[1].get_position().y1) / 2

    # fig.add_artist(
    #     mpl.patches.FancyArrowPatch(
    #         posA=(0.5, gap_y + 0.03),
    #         posB=(0.5, gap_y - 0.03),
    #         transform=fig.transFigure,
    #         arrowstyle="->,head_width=0.3,head_length=0.4",
    #         color="#555555",
    #         linewidth=1.0,
    #         clip_on=False,
    #     )
    # )
    # fig.tight_layout(pad=0.4)
    # fig.subplots_adjust(top=0.92, bottom=0.28, left=0.15, right=0.97, hspace=0.45)
    # fig.subplots_adjust(bottom=0.28, hspace = 0.25)
    _shared_legend(fig, highlight_color=highlight_color,
                   highlight_label=highlight_label, bbox_y=0.04)
    # Save directly without bbox_inches="tight"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figA_llava_misleading_groundable.{ext}", facecolor="white")
    plt.close(fig)
    # _save(fig, output_dir, "figA_llava_misleading_groundable")


def make_fig_BC(
    ds:              Dataset,
    qid_index:       Dict[str, int],
    qids:            List[str],
    output_dir:      str                   = OUTPUT_DIR,
    ci:              float                 = 95.0,
    highlight_spans: List[Tuple[int, int]] = [(10, 17)],
    highlight_color: str                   = HIGHLIGHT_COLOR,
    highlight_label: str                   = HIGHLIGHT_LABEL,
) -> None:

    panel_grid = [
        [
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="misleading_groundable",   subset="fooled",            preferred_stream="mosaic", title="", show_ylabel=True,  show_xlabel=False),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="misleading_ungroundable", subset="fooled",            preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=False),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="correct_answer",          subset="helped",            preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=False),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="irrelevant_word",         subset="fooled",            preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=False),
        ],
        [
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="misleading_groundable",   subset="robust",            preferred_stream="mosaic", title="", show_ylabel=True,  show_xlabel=True),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="misleading_ungroundable", subset="robust",            preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=True),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="correct_answer",          subset="consistently_wrong",preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=True),
            PanelSpec(npz_root=LLAVA_NPZ_ROOT, variant="irrelevant_word",         subset="robust",            preferred_stream="mosaic", title="", show_ylabel=False, show_xlabel=True),
        ],
    ]

    col_titles = ["Mis. grounded", "Mis. ungrounded", "Correct answer", "Irrelevant word"]
    row_labels  = ["Prediction\nchanged", "Prediction\nstable"]

    # --- Create figure WITHOUT any automatic layout manager ---
    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        figsize=(ACL_2COL_W * 1.6, ACL_ROW_H * 2.0),
        sharex=True, sharey=False,
        facecolor="white",
    )

    # --- Draw all panels ---
    labeled: set = set()
    for ri, (row_specs, row_axes) in enumerate(zip(panel_grid, axes)):
        for spec, ax in zip(row_specs, row_axes):
            layers, series, n_sel = collect_curves(
                spec.npz_root, spec.variant, spec.subset,
                spec.preferred_stream, ds, qid_index, qids,
            )
            draw_curves(
                ax, layers, series, n_sel,
                title=spec.title,
                show_ylabel=spec.show_ylabel,
                show_xlabel=spec.show_xlabel,
                ci=ci,
                highlight_spans=highlight_spans,
                highlight_color=highlight_color,
                highlight_label=highlight_label,
                _highlight_labeled=labeled,
            )

        axes[ri][0].annotate(
            row_labels[ri],
            xy=(-0.55, 0.5), xycoords="axes fraction",
            fontweight="bold",
            ha="right", va="center", rotation=90,
            color="#111111",
        )

    # --- Column titles ---
    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            col_title,
            fontweight="bold", fontsize=9.5, loc="center", pad=22,
        )

    plt.setp(axes[0, :], xlabel="")

    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.18, 0.5)

    # --- Fix spacing AFTER drawing, BEFORE reading positions ---
    fig.subplots_adjust(
        left=0.08, right=0.98,
        bottom=0.18, top=0.88,
        wspace=0.25, hspace=0.12,
    )

    # --- Finalize positions ---
    fig.canvas.draw()

    # --- Vertical divider lines between columns ---
    for col_idx in range(1, 4):
        x_right = axes[0, col_idx - 1].get_position().x1
        x_left  = axes[0, col_idx].get_position().x0
        x_mid   = (x_right + x_left) / 2

        y_bottom = axes[1, 0].get_position().y0
        y_top    = axes[0, 0].get_position().y1

        fig.add_artist(mpl.lines.Line2D(
            [x_mid, x_mid], [y_bottom, y_top],
            transform=fig.transFigure,
            color="#CCCCCC", linewidth=0.8, linestyle="--", clip_on=False,
        ))

    _shared_legend(fig, highlight_color=highlight_color,
                   highlight_label=highlight_label, bbox_y=-0.15)
    _save(fig, output_dir, "figBC_llava_changed_vs_stable")


if __name__ == "__main__":
    ds, qid_index, qids = load_shared_data()
    make_fig_A(ds, qid_index, qids)
    # make_fig_BC(ds, qid_index, qids)