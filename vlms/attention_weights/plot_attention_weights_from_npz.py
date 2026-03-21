#!/usr/bin/env python3
# overlay_from_npz_both.py
"""
Regenerate attention overlays from NPZ by re-loading the GUIC dataset image.

Supports BOTH:
- Llava-NeXT attention NPZs
- Qwen attention NPZs

Expected NPZ layouts:
  <npz_root>/<variant>/<qid>/gen_attn_gen_token.npz
  <npz_root>/<variant>/<variant>/<qid>/gen_attn_gen_token.npz

Supported mapping keys:
- packed_mapping_tokens / packed_mapping_summary
- mapping_tokens / mapping_summary

Output:
  out_dir/<variant>/<qid>/overlays/
      attn_<stream>_layerXX.png

Streams:
- Llava-NeXT: base, mosaic
- Qwen: merged
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from PIL import ImageDraw


# -----------------------------
# Visualization helpers
# -----------------------------
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


def keep_top_bottom_k_within_percentiles(
    grid: np.ndarray,
    k: int,
    clip_percentiles=(5, 95),
    *,
    signed=True,
    fill_value=0.0,
):
    g = grid.astype(np.float32, copy=True)
    flat = g.reshape(-1)

    lo = np.percentile(flat, clip_percentiles[0])
    hi = np.percentile(flat, clip_percentiles[1])

    in_band = (flat >= lo) & (flat <= hi) & np.isfinite(flat)
    idx_band = np.nonzero(in_band)[0]

    if k <= 0 or idx_band.size == 0:
        mask = np.zeros_like(flat, dtype=bool)
        return np.full_like(g, fill_value, dtype=np.float32), mask.reshape(g.shape)

    vals_band = flat[idx_band]
    k_eff = min(k, idx_band.size)

    bot_rel = np.argpartition(vals_band, k_eff - 1)[:k_eff]
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
    show_top_bottom_k: int = 0,
    clip_percentiles=(5, 95),
):
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    H, W = img.height, img.width
    gh, gw = grid.shape

    if show_top_bottom_k and show_top_bottom_k > 0:
        grid_kept, mask = keep_top_bottom_k_within_percentiles(
            grid,
            k=show_top_bottom_k,
            clip_percentiles=clip_percentiles,
            signed=signed,
            fill_value=0.0,
        )
    else:
        grid_kept = grid
        mask = np.ones_like(grid_kept, dtype=bool)

    scale_h = H / gh
    scale_w = W / gw
    grid_up = zoom(grid_kept, (scale_h, scale_w), order=0)
    mask_up = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5

    grid_norm = robust_normalize(grid_up, signed=signed)
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


# -----------------------------
# Region drawing helpers
# -----------------------------
def xyxy_to_yxyx(bb):
    x1, y1, x2, y2 = bb
    return (y1, x1, y2, x2)


def xywh_to_yxyx(x, y, w, h):
    return (y, x, y + h, x + w)


def get_text_bbox_xyxy(sample: dict, variant: str, fallback_variant: str = "correct_answer"):
    v = variant
    if v == "notext":
        v = fallback_variant

    if v not in sample:
        return None

    d = sample[v]
    for k in ["bbox", "text_bbox", "text_box", "bbox_xyxy"]:
        if k in d:
            return d[k]
    if "annotations" in d and isinstance(d["annotations"], dict):
        for k in ["bbox", "text_bbox", "bbox_xyxy"]:
            if k in d["annotations"]:
                return d["annotations"][k]
    return None


def build_regions_yxyx(sample: dict, variant: str):
    regions = {}

    tb = get_text_bbox_xyxy(sample, variant, fallback_variant="correct_answer")
    if tb is not None:
        regions["text region"] = xyxy_to_yxyx(tb)

    ca = sample.get("correct_answer", {})
    if all(k in ca for k in ["x", "y", "w", "h"]):
        regions["correct object region"] = xywh_to_yxyx(ca["x"], ca["y"], ca["w"], ca["h"])

    mg = sample.get("misleading_groundable", {})
    if all(k in mg for k in ["x", "y", "w", "h"]):
        regions["misleading object region"] = xywh_to_yxyx(mg["x"], mg["y"], mg["w"], mg["h"])

    return regions


def draw_region_boxes_on_image(img, regions_yxyx: dict, width: int = 4):
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)

    colors = {
        "text region": "blue",
        "correct object region": "green",
        "misleading object region": "red",
    }

    for name, (y0, x0, y1, x1) in regions_yxyx.items():
        c = colors.get(name, "white")
        draw.rectangle([x0, y0, x1, y1], outline=c, width=width)
        draw.text((x0 + 3, y0 + 3), name, fill=c)

    return img2


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


# -----------------------------
# NPZ loading
# -----------------------------
def _json_from_npz_field(data, *names):
    for name in names:
        if name in data:
            val = data[name]
            try:
                return json.loads(str(val))
            except Exception:
                try:
                    if isinstance(val, np.ndarray) and val.shape == ():
                        return json.loads(str(val.item()))
                except Exception:
                    pass
    return None


def load_npz(npz_path: Path):
    data = np.load(str(npz_path), allow_pickle=True)
    attn = data["attn"].astype(np.float32)
    img_pos = data["image_placeholder_positions"].astype(np.int64)

    mapping_tokens = _json_from_npz_field(data, "packed_mapping_tokens", "mapping_tokens")
    summary = _json_from_npz_field(data, "packed_mapping_summary", "mapping_summary")
    meta = _json_from_npz_field(data, "meta")

    if mapping_tokens is None:
        raise RuntimeError(f"No mapping tokens found in {npz_path}")
    if summary is None:
        raise RuntimeError(f"No mapping summary found in {npz_path}")
    if meta is None:
        meta = {}

    return attn, img_pos, mapping_tokens, summary, meta


# -----------------------------
# Stream detection + grid conversion
# -----------------------------
def infer_streams(mapping_tokens: List[Dict[str, Any]]) -> List[str]:
    kinds = {t.get("kind") for t in mapping_tokens}
    streams = []
    if "base_patch" in kinds:
        streams.append("base")
    if "mosaic_patch" in kinds:
        streams.append("mosaic")
    if "merged_patch" in kinds:
        streams.append("merged")
    return streams


def token_scores_to_stream_grids(mapping_tokens, token_scores, summary):
    """
    Returns dict stream -> grid
    Supports:
      - Llava-NeXT: base, mosaic
      - Qwen: merged
    """
    grids = {}
    kinds = {t.get("kind") for t in mapping_tokens}

    if "base_patch" in kinds:
        patches_per_side = summary["patches_per_side"]
        base_grid = np.zeros((patches_per_side, patches_per_side), dtype=np.float32)
        for t in mapping_tokens:
            if t.get("kind") == "base_patch":
                idx = t["token_idx"]
                base_grid[t["row"], t["col"]] += float(token_scores[idx])
        grids["base"] = base_grid

    if "mosaic_patch" in kinds:
        mosaic_h, mosaic_w = summary["mosaic_unpadded_hw_in_patches"]
        mosaic_grid = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
        for t in mapping_tokens:
            if t.get("kind") == "mosaic_patch":
                idx = t["token_idx"]
                mosaic_grid[t["row"], t["col"]] += float(token_scores[idx])
        grids["mosaic"] = mosaic_grid

    if "merged_patch" in kinds:
        merged_h, merged_w = summary["merged_grid_hw"]
        merged_grid = np.zeros((merged_h, merged_w), dtype=np.float32)
        for t in mapping_tokens:
            if t.get("kind") == "merged_patch":
                idx = t["token_idx"]
                merged_grid[t["row"], t["col"]] += float(token_scores[idx])
        grids["merged"] = merged_grid

    return grids


# -----------------------------
# QID normalization / NPZ finder
# -----------------------------
def find_npz_for_qid(npz_root: Path, variant: str, qid: str) -> Optional[Path]:
    candidates = [qid]
    qid_strip = qid.lstrip("0") or "0"
    if qid_strip != qid:
        candidates.append(qid_strip)
    if qid.isdigit():
        candidates.append(qid.zfill(8))

    patterns = [
        "{variant}/{qid}/gen_attn_gen_token.npz",
        "{variant}/{variant}/{qid}/gen_attn_gen_token.npz",
    ]

    for c in candidates:
        for pat in patterns:
            p = npz_root / Path(pat.format(variant=variant, qid=c))
            if p.exists():
                return p

    return None


# -----------------------------
# Image retrieval
# -----------------------------
def build_qid_index(ds: Dataset) -> Dict[str, int]:
    idx = {}
    for i in range(len(ds)):
        qid = str(ds[i].get("question_id"))
        idx[qid] = i
        if qid.isdigit():
            idx[str(int(qid))] = i
            idx[qid.zfill(8)] = i
    return idx


def get_image_and_sample_for_qid(ds, qid_index, qid, variant):
    if qid not in qid_index:
        if qid.isdigit():
            if str(int(qid)) in qid_index:
                qid = str(int(qid))
            elif qid.zfill(8) in qid_index:
                qid = qid.zfill(8)

    if qid not in qid_index:
        raise KeyError(f"qid {qid} not found in dataset index.")

    sample = ds[qid_index[qid]]

    if variant == "notext":
        img = sample["notext"]["image"]
    else:
        if variant not in sample:
            raise KeyError(f"variant '{variant}' not found for qid={qid}.")
        img = sample[variant]["image"]

    return img.convert("RGB"), sample


# -----------------------------
# Overlay generation
# -----------------------------
def regenerate_overlays_from_npz(
    npz_path: Path,
    image,
    out_dir: Path,
    *,
    show_top_bottom_k_base: int = 50,
    show_top_bottom_k_mosaic: int = 50,
    show_top_bottom_k_merged: int = 50,
    clip_percentiles=(5, 99),
):
    attn, img_pos, mapping_tokens, summary, meta = load_npz(npz_path)

    # summary key differs slightly by family but we normalize to one expected field
    if "total_packed_image_tokens" in summary:
        expected = int(summary["total_packed_image_tokens"])
    elif "num_image_tokens" in summary:
        expected = int(summary["num_image_tokens"])
    else:
        raise RuntimeError(f"Could not infer expected image token count from summary in {npz_path}")

    if len(img_pos) != expected:
        raise RuntimeError(
            f"Mapping mismatch for {npz_path}: len(img_pos)={len(img_pos)} != expected={expected}"
        )

    stream_topk = {
        "base": show_top_bottom_k_base,
        "mosaic": show_top_bottom_k_mosaic,
        "merged": show_top_bottom_k_merged,
    }

    L = attn.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(L):
        token_scores = attn[layer_idx, img_pos].astype(np.float32)
        token_scores /= (token_scores.sum() + 1e-12)

        stream_grids = token_scores_to_stream_grids(
            mapping_tokens=mapping_tokens,
            token_scores=token_scores,
            summary=summary,
        )

        for stream, grid in stream_grids.items():
            overlay_grid_block_on_image(
                img=image,
                grid=grid,
                out_path=str(out_dir / f"attn_{stream}_layer{layer_idx:02d}.png"),
                title=f"Layer {layer_idx} {stream.upper()}",
                signed=False,
                cmap="jet",
                alpha=0.6,
                show_top_bottom_k=stream_topk.get(stream, 50),
                clip_percentiles=clip_percentiles,
            )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=str, default="./qwen-vl_attentions",
                        help="Root containing attention NPZs")
    parser.add_argument("--variant", type=str, default="correct_answer")
    parser.add_argument("--qid", type=str, default="02824734",
                        help="If provided, only process this qid. Otherwise process all qids found under variant.")
    parser.add_argument("--out_dir", type=str, default="plots_qwen")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--show_top_bottom_k_base", type=int, default=50)
    parser.add_argument("--show_top_bottom_k_mosaic", type=int, default=50)
    parser.add_argument("--show_top_bottom_k_merged", type=int, default=50)
    parser.add_argument("--clip_lo", type=float, default=5.0)
    parser.add_argument("--clip_hi", type=float, default=98.0)
    parser.add_argument("--plot_overlay", action="store_true")

    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    variant = args.variant
    out_root = Path(args.out_dir)

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    qid_index = build_qid_index(ds)

    if args.qid:
        qids = [args.qid]
    else:
        candidate_dirs = [
            npz_root / variant,
            npz_root / variant / variant,
        ]
        var_dir = next((d for d in candidate_dirs if d.exists()), None)
        if var_dir is None:
            raise RuntimeError(f"Variant dir does not exist under either {candidate_dirs[0]} or {candidate_dirs[1]}")
        qids = sorted([p.name for p in var_dir.iterdir() if p.is_dir()])

    n_ok = 0
    for qid in qids:
        npz_path = find_npz_for_qid(npz_root, variant, qid)
        if npz_path is None:
            print(f"[WARN] NPZ not found for qid={qid} under {npz_root / variant}. Skipping.")
            continue

        try:
            img, sample = get_image_and_sample_for_qid(ds, qid_index, qid, variant)
            regions = build_regions_yxyx(sample, variant)
            img_boxed = draw_region_boxes_on_image(img, regions, width=4)
        except Exception as e:
            print(f"[WARN] Failed to load image for qid={qid}, variant={variant}: {e}. Skipping.")
            continue

        out_dir = out_root / variant / qid / "overlays"
        if args.plot_overlay:
            try:
                regenerate_overlays_from_npz(
                    npz_path=npz_path,
                    image=img_boxed,
                    out_dir=out_dir,
                    show_top_bottom_k_base=args.show_top_bottom_k_base,
                    show_top_bottom_k_mosaic=args.show_top_bottom_k_mosaic,
                    show_top_bottom_k_merged=args.show_top_bottom_k_merged,
                    clip_percentiles=(args.clip_lo, args.clip_hi),
                )
                n_ok += 1
                print(f"[OK] {qid} -> {out_dir}")
            except Exception as e:
                print(f"[WARN] Failed overlays for qid={qid}: {e}")

    if args.plot_overlay:
        print(f"Done. Wrote overlays for {n_ok}/{len(qids)} qids.")


if __name__ == "__main__":
    main()