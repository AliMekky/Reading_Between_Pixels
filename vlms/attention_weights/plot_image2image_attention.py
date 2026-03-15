# plot_region_heatmap_all_layers_v2.py
#
# Changes vs your current script:
# 1) OPTIONAL renormalization within the plotted matrix (row-wise) so non-diagonal structure is visible:
#       --normalize none|row
#    For your use-case, use: --normalize row
#
# 2) Robust visualization scaling:
#       --vmax_percentile  (default 99.5)
#    Uses vmax = percentile(A_plot, vmax_percentile) instead of max, so diagonal spikes don’t wash out everything.
#
# 3) Spatial reordering for base/mosaic too (not only full):
#    - base: ordered by (row,col) from mapping_tokens (base_patch)
#    - mosaic: ordered by (row,col) from mapping_tokens (mosaic_patch)
#    This makes region rectangles less “mysterious” and more spatially coherent.
#
# 4) Fixed your NPZ path bug:
#    You currently have: npz_root/variant/variant/qid/...
#    Default fixed to:   npz_root/variant/qid/...
#    (Still configurable via --npz_path_pattern)
#
# Output:
#   <out_dir>/<qid>_<variant>_<block>/heatmap_layerXX.png

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk


# ----------------------------
# GUIC bbox conversions
# ----------------------------
def guic_text_bbox_to_yxyx(b):  # [x1,y1,x2,y2]
    x1, y1, x2, y2 = b
    return (float(y1), float(x1), float(y2), float(x2))


def guic_obj_bbox_to_yxyx(x, y, w, h):  # (x,y,w,h)
    return (float(y), float(x), float(y + h), float(x + w))


# ----------------------------
# Load NPZ + mapping
# ----------------------------
def load_npz(npz_path: Path) -> Dict[str, Any]:
    z = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    summary = json.loads(str(z["packed_mapping_summary"]))
    tokens = json.loads(str(z["packed_mapping_tokens"]))

    base_idx = z["base_token_idx"].astype(np.int64)        # packed ids, len=B
    mosaic_idx = z["mosaic_token_idx"].astype(np.int64)    # packed ids, len=M

    base_attn = z["base_attn_LBB"].astype(np.float32)        # (L,B,B)
    mosaic_attn = z["mosaic_attn_LMM"].astype(np.float32)    # (L,M,M)
    img_attn = z["img_attn_LNN"].astype(np.float32) if "img_attn_LNN" in z else None  # (L,N,N) or None

    return dict(
        meta=meta,
        summary=summary,
        tokens=tokens,
        base_idx=base_idx,
        mosaic_idx=mosaic_idx,
        base_attn=base_attn,
        mosaic_attn=mosaic_attn,
        img_attn=img_attn,
    )


# ----------------------------
# Token selection by overlap (packed token_idx space)
# ----------------------------
def token_region_indices_patch_cover(mapping_tokens, region_yxyx, kind, min_cover=0.2):
    """
    Select tokens of `kind` whose bbox overlaps region bbox by at least `min_cover`
    fraction of the TOKEN bbox area:
        cover = inter_area / token_area
    """
    ry0, rx0, ry1, rx1 = map(float, region_yxyx)
    out = []
    for t in mapping_tokens:
        if t.get("kind") != kind:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue

        ty0, tx0, ty1, tx1 = map(float, bb)

        iy0, ix0 = max(ry0, ty0), max(rx0, tx0)
        iy1, ix1 = min(ry1, ty1), min(rx1, tx1)
        inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)

        token_area = max(0.0, ty1 - ty0) * max(0.0, tx1 - tx0) + 1e-12
        cover = inter / token_area

        if cover >= float(min_cover):
            out.append(int(t["token_idx"]))
    return np.asarray(out, dtype=np.int64)

def token_region_indices(mapping_tokens: List[Dict[str, Any]],
                         region_yxyx: Tuple[float, float, float, float],
                         kind: str,
                         min_cover: float = 0.1) -> np.ndarray:
    ry0, rx0, ry1, rx1 = region_yxyx
    out: List[int] = []
    for t in mapping_tokens:
        if t.get("kind") != kind:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue
        ty0, tx0, ty1, tx1 = map(float, bb)

        iy0, ix0 = max(ry0, ty0), max(rx0, tx0)
        iy1, ix1 = min(ry1, ty1), min(rx1, tx1)
        ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
        inter = ih * iw

        token_area = max(0.0, ty1 - ty0) * max(0.0, tx1 - tx0) + 1e-12
        cover = inter / token_area

        if cover >= min_cover:
            out.append(int(t["token_idx"]))
    return np.asarray(out, dtype=np.int64)


# ----------------------------
# Ordering utilities
# ----------------------------
def build_full_visual_order(mapping_tokens: List[Dict[str, Any]],
                            include_cls: bool = False,
                            include_newline: bool = False) -> np.ndarray:
    """
    Packed token_idx order for full (N,N) plotting.
    Order:
      base_cls (optional)
      base_patch by (row,col)
      mosaic_patch by (row,col)
      newline (optional) by row
    """
    base_cls = []
    base_patch = []
    mosaic_patch = []
    newline = []

    for t in mapping_tokens:
        k = t["kind"]
        idx = int(t["token_idx"])
        if k == "base_cls":
            if include_cls:
                base_cls.append((0, 0, idx))
        elif k == "base_patch":
            base_patch.append((int(t["row"]), int(t["col"]), idx))
        elif k == "mosaic_patch":
            mosaic_patch.append((int(t["row"]), int(t["col"]), idx))
        elif k == "newline":
            if include_newline:
                r = int(t["row"]) if t.get("row") is not None else 10**9
                newline.append((r, 0, idx))

    base_patch.sort()
    mosaic_patch.sort()
    newline.sort()

    order = [x[2] for x in base_cls] + [x[2] for x in base_patch] + [x[2] for x in mosaic_patch] + [x[2] for x in newline]
    return np.asarray(order, dtype=np.int64)


def build_local_spatial_order(mapping_tokens: List[Dict[str, Any]],
                              packed_list: np.ndarray,
                              kind: str) -> np.ndarray:
    """
    Build a local index permutation (0..len(packed_list)-1) ordered by (row,col)
    using mapping_tokens' packed token_idx and the packed_list mapping.
    """
    packed_to_local = {int(v): i for i, v in enumerate(packed_list.tolist())}

    coords = []
    for t in mapping_tokens:
        if t.get("kind") != kind:
            continue
        tid = int(t["token_idx"])
        if tid not in packed_to_local:
            continue
        r = int(t["row"])
        c = int(t["col"])
        coords.append((r, c, packed_to_local[tid]))

    coords.sort()
    order_local = [x[2] for x in coords]
    return np.asarray(order_local, dtype=np.int64)


def reorder_matrix(A: np.ndarray, order: np.ndarray) -> np.ndarray:
    return A[order][:, order]


def region_indices_in_order(region_idx: np.ndarray, order: np.ndarray) -> np.ndarray:
    pos = {int(tok): i for i, tok in enumerate(order.tolist())}
    return np.asarray([pos[int(x)] for x in region_idx.tolist() if int(x) in pos], dtype=np.int64)


# ----------------------------
# Normalization
# ----------------------------
def normalize_matrix(A: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
      - "none": no change
      - "row":  row-normalize (sum over columns = 1)
    """
    if mode == "none":
        return A
    if mode == "row":
        s = A.sum(axis=1, keepdims=True) + 1e-12
        return A / s
    raise ValueError(f"Unknown normalize mode: {mode}")


# ----------------------------
# Downsampling (for display)
# ----------------------------
def downsample_matrix_and_positions(A: np.ndarray,
                                    max_tokens: int,
                                    *pos_arrays: np.ndarray):
    N = A.shape[0]
    ds_factor = 1
    if N <= max_tokens:
        return (A, ds_factor, *pos_arrays)

    ds_factor = int(np.ceil(N / max_tokens))
    A_ds = A[::ds_factor, ::ds_factor]

    def ds_pos(p):
        if p.size == 0:
            return p
        return np.unique((p // ds_factor).astype(np.int64))

    pos_ds = [ds_pos(p) for p in pos_arrays]
    return (A_ds, ds_factor, *pos_ds)


# ----------------------------
# Overlay helpers
# ----------------------------
def draw_region_blocks(ax, src_pos: np.ndarray, dst_pos: np.ndarray, label: str, linewidth: float = 2.0):
    if src_pos.size == 0 or dst_pos.size == 0:
        return
    y0, y1 = int(src_pos.min()), int(src_pos.max())
    x0, x1 = int(dst_pos.min()), int(dst_pos.max())

    rect = plt.Rectangle((x0, y0), (x1 - x0 + 1), (y1 - y0 + 1),
                         fill=False, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(x0, max(0, y0 - 3), label, fontsize=9, va="bottom", ha="left", clip_on=True)


def percentile_vmax(A: np.ndarray, p: float) -> float:
    flat = A.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    vmax = float(np.percentile(flat, p))
    if vmax <= 0:
        vmax = float(np.max(flat)) if flat.size else 1.0
    return vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_root", type=str, default="llava-next_image2image_attentions")
    ap.add_argument("--variant", type=str, default="misleading_groundable",
                    choices=["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word", "notext"])
    ap.add_argument("--qid", type=str, default="17845553")
    ap.add_argument("--block", type=str, default="base", choices=["full", "base", "mosaic"])
    ap.add_argument("--min_cover", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default="heatmaps_v2")
    ap.add_argument("--max_tokens", type=int, default=3000)
    ap.add_argument("--include_cls", action="store_true")
    ap.add_argument("--include_newline", action="store_true")

    # NEW:
    ap.add_argument("--normalize", type=str, default="row", choices=["none", "row"],
                    help="Recommended: row (renormalize within plotted matrix)")
    ap.add_argument("--vmax_percentile", type=float, default=99.5,
                    help="Color scaling: vmax = percentile(A, p). Helps reveal non-diagonal structure.")
    ap.add_argument("--npz_path_pattern", type=str, default="{root}/{variant}/{variant}/{qid}/prefill_img_self.npz",
                    help="Format string with {root} {variant} {qid}. Use this if your directory layout differs.")
    args = ap.parse_args()

    # Load GUIC example to fetch bboxes
    ds = load_from_disk("../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC")
    ex = None
    for e in ds:
        if str(e.get("question_id")) == str(args.qid):
            ex = e
            break
    if ex is None:
        raise RuntimeError(f"qid {args.qid} not found in GUIC split.")

    # Load NPZ
    npz_path = Path(args.npz_path_pattern.format(root=args.npz_root, variant=args.variant, qid=args.qid))
    if not npz_path.exists():
        raise RuntimeError(f"NPZ not found: {npz_path}")

    npz = load_npz(npz_path)
    tokens = npz["tokens"]

    # Region bboxes (original image coords)
    if args.variant == "notext":
        if "notext" not in ex or "bbox" not in ex["notext"]:
            raise RuntimeError("notext bbox missing in example.")
        text_bbox = ex["notext"]["bbox"]
    else:
        text_bbox = ex[args.variant]["bbox"]  # [x1,y1,x2,y2]

    corr_xywh = (ex["correct_answer"]["x"], ex["correct_answer"]["y"], ex["correct_answer"]["w"], ex["correct_answer"]["h"])
    misl_xywh = (ex["misleading_groundable"]["x"], ex["misleading_groundable"]["y"], ex["misleading_groundable"]["w"], ex["misleading_groundable"]["h"])

    text_yxyx = guic_text_bbox_to_yxyx(text_bbox)
    corr_yxyx = guic_obj_bbox_to_yxyx(*corr_xywh)
    misl_yxyx = guic_obj_bbox_to_yxyx(*misl_xywh)

    # Packed token sets (base/mosaic)
    text_base_p = token_region_indices_patch_cover(tokens, text_yxyx, "base_patch", 0.2)
    corr_base_p = token_region_indices_patch_cover(tokens, corr_yxyx, "base_patch", 0.2)
    misl_base_p = token_region_indices_patch_cover(tokens, misl_yxyx, "base_patch", 0.2)

    text_mos_p = token_region_indices_patch_cover(tokens, text_yxyx, "mosaic_patch", 0.2)
    corr_mos_p = token_region_indices_patch_cover(tokens, corr_yxyx, "mosaic_patch", 0.2)
    misl_mos_p = token_region_indices_patch_cover(tokens, misl_yxyx, "mosaic_patch", 0.2)

    text_all_p = np.unique(np.concatenate([text_base_p, text_mos_p])).astype(np.int64)
    corr_all_p = np.unique(np.concatenate([corr_base_p, corr_mos_p])).astype(np.int64)
    misl_all_p = np.unique(np.concatenate([misl_base_p, misl_mos_p])).astype(np.int64)

    # Select attention tensor and build ordering + region positions
    if args.block == "full":
        if npz["img_attn"] is None:
            raise RuntimeError("img_attn_LNN not saved. Re-run caching with --save_full_img_attn.")
        A_L = npz["img_attn"]  # (L,N,N)
        L = A_L.shape[0]
        N = A_L.shape[1]

        order = build_full_visual_order(tokens, include_cls=args.include_cls, include_newline=args.include_newline)
        order = order[(order >= 0) & (order < N)]

        text_pos = region_indices_in_order(text_all_p, order)
        corr_pos = region_indices_in_order(corr_all_p, order)
        misl_pos = region_indices_in_order(misl_all_p, order)

        reorder_kind = "packed"

    elif args.block == "base":
        A_L = npz["base_attn"]  # (L,B,B)
        L = A_L.shape[0]
        base_packed = npz["base_idx"]  # packed token ids in base local order

        # spatial reorder in base local space
        order_local = build_local_spatial_order(tokens, base_packed, kind="base_patch")
        if order_local.size == 0:
            order_local = np.arange(base_packed.size, dtype=np.int64)

        packed_to_local = {int(v): i for i, v in enumerate(base_packed.tolist())}
        text_pos = np.asarray([packed_to_local[int(x)] for x in text_base_p.tolist() if int(x) in packed_to_local], dtype=np.int64)
        corr_pos = np.asarray([packed_to_local[int(x)] for x in corr_base_p.tolist() if int(x) in packed_to_local], dtype=np.int64)
        misl_pos = np.asarray([packed_to_local[int(x)] for x in misl_base_p.tolist() if int(x) in packed_to_local], dtype=np.int64)

        # map region positions into the reordered axis
        inv = {int(old): i for i, old in enumerate(order_local.tolist())}
        text_pos = np.asarray([inv[int(i)] for i in text_pos.tolist() if int(i) in inv], dtype=np.int64)
        corr_pos = np.asarray([inv[int(i)] for i in corr_pos.tolist() if int(i) in inv], dtype=np.int64)
        misl_pos = np.asarray([inv[int(i)] for i in misl_pos.tolist() if int(i) in inv], dtype=np.int64)

        reorder_kind = "local"
        order = order_local

    else:  # mosaic
        A_L = npz["mosaic_attn"]  # (L,M,M)
        L = A_L.shape[0]
        mos_packed = npz["mosaic_idx"]

        # spatial reorder in mosaic local space
        order_local = build_local_spatial_order(tokens, mos_packed, kind="mosaic_patch")
        if order_local.size == 0:
            order_local = np.arange(mos_packed.size, dtype=np.int64)

        packed_to_local = {int(v): i for i, v in enumerate(mos_packed.tolist())}
        text_pos = np.asarray([packed_to_local[int(x)] for x in text_mos_p.tolist() if int(x) in packed_to_local], dtype=np.int64)
        corr_pos = np.asarray([packed_to_local[int(x)] for x in corr_mos_p.tolist() if int(x) in packed_to_local], dtype=np.int64)
        misl_pos = np.asarray([packed_to_local[int(x)] for x in misl_mos_p.tolist() if int(x) in packed_to_local], dtype=np.int64)

        inv = {int(old): i for i, old in enumerate(order_local.tolist())}
        text_pos = np.asarray([inv[int(i)] for i in text_pos.tolist() if int(i) in inv], dtype=np.int64)
        corr_pos = np.asarray([inv[int(i)] for i in corr_pos.tolist() if int(i) in inv], dtype=np.int64)
        misl_pos = np.asarray([inv[int(i)] for i in misl_pos.tolist() if int(i) in inv], dtype=np.int64)

        reorder_kind = "local"
        order = order_local

    # Output dir
    out_dir = Path(args.out_dir) / f"{args.qid}_{args.variant}_{args.block}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot ALL layers
    for layer in range(L):
        A = A_L[layer]

        # reorder
        A = reorder_matrix(A, order)

        # normalize within plotted matrix (recommended)
        A = normalize_matrix(A, args.normalize)

        # downsample for display
        A_plot, ds_factor, text_pos_ds, corr_pos_ds, misl_pos_ds = downsample_matrix_and_positions(
            A, args.max_tokens, text_pos, corr_pos, misl_pos
        )

        # robust color scaling
        vmax = percentile_vmax(A_plot, args.vmax_percentile)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(A_plot, aspect="auto", vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(
            f"{args.qid} | {args.variant} | block={args.block} | layer={layer} | ds={ds_factor} | "
            f"min_cover={args.min_cover} | norm={args.normalize} | vmax_p={args.vmax_percentile}"
        )
        ax.set_xlabel("Destination image tokens")
        ax.set_ylabel("Source image tokens")

        draw_region_blocks(ax, text_pos_ds, corr_pos_ds, "text→correct")
        draw_region_blocks(ax, text_pos_ds, misl_pos_ds, "text→mislead")
        draw_region_blocks(ax, text_pos_ds, text_pos_ds, "text→text")

        plt.tight_layout()
        layer_path = out_dir / f"heatmap_layer{layer:02d}.png"
        plt.savefig(str(layer_path), dpi=200)
        plt.close(fig)

    print(f"Saved {L} heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()