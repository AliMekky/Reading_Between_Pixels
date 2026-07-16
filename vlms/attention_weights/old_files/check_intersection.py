# debug_mapping_overlay.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    return dict(meta=meta, summary=summary, tokens=tokens)


# ----------------------------
# Geometry
# ----------------------------
def area_yxyx(bb: Tuple[float, float, float, float]) -> float:
    y0, x0, y1, x1 = bb
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)

def inter_area_yxyx(a, b) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    return max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)

def iou_yxyx(a, b) -> float:
    inter = inter_area_yxyx(a, b)
    if inter <= 0:
        return 0.0
    ua = area_yxyx(a) + area_yxyx(b) - inter + 1e-12
    return inter / ua

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

# ----------------------------
# Token selection: IoU threshold
# ----------------------------
def token_region_indices_iou(mapping_tokens: List[Dict[str, Any]],
                             region_yxyx: Tuple[float, float, float, float],
                             kind: str,
                             min_iou: float) -> np.ndarray:
    out = []
    for t in mapping_tokens:
        if t.get("kind") != kind:
            continue
        bb = t.get("bbox")
        if bb is None:
            continue
        token_bb = tuple(map(float, bb))  # (y0,x0,y1,x1)
        if iou_yxyx(token_bb, region_yxyx) >= min_iou:
            out.append(int(t["token_idx"]))
    return np.asarray(out, dtype=np.int64)


def get_token_bboxes_by_idx(mapping_tokens: List[Dict[str, Any]], idx_set: set) -> List[Tuple[float,float,float,float]]:
    bbs = []
    for t in mapping_tokens:
        if int(t["token_idx"]) in idx_set and t.get("bbox") is not None:
            bbs.append(tuple(map(float, t["bbox"])))
    return bbs


# ----------------------------
# Plotting
# ----------------------------
def draw_bbox(ax, bb_yxyx, color, lw=3, label=None):
    y0, x0, y1, x1 = bb_yxyx
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)
    if label:
        ax.text(x0, y0 - 3, label, color=color, fontsize=10, va="bottom")

def draw_token_boxes(ax, bbs, color, lw=1, alpha=0.7):
    for (y0,x0,y1,x1) in bbs:
        rect = Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=lw, alpha=alpha)
        ax.add_patch(rect)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_disk", type=str, default="../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC")
    ap.add_argument("--npz_root", type=str, required=True)
    ap.add_argument("--variant", type=str, required=True,
                    choices=["correct_answer","misleading_groundable","misleading_ungroundable","irrelevant_word","notext"])
    ap.add_argument("--qid", type=str, required=True)
    ap.add_argument("--npz_path_pattern", type=str, default="{root}/{variant}/{variant}/{qid}/prefill_img_self.npz")
    ap.add_argument("--min_iou", type=float, default=0.08)
    ap.add_argument("--out_dir", type=str, default="mapping_debug")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_disk)

    ex = None
    for e in ds:
        if str(e.get("question_id")) == str(args.qid):
            ex = e
            break
    if ex is None:
        raise RuntimeError(f"qid {args.qid} not found in dataset.")

    # image for THIS variant (what model saw)
    if args.variant == "notext":
        img = ex["notext"]["image"].convert("RGB")
        if "bbox" not in ex["notext"]:
            raise RuntimeError("notext bbox missing.")
        text_bbox_xyxy = ex["notext"]["bbox"]
    else:
        img = ex[args.variant]["image"].convert("RGB")
        text_bbox_xyxy = ex[args.variant]["bbox"]

    corr_xywh = (ex["correct_answer"]["x"], ex["correct_answer"]["y"], ex["correct_answer"]["w"], ex["correct_answer"]["h"])
    misl_xywh = (ex["misleading_groundable"]["x"], ex["misleading_groundable"]["y"], ex["misleading_groundable"]["w"], ex["misleading_groundable"]["h"])

    text_yxyx = guic_text_bbox_to_yxyx(text_bbox_xyxy)
    corr_yxyx = guic_obj_bbox_to_yxyx(*corr_xywh)
    misl_yxyx = guic_obj_bbox_to_yxyx(*misl_xywh)

    npz_path = Path(args.npz_path_pattern.format(root=args.npz_root, variant=args.variant, qid=args.qid))
    if not npz_path.exists():
        raise RuntimeError(f"NPZ not found: {npz_path}")

    npz = load_npz(npz_path)
    tokens = npz["tokens"]

    # Select tokens by IoU for base and mosaic separately
    text_base = token_region_indices_patch_cover(tokens, text_yxyx, "base_patch", min_cover=0.2)
    corr_base = token_region_indices_patch_cover(tokens, corr_yxyx, "base_patch", min_cover=0.2)
    misl_base = token_region_indices_patch_cover(tokens, misl_yxyx, "base_patch", min_cover=0.2)

    text_mos  = token_region_indices_patch_cover(tokens, text_yxyx, "mosaic_patch", min_cover=0.2)
    corr_mos = token_region_indices_patch_cover(tokens, corr_yxyx, "mosaic_patch", min_cover=0.2)
    misl_mos = token_region_indices_patch_cover(tokens, misl_yxyx, "mosaic_patch", min_cover=0.2)

    # Print sanity stats
    def stats(name, arr):
        if arr.size == 0:
            return f"{name}: n=0"
        return f"{name}: n={arr.size}, idx[min,max]=({arr.min()},{arr.max()})"
    print(stats("text_base", text_base))
    print(stats("corr_base", corr_base))
    print(stats("misl_base", misl_base))
    print(stats("text_mos", text_mos))
    print(stats("corr_mos", corr_mos))
    print(stats("misl_mos", misl_mos))

    # Get actual token bboxes
    text_bbs = get_token_bboxes_by_idx(tokens, set(text_base.tolist()) | set(text_mos.tolist()))
    corr_bbs = get_token_bboxes_by_idx(tokens, set(corr_base.tolist()) | set(corr_mos.tolist()))
    misl_bbs = get_token_bboxes_by_idx(tokens, set(misl_base.tolist()) | set(misl_mos.tolist()))

    out_dir = Path(args.out_dir) / f"{args.qid}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{args.qid} | {args.variant} | min_iou={args.min_iou}")

    # GUIC GT boxes
    draw_bbox(ax, text_yxyx, color="blue", lw=3, label="GT text")
    draw_bbox(ax, corr_yxyx, color="green", lw=3, label="GT correct obj")
    draw_bbox(ax, misl_yxyx, color="red", lw=3, label="GT misleading obj")

    # token boxes
    draw_token_boxes(ax, text_bbs, color="cyan", lw=1, alpha=0.7)
    draw_token_boxes(ax, corr_bbs, color="lime", lw=1, alpha=0.7)
    draw_token_boxes(ax, misl_bbs, color="orange", lw=1, alpha=0.7)

    plt.tight_layout()
    out_path = out_dir / "token_mapping_overlay.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved overlay: {out_path}")

if __name__ == "__main__":
    main()