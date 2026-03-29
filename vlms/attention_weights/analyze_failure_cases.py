#!/usr/bin/env python3
# analyze_fooled_delta_no_skip.py
"""
NO-SKIP version of analyze_fooled_delta.py

Behavioral goals:
- Do NOT drop qids from the loop due to missing NPZs, mapping mismatch, missing dataset row, etc.
- For any qid that cannot produce a valid curve, append an all-NaN curve (per region, per stream).
- Preserve your current subset criterion behavior:
    - Default keeps the provided behavior: "helped" subset (for correct_answer use-case)
    - Optionally enable fooled subset via --subset fooled
  Non-selected qids get all-NaN curves so aggregation doesn't silently change by skipping.

Metric:
- density_ratio(region/stream) =
    (avg attn per region token within stream) / (avg attn per stream token)
- Validity gating uses stream MASS:
    if stream_mass < min_denom => ratio is NaN at that layer

Outputs:
  out_dir/<variant>/
    subset_only_base.png
    subset_only_mosaic.png
    subset_notext_only_base.png
    subset_notext_only_mosaic.png
    subset_delta_vs_notext_base.png
    subset_delta_vs_notext_mosaic.png
    report_subset.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


# -------------------------
# dataset utils
# -------------------------
def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
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
    idx = {}
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
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.append(s)
    return qids


# -------------------------
# bbox helpers / regions
# -------------------------
def xyxy_to_yxyx(bb):
    x1, y1, x2, y2 = bb
    return (float(y1), float(x1), float(y2), float(x2))


def xywh_to_yxyx(x, y, w, h):
    return (float(y), float(x), float(y + h), float(x + w))


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


def build_regions_yxyx(sample: dict, variant: str) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Regions:
      - text region: bbox of current variant (fallback correct_answer for notext)
      - correct object region: correct_answer xywh if present
      - misleading object region: misleading_groundable xywh if present
    """
    regions: Dict[str, Tuple[float, float, float, float]] = {}

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


# -------------------------
# NPZ loading / qid pathing
# -------------------------
def find_npz(npz_root: Path, variant: str, qid: str) -> Optional[Path]:
    """
    On-disk layout:
      <npz_root>/<variant>/<variant>/<qid>/gen_attn_gen_token.npz
    """
    base = npz_root / variant / variant
    candidates = [qid]
    qid_strip = qid.lstrip("0") or "0"
    if qid_strip != qid:
        candidates.append(qid_strip)
    if qid.isdigit():
        candidates.append(qid.zfill(8))

    for c in candidates:
        p = base / c / "gen_attn_gen_token.npz"
        if p.exists():
            return p
    return None


def load_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(str(npz_path), allow_pickle=True)
    out = {
        "attn": data["attn"].astype(np.float32),  # (L, S_prompt)
        "img_pos": data["image_placeholder_positions"].astype(np.int64),  # (N_img,)
        "mapping_tokens": json.loads(str(data["packed_mapping_tokens"])),
        "summary": json.loads(str(data["packed_mapping_summary"])),
        "meta": json.loads(str(data["meta"])) if "meta" in data else {},
    }
    return out


# -------------------------
# token->region overlap + density_ratio(region/stream)
# -------------------------
def overlap_frac_yxyx(a, b) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b

    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
    inter = ih * iw

    at = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    if at <= 0.0:
        return 0.0
    return float(inter / at)


def infer_stream_names(mapping_tokens: List[dict]) -> List[str]:
    kinds = {t.get("kind") for t in mapping_tokens}

    if "merged_patch" in kinds:
        return ["merged"]

    has_base = "base_patch" in kinds
    has_mosaic = "mosaic_patch" in kinds

    streams = []
    if has_base:
        streams.append("base")
    if has_mosaic:
        streams.append("mosaic")
    return streams


def get_stream_token_indices(mapping_tokens: List[dict]) -> Dict[str, np.ndarray]:
    kinds = {t.get("kind") for t in mapping_tokens}

    if "merged_patch" in kinds:
        merged = [
            int(t["token_idx"])
            for t in mapping_tokens
            if t.get("kind") == "merged_patch"
        ]
        return {"merged": np.asarray(merged, dtype=np.int64)}

    out = {}
    base = [
        int(t["token_idx"])
        for t in mapping_tokens
        if t.get("kind") == "base_patch"
    ]
    mosaic = [
        int(t["token_idx"])
        for t in mapping_tokens
        if t.get("kind") == "mosaic_patch"
    ]

    if len(base) > 0:
        out["base"] = np.asarray(base, dtype=np.int64)
    if len(mosaic) > 0:
        out["mosaic"] = np.asarray(mosaic, dtype=np.int64)

    return out


def get_region_token_mask(
    mapping_tokens: List[dict],
    region_yxyx: Tuple[float, float, float, float],
    *,
    min_overlap_frac: float,
) -> np.ndarray:
    n = max(int(t["token_idx"]) for t in mapping_tokens) + 1
    mask = np.zeros((n,), dtype=bool)
    for t in mapping_tokens:
        bb = t.get("bbox")
        if bb is None:
            continue
        tok_idx = int(t["token_idx"])
        if overlap_frac_yxyx(tuple(bb), region_yxyx) >= min_overlap_frac:
            mask[tok_idx] = True
    return mask


def extract_density_ratio_region_given_stream(
    attn: np.ndarray,               # (L, S_prompt)
    img_pos: np.ndarray,            # (N_img,), prompt indices for packed image tokens in token_idx order
    mapping_tokens: List[dict],
    regions_yxyx: Dict[str, Tuple[float, float, float, float]],
    *,
    min_overlap_frac: float,
    min_denom: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns out[stream][region] = (L,) density_ratio(region/stream)
    with NaN where invalid.

    density_ratio(region/stream) =
        (sum(attn over (stream∩region)) / |stream∩region|)
        /
        (sum(attn over stream) / |stream|)
    Validity gating uses stream MASS: sum(attn over stream) < min_denom => NaN
    """
    L = attn.shape[0]
    token_scores = attn[:, img_pos]  # (L, N_img)

    stream_to_idx = get_stream_token_indices(mapping_tokens)
    out = {stream: {} for stream in stream_to_idx.keys()}

    region_masks = {
        name: get_region_token_mask(mapping_tokens, bb, min_overlap_frac=min_overlap_frac)
        for name, bb in regions_yxyx.items()
    }

    for stream, idx in stream_to_idx.items():
        if idx.size == 0:
            for rname in regions_yxyx.keys():
                out[stream][rname] = np.full((L,), np.nan, dtype=np.float32)
            continue

        stream_mass = token_scores[:, idx].sum(axis=1)  # (L,)
        ok = stream_mass >= float(min_denom)

        stream_den = stream_mass / float(idx.size)  # (L,)

        for rname, rmask in region_masks.items():
            in_region = rmask[idx]
            if not np.any(in_region):
                out[stream][rname] = np.full((L,), np.nan, dtype=np.float32)
                continue

            r_idx = idx[in_region]
            region_mass = token_scores[:, r_idx].sum(axis=1)         # (L,)
            region_den = region_mass / float(r_idx.size)             # (L,)

            ratio = np.full((L,), np.nan, dtype=np.float32)
            ratio[ok] = (region_den[ok] / (stream_den[ok] + 1e-12)).astype(np.float32)
            out[stream][rname] = ratio

    return out


# -------------------------
# subset definitions
# -------------------------
def parse_pred_letter(meta: Dict[str, Any]) -> Optional[str]:
    s = str(meta.get("generated_token_text", ""))
    for ch in s:
        if ch in ("A", "B", "C", "D"):
            return ch
    return None


def pred_key_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    letter = parse_pred_letter(meta)
    if letter is None:
        return None
    opt = meta.get("option_meta", {})
    l2k = opt.get("label_to_key", {})
    return l2k.get(letter, None)


def is_fooled(meta: Dict[str, Any], meta_n: Dict[str, Any], variant: str) -> bool:
    correct = meta.get("correct_letter", None)
    pred_letter = parse_pred_letter(meta)
    if pred_letter is None or correct is None:
        return False
    if pred_letter == correct:
        return False
    pn = parse_pred_letter(meta_n)
    pk = pred_key_from_meta(meta)
    return (pred_letter != correct) and (pn == correct) and (pk == variant)


def is_robust(meta: Dict[str, Any], meta_n: Dict[str, Any], variant: str) -> bool:
    correct = meta["correct_letter"]
    pred_letter_v = parse_pred_letter(meta)
    pred_letter_n = parse_pred_letter(meta_n)
    if pred_letter_v is None or pred_letter_n is None or correct is None:
        return False
    return (pred_letter_v == correct) and (pred_letter_n == correct)


def is_wrongly_consistent(meta: Dict[str, Any], meta_n: Dict[str, Any], variant: str) -> bool:
    for k, v in meta['option_meta']['label_to_key'].items():
        if v == "misleading_groundable":
            misleading_groundable_letter = k

    correct = meta["correct_letter"]
    pred_letter_v = parse_pred_letter(meta)
    pred_letter_n = parse_pred_letter(meta_n)
    print(meta)
    if pred_letter_v is None or pred_letter_n is None or correct is None:
        return False
    return (pred_letter_v != correct) and (pred_letter_n != correct) and (pred_letter_v == misleading_groundable_letter)


def is_helped(meta_v: Dict[str, Any], meta_n: Dict[str, Any]) -> bool:
    correct = meta_v.get("correct_letter", None)
    pv = parse_pred_letter(meta_v)
    pn = parse_pred_letter(meta_n)
    if pv is None or pn is None or correct is None:
        return False
    return (pn != correct) and (pv == correct)


# -------------------------
# stats + plotting
# -------------------------
def nanmean_ci(x: np.ndarray, ci: float = 95.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    lo = np.nanpercentile(x, (100.0 - ci) / 2.0, axis=0)
    hi = np.nanpercentile(x, 100.0 - (100.0 - ci) / 2.0, axis=0)
    return mean, lo, hi


def plot_three_lines_with_ci(
    title: str,
    x: np.ndarray,
    series: Dict[str, np.ndarray],        # name -> (N,L)
    out_path: Path,
    ylabel: str,
    ci: float = 95.0,
):
    plt.figure(figsize=(12, 5))
    for name, mat in series.items():
        m, lo, hi = nanmean_ci(mat, ci=ci)
        plt.plot(x, m, label=name)
        plt.fill_between(x, lo, hi, alpha=0.2)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_notext_only_with_ci(
    title: str,
    x: np.ndarray,
    series: Dict[str, np.ndarray],        # name -> (N,L)
    out_path: Path,
    ylabel: str,
    ci: float = 95.0,
):
    plt.figure(figsize=(12, 5))
    for name, mat in series.items():
        m, lo, hi = nanmean_ci(mat, ci=ci)
        plt.plot(x, m, label=name)
        plt.fill_between(x, lo, hi, alpha=0.2)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=str, default="/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/attention_weights/qwen-vl_attentions")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument(
        "--variant",
        type=str,
        default="correct_answer",
        choices=["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"],
    )
    parser.add_argument(
        "--qid_file",
        type=str,
        default="../inference/no_overlap_question_ids.txt",
        help="Whitelist (one qid per line). If empty, qids are taken from variant NPZ directory.",
    )
    parser.add_argument("--out_dir", type=str, default="qwen_plots")

    parser.add_argument("--min_overlap_frac", type=float, default=0.25)
    parser.add_argument("--min_denom", type=float, default=1e-4)
    parser.add_argument("--ci", type=float, default=95.0)

    parser.add_argument(
        "--subset",
        type=str,
        default="helped",
        choices=["helped", "fooled", "all", "robust", "wrongly_consistent"],
        help="Which subset to aggregate. Non-selected qids are kept as all-NaN (no skip).",
    )

    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    out_root = Path(args.out_dir) / args.variant
    out_root.mkdir(parents=True, exist_ok=True)

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]
    qid_index = build_qid_index(ds)

    qids_from_file = load_qid_list(args.qid_file)
    if qids_from_file is not None and len(qids_from_file) > 0:
        qids = qids_from_file
        qid_source = "qid_file"
    else:
        var_dir = npz_root / args.variant / args.variant
        if not var_dir.exists():
            raise RuntimeError(f"Variant dir not found: {var_dir}")
        qids = sorted([p.name for p in var_dir.iterdir() if p.is_dir()])
        qid_source = "variant_dir"

    streams = None
    L_ref: Optional[int] = None
    for qid in qids:
        pv = find_npz(npz_root, args.variant, qid)
        pn = find_npz(npz_root, "notext", qid)
        if pv is None or pn is None:
            continue
        try:
            dv = load_npz(pv)
            if streams is None:
                streams = infer_stream_names(dv["mapping_tokens"])
            dn = load_npz(pn)
            L_ref = int(dv["attn"].shape[0])
            if int(dn["attn"].shape[0]) != L_ref:
                L_ref = None
                continue
            break
        except Exception:
            continue

    if L_ref is None:
        raise RuntimeError("Could not infer L_ref from any available (variant, notext) NPZ pair.")

    layers = np.arange(L_ref, dtype=int)
    region_names = ["text region", "correct object region", "misleading object region"]

    def nan_vec():
        return np.full((L_ref,), np.nan, dtype=np.float32)

    store_V: Dict[str, Dict[str, List[np.ndarray]]] = {s: {r: [] for r in region_names} for s in streams}
    store_N: Dict[str, Dict[str, List[np.ndarray]]] = {s: {r: [] for r in region_names} for s in streams}

    selected_mask: List[bool] = []
    used_qids: List[str] = []

    counts: Dict[str, int] = {
        "N_total_qids_iterated": 0,
        "N_selected": 0,
        "missing_variant_npz": 0,
        "missing_notext_npz": 0,
        "npz_load_error": 0,
        "mapping_mismatch": 0,
        "missing_dataset_row": 0,
        "layer_mismatch": 0,
        "regions_missing_or_empty": 0,
    }

    for qid in qids:
        counts["N_total_qids_iterated"] += 1
        used_qids.append(qid)

        is_selected = False

        pv = find_npz(npz_root, args.variant, qid)
        pn = find_npz(npz_root, "notext", qid)
        if pv is None:
            counts["missing_variant_npz"] += 1
        if pn is None:
            counts["missing_notext_npz"] += 1

        if pv is None or pn is None:
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        try:
            dv = load_npz(pv)
            dn = load_npz(pn)
        except Exception:
            counts["npz_load_error"] += 1
            for s in streams:
                print(s)
                for r in region_names:
                    print(r)
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        if int(dv["attn"].shape[0]) != L_ref or int(dn["attn"].shape[0]) != L_ref:
            counts["layer_mismatch"] += 1
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        try:
            exp_v = int(dv["summary"]["total_packed_image_tokens"])
            exp_n = int(dn["summary"]["total_packed_image_tokens"])
        except Exception:
            exp_v, exp_n = -1, -1

        if len(dv["img_pos"]) != exp_v or len(dn["img_pos"]) != exp_n:
            counts["mapping_mismatch"] += 1
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        qid_key = qid
        if qid_key not in qid_index:
            if qid_key.isdigit() and str(int(qid_key)) in qid_index:
                qid_key = str(int(qid_key))
            elif qid_key.isdigit() and qid_key.zfill(8) in qid_index:
                qid_key = qid_key.zfill(8)
            else:
                counts["missing_dataset_row"] += 1
                for s in streams:
                    for r in region_names:
                        store_V[s][r].append(nan_vec())
                        store_N[s][r].append(nan_vec())
                selected_mask.append(False)
                continue

        sample = ds[qid_index[qid_key]]
        regions = build_regions_yxyx(sample, args.variant)

        if args.subset == "all":
            is_selected = True
        elif args.subset == "fooled":
            is_selected = is_fooled(dv["meta"], dn["meta"], args.variant)
        elif args.subset == "robust":
            is_selected = is_robust(dv["meta"], dn["meta"], args.variant)
        elif args.subset == "wrongly_consistent":
            is_selected = is_wrongly_consistent(dv["meta"], dn["meta"], args.variant)
        else:
            is_selected = is_helped(dv["meta"], dn["meta"])

        if is_selected:
            counts["N_selected"] += 1

        if not any(r in regions for r in region_names):
            counts["regions_missing_or_empty"] += 1
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        if not is_selected:
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        pv_cur = extract_density_ratio_region_given_stream(
            dv["attn"], dv["img_pos"], dv["mapping_tokens"], regions,
            min_overlap_frac=args.min_overlap_frac,
            min_denom=args.min_denom,
        )
        pn_cur = extract_density_ratio_region_given_stream(
            dn["attn"], dn["img_pos"], dn["mapping_tokens"], regions,
            min_overlap_frac=args.min_overlap_frac,
            min_denom=args.min_denom,
        )

        for s in streams:
            for r in region_names:
                store_V[s][r].append(pv_cur.get(s, {}).get(r, nan_vec()))
                store_N[s][r].append(pn_cur.get(s, {}).get(r, nan_vec()))

        selected_mask.append(True)

    N = len(used_qids)

    def stack(store: Dict[str, List[np.ndarray]], rname: str) -> np.ndarray:
        return np.stack(store.get(rname, [nan_vec()] * N), axis=0).astype(np.float32)

    for stream in streams:
        series = {rn: stack(store_V[stream], rn) for rn in region_names}
        plot_three_lines_with_ci(
            title=f"{args.variant} | {stream.upper()} density_ratio(region/{stream}) | subset={args.subset} (N_total={N}, N_selected={counts['N_selected']})",
            x=layers,
            series=series,
            out_path=out_root / f"subset_only_{stream}.png",
            ylabel=f"(avg attn per region token) / (avg attn per {stream} token)",
            ci=args.ci,
        )

    for stream in streams:
        series_notext = {rn: stack(store_N[stream], rn) for rn in region_names}
        plot_notext_only_with_ci(
            title=f"{args.variant} | {stream.upper()} density_ratio(region/{stream}) | NOTEXT only | subset={args.subset} (N_total={N}, N_selected={counts['N_selected']})",
            x=layers,
            series=series_notext,
            out_path=out_root / f"subset_notext_only_{stream}.png",
            ylabel=f"(avg attn per region token) / (avg attn per {stream} token)",
            ci=args.ci,
        )

    for stream in streams:
        series_delta: Dict[str, np.ndarray] = {}
        for rn in region_names:
            v = stack(store_V[stream], rn)
            n0 = stack(store_N[stream], rn)
            series_delta[rn] = v - n0
        plot_three_lines_with_ci(
            title=f"{args.variant} | {stream.upper()} Δdensity_ratio = variant - notext | subset={args.subset} (paired, NaN-masked)",
            x=layers,
            series=series_delta,
            out_path=out_root / f"subset_delta_vs_notext_{stream}.png",
            ylabel="Δ[(avg attn per region token)/(avg attn per stream token)]",
            ci=args.ci,
        )

    outputs = {}
    for stream in streams:
        outputs[f"subset_only_{stream}"] = f"subset_only_{stream}.png"
        outputs[f"subset_notext_only_{stream}"] = f"subset_notext_only_{stream}.png"
        outputs[f"subset_delta_vs_notext_{stream}"] = f"subset_delta_vs_notext_{stream}.png"

    report = {
        "variant": args.variant,
        "subset": args.subset,
        "qid_source": qid_source,
        "qid_file": args.qid_file,
        "npz_root": str(npz_root),
        "N_total": N,
        "N_selected": counts["N_selected"],
        "L_ref": L_ref,
        "settings": {
            "min_overlap_frac": float(args.min_overlap_frac),
            "min_denom": float(args.min_denom),
            "ci": float(args.ci),
            "metric": "density_ratio(region/stream) = (avg attn per region token within stream) / (avg attn per stream token)",
            "gating": "NaN where stream mass < min_denom",
            "no_skip_policy": "Iterate all qids; on any failure append all-NaN curves; subset enforced by NaN masking (non-selected -> NaN)",
        },
        "counts": counts,
        "used_qids_first_20": used_qids[:20],
        "outputs": outputs,
    }
    with open(out_root / "report_subset.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()