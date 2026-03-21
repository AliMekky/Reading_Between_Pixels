# # # failure_case_analysis_guic.py
# # """
# # Failure-case analysis for GUIC attention NPZs (Llava-NeXT cached attn).

# # For each variant:
# #   - Loads NPZs for qids in --qid_file
# #   - Computes per-layer conditional mass within each region, separately for BASE and MOSAIC streams:
# #         p(region | stream) = sum_{img tokens in region & in stream} attn
# #                               / sum_{img tokens in stream} attn
# #     where attn is generated-token -> prompt-token attention (head-avg) stored in NPZ (L, S_prompt).

# #   - Stratifies samples into:
# #         success: generated_letter == correct_letter
# #         failure: generated_letter != correct_letter
# #         fooled_by_variant: predicted option key == variant key (for misleading_* and irrelevant_word)

# #   - Also computes paired delta-vs-notext for each qid:
# #         delta = metric(variant) - metric(notext)

# # Outputs:
# #   - Plots per variant:
# #       * mosaic_conditional_success_vs_failure.png
# #       * mosaic_delta_vs_notext_success_vs_failure.png
# #     (each plot overlays 3 lines: text region, correct object region, misleading object region)
# #   - JSON report with counts
# # """

# # import argparse
# # import json
# # from pathlib import Path
# # from typing import Dict, Any, List, Optional, Tuple, Set

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from datasets import load_dataset, load_from_disk, Dataset, DatasetDict


# # # -----------------------------
# # # Dataset loading
# # # -----------------------------
# # def sanitize_repo_id(repo_id: str) -> str:
# #     return repo_id.replace("/", "__").replace(" ", "_")


# # def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
# #     local_cache_root = Path(local_cache_root)
# #     local_cache_root.mkdir(parents=True, exist_ok=True)
# #     cache_dir = local_cache_root / sanitize_repo_id(dataset_id)

# #     if cache_dir.exists():
# #         return load_from_disk(str(cache_dir))

# #     ds = load_dataset(dataset_id, split=split)
# #     try:
# #         ds.save_to_disk(str(cache_dir))
# #     except Exception:
# #         pass
# #     return ds


# # def build_qid_index(ds: Dataset) -> Dict[str, int]:
# #     idx = {}
# #     for i in range(len(ds)):
# #         qid = str(ds[i].get("question_id"))
# #         idx[qid] = i
# #         if qid.isdigit():
# #             idx[str(int(qid))] = i
# #             idx[qid.zfill(8)] = i
# #     return idx


# # def load_qid_whitelist(path: str) -> List[str]:
# #     qids = []
# #     with open(path, "r") as f:
# #         for line in f:
# #             s = line.strip()
# #             if s:
# #                 qids.append(s)
# #     return qids


# # # -----------------------------
# # # NPZ helpers
# # # -----------------------------
# # def load_npz(npz_path: Path) -> Dict[str, Any]:
# #     data = np.load(str(npz_path), allow_pickle=True)
# #     out: Dict[str, Any] = {
# #         "attn": data["attn"].astype(np.float32),  # (L, S_prompt)
# #         "image_pos": data["image_placeholder_positions"].astype(np.int64),  # (N_img,)
# #         "mapping_tokens": json.loads(str(data["packed_mapping_tokens"])),
# #         "mapping_summary": json.loads(str(data["packed_mapping_summary"])),
# #         "meta": json.loads(str(data["meta"])) if "meta" in data else {},
# #     }
# #     return out


# # def find_npz(npz_root: Path, variant: str, qid: str) -> Optional[Path]:
# #     base = npz_root / variant / variant
# #     candidates = [qid]
# #     qid_strip = qid.lstrip("0") or "0"
# #     if qid_strip != qid:
# #         candidates.append(qid_strip)
# #     if qid.isdigit():
# #         candidates.append(qid.zfill(8))

# #     for c in candidates:
# #         p = base / c / "gen_attn_gen_token.npz"
# #         if p.exists():
# #             return p
# #     return None


# # # -----------------------------
# # # Region extraction from GUIC sample
# # # -----------------------------
# # def xyxy_to_yxyx(bb):
# #     x1, y1, x2, y2 = bb
# #     return (float(y1), float(x1), float(y2), float(x2))


# # def xywh_to_yxyx(x, y, w, h):
# #     return (float(y), float(x), float(y + h), float(x + w))


# # def get_text_bbox_xyxy(sample: dict, variant: str, fallback_variant: str = "correct_answer"):
# #     v = variant
# #     if v == "notext":
# #         v = fallback_variant
# #     if v not in sample:
# #         return None

# #     d = sample[v]
# #     # robust key search
# #     for k in ["bbox", "text_bbox", "text_box", "bbox_xyxy"]:
# #         if k in d:
# #             return d[k]
# #     if "annotations" in d and isinstance(d["annotations"], dict):
# #         for k in ["bbox", "text_bbox", "bbox_xyxy"]:
# #             if k in d["annotations"]:
# #                 return d["annotations"][k]
# #     return None


# # def build_regions_yxyx(sample: dict, variant: str) -> Dict[str, Tuple[float, float, float, float]]:
# #     regions: Dict[str, Tuple[float, float, float, float]] = {}

# #     tb = get_text_bbox_xyxy(sample, variant, fallback_variant="correct_answer")
# #     if tb is not None:
# #         regions["text region"] = xyxy_to_yxyx(tb)

# #     ca = sample.get("correct_answer", {})
# #     if all(k in ca for k in ["x", "y", "w", "h"]):
# #         regions["correct object region"] = xywh_to_yxyx(ca["x"], ca["y"], ca["w"], ca["h"])

# #     mg = sample.get("misleading_groundable", {})
# #     if all(k in mg for k in ["x", "y", "w", "h"]):
# #         regions["misleading object region"] = xywh_to_yxyx(mg["x"], mg["y"], mg["w"], mg["h"])

# #     return regions


# # # -----------------------------
# # # Overlap / token selection
# # # -----------------------------
# # def area_yxyx(bb):
# #     y0, x0, y1, x1 = bb
# #     return max(0.0, y1 - y0) * max(0.0, x1 - x0)


# # def intersect_yxyx(a, b):
# #     ay0, ax0, ay1, ax1 = a
# #     by0, bx0, by1, bx1 = b
# #     y0 = max(ay0, by0)
# #     x0 = max(ax0, bx0)
# #     y1 = min(ay1, by1)
# #     x1 = min(ax1, bx1)
# #     if y1 <= y0 or x1 <= x0:
# #         return (0.0, 0.0, 0.0, 0.0)
# #     return (y0, x0, y1, x1)


# # def token_in_region_fraction(token_bbox_yxyx, region_bbox_yxyx) -> float:
# #     inter = intersect_yxyx(token_bbox_yxyx, region_bbox_yxyx)
# #     a_int = area_yxyx(inter)
# #     a_tok = area_yxyx(token_bbox_yxyx)
# #     if a_tok <= 1e-12:
# #         return 0.0
# #     return a_int / a_tok


# # def stream_token_indices(mapping_tokens: List[Dict[str, Any]], stream: str) -> np.ndarray:
# #     # returns token_idx positions within the packed image token space (0..N_img-1)
# #     kinds = {"base": {"base_patch"}, "mosaic": {"mosaic_patch"}}
# #     want = kinds[stream]
# #     idxs = []
# #     for t in mapping_tokens:
# #         if t.get("kind") in want:
# #             idxs.append(int(t["token_idx"]))
# #     return np.asarray(idxs, dtype=np.int64)


# # def region_token_indices(
# #     mapping_tokens: List[Dict[str, Any]],
# #     stream: str,
# #     region_bbox_yxyx: Tuple[float, float, float, float],
# #     min_overlap_frac: float,
# # ) -> np.ndarray:
# #     want = {"base": {"base_patch"}, "mosaic": {"mosaic_patch"}}[stream]
# #     idxs = []
# #     for t in mapping_tokens:
# #         if t.get("kind") not in want:
# #             continue
# #         bb = t.get("bbox")
# #         if bb is None:
# #             continue
# #         frac = token_in_region_fraction(tuple(bb), region_bbox_yxyx)
# #         if frac >= min_overlap_frac:
# #             idxs.append(int(t["token_idx"]))
# #     return np.asarray(idxs, dtype=np.int64)


# # # -----------------------------
# # # Metrics
# # # -----------------------------
# # def compute_p_region_given_stream(
# #     attn: np.ndarray,                   # (L, S_prompt)
# #     image_pos_prompt: np.ndarray,        # (N_img,) prompt positions for image tokens
# #     mapping_tokens: List[Dict[str, Any]],
# #     regions_yxyx: Dict[str, Tuple[float, float, float, float]],
# #     *,
# #     stream: str,
# #     min_overlap_frac: float,
# #     min_denom: float,
# # ) -> Dict[str, np.ndarray]:
# #     """
# #     Returns dict region_name -> (L,) with NaNs where denom too small or region missing.
# #     """
# #     L = attn.shape[0]
# #     img_mass = attn[:, image_pos_prompt]  # (L, N_img) aligned with packed image tokens order

# #     # denom is stream mass
# #     s_idx = stream_token_indices(mapping_tokens, stream)
# #     denom = img_mass[:, s_idx].sum(axis=1)  # (L,)

# #     out: Dict[str, np.ndarray] = {}
# #     for rname, rbb in regions_yxyx.items():
# #         r_idx = region_token_indices(mapping_tokens, stream, rbb, min_overlap_frac)
# #         if r_idx.size == 0:
# #             out[rname] = np.full((L,), np.nan, dtype=np.float32)
# #             continue
# #         num = img_mass[:, r_idx].sum(axis=1)  # (L,)
# #         p = np.full((L,), np.nan, dtype=np.float32)
# #         ok = denom >= min_denom
# #         p[ok] = (num[ok] / denom[ok]).astype(np.float32)
# #         out[rname] = p
# #     return out


# # def agg_nanmean_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #     """
# #     curves: list of (L,) arrays possibly with NaNs
# #     returns mean, lo, hi where lo/hi are 2.5/97.5 percentiles per layer.
# #     """
# #     X = np.stack(curves, axis=0)  # (N, L)
# #     mean = np.nanmean(X, axis=0)
# #     lo = np.nanpercentile(X, 2.5, axis=0)
# #     hi = np.nanpercentile(X, 97.5, axis=0)
# #     return mean, lo, hi


# # # -----------------------------
# # # Outcome parsing
# # # -----------------------------
# # def normalize_letter(x: Any) -> Optional[str]:
# #     if x is None:
# #         return None
# #     s = str(x).strip()
# #     if not s:
# #         return None
# #     s = s.upper()
# #     # generated token might decode to " A" etc; take first alpha A-D
# #     for ch in s:
# #         if ch in ["A", "B", "C", "D"]:
# #             return ch
# #     return None


# # def predicted_key_from_meta(meta: Dict[str, Any]) -> Optional[str]:
# #     gen = normalize_letter(meta.get("generated_token_text"))
# #     opt = meta.get("option_meta", {})
# #     l2k = opt.get("label_to_key", None)
# #     if gen is None or not isinstance(l2k, dict):
# #         return None
# #     return l2k.get(gen)


# # # -----------------------------
# # # Plotting
# # # -----------------------------
# # def plot_grouped_curves(
# #     title: str,
# #     out_path: Path,
# #     x: np.ndarray,
# #     curves_by_group: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
# #     region_order: List[str],
# # ):
# #     """
# #     curves_by_group[group][region] = (mean, lo, hi)
# #     Draw one figure with separate lines per region, per group (solid vs dashed).
# #     """
# #     plt.figure(figsize=(10, 5))
# #     for group, regdict in curves_by_group.items():
# #         for rname in region_order:
# #             if rname not in regdict:
# #                 continue
# #             mean, lo, hi = regdict[rname]
# #             # style: success solid, failure dashed, fooled dotted
# #             if group == "success":
# #                 ls = "-"
# #             elif group == "failure":
# #                 ls = "--"
# #             else:
# #                 ls = ":"
# #             plt.plot(x, mean, linestyle=ls, label=f"{group}: {rname}")
# #             plt.fill_between(x, lo, hi, alpha=0.12)

# #     plt.xlabel("layer")
# #     plt.ylabel("mean of per-sample ratios (percentile band)")
# #     plt.title(title)
# #     plt.legend(fontsize=9, ncol=2)
# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()


# # # -----------------------------
# # # Main
# # # -----------------------------
# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--npz_root", type=str, required=True)
# #     parser.add_argument("--qid_file", type=str, required=True)

# #     parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
# #     parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
# #     parser.add_argument("--split", type=str, default="test")

# #     parser.add_argument("--min_overlap_frac", type=float, default=0.25)
# #     parser.add_argument("--min_denom", type=float, default=1e-4)

# #     parser.add_argument("--out_dir", type=str, default="failure_case_plots")
# #     parser.add_argument("--variants", type=str, default="notext,correct_answer,misleading_groundable,misleading_ungroundable,irrelevant_word")

# #     args = parser.parse_args()

# #     npz_root = Path(args.npz_root)
# #     out_root = Path(args.out_dir)
# #     out_root.mkdir(parents=True, exist_ok=True)

# #     variants = [v.strip() for v in args.variants.split(",") if v.strip()]
# #     if "notext" not in variants:
# #         variants = ["notext"] + variants  # ensure notext is available for deltas

# #     qids = load_qid_whitelist(args.qid_file)

# #     ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
# #     if isinstance(ds, DatasetDict):
# #         ds = ds[args.split]
# #     qid_index = build_qid_index(ds)

# #     region_order = ["text region", "correct object region", "misleading object region"]

# #     report: Dict[str, Any] = {
# #         "npz_root": str(npz_root),
# #         "qid_file": args.qid_file,
# #         "min_overlap_frac": args.min_overlap_frac,
# #         "min_denom": args.min_denom,
# #         "variants": variants,
# #         "per_variant": {},
# #     }

# #     # Preload notext metrics per qid so we can compute paired deltas
# #     notext_cache: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}  # qid -> stream -> region -> (L,)
# #     notext_outcome: Dict[str, str] = {}  # qid -> "success"/"failure"
# #     for qid in qids:
# #         npz_path = find_npz(npz_root, "notext", qid)
# #         if npz_path is None:
# #             continue
# #         if qid not in qid_index:
# #             continue

# #         sample = ds[qid_index[qid]]
# #         regions = build_regions_yxyx(sample, "notext")

# #         data = load_npz(npz_path)
# #         meta = data["meta"]
# #         correct = normalize_letter(meta.get("correct_letter"))
# #         gen = normalize_letter(meta.get("generated_token_text"))
# #         outcome = "success" if (correct is not None and gen == correct) else "failure"

# #         base_cur = compute_p_region_given_stream(
# #             data["attn"], data["image_pos"], data["mapping_tokens"], regions,
# #             stream="base", min_overlap_frac=args.min_overlap_frac, min_denom=args.min_denom
# #         )
# #         mosaic_cur = compute_p_region_given_stream(
# #             data["attn"], data["image_pos"], data["mapping_tokens"], regions,
# #             stream="mosaic", min_overlap_frac=args.min_overlap_frac, min_denom=args.min_denom
# #         )
# #         notext_cache[qid] = {"base": base_cur, "mosaic": mosaic_cur}
# #         notext_outcome[qid] = outcome

# #     # Process each variant (excluding notext as target)
# #     for variant in [v for v in variants if v != "notext"]:
# #         # store per-sample curves by group
# #         per_group_mosaic: Dict[str, Dict[str, List[np.ndarray]]] = {
# #             "success": {r: [] for r in region_order},
# #             "failure": {r: [] for r in region_order},
# #             "fooled":  {r: [] for r in region_order},
# #         }
# #         per_group_delta_mosaic: Dict[str, Dict[str, List[np.ndarray]]] = {
# #             "success": {r: [] for r in region_order},
# #             "failure": {r: [] for r in region_order},
# #             "fooled":  {r: [] for r in region_order},
# #         }

# #         counts = {"used": 0, "success": 0, "failure": 0, "fooled": 0, "missing_npz": 0, "missing_notext_pair": 0}

# #         for qid in qids:
# #             npz_path = find_npz(npz_root, variant, qid)
# #             if npz_path is None:
# #                 counts["missing_npz"] += 1
# #                 continue
# #             if qid not in qid_index:
# #                 continue

# #             sample = ds[qid_index[qid]]
# #             regions = build_regions_yxyx(sample, variant)
# #             data = load_npz(npz_path)
# #             meta = data["meta"]

# #             correct = normalize_letter(meta.get("correct_letter"))
# #             gen = normalize_letter(meta.get("generated_token_text"))
# #             outcome = "success" if (correct is not None and gen == correct) else "failure"

# #             pred_key = predicted_key_from_meta(meta)
# #             fooled = (pred_key == variant)

# #             # compute mosaic curves for this sample
# #             mosaic_cur = compute_p_region_given_stream(
# #                 data["attn"], data["image_pos"], data["mapping_tokens"], regions,
# #                 stream="mosaic", min_overlap_frac=args.min_overlap_frac, min_denom=args.min_denom
# #             )

# #             counts["used"] += 1
# #             counts[outcome] += 1
# #             if fooled:
# #                 counts["fooled"] += 1

# #             g = "success" if outcome == "success" else "failure"
# #             for r in region_order:
# #                 if r in mosaic_cur:
# #                     per_group_mosaic[g][r].append(mosaic_cur[r])
# #                     if fooled:
# #                         per_group_mosaic["fooled"][r].append(mosaic_cur[r])

# #             # paired delta vs notext (only if notext exists for this qid)
# #             if qid not in notext_cache:
# #                 counts["missing_notext_pair"] += 1
# #                 continue

# #             notext_mosaic = notext_cache[qid]["mosaic"]
# #             for r in region_order:
# #                 if r in mosaic_cur and r in notext_mosaic:
# #                     delta = mosaic_cur[r] - notext_mosaic[r]
# #                     per_group_delta_mosaic[g][r].append(delta)
# #                     if fooled:
# #                         per_group_delta_mosaic["fooled"][r].append(delta)

# #         # aggregate and plot
# #         def aggregate_groups(group_dict: Dict[str, Dict[str, List[np.ndarray]]]):
# #             out = {}
# #             for gname, regdict in group_dict.items():
# #                 out[gname] = {}
# #                 for rname, curves in regdict.items():
# #                     if len(curves) == 0:
# #                         continue
# #                     out[gname][rname] = agg_nanmean_curves(curves)
# #             return out

# #         agg_mosaic = aggregate_groups(per_group_mosaic)
# #         agg_delta = aggregate_groups(per_group_delta_mosaic)

# #         # infer L from any available curve
# #         L = None
# #         for g in agg_mosaic.values():
# #             for tpl in g.values():
# #                 L = int(tpl[0].shape[0])
# #                 break
# #             if L is not None:
# #                 break
# #         if L is None:
# #             report["per_variant"][variant] = {"counts": counts, "note": "no curves to plot"}
# #             continue

# #         x = np.arange(L)

# #         vdir = out_root / variant
# #         vdir.mkdir(parents=True, exist_ok=True)

# #         plot_grouped_curves(
# #             title=f"{variant} | MOSAIC p(region | mosaic): success vs failure (N_used={counts['used']})",
# #             out_path=vdir / "mosaic_conditional_success_vs_failure.png",
# #             x=x,
# #             curves_by_group={k: v for k, v in agg_mosaic.items() if k in ["success", "failure", "fooled"]},
# #             region_order=region_order,
# #         )

# #         plot_grouped_curves(
# #             title=f"{variant} | MOSAIC delta vs notext: success vs failure (paired qids)",
# #             out_path=vdir / "mosaic_delta_vs_notext_success_vs_failure.png",
# #             x=x,
# #             curves_by_group={k: v for k, v in agg_delta.items() if k in ["success", "failure", "fooled"]},
# #             region_order=region_order,
# #         )

# #         report["per_variant"][variant] = {
# #             "counts": counts,
# #             "outputs": {
# #                 "mosaic_conditional": str((vdir / "mosaic_conditional_success_vs_failure.png").name),
# #                 "mosaic_delta_vs_notext": str((vdir / "mosaic_delta_vs_notext_success_vs_failure.png").name),
# #             },
# #         }

# #     with open(out_root / "report.json", "w") as f:
# #         json.dump(report, f, indent=2)

# #     print(f"Wrote outputs to: {out_root}")
# #     print(f"Report: {out_root/'report.json'}")


# # if __name__ == "__main__":
# #     main()


# # analyze_fooled_delta.py
# """
# Fooled-only analysis + paired delta vs NOTEXT baseline.

# For a given GUIC variant V (correct_answer / misleading_groundable / misleading_ungroundable / irrelevant_word):
# - "fooled" samples are those where the generated answer letter maps to key == V
#   and it is not the correct letter.
# - We compute p(region | stream) curves (BASE and MOSAIC) across layers on the fooled subset.
# - We also compute paired deltas vs NOTEXT on the SAME qids:
#     Δp(layer) = p_V(layer) - p_NOTEXT(layer)

# Outputs per variant:
#   - fooled_only_base.png
#   - fooled_only_mosaic.png
#   - fooled_delta_vs_notext_base.png
#   - fooled_delta_vs_notext_mosaic.png
#   - report_fooled.json

# Assumptions about NPZ:
# - meta contains:
#     meta["correct_letter"]
#     meta["option_meta"]["label_to_key"]  (maps 'A'/'B'/'C'/'D' -> variant key)
#     meta["generated_token_text"]         (the generated answer token text; we parse first A-D)
# - NPZ contains:
#     attn (L, S_prompt)
#     image_placeholder_positions (N_img,)
#     packed_mapping_tokens (json list)
#     packed_mapping_summary (json dict)
# """

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple

# import numpy as np
# import matplotlib.pyplot as plt
# from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
# from PIL import Image


# # -------------------------
# # dataset utils
# # -------------------------
# def sanitize_repo_id(repo_id: str) -> str:
#     return repo_id.replace("/", "__").replace(" ", "_")

# def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
#     local_cache_root = Path(local_cache_root)
#     local_cache_root.mkdir(parents=True, exist_ok=True)
#     cache_dir = local_cache_root / sanitize_repo_id(dataset_id)

#     if cache_dir.exists():
#         return load_from_disk(str(cache_dir))

#     ds = load_dataset(dataset_id, split=split)
#     try:
#         ds.save_to_disk(str(cache_dir))
#     except Exception:
#         pass
#     return ds

# def build_qid_index(ds: Dataset) -> Dict[str, int]:
#     idx = {}
#     for i in range(len(ds)):
#         qid = str(ds[i].get("question_id"))
#         idx[qid] = i
#         if qid.isdigit():
#             idx[str(int(qid))] = i
#             idx[qid.zfill(8)] = i
#     return idx

# def load_qid_whitelist(path: str) -> Optional[set]:
#     if not path:
#         return None
#     qids = set()
#     with open(path, "r") as f:
#         for line in f:
#             s = line.strip()
#             if s:
#                 qids.add(s)
#     return qids


# # -------------------------
# # bbox helpers / regions
# # -------------------------
# def xyxy_to_yxyx(bb):
#     x1, y1, x2, y2 = bb
#     return (float(y1), float(x1), float(y2), float(x2))

# def xywh_to_yxyx(x, y, w, h):
#     return (float(y), float(x), float(y + h), float(x + w))

# def get_text_bbox_xyxy(sample: dict, variant: str, fallback_variant: str = "correct_answer"):
#     v = variant
#     if v == "notext":
#         v = fallback_variant
#     if v not in sample:
#         return None
#     d = sample[v]
#     # GUIC schema sometimes differs; try a few candidates
#     for k in ["bbox", "text_bbox", "text_box", "bbox_xyxy"]:
#         if k in d:
#             return d[k]
#     if "annotations" in d and isinstance(d["annotations"], dict):
#         for k in ["bbox", "text_bbox", "bbox_xyxy"]:
#             if k in d["annotations"]:
#                 return d["annotations"][k]
#     return None

# def build_regions_yxyx(sample: dict, variant: str) -> Dict[str, Tuple[float, float, float, float]]:
#     """
#     Keep consistent regions across variants:
#       - text region: text bbox of *current variant* (fallback correct_answer for notext)
#       - correct object: correct_answer xywh if exists
#       - misleading object: misleading_groundable xywh if exists
#     """
#     regions = {}

#     tb = get_text_bbox_xyxy(sample, variant, fallback_variant="correct_answer")
#     if tb is not None:
#         regions["text region"] = xyxy_to_yxyx(tb)

#     ca = sample.get("correct_answer", {})
#     if all(k in ca for k in ["x", "y", "w", "h"]):
#         regions["correct object region"] = xywh_to_yxyx(ca["x"], ca["y"], ca["w"], ca["h"])

#     mg = sample.get("misleading_groundable", {})
#     if all(k in mg for k in ["x", "y", "w", "h"]):
#         regions["misleading object region"] = xywh_to_yxyx(mg["x"], mg["y"], mg["w"], mg["h"])

#     return regions


# # -------------------------
# # NPZ loading / qid pathing
# # -------------------------
# def find_npz(npz_root: Path, variant: str, qid: str) -> Optional[Path]:
#     """
#     Your on-disk layout looks like:
#       llava-next_attentions/<variant>/<variant>/<qid>/gen_attn_gen_token.npz
#     (based on your printed paths)
#     """
#     base = npz_root / variant / variant
#     candidates = [qid]
#     qid_strip = qid.lstrip("0") or "0"
#     if qid_strip != qid:
#         candidates.append(qid_strip)
#     if qid.isdigit():
#         candidates.append(qid.zfill(8))

#     for c in candidates:
#         p = base / c / "gen_attn_gen_token.npz"
#         if p.exists():
#             return p
#     return None

# def load_npz(npz_path: Path) -> Dict[str, Any]:
#     data = np.load(str(npz_path), allow_pickle=True)
#     out = {
#         "attn": data["attn"].astype(np.float32),  # (L, S_prompt)
#         "img_pos": data["image_placeholder_positions"].astype(np.int64),
#         "mapping_tokens": json.loads(str(data["packed_mapping_tokens"])),
#         "summary": json.loads(str(data["packed_mapping_summary"])),
#         "meta": json.loads(str(data["meta"])) if "meta" in data else {},
#     }
#     return out


# # -------------------------
# # token->region overlap + p(region|stream)
# # -------------------------
# def overlap_frac_yxyx(a, b) -> float:
#     """
#     a,b: (y0,x0,y1,x1) in same pixel coordinate space
#     Returns intersection / area(token_bbox)  (token-centric overlap)
#     """
#     ay0, ax0, ay1, ax1 = a
#     by0, bx0, by1, bx1 = b

#     iy0, ix0 = max(ay0, by0), max(ax0, bx0)
#     iy1, ix1 = min(ay1, by1), min(ax1, bx1)
#     ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
#     inter = ih * iw

#     at = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
#     if at <= 0:
#         return 0.0
#     return float(inter / at)

# def get_stream_token_indices(mapping_tokens: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Return indices (into packed image token list, i.e., token_idx values) for:
#       - base_patch tokens
#       - mosaic_patch tokens
#     """
#     base = []
#     mosaic = []
#     for t in mapping_tokens:
#         if t.get("kind") == "base_patch":
#             base.append(int(t["token_idx"]))
#         elif t.get("kind") == "mosaic_patch":
#             mosaic.append(int(t["token_idx"]))
#     return np.asarray(base, dtype=np.int64), np.asarray(mosaic, dtype=np.int64)

# def get_region_token_mask(
#     mapping_tokens: List[dict],
#     region_yxyx: Tuple[float, float, float, float],
#     *,
#     min_overlap_frac: float,
# ) -> np.ndarray:
#     """
#     Boolean mask over packed image tokens (length = total_packed_image_tokens),
#     True where token bbox overlaps region bbox by >= min_overlap_frac.
#     """
#     n = max(int(t["token_idx"]) for t in mapping_tokens) + 1
#     mask = np.zeros((n,), dtype=bool)
#     for t in mapping_tokens:
#         bb = t.get("bbox")
#         if bb is None:
#             continue
#         tok_idx = int(t["token_idx"])
#         if overlap_frac_yxyx(tuple(bb), region_yxyx) >= min_overlap_frac:
#             mask[tok_idx] = True
#     return mask

# def extract_p_region_given_stream(
#     attn: np.ndarray,               # (L, S_prompt)
#     img_pos: np.ndarray,            # (N_img,), prompt indices corresponding to packed image tokens in order token_idx
#     mapping_tokens: List[dict],
#     regions_yxyx: Dict[str, Tuple[float, float, float, float]],
#     *,
#     min_overlap_frac: float,
#     min_denom: float,
# ) -> Dict[str, Dict[str, np.ndarray]]:
#     """
#     Returns:
#       out[stream]["region name"] = (L,) per-layer p(region | stream)
#     where stream in {"base","mosaic"} is defined over image tokens.
#     """
#     L = attn.shape[0]

#     # Attention restricted to image tokens, indexed by token_idx order
#     # token_scores[layer, token_idx] = attn[layer, img_pos[token_idx]]
#     token_scores = attn[:, img_pos]  # (L, N_img == total_packed_image_tokens)

#     base_idx, mosaic_idx = get_stream_token_indices(mapping_tokens)

#     out = {"base": {}, "mosaic": {}}

#     # precompute region masks over token_idx space
#     region_masks = {}
#     for name, bb in regions_yxyx.items():
#         region_masks[name] = get_region_token_mask(mapping_tokens, bb, min_overlap_frac=min_overlap_frac)

#     for stream, idx in [("base", base_idx), ("mosaic", mosaic_idx)]:
#         if idx.size == 0:
#             for rname in regions_yxyx.keys():
#                 out[stream][rname] = np.full((L,), np.nan, dtype=np.float32)
#             continue

#         denom = token_scores[:, idx].sum(axis=1)  # (L,)
#         denom = np.maximum(denom, min_denom)

#         for rname, rmask in region_masks.items():
#             # stream ∩ region
#             in_region = rmask[idx]
#             if not np.any(in_region):
#                 out[stream][rname] = np.full((L,), np.nan, dtype=np.float32)
#                 continue
#             num = token_scores[:, idx[in_region]].sum(axis=1)  # (L,)
#             out[stream][rname] = (num / denom).astype(np.float32)

#     return out


# # -------------------------
# # fooled definition
# # -------------------------
# def parse_pred_letter(meta: Dict[str, Any]) -> Optional[str]:
#     """
#     meta["generated_token_text"] might be "A", "A</s>", " A", etc.
#     We accept first occurrence of A/B/C/D.
#     """
#     s = str(meta.get("generated_token_text", ""))
#     for ch in s:
#         if ch in ("A", "B", "C", "D"):
#             return ch
#     return None

# def pred_key_from_meta(meta: Dict[str, Any]) -> Optional[str]:
#     letter = parse_pred_letter(meta)
#     if letter is None:
#         return None
#     opt = meta.get("option_meta", {})
#     l2k = opt.get("label_to_key", {})
#     return l2k.get(letter, None)

# def is_fooled(meta: Dict[str, Any], variant: str) -> bool:
#     """
#     Fooled for variant V means model selected the option whose key==V,
#     and that option is not the correct one.
#     """
#     correct = meta.get("correct_letter", None)
#     pred_letter = parse_pred_letter(meta)
#     if pred_letter is None or correct is None:
#         return False
#     if pred_letter == correct:
#         return False
#     pk = pred_key_from_meta(meta)
#     return pk == variant



# def is_helped(meta_c: Dict[str, Any], meta_n: Dict[str, Any],variant: str) -> bool:
#     """
#     used only for the correct variant
#         """
#     correct = meta_c.get("correct_letter", None)
#     pred_letter_c = parse_pred_letter(meta_c)
#     pred_letter_n = parse_pred_letter(meta_n)

#     if pred_letter_c is None or pred_letter_n is None or correct is None:
#         return False
    
#     if pred_letter_n != correct and pred_letter_c == correct:
#         return True

#     return False


# # -------------------------
# # stats + plotting
# # -------------------------
# def nanmean_ci(x: np.ndarray, ci: float = 95.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     x: (N, L)
#     returns (mean, lo, hi) using percentile CI across samples per layer
#     """
#     mean = np.nanmean(x, axis=0)
#     lo = np.nanpercentile(x, (100.0 - ci) / 2.0, axis=0)
#     hi = np.nanpercentile(x, 100.0 - (100.0 - ci) / 2.0, axis=0)
#     return mean, lo, hi

# def plot_three_lines_with_ci(
#     title: str,
#     x: np.ndarray,
#     series: Dict[str, np.ndarray],        # name -> (N,L)
#     out_path: Path,
#     ylabel: str,
#     ci: float = 95.0,
# ):
#     plt.figure(figsize=(12, 5))
#     for name, mat in series.items():
#         m, lo, hi = nanmean_ci(mat, ci=ci)
#         plt.plot(x, m, label=name)
#         plt.fill_between(x, lo, hi, alpha=0.2)
#     plt.xlabel("layer")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# # -------------------------
# # main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_root", type=str, required=True)
#     parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
#     parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
#     parser.add_argument("--split", type=str, default="test")

#     parser.add_argument("--variant", type=str, required=True,
#                         choices=["correct_answer","misleading_groundable","misleading_ungroundable","irrelevant_word"])
#     parser.add_argument("--qid_file", type=str, default="", help="Optional whitelist (e.g., no-overlap list)")
#     parser.add_argument("--out_dir", type=str, default="plots_fooled")

#     parser.add_argument("--min_overlap_frac", type=float, default=0.25)
#     parser.add_argument("--min_denom", type=float, default=1e-4)
#     parser.add_argument("--ci", type=float, default=95.0)

#     args = parser.parse_args()

#     npz_root = Path(args.npz_root)
#     out_root = Path(args.out_dir) / args.variant
#     out_root.mkdir(parents=True, exist_ok=True)

#     ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
#     if isinstance(ds, DatasetDict):
#         ds = ds[args.split]
#     qid_index = build_qid_index(ds)

#     whitelist = load_qid_whitelist(args.qid_file)

#     # enumerate qids we have NPZ for under this variant
#     var_dir = npz_root / args.variant / args.variant
#     if not var_dir.exists():
#         raise RuntimeError(f"Variant dir not found: {var_dir}")
#     all_qids = sorted([p.name for p in var_dir.iterdir() if p.is_dir()])

#     if whitelist is not None:
#         all_qids = [q for q in all_qids if q in whitelist or (q.lstrip("0") in whitelist) or (q.zfill(8) in whitelist)]

#     fooled_qids = []
#     # store per-sample curves for variant and for notext (paired)
#     # structure: stream -> region -> list of (L,) arrays
#     store_V = {"base": {}, "mosaic": {}}
#     store_N = {"base": {}, "mosaic": {}}

#     L_ref = None

#     for qid in all_qids:
#         npz_v = find_npz(npz_root, args.variant, qid)
#         npz_n = find_npz(npz_root, "notext", qid)
#         if npz_v is None or npz_n is None:
#             continue

#         dv = load_npz(npz_v)
#         dn = load_npz(npz_n)

#         # fooled subset defined by the variant's prediction/meta
#         # if not is_fooled(dv["meta"], args.variant):
#         #     continue

#         if not is_helped(dv["meta"], dn["meta"], args.variant):
#             continue

#         # mapping alignment sanity (required)
#         exp_v = int(dv["summary"]["total_packed_image_tokens"])
#         exp_n = int(dn["summary"]["total_packed_image_tokens"])
#         if len(dv["img_pos"]) != exp_v or len(dn["img_pos"]) != exp_n:
#             continue

#         # fetch sample + regions
#         qid_key = qid
#         if qid_key not in qid_index:
#             if qid_key.isdigit() and str(int(qid_key)) in qid_index:
#                 qid_key = str(int(qid_key))
#             elif qid_key.isdigit() and qid_key.zfill(8) in qid_index:
#                 qid_key = qid_key.zfill(8)
#             else:
#                 continue
#         sample = ds[qid_index[qid_key]]
#         regions = build_regions_yxyx(sample, args.variant)

#         # compute p(region|stream) for variant and notext
#         pv = extract_p_region_given_stream(
#             dv["attn"], dv["img_pos"], dv["mapping_tokens"], regions,
#             min_overlap_frac=args.min_overlap_frac,
#             min_denom=args.min_denom,
#         )
#         pn = extract_p_region_given_stream(
#             dn["attn"], dn["img_pos"], dn["mapping_tokens"], regions,
#             min_overlap_frac=args.min_overlap_frac,
#             min_denom=args.min_denom,
#         )

#         # establish L
#         if L_ref is None:
#             L_ref = dv["attn"].shape[0]
#         if dv["attn"].shape[0] != L_ref or dn["attn"].shape[0] != L_ref:
#             continue

#         fooled_qids.append(qid)

#         for stream in ("base","mosaic"):
#             for rname, vec in pv[stream].items():
#                 store_V[stream].setdefault(rname, []).append(vec)
#             for rname, vec in pn[stream].items():
#                 store_N[stream].setdefault(rname, []).append(vec)

#     N = len(fooled_qids)
#     if N == 0:
#         raise RuntimeError(f"No fooled samples found for variant={args.variant}. Check NPZ/meta parsing.")

#     layers = np.arange(L_ref, dtype=int)

#     # pack to (N,L)
#     def stack(store: Dict[str, List[np.ndarray]], rname: str) -> np.ndarray:
#         lst = store.get(rname, [])
#         if len(lst) == 0:
#             return np.full((N, L_ref), np.nan, dtype=np.float32)
#         return np.stack(lst, axis=0).astype(np.float32)

#     # choose exactly these three labels (some may be all-nan depending on availability)
#     region_names = ["text region", "correct object region", "misleading object region"]

#     # Plot fooled-only curves (3 lines) for BASE and MOSAIC
#     for stream in ("base","mosaic"):
#         series = {rn: stack(store_V[stream], rn) for rn in region_names}
#         plot_three_lines_with_ci(
#             title=f"{args.variant} | {stream.upper()} p(region | {stream}) | FOOLED only (N={N})",
#             x=layers,
#             series=series,
#             out_path=out_root / f"fooled_only_{stream}.png",
#             ylabel=f"p(region | {stream})  (mean of per-sample ratios)",
#             ci=args.ci,
#         )

#     # Plot deltas vs notext (paired within the fooled qids)
#     for stream in ("base","mosaic"):
#         series_delta = {}
#         for rn in region_names:
#             v = stack(store_V[stream], rn)
#             n0 = stack(store_N[stream], rn)
#             series_delta[rn] = v - n0
#         plot_three_lines_with_ci(
#             title=f"{args.variant} | {stream.upper()} Δp(region | {stream}) = variant - notext | FOOLED only (N={N})",
#             x=layers,
#             series=series_delta,
#             out_path=out_root / f"fooled_delta_vs_notext_{stream}.png",
#             ylabel=f"Δp(region | {stream})",
#             ci=args.ci,
#         )

#     # write report
#     report = {
#         "variant": args.variant,
#         "npz_root": str(npz_root),
#         "qid_file": args.qid_file,
#         "N_fooled": N,
#         "fooled_qids_first_20": fooled_qids[:20],
#         "settings": {
#             "min_overlap_frac": args.min_overlap_frac,
#             "min_denom": args.min_denom,
#             "ci": args.ci,
#             "aggregation": "mean of per-sample ratios; CI via nanpercentile; deltas are paired vs notext on same fooled qids",
#         },
#         "outputs": {
#             "fooled_only_base": str((out_root / "fooled_only_base.png").name),
#             "fooled_only_mosaic": str((out_root / "fooled_only_mosaic.png").name),
#             "fooled_delta_vs_notext_base": str((out_root / "fooled_delta_vs_notext_base.png").name),
#             "fooled_delta_vs_notext_mosaic": str((out_root / "fooled_delta_vs_notext_mosaic.png").name),
#         },
#     }
#     with open(out_root / "report_fooled.json", "w") as f:
#         json.dump(report, f, indent=2)

#     print(json.dumps(report, indent=2))


# if __name__ == "__main__":
#     main()

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

    # base_idx, mosaic_idx = get_stream_token_indices(mapping_tokens)
    # out = {"base": {}, "mosaic": {}}

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

        # stream mass per layer (gate)
        stream_mass = token_scores[:, idx].sum(axis=1)  # (L,)
        ok = stream_mass >= float(min_denom)

        # ---- Option C (density ratio) ----
        stream_den = stream_mass / float(idx.size)  # (L,)

        for rname, rmask in region_masks.items():
            # stream ∩ region
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
    # pk = parse_pred_letter(meta)

    # if (pk != correct) and (pn == correct):
        # print(f"qid = {meta.get('question_id')} meta variant={meta.get('variant')} pred_letter={pred_letter} correct={correct} pk={pk} pn={pn}")
    return (pred_letter != correct) and (pn == correct) and (pk == variant)

def is_robust(meta: Dict[str, Any], meta_n: Dict[str, Any], variant: str) -> bool:
    correct = meta["correct_letter"]
    pred_letter_v = parse_pred_letter(meta)
    pred_letter_n = parse_pred_letter(meta_n)
    if pred_letter_v is None or pred_letter_n is None or correct is None:
        return False
    return (pred_letter_v == correct) and (pred_letter_n == correct)

def is_wrongly_consistent(meta: Dict[str, Any], meta_n: Dict[str, Any], variant: str) -> bool:
    # print(meta)
    # print('---')
    # print(meta_n)
    # print('---')
    for k, v in meta['option_meta']['label_to_key'].items():
        if v == "misleading_groundable":
            misleading_groundable_letter = k

    
    correct = meta["correct_letter"]
    pred_letter_v = parse_pred_letter(meta)
    pred_letter_n = parse_pred_letter(meta_n)
    print(meta)
    # print(f"qid = {meta.get('question_id')} meta variant={meta.get('variant')} pred_letter_v={pred_letter_v} pred_letter_n={pred_letter_n} correct={correct}")
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
        default="misleading_ungroundable",
        choices=["misleading_groundable", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"],
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
        default="fooled",
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

    # Determine qids list
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
    # Establish L_ref by scanning for first available pair with valid shapes
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
    # streams = ["base", "mosaic"]

    def nan_vec():
        return np.full((L_ref,), np.nan, dtype=np.float32)

    # Store (N,L) per stream/region for variant and notext
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

        # default: not selected until proven otherwise
        is_selected = False

        pv = find_npz(npz_root, args.variant, qid)
        pn = find_npz(npz_root, "notext", qid)
        if pv is None:
            counts["missing_variant_npz"] += 1
        if pn is None:
            counts["missing_notext_npz"] += 1

        if pv is None or pn is None:
            # append NaNs (no skip)
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

        # Layer mismatch -> NaNs (no skip)
        if int(dv["attn"].shape[0]) != L_ref or int(dn["attn"].shape[0]) != L_ref:
            counts["layer_mismatch"] += 1
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        # Mapping sanity (must align img_pos length with summary total tokens)
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

        # Fetch dataset sample (no skip: missing row -> NaNs)
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

        # Subset selection (but no skip: non-selected become NaNs)
        if args.subset == "all":
            is_selected = True
        elif args.subset == "fooled":
            is_selected = is_fooled(dv["meta"], dn["meta"], args.variant)
        elif args.subset == "robust":
            is_selected = is_robust(dv["meta"], dn["meta"], args.variant)
        elif args.subset == "wrongly_consistent":
            is_selected = is_wrongly_consistent(dv["meta"], dn["meta"], args.variant)
        else:  # helped
            is_selected = is_helped(dv["meta"], dn["meta"])

        if is_selected:
            counts["N_selected"] += 1

        # If regions missing, still append NaNs (no skip)
        # (We only care about the 3 standard region names)
        if not any(r in regions for r in region_names):
            counts["regions_missing_or_empty"] += 1
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        if not is_selected:
            # Non-selected: append NaNs (kept in arrays; no skip)
            for s in streams:
                for r in region_names:
                    store_V[s][r].append(nan_vec())
                    store_N[s][r].append(nan_vec())
            selected_mask.append(False)
            continue

        # Compute metrics for selected
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

        # Append (always append all three region names; missing -> NaN)
        for s in streams:
            for r in region_names:
                store_V[s][r].append(pv_cur.get(s, {}).get(r, nan_vec()))
                store_N[s][r].append(pn_cur.get(s, {}).get(r, nan_vec()))

        selected_mask.append(True)

    # Stack helpers
    N = len(used_qids)

    def stack(store: Dict[str, List[np.ndarray]], rname: str) -> np.ndarray:
        # (N,L)
        return np.stack(store.get(rname, [nan_vec()] * N), axis=0).astype(np.float32)

    # Plot subset-only (implemented by NaN-ing non-selected; so nanmean is over selected)
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

    # Plot paired deltas (variant - notext) on the selected subset, via NaN masking
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
        "outputs": {
            "subset_only_base": "subset_only_base.png",
            "subset_only_mosaic": "subset_only_mosaic.png",
            "subset_delta_vs_notext_base": "subset_delta_vs_notext_base.png",
            "subset_delta_vs_notext_mosaic": "subset_delta_vs_notext_mosaic.png",
        },
    }
    with open(out_root / "report_subset.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()