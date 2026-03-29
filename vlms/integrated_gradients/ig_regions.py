# #!/usr/bin/env python3

# import json
# import argparse
# from pathlib import Path
# from typing import Any, Dict, Tuple, Optional, List

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes, label
# from datasets import load_dataset, load_from_disk


# # ============================================================
# # Defaults
# # ============================================================

# DEFAULT_HF_DATASET = "AHAAM/GUIC"
# DEFAULT_SPLIT = "test"
# DEFAULT_HF_CACHE_DIR = "./hf_dataset_GUIC"

# # The text target will come from this chosen variant
# DEFAULT_VARIANT = "misleading_ungroundable"  # correct_answer / misleading_groundable / misleading_ungroundable / irrelevant_word

# # Which saved grid to use from the npz
# DEFAULT_GRID_SOURCE = "base"  # mosaic / base

# # Region-growing parameters
# DEFAULT_SMOOTH_SIGMA = 0.7
# DEFAULT_ALPHA = 0.70
# DEFAULT_EDGE_TAU = 0.08
# DEFAULT_FLOOR_PERCENTILE = 65.0
# DEFAULT_MIN_REGION_SIZE = 3
# DEFAULT_EXPAND_RATIO = 0.20
# DEFAULT_EXPAND_MIN_PIXELS = 10
# DEFAULT_DILATE_ITERS = 1

# # Multi-seed parameters
# DEFAULT_TOPK_GLOBAL = 50

# DEFAULT_OUT_DIR = "./one_question_three_regions_mask_based_strict_sign"


# # ============================================================
# # Dataset helpers
# # ============================================================

# def sanitize_repo_id(repo_id: str) -> str:
#     return repo_id.replace("/", "__").replace(" ", "_")


# def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test"):
#     local_cache_root = Path(local_cache_root)
#     local_cache_root.mkdir(parents=True, exist_ok=True)
#     safe_name = sanitize_repo_id(dataset_id)
#     cache_dir = local_cache_root / safe_name

#     if cache_dir.exists():
#         return load_from_disk(str(cache_dir))

#     ds = load_dataset(dataset_id, split=split)
#     try:
#         ds.save_to_disk(str(cache_dir))
#     except Exception:
#         pass
#     return ds


# def get_sample_by_qid(ds, qid: str) -> Dict[str, Any]:
#     for i in range(len(ds)):
#         sample = ds[i]
#         if str(sample.get("question_id")) == str(qid):
#             return sample
#     raise ValueError(f"question_id={qid} not found in dataset")


# # ============================================================
# # NPZ loading
# # ============================================================

# def load_npz_with_json(npz_path: str) -> Dict[str, Any]:
#     z = np.load(npz_path, allow_pickle=True)
#     out = {}

#     for k in z.files:
#         v = z[k]
#         if isinstance(v, np.ndarray) and v.shape == ():
#             vv = v.item()
#             if isinstance(vv, str):
#                 try:
#                     out[k] = json.loads(vv)
#                     continue
#                 except Exception:
#                     out[k] = vv
#                     continue
#         out[k] = v

#     return out


# def token_scores_to_grids(mapping_tokens, token_scores, summary):
#     patches_per_side = summary["patches_per_side"]
#     mosaic_h, mosaic_w = summary["mosaic_unpadded_hw_in_patches"]

#     base_grid = np.zeros((patches_per_side, patches_per_side), dtype=np.float32)
#     mosaic_grid = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

#     for t in mapping_tokens:
#         idx = int(t["token_idx"])
#         if idx < 0 or idx >= len(token_scores):
#             continue

#         s = float(token_scores[idx])

#         if t["kind"] == "base_patch":
#             r, c = int(t["row"]), int(t["col"])
#             base_grid[r, c] += s
#         elif t["kind"] == "mosaic_patch":
#             r, c = int(t["row"]), int(t["col"])
#             mosaic_grid[r, c] += s

#     return base_grid, mosaic_grid


# def load_signed_grid_from_npz(npz_path: str, grid_source: str = DEFAULT_GRID_SOURCE) -> np.ndarray:
#     data = load_npz_with_json(npz_path)

#     if grid_source == "mosaic" and "mosaic_grid_signed" in data:
#         return data["mosaic_grid_signed"].astype(np.float32)

#     if grid_source == "base" and "base_grid_signed" in data:
#         return data["base_grid_signed"].astype(np.float32)

#     required = {"token_scores", "mapping_tokens", "mapping_summary"}
#     if not required.issubset(set(data.keys())):
#         raise ValueError(
#             f"NPZ must contain {grid_source}_grid_signed or token_scores + mapping_tokens + mapping_summary"
#         )

#     token_scores = data["token_scores"].astype(np.float32)
#     mapping_tokens = data["mapping_tokens"]
#     mapping_summary = data["mapping_summary"]
#     base_grid, mosaic_grid = token_scores_to_grids(mapping_tokens, token_scores, mapping_summary)

#     return mosaic_grid if grid_source == "mosaic" else base_grid


# # ============================================================
# # Geometry helpers
# # ============================================================

# def bbox_xywh_to_yxyx(box_xywh):
#     x, y, w, h = box_xywh
#     return float(y), float(x), float(y + h), float(x + w)


# def bbox_xyxy_to_yxyx(box_xyxy):
#     x1, y1, x2, y2 = box_xyxy
#     return float(y1), float(x1), float(y2), float(x2)


# def grid_cell_bbox_to_image(
#     r0: int, c0: int, r1: int, c1: int,
#     image_h: int, image_w: int,
#     grid_h: int, grid_w: int,
# ) -> Tuple[float, float, float, float]:
#     y0 = r0 * image_h / grid_h
#     x0 = c0 * image_w / grid_w
#     y1 = r1 * image_h / grid_h
#     x1 = c1 * image_w / grid_w
#     return y0, x0, y1, x1


# def annotation_bbox_to_grid_mask(
#     ann_bbox_yxyx: Tuple[float, float, float, float],
#     image_h: int,
#     image_w: int,
#     grid_h: int,
#     grid_w: int,
# ) -> np.ndarray:
#     y0, x0, y1, x1 = ann_bbox_yxyx
#     mask = np.zeros((grid_h, grid_w), dtype=bool)

#     for r in range(grid_h):
#         for c in range(grid_w):
#             gy0, gx0, gy1, gx1 = grid_cell_bbox_to_image(r, c, r + 1, c + 1, image_h, image_w, grid_h, grid_w)
#             cy = 0.5 * (gy0 + gy1)
#             cx = 0.5 * (gx0 + gx1)
#             if x0 <= cx <= x1 and y0 <= cy <= y1:
#                 mask[r, c] = True

#     return mask


# def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
#     ys, xs = np.where(mask)
#     if len(ys) == 0:
#         return None
#     return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


# def expand_bbox_yxyx(
#     bbox: Tuple[float, float, float, float],
#     image_h: int,
#     image_w: int,
#     expand_ratio: float = DEFAULT_EXPAND_RATIO,
#     expand_min_pixels: int = DEFAULT_EXPAND_MIN_PIXELS,
# ) -> Tuple[float, float, float, float]:
#     y0, x0, y1, x1 = bbox
#     h = y1 - y0
#     w = x1 - x0

#     dy = max(expand_min_pixels, expand_ratio * h)
#     dx = max(expand_min_pixels, expand_ratio * w)

#     ny0 = max(0.0, y0 - dy)
#     nx0 = max(0.0, x0 - dx)
#     ny1 = min(float(image_h), y1 + dy)
#     nx1 = min(float(image_w), x1 + dx)

#     return ny0, nx0, ny1, nx1


# def bbox_iou_yxyx(a, b) -> float:
#     ay0, ax0, ay1, ax1 = a
#     by0, bx0, by1, bx1 = b

#     iy0 = max(ay0, by0)
#     ix0 = max(ax0, bx0)
#     iy1 = min(ay1, by1)
#     ix1 = min(ax1, bx1)

#     ih = max(0.0, iy1 - iy0)
#     iw = max(0.0, ix1 - ix0)
#     inter = ih * iw

#     area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
#     area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
#     union = area_a + area_b - inter + 1e-12
#     return float(inter / union)


# def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
#     inter = np.logical_and(a, b).sum()
#     union = np.logical_or(a, b).sum()
#     return 0.0 if union == 0 else float(inter / union)


# # ============================================================
# # GUIC helpers
# # ============================================================

# def get_variant_image_and_boxes(sample: Dict[str, Any], variant: str):
#     entry = sample[variant]
#     image = entry["image"].convert("RGB")
#     W, H = image.size

#     text_bbox = None
#     if entry.get("bbox", None) is not None:
#         text_bbox = bbox_xyxy_to_yxyx(entry["bbox"])
#     if variant == "notext":
#         text_bbox = bbox_xyxy_to_yxyx(sample["correct_answer"]["bbox"])

#     object_bbox = None
#     if variant in ("correct_answer", "misleading_groundable"):
#         x = entry.get("x", None)
#         y = entry.get("y", None)
#         w = entry.get("w", None)
#         h = entry.get("h", None)
#         if None not in (x, y, w, h):
#             object_bbox = bbox_xywh_to_yxyx((x, y, w, h))

#     return image, H, W, text_bbox, object_bbox


# def build_three_targets(sample: Dict[str, Any], chosen_variant: str) -> List[Dict[str, Any]]:
#     targets: List[Dict[str, Any]] = []

#     _, _, _, _, correct_obj = get_variant_image_and_boxes(sample, "correct_answer")
#     if correct_obj is None:
#         raise ValueError("correct_answer object bbox not found")

#     targets.append({
#         "name": "correct_object",
#         "variant": "correct_answer",
#         "bbox_yxyx": correct_obj,
#     })

#     _, _, _, _, misleading_obj = get_variant_image_and_boxes(sample, "misleading_groundable")
#     if misleading_obj is None:
#         raise ValueError("misleading_groundable object bbox not found")

#     targets.append({
#         "name": "misleading_object",
#         "variant": "misleading_groundable",
#         "bbox_yxyx": misleading_obj,
#     })

#     _, _, _, chosen_text_bbox, _ = get_variant_image_and_boxes(sample, chosen_variant)
#     if chosen_text_bbox is None:
#         raise ValueError(f"{chosen_variant} text bbox not found")

#     targets.append({
#         "name": f"{chosen_variant}_text",
#         "variant": chosen_variant,
#         "bbox_yxyx": chosen_text_bbox,
#     })

#     return targets


# # ============================================================
# # Dynamic sign + multi-seed extraction (strict sign)
# # ============================================================

# def prepare_score_map(grid: np.ndarray, sign: str) -> np.ndarray:
#     g = grid.astype(np.float32)

#     if sign == "positive":
#         return np.clip(g, 0, None)
#     if sign == "negative":
#         return np.clip(-g, 0, None)

#     raise ValueError("sign must be 'positive' or 'negative'")


# def choose_dynamic_sign_from_bbox(
#     signed_grid: np.ndarray,
#     ann_bbox_yxyx: Tuple[float, float, float, float],
#     image_h: int,
#     image_w: int,
#     eps: float = 1e-8,
# ) -> Dict[str, Any]:
#     gh, gw = signed_grid.shape
#     ann_mask = annotation_bbox_to_grid_mask(ann_bbox_yxyx, image_h, image_w, gh, gw)

#     vals = signed_grid[ann_mask]
#     if vals.size == 0:
#         return {
#             "chosen_sign": "positive",
#             "net_attr": 0.0,
#             "pos_sum": 0.0,
#             "neg_sum": 0.0,
#             "n_tokens": 0,
#             "ann_mask": ann_mask,
#         }

#     net_attr = float(vals.sum())
#     pos_sum = float(np.clip(vals, 0, None).sum())
#     neg_sum = float(np.clip(-vals, 0, None).sum())

#     if abs(net_attr) < eps:
#         chosen_sign = "positive" if pos_sum >= neg_sum else "negative"
#     else:
#         chosen_sign = "positive" if net_attr > 0 else "negative"

#     return {
#         "chosen_sign": chosen_sign,
#         "net_attr": net_attr,
#         "pos_sum": pos_sum,
#         "neg_sum": neg_sum,
#         "n_tokens": int(vals.size),
#         "ann_mask": ann_mask,
#     }


# def sign_mask_from_grid(signed_grid: np.ndarray, chosen_sign: str) -> np.ndarray:
#     if chosen_sign == "positive":
#         return signed_grid > 0
#     return signed_grid < 0


# def get_topk_seed_points(score_map: np.ndarray, topk: int) -> List[Tuple[int, int, float]]:
#     flat = score_map.reshape(-1)
#     valid = np.isfinite(flat) & (flat > 0)
#     valid_idx = np.flatnonzero(valid)

#     if valid_idx.size == 0:
#         return []

#     k = min(topk, valid_idx.size)
#     vals = flat[valid_idx]
#     rel = np.argpartition(vals, -k)[-k:]
#     chosen = valid_idx[rel]

#     seeds = []
#     for idx in chosen:
#         r, c = np.unravel_index(idx, score_map.shape)
#         seeds.append((int(r), int(c), float(score_map[r, c])))

#     seeds.sort(key=lambda t: t[2], reverse=True)
#     return seeds


# def filter_seeds_by_mask(
#     seeds: List[Tuple[int, int, float]],
#     mask: np.ndarray,
# ) -> List[Tuple[int, int, float]]:
#     kept = []
#     for r, c, s in seeds:
#         if mask[r, c]:
#             kept.append((r, c, s))
#     return kept


# def find_strongest_seed_in_mask(score_map: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, float]]:
#     if mask.sum() == 0:
#         return None

#     vals = score_map.copy()
#     vals[~mask] = -np.inf
#     idx = np.argmax(vals)
#     peak_val = vals.reshape(-1)[idx]

#     if not np.isfinite(peak_val):
#         return None

#     r, c = np.unravel_index(idx, vals.shape)
#     return int(r), int(c), float(score_map[r, c])


# def region_grow_from_peak_strict(
#     signed_grid: np.ndarray,
#     score_map: np.ndarray,
#     chosen_sign: str,
#     seed_rc: Tuple[int, int],
#     peak_score: float,
#     alpha: float = DEFAULT_ALPHA,
#     edge_tau: float = DEFAULT_EDGE_TAU,
#     floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
# ) -> np.ndarray:
#     H, W = score_map.shape
#     region = np.zeros((H, W), dtype=bool)
#     visited = np.zeros((H, W), dtype=bool)

#     vals = score_map[score_map > 0]
#     if vals.size == 0:
#         return region

#     floor = float(np.percentile(vals, floor_percentile))
#     threshold = max(floor, peak_score * alpha)

#     sy, sx = seed_rc
#     stack = [(sy, sx, float(score_map[sy, sx]))]
#     neigh = [
#         (-1, -1), (-1, 0), (-1, 1),
#         ( 0, -1),          ( 0, 1),
#         ( 1, -1), ( 1, 0), ( 1, 1),
#     ]

#     while stack:
#         y, x, parent_score = stack.pop()

#         if visited[y, x]:
#             continue
#         visited[y, x] = True

#         raw_val = float(signed_grid[y, x])
#         if chosen_sign == "positive" and raw_val <= 0:
#             continue
#         if chosen_sign == "negative" and raw_val >= 0:
#             continue

#         s = float(score_map[y, x])

#         if s < threshold:
#             continue
#         if abs(parent_score - s) > edge_tau and (y, x) != (sy, sx):
#             continue

#         region[y, x] = True

#         for dy, dx in neigh:
#             ny, nx = y + dy, x + dx
#             if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
#                 stack.append((ny, nx, s))

#     return region


# def compute_component_stats(
#     region_mask: np.ndarray,
#     score_map: np.ndarray,
#     ann_mask: np.ndarray,
#     image_h: int,
#     image_w: int,
# ) -> Dict[str, Any]:
#     structure = np.ones((3, 3), dtype=np.int32)
#     comp_map, n_comp = label(region_mask.astype(np.uint8), structure=structure)

#     components = []
#     for comp_id in range(1, n_comp + 1):
#         comp_mask = comp_map == comp_id
#         n_tokens = int(comp_mask.sum())
#         if n_tokens == 0:
#             continue

#         vals = score_map[comp_mask]
#         bbox_grid = mask_to_bbox(comp_mask)
#         if bbox_grid is None:
#             continue

#         gy0, gx0, gy1, gx1 = bbox_grid
#         bbox_img = grid_cell_bbox_to_image(
#             gy0, gx0, gy1, gx1,
#             image_h=image_h,
#             image_w=image_w,
#             grid_h=region_mask.shape[0],
#             grid_w=region_mask.shape[1],
#         )

#         components.append({
#             "component_id": int(comp_id),
#             "n_tokens": n_tokens,
#             "sum_attr": float(vals.sum()),
#             "mean_attr": float(vals.mean()),
#             "max_attr": float(vals.max()),
#             "bbox_grid": [int(gy0), int(gx0), int(gy1), int(gx1)],
#             "bbox_yxyx": [float(x) for x in bbox_img],
#             "mask_iou_vs_annotation": float(mask_iou(comp_mask, ann_mask)),
#         })

#     components.sort(key=lambda x: x["sum_attr"], reverse=True)
#     return {
#         "component_map": comp_map.astype(np.int32),
#         "components": components,
#     }


# def extract_region_from_bbox_dynamic_multiseed_strict(
#     signed_grid: np.ndarray,
#     ann_bbox_yxyx: Tuple[float, float, float, float],
#     image_h: int,
#     image_w: int,
#     smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
#     alpha: float = DEFAULT_ALPHA,
#     edge_tau: float = DEFAULT_EDGE_TAU,
#     floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
#     min_region_size: int = DEFAULT_MIN_REGION_SIZE,
#     expand_ratio: float = DEFAULT_EXPAND_RATIO,
#     expand_min_pixels: int = DEFAULT_EXPAND_MIN_PIXELS,
#     dilate_iters: int = DEFAULT_DILATE_ITERS,
#     topk_global: int = DEFAULT_TOPK_GLOBAL,
# ) -> Optional[Dict[str, Any]]:
#     gh, gw = signed_grid.shape

#     sign_info = choose_dynamic_sign_from_bbox(
#         signed_grid=signed_grid,
#         ann_bbox_yxyx=ann_bbox_yxyx,
#         image_h=image_h,
#         image_w=image_w,
#     )
#     chosen_sign = sign_info["chosen_sign"]

#     expanded_bbox = expand_bbox_yxyx(
#         ann_bbox_yxyx,
#         image_h=image_h,
#         image_w=image_w,
#         expand_ratio=expand_ratio,
#         expand_min_pixels=expand_min_pixels,
#     )
#     expanded_mask = annotation_bbox_to_grid_mask(expanded_bbox, image_h, image_w, gh, gw)

#     score_map = prepare_score_map(signed_grid, chosen_sign)
#     if smooth_sigma > 0:
#         score_map = gaussian_filter(score_map, sigma=smooth_sigma)

#     # Enforce the selected sign on the smoothed map
#     raw_sign_mask = sign_mask_from_grid(signed_grid, chosen_sign)
#     score_map = score_map * raw_sign_mask.astype(np.float32)

#     global_seeds = get_topk_seed_points(score_map, topk=topk_global)
#     seeds_in_box = filter_seeds_by_mask(global_seeds, expanded_mask)

#     fallback_used = False
#     if len(seeds_in_box) == 0:
#         best = find_strongest_seed_in_mask(score_map, expanded_mask)
#         if best is None:
#             return None
#         seeds_in_box = [best]
#         fallback_used = True

#     union_region = np.zeros_like(score_map, dtype=bool)
#     used_seeds = []

#     for r, c, peak_score in seeds_in_box:
#         reg = region_grow_from_peak_strict(
#             signed_grid=signed_grid,
#             score_map=score_map,
#             chosen_sign=chosen_sign,
#             seed_rc=(r, c),
#             peak_score=peak_score,
#             alpha=alpha,
#             edge_tau=edge_tau,
#             floor_percentile=floor_percentile,
#         )
#         if reg.sum() == 0:
#             continue

#         union_region |= reg
#         used_seeds.append({
#             "rc": [int(r), int(c)],
#             "score": float(peak_score),
#         })

#     if union_region.sum() == 0:
#         return None

#     if dilate_iters > 0:
#         union_region = binary_dilation(union_region, iterations=dilate_iters)

#     union_region = binary_fill_holes(union_region)

#     # Re-enforce sign after morphology
#     union_region = union_region & raw_sign_mask

#     if union_region.sum() < min_region_size:
#         return None

#     bbox_grid = mask_to_bbox(union_region)
#     if bbox_grid is None:
#         return None

#     gy0, gx0, gy1, gx1 = bbox_grid
#     found_bbox_yxyx = grid_cell_bbox_to_image(gy0, gx0, gy1, gx1, image_h, image_w, gh, gw)

#     ann_mask = sign_info["ann_mask"]

#     component_info = compute_component_stats(
#         region_mask=union_region,
#         score_map=score_map,
#         ann_mask=ann_mask,
#         image_h=image_h,
#         image_w=image_w,
#     )

#     vals_selected = score_map[union_region]
#     vals_signed = signed_grid[union_region]

#     return {
#         "chosen_sign": chosen_sign,
#         "net_attr": sign_info["net_attr"],
#         "pos_sum": sign_info["pos_sum"],
#         "neg_sum": sign_info["neg_sum"],
#         "n_tokens_in_ann": sign_info["n_tokens"],
#         "expanded_bbox_yxyx": expanded_bbox,
#         "expanded_mask": expanded_mask,
#         "ann_mask": ann_mask,
#         "score_map": score_map,
#         "region_mask": union_region,
#         "bbox_grid": bbox_grid,
#         "found_bbox_yxyx": found_bbox_yxyx,
#         "region_size": int(union_region.sum()),
#         # chosen-sign-only stats
#         "region_sum": float(vals_selected.sum()),
#         "region_mean": float(vals_selected.mean()),
#         "region_max": float(vals_selected.max()),
#         # debug signed stats
#         "region_signed_sum": float(vals_signed.sum()),
#         "region_pos_sum": float(np.clip(vals_signed, 0, None).sum()),
#         "region_neg_sum": float(np.clip(-vals_signed, 0, None).sum()),
#         "bbox_iou": bbox_iou_yxyx(found_bbox_yxyx, ann_bbox_yxyx),
#         "mask_iou": mask_iou(union_region, ann_mask),
#         "used_seeds": used_seeds,
#         "n_global_seeds": len(global_seeds),
#         "n_seeds_in_box": len(seeds_in_box),
#         "fallback_used": fallback_used,
#         "component_map": component_info["component_map"],
#         "components": component_info["components"],
#     }


# # ============================================================
# # Visualization
# # ============================================================

# TARGET_COLORS = {
#     "correct_object": (0, 255, 0),
#     "misleading_object": (255, 0, 0),
#     "correct_answer_text": (0, 0, 255),
#     "misleading_groundable_text": (255, 140, 0),
#     "misleading_ungroundable_text": (128, 0, 255),
#     "irrelevant_word_text": (0, 200, 200),
# }

# COMPONENT_COLORS = [
#     "lime", "yellow", "magenta", "cyan", "orange", "white"
# ]


# def draw_box(draw: ImageDraw.ImageDraw, box_yxyx, color, width=3, label=None):
#     if box_yxyx is None:
#         return

#     y0, x0, y1, x1 = box_yxyx
#     draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

#     if label:
#         draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)


# def overlay_single_target(
#     image: Image.Image,
#     region_mask_grid: np.ndarray,
#     ann_bbox_yxyx,
#     expanded_bbox_yxyx,
#     out_path: str,
#     ann_color,
#     chosen_sign,
#     title="",
#     show_search_box: bool = False,
# ):
#     img = np.array(image).copy()
#     H, W = img.shape[:2]
#     gh, gw = region_mask_grid.shape

#     overlay = img.copy()

#     for r in range(gh):
#         for c in range(gw):
#             if region_mask_grid[r, c]:
#                 y0 = int(round(r * H / gh))
#                 y1 = int(round((r + 1) * H / gh))
#                 x0 = int(round(c * W / gw))
#                 x1 = int(round((c + 1) * W / gw))
#                 if chosen_sign == "positive":
#                     overlay[y0:y1, x0:x1, 1] = 255   # green
#                 else:
#                     overlay[y0:y1, x0:x1, 0] = 255   # red

#     blended = (0.3 * img + 0.7 * overlay).astype(np.uint8)
#     im = Image.fromarray(blended)
#     draw = ImageDraw.Draw(im)

#     draw_box(draw, ann_bbox_yxyx, ann_color, width=3, label="annotated")

#     if show_search_box:
#         draw_box(draw, expanded_bbox_yxyx, (255, 255, 0), width=2, label="search")

#     plt.figure(figsize=(8, 6))
#     plt.imshow(im)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def overlay_all_three_on_base(
#     image: Image.Image,
#     items: List[Dict[str, Any]],
#     out_path: str,
#     title="",
# ):
#     img = np.array(image).copy()
#     H, W = img.shape[:2]
#     overlay = img.copy()

#     for item in items:
#         region_mask = item.get("region_mask_array", None)
#         if region_mask is None:
#             continue

#         gh, gw = region_mask.shape
#         name = item["name"]

#         for r in range(gh):
#             for c in range(gw):
#                 if region_mask[r, c]:
#                     y0 = int(round(r * H / gh))
#                     y1 = int(round((r + 1) * H / gh))
#                     x0 = int(round(c * W / gw))
#                     x1 = int(round((c + 1) * W / gw))

#                     sign = item.get("chosen_sign", "positive")

#                     if sign == "positive":
#                         overlay[y0:y1, x0:x1, 1] = 255
#                     else:
#                         overlay[y0:y1, x0:x1, 0] = 255

#     blended = (0.1 * img + 0.9 * overlay).astype(np.uint8)
#     im = Image.fromarray(blended)
#     draw = ImageDraw.Draw(im)

#     for item in items:
#         name = item["name"]
#         ann_bbox = item["annotation_bbox_yxyx"]
#         ann_color = TARGET_COLORS.get(name, (255, 255, 255))
#         draw_box(draw, ann_bbox, ann_color, width=3, label=f"{name}:ann")

#     plt.figure(figsize=(10, 8))
#     plt.imshow(im)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def save_grid_debug(
#     signed_grid: np.ndarray,
#     result: Dict[str, Any],
#     out_path: str,
#     title: str = "",
# ):
#     plt.figure(figsize=(8, 6))

#     vmax = np.percentile(np.abs(signed_grid), 95)
#     vmax = max(vmax, 1e-6)
#     plt.imshow(signed_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

#     ey, ex = np.where(result["expanded_mask"])
#     if len(ey) > 0:
#         plt.scatter(ex, ey, s=8, c="yellow", alpha=0.25, label="expanded bbox")

#     ry, rx = np.where(result["region_mask"])
#     if len(ry) > 0:
#         plt.scatter(rx, ry, s=10, facecolors="none", edgecolors="lime", linewidths=0.8, label="union region")

#     for seed in result["used_seeds"]:
#         sr, sc = seed["rc"]
#         plt.scatter([sc], [sr], c="magenta", s=40, marker="x")

#     plt.colorbar()
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def save_component_overlay(
#     image: Image.Image,
#     result: Dict[str, Any],
#     ann_bbox_yxyx,
#     out_path: str,
#     title: str = "",
# ):
#     img = np.array(image).copy()
#     H, W = img.shape[:2]
#     gh, gw = result["component_map"].shape
#     comp_map = result["component_map"]

#     overlay = img.copy()
#     for comp_id in range(1, int(comp_map.max()) + 1):
#         mask = comp_map == comp_id
#         for r, c in zip(*np.where(mask)):
#             y0 = int(round(r * H / gh))
#             y1 = int(round((r + 1) * H / gh))
#             x0 = int(round(c * W / gw))
#             x1 = int(round((c + 1) * W / gw))
#             overlay[y0:y1, x0:x1, 1] = np.maximum(overlay[y0:y1, x0:x1, 1], 140)

#     blended = (0.30 * img + 0.70 * overlay).astype(np.uint8)
#     im = Image.fromarray(blended)
#     draw = ImageDraw.Draw(im)

#     draw_box(draw, ann_bbox_yxyx, (255, 0, 0), width=3, label="annotated")

#     for comp in result["components"]:
#         box = comp["bbox_yxyx"]
#         comp_id = comp["component_id"]
#         color_name = COMPONENT_COLORS[(comp_id - 1) % len(COMPONENT_COLORS)]
#         draw_box(draw, box, color_name, width=3, label=f"c{comp_id}")

#     plt.figure(figsize=(8, 6))
#     plt.imshow(im)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# # ============================================================
# # Saving mask artifacts
# # ============================================================

# def save_result_npz(
#     out_path: str,
#     result: Dict[str, Any],
# ):
#     np.savez_compressed(
#         out_path,
#         region_mask=result["region_mask"].astype(np.uint8),
#         component_map=result["component_map"].astype(np.int32),
#         ann_mask=result["ann_mask"].astype(np.uint8),
#         expanded_mask=result["expanded_mask"].astype(np.uint8),
#         score_map=result["score_map"].astype(np.float32),
#     )


# # ============================================================
# # Main
# # ============================================================

# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--question_id", type=str, default="06199707")
#     ap.add_argument("--npz", type=str, default="/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/integrated_gradients/llava-next_ig_token_outputs/misleading_groundable/misleading_groundable/06199707/ig_prefill_next_token.npz")

#     ap.add_argument(
#         "--variant",
#         type=str,
#         default="misleading_groundable",
#         choices=["notext", "correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"],
#         help="Chosen variant whose text bbox is used as the 3rd target",
#     )

#     ap.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET)
#     ap.add_argument("--split", type=str, default=DEFAULT_SPLIT)
#     ap.add_argument("--hf_cache_dir", type=str, default=DEFAULT_HF_CACHE_DIR)

#     ap.add_argument("--grid_source", type=str, default=DEFAULT_GRID_SOURCE, choices=["mosaic", "base"])

#     ap.add_argument("--smooth_sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
#     ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
#     ap.add_argument("--edge_tau", type=float, default=DEFAULT_EDGE_TAU)
#     ap.add_argument("--floor_percentile", type=float, default=DEFAULT_FLOOR_PERCENTILE)
#     ap.add_argument("--min_region_size", type=int, default=DEFAULT_MIN_REGION_SIZE)
#     ap.add_argument("--expand_ratio", type=float, default=DEFAULT_EXPAND_RATIO)
#     ap.add_argument("--expand_min_pixels", type=int, default=DEFAULT_EXPAND_MIN_PIXELS)
#     ap.add_argument("--dilate_iters", type=int, default=DEFAULT_DILATE_ITERS)
#     ap.add_argument("--topk_global", type=int, default=DEFAULT_TOPK_GLOBAL)

#     ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

#     args = ap.parse_args()

#     out_dir = Path(args.out_dir) / str(args.question_id) / args.variant
#     out_dir.mkdir(parents=True, exist_ok=True)

#     ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
#     sample = get_sample_by_qid(ds, args.question_id)

#     base_image = sample["notext"]["image"].convert("RGB")
#     grid = load_signed_grid_from_npz(args.npz, grid_source=args.grid_source)

#     targets = build_three_targets(sample, args.variant)

#     all_results_json: List[Dict[str, Any]] = []
#     all_results_plot: List[Dict[str, Any]] = []

#     for target in targets:
#         name = target["name"]
#         variant = target["variant"]
#         ann_bbox = target["bbox_yxyx"]

#         variant_image, H, W, _, _ = get_variant_image_and_boxes(sample, variant)

#         result = extract_region_from_bbox_dynamic_multiseed_strict(
#             signed_grid=grid,
#             ann_bbox_yxyx=ann_bbox,
#             image_h=H,
#             image_w=W,
#             smooth_sigma=args.smooth_sigma,
#             alpha=args.alpha,
#             edge_tau=args.edge_tau,
#             floor_percentile=args.floor_percentile,
#             min_region_size=args.min_region_size,
#             expand_ratio=args.expand_ratio,
#             expand_min_pixels=args.expand_min_pixels,
#             dilate_iters=args.dilate_iters,
#             topk_global=args.topk_global,
#         )

#         item_json: Dict[str, Any] = {
#             "name": name,
#             "variant": variant,
#             "annotation_bbox_yxyx": list(ann_bbox),
#             "status": "ok" if result is not None else "no_region_found",
#         }

#         item_plot: Dict[str, Any] = {
#             "name": name,
#             "annotation_bbox_yxyx": list(ann_bbox),
#         }

#         if result is not None:
#             item_json.update({
#                 "chosen_sign": result["chosen_sign"],
#                 "net_attr": result["net_attr"],
#                 "pos_sum": result["pos_sum"],
#                 "neg_sum": result["neg_sum"],
#                 "n_tokens_in_ann": result["n_tokens_in_ann"],
#                 "expanded_bbox_yxyx": list(result["expanded_bbox_yxyx"]),
#                 "found_bbox_yxyx": list(result["found_bbox_yxyx"]),
#                 "region_bbox_grid": list(result["bbox_grid"]),
#                 "region_size": result["region_size"],
#                 # chosen-sign-only stats
#                 "region_sum": result["region_sum"],
#                 "region_mean": result["region_mean"],
#                 "region_max": result["region_max"],
#                 # debug signed stats
#                 "region_signed_sum": result["region_signed_sum"],
#                 "region_pos_sum": result["region_pos_sum"],
#                 "region_neg_sum": result["region_neg_sum"],
#                 "bbox_iou": result["bbox_iou"],
#                 "mask_iou": result["mask_iou"],
#                 "used_seeds": result["used_seeds"],
#                 "n_global_seeds": result["n_global_seeds"],
#                 "n_seeds_in_box": result["n_seeds_in_box"],
#                 "fallback_used": result["fallback_used"],
#                 "components": result["components"],
#             })

#             item_plot["region_mask_array"] = result["region_mask"].copy()
#             item_plot["chosen_sign"] = result["chosen_sign"]

#             overlay_single_target(
#                 image=variant_image,
#                 region_mask_grid=result["region_mask"],
#                 ann_bbox_yxyx=ann_bbox,
#                 expanded_bbox_yxyx=result["expanded_bbox_yxyx"],
#                 out_path=str(out_dir / f"{name}_overlay.png"),
#                 ann_color=TARGET_COLORS.get(name, (255, 255, 255)),
#                 chosen_sign=result["chosen_sign"],
#                 title=(
#                     f"{args.question_id} | {name} | sign={result['chosen_sign']} | "
#                     f"net={result['net_attr']:.4f} | mean={result['region_mean']:.4f}"
#                 ),
#                 show_search_box=False,
#             )

#             save_grid_debug(
#                 signed_grid=grid,
#                 result=result,
#                 out_path=str(out_dir / f"{name}_grid_debug.png"),
#                 title=f"{args.question_id} | {name} | comps={len(result['components'])}",
#             )

#             save_component_overlay(
#                 image=variant_image,
#                 result=result,
#                 ann_bbox_yxyx=ann_bbox,
#                 out_path=str(out_dir / f"{name}_components_overlay.png"),
#                 title=f"{args.question_id} | {name} | components",
#             )

#             save_result_npz(
#                 out_path=str(out_dir / f"{name}_masks.npz"),
#                 result=result,
#             )

#         all_results_json.append(item_json)
#         all_results_plot.append(item_plot)

#     overlay_all_three_on_base(
#         image=base_image,
#         items=all_results_plot,
#         out_path=str(out_dir / "all_three_regions_on_base.png"),
#         title=f"{args.question_id} | correct object, misleading object, {args.variant} text",
#     )

#     report = {
#         "question_id": str(args.question_id),
#         "chosen_text_variant": args.variant,
#         "npz": args.npz,
#         "grid_source": args.grid_source,
#         "grid_shape": list(grid.shape),
#         "params": {
#             "smooth_sigma": args.smooth_sigma,
#             "alpha": args.alpha,
#             "edge_tau": args.edge_tau,
#             "floor_percentile": args.floor_percentile,
#             "min_region_size": args.min_region_size,
#             "expand_ratio": args.expand_ratio,
#             "expand_min_pixels": args.expand_min_pixels,
#             "dilate_iters": args.dilate_iters,
#             "topk_global": args.topk_global,
#         },
#         "results": all_results_json,
#     }

#     with open(out_dir / "report.json", "w") as f:
#         json.dump(report, f, indent=2)

#     print(json.dumps(report, indent=2))


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes, label
from datasets import load_dataset, load_from_disk


# ============================================================
# Defaults
# ============================================================

DEFAULT_HF_DATASET = "AHAAM/GUIC"
DEFAULT_SPLIT = "test"
DEFAULT_HF_CACHE_DIR = "./hf_dataset_GUIC"

DEFAULT_VARIANT = "misleading_ungroundable"
DEFAULT_GRID_SOURCE = "base"  # for Llava: base/mosaic, ignored for Qwen if qwen_grid_signed exists

DEFAULT_SMOOTH_SIGMA = 0.7
DEFAULT_ALPHA = 0.70
DEFAULT_EDGE_TAU = 0.08
DEFAULT_FLOOR_PERCENTILE = 65.0
DEFAULT_MIN_REGION_SIZE = 3
DEFAULT_EXPAND_RATIO = 0.20
DEFAULT_EXPAND_MIN_PIXELS = 10
DEFAULT_DILATE_ITERS = 1
DEFAULT_TOPK_GLOBAL = 50

DEFAULT_OUT_DIR = "./one_question_three_regions_mask_based_strict_sign"


# ============================================================
# Dataset helpers
# ============================================================

def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test"):
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    ds = load_dataset(dataset_id, split=split)
    try:
        ds.save_to_disk(str(cache_dir))
    except Exception:
        pass
    return ds


def get_sample_by_qid(ds, qid: str) -> Dict[str, Any]:
    for i in range(len(ds)):
        sample = ds[i]
        if str(sample.get("question_id")) == str(qid):
            return sample
    raise ValueError(f"question_id={qid} not found in dataset")


# ============================================================
# NPZ loading
# ============================================================

def load_npz_with_json(npz_path: str) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    out = {}

    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            vv = v.item()
            if isinstance(vv, str):
                try:
                    out[k] = json.loads(vv)
                    continue
                except Exception:
                    out[k] = vv
                    continue
        out[k] = v

    return out


def token_scores_to_grid_auto(mapping_tokens, token_scores, summary):
    """
    Reconstruct a signed 2D grid from token_scores + mapping_tokens + mapping_summary.

    Supports:
      - Llava-NeXT:
          kinds: base_patch / mosaic_patch
          summary keys: patches_per_side, mosaic_unpadded_hw_in_patches
      - Qwen:
          kinds: merged_patch
          summary keys: merged_grid_hw
    """
    kinds_present = {str(t.get("kind", "")) for t in mapping_tokens}

    # -------------------------
    # Qwen
    # -------------------------
    if "merged_patch" in kinds_present:
        if "merged_grid_hw" not in summary:
            raise KeyError("Qwen mapping_summary is missing 'merged_grid_hw'")

        gh, gw = summary["merged_grid_hw"]
        grid = np.zeros((int(gh), int(gw)), dtype=np.float32)

        for t in mapping_tokens:
            if t.get("kind") != "merged_patch":
                continue
            idx = int(t["token_idx"])
            if idx < 0 or idx >= len(token_scores):
                continue
            r = int(t["row"])
            c = int(t["col"])
            grid[r, c] += float(token_scores[idx])

        return grid

    # -------------------------
    # Llava
    # -------------------------
    if ("base_patch" in kinds_present) or ("mosaic_patch" in kinds_present):
        if "mosaic_patch" in kinds_present:
            if "mosaic_unpadded_hw_in_patches" not in summary:
                raise KeyError("Llava mapping_summary is missing 'mosaic_unpadded_hw_in_patches'")

            gh, gw = summary["mosaic_unpadded_hw_in_patches"]
            grid = np.zeros((int(gh), int(gw)), dtype=np.float32)

            for t in mapping_tokens:
                if t.get("kind") != "mosaic_patch":
                    continue
                idx = int(t["token_idx"])
                if idx < 0 or idx >= len(token_scores):
                    continue
                r = int(t["row"])
                c = int(t["col"])
                grid[r, c] += float(token_scores[idx])

            return grid

        if "patches_per_side" not in summary:
            raise KeyError("Llava mapping_summary is missing 'patches_per_side'")

        pps = int(summary["patches_per_side"])
        grid = np.zeros((pps, pps), dtype=np.float32)

        for t in mapping_tokens:
            if t.get("kind") != "base_patch":
                continue
            idx = int(t["token_idx"])
            if idx < 0 or idx >= len(token_scores):
                continue
            r = int(t["row"])
            c = int(t["col"])
            grid[r, c] += float(token_scores[idx])

        return grid

    raise ValueError(
        f"Could not infer model/grid type from mapping token kinds: {sorted(kinds_present)}"
    )


def load_signed_grid_from_npz(npz_path: str, grid_source: str = DEFAULT_GRID_SOURCE) -> np.ndarray:
    """
    Load a signed grid from NPZ.

    Priority:
      1) directly saved grids:
         - Llava: mosaic_grid_signed / base_grid_signed
         - Qwen:  qwen_grid_signed
      2) reconstruct from token_scores + mapping_tokens + mapping_summary
    """
    data = load_npz_with_json(npz_path)

    if "qwen_grid_signed" in data:
        return data["qwen_grid_signed"].astype(np.float32)

    if grid_source == "mosaic" and "mosaic_grid_signed" in data:
        return data["mosaic_grid_signed"].astype(np.float32)

    if grid_source == "base" and "base_grid_signed" in data:
        return data["base_grid_signed"].astype(np.float32)

    required = {"token_scores", "mapping_tokens", "mapping_summary"}
    if not required.issubset(set(data.keys())):
        raise ValueError(
            "NPZ must contain a saved signed grid "
            "(qwen_grid_signed / mosaic_grid_signed / base_grid_signed) "
            "or token_scores + mapping_tokens + mapping_summary"
        )

    token_scores = data["token_scores"].astype(np.float32)
    mapping_tokens = data["mapping_tokens"]
    mapping_summary = data["mapping_summary"]

    return token_scores_to_grid_auto(mapping_tokens, token_scores, mapping_summary)


# ============================================================
# Geometry helpers
# ============================================================

def bbox_xywh_to_yxyx(box_xywh):
    x, y, w, h = box_xywh
    return float(y), float(x), float(y + h), float(x + w)


def bbox_xyxy_to_yxyx(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return float(y1), float(x1), float(y2), float(x2)


def grid_cell_bbox_to_image(
    r0: int, c0: int, r1: int, c1: int,
    image_h: int, image_w: int,
    grid_h: int, grid_w: int,
) -> Tuple[float, float, float, float]:
    y0 = r0 * image_h / grid_h
    x0 = c0 * image_w / grid_w
    y1 = r1 * image_h / grid_h
    x1 = c1 * image_w / grid_w
    return y0, x0, y1, x1


def annotation_bbox_to_grid_mask(
    ann_bbox_yxyx: Tuple[float, float, float, float],
    image_h: int,
    image_w: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    y0, x0, y1, x1 = ann_bbox_yxyx
    mask = np.zeros((grid_h, grid_w), dtype=bool)

    for r in range(grid_h):
        for c in range(grid_w):
            gy0, gx0, gy1, gx1 = grid_cell_bbox_to_image(r, c, r + 1, c + 1, image_h, image_w, grid_h, grid_w)
            cy = 0.5 * (gy0 + gy1)
            cx = 0.5 * (gx0 + gx1)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                mask[r, c] = True

    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def expand_bbox_yxyx(
    bbox: Tuple[float, float, float, float],
    image_h: int,
    image_w: int,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
    expand_min_pixels: int = DEFAULT_EXPAND_MIN_PIXELS,
) -> Tuple[float, float, float, float]:
    y0, x0, y1, x1 = bbox
    h = y1 - y0
    w = x1 - x0

    dy = max(expand_min_pixels, expand_ratio * h)
    dx = max(expand_min_pixels, expand_ratio * w)

    ny0 = max(0.0, y0 - dy)
    nx0 = max(0.0, x0 - dx)
    ny1 = min(float(image_h), y1 + dy)
    nx1 = min(float(image_w), x1 + dx)

    return ny0, nx0, ny1, nx1


def bbox_iou_yxyx(a, b) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b

    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)

    ih = max(0.0, iy1 - iy0)
    iw = max(0.0, ix1 - ix0)
    inter = ih * iw

    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(inter / union)


# ============================================================
# GUIC helpers
# ============================================================

def get_variant_image_and_boxes(sample: Dict[str, Any], variant: str):
    entry = sample[variant]
    image = entry["image"].convert("RGB")
    W, H = image.size

    text_bbox = None
    if entry.get("bbox", None) is not None:
        text_bbox = bbox_xyxy_to_yxyx(entry["bbox"])
    if variant == "notext":
        text_bbox = bbox_xyxy_to_yxyx(sample["correct_answer"]["bbox"])

    object_bbox = None
    if variant in ("correct_answer", "misleading_groundable"):
        x = entry.get("x", None)
        y = entry.get("y", None)
        w = entry.get("w", None)
        h = entry.get("h", None)
        if None not in (x, y, w, h):
            object_bbox = bbox_xywh_to_yxyx((x, y, w, h))

    return image, H, W, text_bbox, object_bbox


def build_three_targets(sample: Dict[str, Any], chosen_variant: str) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []

    _, _, _, _, correct_obj = get_variant_image_and_boxes(sample, "correct_answer")
    if correct_obj is None:
        raise ValueError("correct_answer object bbox not found")

    targets.append({
        "name": "correct_object",
        "variant": "correct_answer",
        "bbox_yxyx": correct_obj,
    })

    _, _, _, _, misleading_obj = get_variant_image_and_boxes(sample, "misleading_groundable")
    if misleading_obj is None:
        raise ValueError("misleading_groundable object bbox not found")

    targets.append({
        "name": "misleading_object",
        "variant": "misleading_groundable",
        "bbox_yxyx": misleading_obj,
    })

    _, _, _, chosen_text_bbox, _ = get_variant_image_and_boxes(sample, chosen_variant)
    if chosen_text_bbox is None:
        raise ValueError(f"{chosen_variant} text bbox not found")

    targets.append({
        "name": f"{chosen_variant}_text",
        "variant": chosen_variant,
        "bbox_yxyx": chosen_text_bbox,
    })

    return targets


# ============================================================
# Dynamic sign + multi-seed extraction (strict sign)
# ============================================================

def prepare_score_map(grid: np.ndarray, sign: str) -> np.ndarray:
    g = grid.astype(np.float32)

    if sign == "positive":
        return np.clip(g, 0, None)
    if sign == "negative":
        return np.clip(-g, 0, None)

    raise ValueError("sign must be 'positive' or 'negative'")


def choose_dynamic_sign_from_bbox(
    signed_grid: np.ndarray,
    ann_bbox_yxyx: Tuple[float, float, float, float],
    image_h: int,
    image_w: int,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    gh, gw = signed_grid.shape
    ann_mask = annotation_bbox_to_grid_mask(ann_bbox_yxyx, image_h, image_w, gh, gw)

    vals = signed_grid[ann_mask]
    if vals.size == 0:
        return {
            "chosen_sign": "positive",
            "net_attr": 0.0,
            "pos_sum": 0.0,
            "neg_sum": 0.0,
            "n_tokens": 0,
            "ann_mask": ann_mask,
        }

    net_attr = float(vals.sum())
    pos_sum = float(np.clip(vals, 0, None).sum())
    neg_sum = float(np.clip(-vals, 0, None).sum())

    if abs(net_attr) < eps:
        chosen_sign = "positive" if pos_sum >= neg_sum else "negative"
    else:
        chosen_sign = "positive" if net_attr > 0 else "negative"

    return {
        "chosen_sign": chosen_sign,
        "net_attr": net_attr,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "n_tokens": int(vals.size),
        "ann_mask": ann_mask,
    }


def sign_mask_from_grid(signed_grid: np.ndarray, chosen_sign: str) -> np.ndarray:
    if chosen_sign == "positive":
        return signed_grid > 0
    return signed_grid < 0


def get_topk_seed_points(score_map: np.ndarray, topk: int) -> List[Tuple[int, int, float]]:
    flat = score_map.reshape(-1)
    valid = np.isfinite(flat) & (flat > 0)
    valid_idx = np.flatnonzero(valid)

    if valid_idx.size == 0:
        return []

    k = min(topk, valid_idx.size)
    vals = flat[valid_idx]
    rel = np.argpartition(vals, -k)[-k:]
    chosen = valid_idx[rel]

    seeds = []
    for idx in chosen:
        r, c = np.unravel_index(idx, score_map.shape)
        seeds.append((int(r), int(c), float(score_map[r, c])))

    seeds.sort(key=lambda t: t[2], reverse=True)
    return seeds


def filter_seeds_by_mask(
    seeds: List[Tuple[int, int, float]],
    mask: np.ndarray,
) -> List[Tuple[int, int, float]]:
    kept = []
    for r, c, s in seeds:
        if mask[r, c]:
            kept.append((r, c, s))
    return kept


def find_strongest_seed_in_mask(score_map: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int, float]]:
    if mask.sum() == 0:
        return None

    vals = score_map.copy()
    vals[~mask] = -np.inf
    idx = np.argmax(vals)
    peak_val = vals.reshape(-1)[idx]

    if not np.isfinite(peak_val):
        return None

    r, c = np.unravel_index(idx, vals.shape)
    return int(r), int(c), float(score_map[r, c])


def region_grow_from_peak_strict(
    signed_grid: np.ndarray,
    score_map: np.ndarray,
    chosen_sign: str,
    seed_rc: Tuple[int, int],
    peak_score: float,
    alpha: float = DEFAULT_ALPHA,
    edge_tau: float = DEFAULT_EDGE_TAU,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
) -> np.ndarray:
    H, W = score_map.shape
    region = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)

    vals = score_map[score_map > 0]
    if vals.size == 0:
        return region

    floor = float(np.percentile(vals, floor_percentile))
    threshold = max(floor, peak_score * alpha)

    sy, sx = seed_rc
    stack = [(sy, sx, float(score_map[sy, sx]))]
    neigh = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    while stack:
        y, x, parent_score = stack.pop()

        if visited[y, x]:
            continue
        visited[y, x] = True

        raw_val = float(signed_grid[y, x])
        if chosen_sign == "positive" and raw_val <= 0:
            continue
        if chosen_sign == "negative" and raw_val >= 0:
            continue

        s = float(score_map[y, x])

        if s < threshold:
            continue
        if abs(parent_score - s) > edge_tau and (y, x) != (sy, sx):
            continue

        region[y, x] = True

        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                stack.append((ny, nx, s))

    return region


def compute_component_stats(
    region_mask: np.ndarray,
    score_map: np.ndarray,
    ann_mask: np.ndarray,
    image_h: int,
    image_w: int,
) -> Dict[str, Any]:
    structure = np.ones((3, 3), dtype=np.int32)
    comp_map, n_comp = label(region_mask.astype(np.uint8), structure=structure)

    components = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = comp_map == comp_id
        n_tokens = int(comp_mask.sum())
        if n_tokens == 0:
            continue

        vals = score_map[comp_mask]
        bbox_grid = mask_to_bbox(comp_mask)
        if bbox_grid is None:
            continue

        gy0, gx0, gy1, gx1 = bbox_grid
        bbox_img = grid_cell_bbox_to_image(
            gy0, gx0, gy1, gx1,
            image_h=image_h,
            image_w=image_w,
            grid_h=region_mask.shape[0],
            grid_w=region_mask.shape[1],
        )

        components.append({
            "component_id": int(comp_id),
            "n_tokens": n_tokens,
            "sum_attr": float(vals.sum()),
            "mean_attr": float(vals.mean()),
            "max_attr": float(vals.max()),
            "bbox_grid": [int(gy0), int(gx0), int(gy1), int(gx1)],
            "bbox_yxyx": [float(x) for x in bbox_img],
            "mask_iou_vs_annotation": float(mask_iou(comp_mask, ann_mask)),
        })

    components.sort(key=lambda x: x["sum_attr"], reverse=True)
    return {
        "component_map": comp_map.astype(np.int32),
        "components": components,
    }


def extract_region_from_bbox_dynamic_multiseed_strict(
    signed_grid: np.ndarray,
    ann_bbox_yxyx: Tuple[float, float, float, float],
    image_h: int,
    image_w: int,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    alpha: float = DEFAULT_ALPHA,
    edge_tau: float = DEFAULT_EDGE_TAU,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    min_region_size: int = DEFAULT_MIN_REGION_SIZE,
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
    expand_min_pixels: int = DEFAULT_EXPAND_MIN_PIXELS,
    dilate_iters: int = DEFAULT_DILATE_ITERS,
    topk_global: int = DEFAULT_TOPK_GLOBAL,
) -> Optional[Dict[str, Any]]:
    gh, gw = signed_grid.shape

    sign_info = choose_dynamic_sign_from_bbox(
        signed_grid=signed_grid,
        ann_bbox_yxyx=ann_bbox_yxyx,
        image_h=image_h,
        image_w=image_w,
    )
    chosen_sign = sign_info["chosen_sign"]

    expanded_bbox = expand_bbox_yxyx(
        ann_bbox_yxyx,
        image_h=image_h,
        image_w=image_w,
        expand_ratio=expand_ratio,
        expand_min_pixels=expand_min_pixels,
    )
    expanded_mask = annotation_bbox_to_grid_mask(expanded_bbox, image_h, image_w, gh, gw)

    score_map = prepare_score_map(signed_grid, chosen_sign)
    if smooth_sigma > 0:
        score_map = gaussian_filter(score_map, sigma=smooth_sigma)

    raw_sign_mask = sign_mask_from_grid(signed_grid, chosen_sign)
    score_map = score_map * raw_sign_mask.astype(np.float32)

    global_seeds = get_topk_seed_points(score_map, topk=topk_global)
    seeds_in_box = filter_seeds_by_mask(global_seeds, expanded_mask)

    fallback_used = False
    if len(seeds_in_box) == 0:
        best = find_strongest_seed_in_mask(score_map, expanded_mask)
        if best is None:
            return None
        seeds_in_box = [best]
        fallback_used = True

    union_region = np.zeros_like(score_map, dtype=bool)
    used_seeds = []

    for r, c, peak_score in seeds_in_box:
        reg = region_grow_from_peak_strict(
            signed_grid=signed_grid,
            score_map=score_map,
            chosen_sign=chosen_sign,
            seed_rc=(r, c),
            peak_score=peak_score,
            alpha=alpha,
            edge_tau=edge_tau,
            floor_percentile=floor_percentile,
        )
        if reg.sum() == 0:
            continue

        union_region |= reg
        used_seeds.append({
            "rc": [int(r), int(c)],
            "score": float(peak_score),
        })

    if union_region.sum() == 0:
        return None

    if dilate_iters > 0:
        union_region = binary_dilation(union_region, iterations=dilate_iters)

    union_region = binary_fill_holes(union_region)
    union_region = union_region & raw_sign_mask

    if union_region.sum() < min_region_size:
        return None

    bbox_grid = mask_to_bbox(union_region)
    if bbox_grid is None:
        return None

    gy0, gx0, gy1, gx1 = bbox_grid
    found_bbox_yxyx = grid_cell_bbox_to_image(gy0, gx0, gy1, gx1, image_h, image_w, gh, gw)

    ann_mask = sign_info["ann_mask"]

    component_info = compute_component_stats(
        region_mask=union_region,
        score_map=score_map,
        ann_mask=ann_mask,
        image_h=image_h,
        image_w=image_w,
    )

    vals_selected = score_map[union_region]
    vals_signed = signed_grid[union_region]

    return {
        "chosen_sign": chosen_sign,
        "net_attr": sign_info["net_attr"],
        "pos_sum": sign_info["pos_sum"],
        "neg_sum": sign_info["neg_sum"],
        "n_tokens_in_ann": sign_info["n_tokens"],
        "expanded_bbox_yxyx": expanded_bbox,
        "expanded_mask": expanded_mask,
        "ann_mask": ann_mask,
        "score_map": score_map,
        "region_mask": union_region,
        "bbox_grid": bbox_grid,
        "found_bbox_yxyx": found_bbox_yxyx,
        "region_size": int(union_region.sum()),
        "region_sum": float(vals_selected.sum()),
        "region_mean": float(vals_selected.mean()),
        "region_max": float(vals_selected.max()),
        "region_signed_sum": float(vals_signed.sum()),
        "region_pos_sum": float(np.clip(vals_signed, 0, None).sum()),
        "region_neg_sum": float(np.clip(-vals_signed, 0, None).sum()),
        "bbox_iou": bbox_iou_yxyx(found_bbox_yxyx, ann_bbox_yxyx),
        "mask_iou": mask_iou(union_region, ann_mask),
        "used_seeds": used_seeds,
        "n_global_seeds": len(global_seeds),
        "n_seeds_in_box": len(seeds_in_box),
        "fallback_used": fallback_used,
        "component_map": component_info["component_map"],
        "components": component_info["components"],
    }


# ============================================================
# Visualization
# ============================================================

TARGET_COLORS = {
    "correct_object": (0, 255, 0),
    "misleading_object": (255, 0, 0),
    "correct_answer_text": (0, 0, 255),
    "misleading_groundable_text": (255, 140, 0),
    "misleading_ungroundable_text": (128, 0, 255),
    "irrelevant_word_text": (0, 200, 200),
}

COMPONENT_COLORS = [
    "lime", "yellow", "magenta", "cyan", "orange", "white"
]


def draw_box(draw: ImageDraw.ImageDraw, box_yxyx, color, width=3, label=None):
    if box_yxyx is None:
        return

    y0, x0, y1, x1 = box_yxyx
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    if label:
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)


def overlay_single_target(
    image: Image.Image,
    region_mask_grid: np.ndarray,
    ann_bbox_yxyx,
    expanded_bbox_yxyx,
    out_path: str,
    ann_color,
    chosen_sign,
    title="",
    show_search_box: bool = False,
):
    img = np.array(image).copy()
    H, W = img.shape[:2]
    gh, gw = region_mask_grid.shape

    overlay = img.copy()

    for r in range(gh):
        for c in range(gw):
            if region_mask_grid[r, c]:
                y0 = int(round(r * H / gh))
                y1 = int(round((r + 1) * H / gh))
                x0 = int(round(c * W / gw))
                x1 = int(round((c + 1) * W / gw))
                if chosen_sign == "positive":
                    overlay[y0:y1, x0:x1, 1] = 255
                else:
                    overlay[y0:y1, x0:x1, 0] = 255

    blended = (0.3 * img + 0.7 * overlay).astype(np.uint8)
    im = Image.fromarray(blended)
    draw = ImageDraw.Draw(im)

    draw_box(draw, ann_bbox_yxyx, ann_color, width=3, label="annotated")

    if show_search_box:
        draw_box(draw, expanded_bbox_yxyx, (255, 255, 0), width=2, label="search")

    plt.figure(figsize=(8, 6))
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def overlay_all_three_on_base(
    image: Image.Image,
    items: List[Dict[str, Any]],
    out_path: str,
    title="",
):
    img = np.array(image).copy()
    H, W = img.shape[:2]
    overlay = img.copy()

    for item in items:
        region_mask = item.get("region_mask_array", None)
        if region_mask is None:
            continue

        gh, gw = region_mask.shape
        for r in range(gh):
            for c in range(gw):
                if region_mask[r, c]:
                    y0 = int(round(r * H / gh))
                    y1 = int(round((r + 1) * H / gh))
                    x0 = int(round(c * W / gw))
                    x1 = int(round((c + 1) * W / gw))

                    sign = item.get("chosen_sign", "positive")
                    if sign == "positive":
                        overlay[y0:y1, x0:x1, 1] = 255
                    else:
                        overlay[y0:y1, x0:x1, 0] = 255

    blended = (0.1 * img + 0.9 * overlay).astype(np.uint8)
    im = Image.fromarray(blended)
    draw = ImageDraw.Draw(im)

    for item in items:
        name = item["name"]
        ann_bbox = item["annotation_bbox_yxyx"]
        ann_color = TARGET_COLORS.get(name, (255, 255, 255))
        draw_box(draw, ann_bbox, ann_color, width=3, label=f"{name}:ann")

    plt.figure(figsize=(10, 8))
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grid_debug(
    signed_grid: np.ndarray,
    result: Dict[str, Any],
    out_path: str,
    title: str = "",
):
    plt.figure(figsize=(8, 6))

    vmax = np.percentile(np.abs(signed_grid), 95)
    vmax = max(vmax, 1e-6)
    plt.imshow(signed_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ey, ex = np.where(result["expanded_mask"])
    if len(ey) > 0:
        plt.scatter(ex, ey, s=8, c="yellow", alpha=0.25, label="expanded bbox")

    ry, rx = np.where(result["region_mask"])
    if len(ry) > 0:
        plt.scatter(rx, ry, s=10, facecolors="none", edgecolors="lime", linewidths=0.8, label="union region")

    for seed in result["used_seeds"]:
        sr, sc = seed["rc"]
        plt.scatter([sc], [sr], c="magenta", s=40, marker="x")

    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_component_overlay(
    image: Image.Image,
    result: Dict[str, Any],
    ann_bbox_yxyx,
    out_path: str,
    title: str = "",
):
    img = np.array(image).copy()
    H, W = img.shape[:2]
    gh, gw = result["component_map"].shape
    comp_map = result["component_map"]

    overlay = img.copy()
    for comp_id in range(1, int(comp_map.max()) + 1):
        mask = comp_map == comp_id
        for r, c in zip(*np.where(mask)):
            y0 = int(round(r * H / gh))
            y1 = int(round((r + 1) * H / gh))
            x0 = int(round(c * W / gw))
            x1 = int(round((c + 1) * W / gw))
            overlay[y0:y1, x0:x1, 1] = np.maximum(overlay[y0:y1, x0:x1, 1], 140)

    blended = (0.30 * img + 0.70 * overlay).astype(np.uint8)
    im = Image.fromarray(blended)
    draw = ImageDraw.Draw(im)

    draw_box(draw, ann_bbox_yxyx, (255, 0, 0), width=3, label="annotated")

    for comp in result["components"]:
        box = comp["bbox_yxyx"]
        comp_id = comp["component_id"]
        color_name = COMPONENT_COLORS[(comp_id - 1) % len(COMPONENT_COLORS)]
        draw_box(draw, box, color_name, width=3, label=f"c{comp_id}")

    plt.figure(figsize=(8, 6))
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Saving mask artifacts
# ============================================================

def save_result_npz(
    out_path: str,
    result: Dict[str, Any],
):
    np.savez_compressed(
        out_path,
        region_mask=result["region_mask"].astype(np.uint8),
        component_map=result["component_map"].astype(np.int32),
        ann_mask=result["ann_mask"].astype(np.uint8),
        expanded_mask=result["expanded_mask"].astype(np.uint8),
        score_map=result["score_map"].astype(np.float32),
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--question_id", type=str, default="06199707")
    ap.add_argument("--npz", type=str, default="/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/integrated_gradients/llava-next_ig_token_outputs/misleading_groundable/misleading_groundable/06199707/ig_prefill_next_token.npz")

    ap.add_argument(
        "--variant",
        type=str,
        default="misleading_groundable",
        choices=["notext", "correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"],
        help="Chosen variant whose text bbox is used as the 3rd target",
    )

    ap.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET)
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    ap.add_argument("--hf_cache_dir", type=str, default=DEFAULT_HF_CACHE_DIR)
    ap.add_argument("--dataset_is_disk", action="store_true")

    ap.add_argument("--grid_source", type=str, default=DEFAULT_GRID_SOURCE, choices=["mosaic", "base"])

    ap.add_argument("--smooth_sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--edge_tau", type=float, default=DEFAULT_EDGE_TAU)
    ap.add_argument("--floor_percentile", type=float, default=DEFAULT_FLOOR_PERCENTILE)
    ap.add_argument("--min_region_size", type=int, default=DEFAULT_MIN_REGION_SIZE)
    ap.add_argument("--expand_ratio", type=float, default=DEFAULT_EXPAND_RATIO)
    ap.add_argument("--expand_min_pixels", type=int, default=DEFAULT_EXPAND_MIN_PIXELS)
    ap.add_argument("--dilate_iters", type=int, default=DEFAULT_DILATE_ITERS)
    ap.add_argument("--topk_global", type=int, default=DEFAULT_TOPK_GLOBAL)

    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    args = ap.parse_args()

    out_dir = Path(args.out_dir) / str(args.question_id) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.hf_dataset) if args.dataset_is_disk else get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    sample = get_sample_by_qid(ds, args.question_id)

    base_image = sample["notext"]["image"].convert("RGB")
    grid = load_signed_grid_from_npz(args.npz, grid_source=args.grid_source)

    targets = build_three_targets(sample, args.variant)

    all_results_json: List[Dict[str, Any]] = []
    all_results_plot: List[Dict[str, Any]] = []

    for target in targets:
        name = target["name"]
        variant = target["variant"]
        ann_bbox = target["bbox_yxyx"]

        variant_image, H, W, _, _ = get_variant_image_and_boxes(sample, variant)

        result = extract_region_from_bbox_dynamic_multiseed_strict(
            signed_grid=grid,
            ann_bbox_yxyx=ann_bbox,
            image_h=H,
            image_w=W,
            smooth_sigma=args.smooth_sigma,
            alpha=args.alpha,
            edge_tau=args.edge_tau,
            floor_percentile=args.floor_percentile,
            min_region_size=args.min_region_size,
            expand_ratio=args.expand_ratio,
            expand_min_pixels=args.expand_min_pixels,
            dilate_iters=args.dilate_iters,
            topk_global=args.topk_global,
        )

        item_json: Dict[str, Any] = {
            "name": name,
            "variant": variant,
            "annotation_bbox_yxyx": list(ann_bbox),
            "status": "ok" if result is not None else "no_region_found",
        }

        item_plot: Dict[str, Any] = {
            "name": name,
            "annotation_bbox_yxyx": list(ann_bbox),
        }

        if result is not None:
            item_json.update({
                "chosen_sign": result["chosen_sign"],
                "net_attr": result["net_attr"],
                "pos_sum": result["pos_sum"],
                "neg_sum": result["neg_sum"],
                "n_tokens_in_ann": result["n_tokens_in_ann"],
                "expanded_bbox_yxyx": list(result["expanded_bbox_yxyx"]),
                "found_bbox_yxyx": list(result["found_bbox_yxyx"]),
                "region_bbox_grid": list(result["bbox_grid"]),
                "region_size": result["region_size"],
                "region_sum": result["region_sum"],
                "region_mean": result["region_mean"],
                "region_max": result["region_max"],
                "region_signed_sum": result["region_signed_sum"],
                "region_pos_sum": result["region_pos_sum"],
                "region_neg_sum": result["region_neg_sum"],
                "bbox_iou": result["bbox_iou"],
                "mask_iou": result["mask_iou"],
                "used_seeds": result["used_seeds"],
                "n_global_seeds": result["n_global_seeds"],
                "n_seeds_in_box": result["n_seeds_in_box"],
                "fallback_used": result["fallback_used"],
                "components": result["components"],
            })

            item_plot["region_mask_array"] = result["region_mask"].copy()
            item_plot["chosen_sign"] = result["chosen_sign"]

            overlay_single_target(
                image=variant_image,
                region_mask_grid=result["region_mask"],
                ann_bbox_yxyx=ann_bbox,
                expanded_bbox_yxyx=result["expanded_bbox_yxyx"],
                out_path=str(out_dir / f"{name}_overlay.png"),
                ann_color=TARGET_COLORS.get(name, (255, 255, 255)),
                chosen_sign=result["chosen_sign"],
                title=(
                    f"{args.question_id} | {name} | sign={result['chosen_sign']} | "
                    f"net={result['net_attr']:.4f} | mean={result['region_mean']:.4f}"
                ),
                show_search_box=False,
            )

            save_grid_debug(
                signed_grid=grid,
                result=result,
                out_path=str(out_dir / f"{name}_grid_debug.png"),
                title=f"{args.question_id} | {name} | comps={len(result['components'])}",
            )

            save_component_overlay(
                image=variant_image,
                result=result,
                ann_bbox_yxyx=ann_bbox,
                out_path=str(out_dir / f"{name}_components_overlay.png"),
                title=f"{args.question_id} | {name} | components",
            )

            save_result_npz(
                out_path=str(out_dir / f"{name}_masks.npz"),
                result=result,
            )

        all_results_json.append(item_json)
        all_results_plot.append(item_plot)

    overlay_all_three_on_base(
        image=base_image,
        items=all_results_plot,
        out_path=str(out_dir / "all_three_regions_on_base.png"),
        title=f"{args.question_id} | correct object, misleading object, {args.variant} text",
    )

    report = {
        "question_id": str(args.question_id),
        "chosen_text_variant": args.variant,
        "npz": args.npz,
        "grid_source": args.grid_source,
        "grid_shape": list(grid.shape),
        "params": {
            "smooth_sigma": args.smooth_sigma,
            "alpha": args.alpha,
            "edge_tau": args.edge_tau,
            "floor_percentile": args.floor_percentile,
            "min_region_size": args.min_region_size,
            "expand_ratio": args.expand_ratio,
            "expand_min_pixels": args.expand_min_pixels,
            "dilate_iters": args.dilate_iters,
            "topk_global": args.topk_global,
        },
        "results": all_results_json,
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()