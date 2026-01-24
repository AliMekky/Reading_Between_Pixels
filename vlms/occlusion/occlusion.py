# import json
# from pathlib import Path
# from typing import Dict, List
# from collections import defaultdict

# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# from torch.nn.functional import log_softmax

# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom

# # Your VLM evaluator wrapper
# from infere_vlms import get_evaluator


# class ImageVariantAnalyzer:
#     """
#     Phase 1-style analysis, but using occlusion-style image attribution
#     on the IMAGE for LLaVA / LLaVA-Next-style models.

#     Pipeline:
#       1) Load existing JSON results for variants:
#          - notext
#          - correct
#          - incorrect
#          - irrelevant
#       2) Identify cases where predictions differ across variants.
#       3) For those cases, compute occlusion attributions per variant:
#            - base log p(answer token)
#            - mask patches, recompute log p
#            - score = base_logp - masked_logp
#       4) Visualize per-case, per-variant heatmaps and overlays.
#       5) Compute simple stats per pattern & variant.
#     """

#     def __init__(self, model_type: str = "llava", model_id: str = None, device: str = "auto"):
#         print(f"Loading {model_type} model...")
#         self.evaluator = get_evaluator(model_type, model_id, device)
#         self.model = self.evaluator.model
#         self.processor = self.evaluator.processor
#         self.device = self.evaluator.device

#     # ------------------------------------------------------------------ #
#     # BASIC UTILITIES
#     # ------------------------------------------------------------------ #
#     def _get_device(self) -> torch.device:
#         if hasattr(self.model, "device"):
#             return self.model.device
#         return next(self.model.parameters()).device

#     def _topk_mask(self, attr: np.ndarray, keep_ratio: float = 0.02):
#         flat = attr.flatten()
#         if flat.size == 0:
#             return attr
#         k = max(int(flat.size * keep_ratio), 1)
#         thresh = np.partition(flat, -k)[-k]

#         mask = attr >= thresh
#         out = np.zeros_like(attr, dtype=np.float32)
#         if np.any(mask):
#             out[mask] = attr[mask] / attr[mask].max()
#         return out

#     def _sparse_signed_delta(self, attr1: np.ndarray, attr2: np.ndarray, keep_ratio: float = 0.01):
#         """
#         Build a sparse signed delta map:

#             delta = attr1 - attr2

#         - magnitude: keep only top `keep_ratio` fraction of |delta|
#         - sign: positive = regions where variant1 gained importance,
#                 negative = regions where variant1 lost importance.

#         Returns a 2D array in [-1, 1].
#         """
#         delta = attr1.astype(np.float32) - attr2.astype(np.float32)
#         mag = np.abs(delta)
#         mag_sparse = self._topk_mask(mag, keep_ratio=keep_ratio)  # [0,1] for top-k, 0 elsewhere

#         out = np.zeros_like(delta, dtype=np.float32)
#         pos_mask = delta > 0
#         neg_mask = delta < 0

#         out[pos_mask] = mag_sparse[pos_mask]    # +[0,1]
#         out[neg_mask] = -mag_sparse[neg_mask]   # -[0,1]

#         return out  # in [-1, 1]

#     def _example_id_from_filename(self, fname: str) -> str:
#         """
#         Map variant-specific filenames to a shared example id.

#         Examples:
#           '0_3.jpg'      -> '0'
#           '1_0.jpg'      -> '1'
#           'image_0.jpg'  -> '0'
#           'image_10.jpg' -> '10'
#         """
#         fname = Path(fname).name  # strip any path

#         if fname.startswith("image_"):
#             # 'image_0.jpg' -> '0'
#             core = fname[len("image_"):]
#             ex_id = core.split(".")[0]
#         else:
#             # '0_3.jpg' -> '0'
#             ex_id = fname.split("_")[0]

#         return ex_id

#     def _prepare_inputs_inference(self, image: Image.Image, prompt: str) -> Dict:
#         """
#         Original inference-style input builder (user-only, add_generation_prompt=True).
#         Not used in occlusion, but kept for reference.
#         """
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image"},
#                 ],
#             },
#         ]

#         formatted_prompt = self.processor.apply_chat_template(
#             conversation,
#             add_generation_prompt=True,
#         )

#         inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt")
#         inputs = inputs.to(self.device)

#         if "pixel_values" in inputs:
#             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

#         return inputs

#     def _prepare_inputs_with_response(self, image: Image.Image, question: str, response: str) -> Dict:
#         """
#         Prepare inputs including both:
#           - user question (with image)
#           - assistant response (text)

#         This is what we use for log-prob computations:
#           - add_generation_prompt=False because we already include the answer.
#         """
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                     {"type": "image"},
#                 ],
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": response},
#                 ],
#             },
#         ]

#         formatted_prompt = self.processor.apply_chat_template(
#             conversation,
#             add_generation_prompt=False,
#         )

#         inputs = self.processor(
#             images=image,
#             text=formatted_prompt,
#             return_tensors="pt",
#         )
#         inputs = inputs.to(self.device)

#         if "pixel_values" in inputs:
#             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

#         return inputs

#     # ------------------------------------------------------------------ #
#     # LOAD RESULTS & IDENTIFY CONFLICT CASES
#     # ------------------------------------------------------------------ #
#     def load_results(self, results_dir: str) -> Dict[str, List[Dict]]:
#         """
#         Load inference results from all variants.
#         """
#         results_dir = Path(results_dir)

#         variant_files = {
#             "notext": "llava-next_notext_results.json",
#             "correct": "llava-next_correct_results.json",
#             "incorrect": "llava-next_misleading_results.json",
#             "irrelevant": "llava-next_irrelevant_results.json",
#         }

#         all_results = {}

#         for variant, filename in variant_files.items():
#             filepath = results_dir / filename
#             if filepath.exists():
#                 with open(filepath, "r") as f:
#                     all_results[variant] = json.load(f)
#                 print(f"Loaded {len(all_results[variant])} results from {filename}")
#             else:
#                 print(f"Warning: {filename} not found")

#         return all_results

#     def identify_differing_cases(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
#         print("\n" + "=" * 60)
#         print("IDENTIFYING CASES WITH VARIANT DIFFERENCES")
#         print("=" * 60)

#         # Group by logical example id, not raw filename
#         grouped = defaultdict(dict)

#         for variant, results in all_results.items():
#             for item in results:
#                 ex_id = self._example_id_from_filename(item["image"])
#                 grouped[ex_id][variant] = item

#         differing_cases = []

#         for ex_id, variants in grouped.items():
#             predictions = {
#                 variant: data["predicted_answer"]
#                 for variant, data in variants.items()
#             }
#             unique_predictions = set(predictions.values())

#             if len(unique_predictions) > 1:
#                 correctness = {
#                     variant: data["is_correct"]
#                     for variant, data in variants.items()
#                 }
#                 case = {
#                     "image_id": ex_id,  # logical example id
#                     "image_files": {v: d["image"] for v, d in variants.items()},
#                     "predictions": predictions,
#                     "correctness": correctness,
#                     "variants": variants,
#                     "pattern": self._categorize_pattern(predictions, correctness),
#                 }

#                 differing_cases.append(case)

#         print(f"\nFound {len(differing_cases)} cases with variant differences")
#         print(f"Total cases: {len(grouped)}")
#         print(f"Proportion: {len(differing_cases) / max(len(grouped), 1) * 100:.1f}%")

#         pattern_counts = defaultdict(int)
#         for case in differing_cases:
#             pattern_counts[case["pattern"]] += 1

#         print("\nPattern Distribution:")
#         print("-" * 60)
#         for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
#             print(f"  {pattern:40s}: {count:4d} cases")

#         return differing_cases

#     def _normalize_attr_for_viz(self, attr: np.ndarray):
#         """
#         Normalize a signed 2D attribution map for visualization.

#         - subtract mean
#         - divide by std
#         - clip to robust percentiles

#         Returns (attr_norm, vmin, vmax).
#         """
#         attr = attr.astype(np.float32)
#         attr = attr - np.mean(attr)

#         std = np.std(attr)
#         if std > 1e-8:
#             attr = attr / std

#         lo, hi = np.percentile(attr, [5, 95])
#         if hi - lo < 1e-6:
#             lo, hi = -1.0, 1.0

#         return attr, float(lo), float(hi)

#     def _categorize_pattern(self, predictions: Dict[str, str],
#                             correctness: Dict[str, bool]) -> str:
#         """
#         Categorize behavioral pattern based on correctness of each variant.
#         """
#         nt  = correctness.get("notext")
#         cor = correctness.get("correct")
#         inc = correctness.get("incorrect")
#         irr = correctness.get("irrelevant")

#         # Fooled by incorrect text
#         if nt is True and inc is False:
#             return "fooled_by_incorrect"

#         # Fooled by irrelevant text
#         if nt is True and irr is False:
#             return "fooled_by_irrelevant"

#         # Helped by correct text
#         if nt is False and cor is True:
#             return "helped_by_correct"

#         # Resisted incorrect text
#         if inc is False and cor is True:
#             return "resisted_incorrect"

#         vals = [v for v in correctness.values() if isinstance(v, bool)]
#         if vals and all(vals):
#             return "all_correct"

#         if vals and not any(vals):
#             return "all_incorrect"

#         return "mixed_behavior"

#     # ------------------------------------------------------------------ #
#     #  LOG PROB + OCCLUSION
#     # ------------------------------------------------------------------ #
#     def _forward_log_prob(self, image: Image.Image, question: str, response: str) -> float:
#         """
#         Compute log p(target_token | question, image, fixed response)
#         using the same chat template setup as inference.

#         Target token = first token of response (or last input token if empty).
#         """
#         self.model.eval()

#         # Use the same "question + image, assistant: response" setup
#         inputs = self._prepare_inputs_with_response(image, question, response)

#         input_ids = inputs["input_ids"]
#         resp_ids = self.processor.tokenizer.encode(response, add_special_tokens=False)
#         if len(resp_ids) == 0:
#             target_id = input_ids[0, -1].item()
#         else:
#             target_id = resp_ids[0]

#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits  # (1, T, V)
#             log_probs = log_softmax(logits[:, -1, :], dim=-1)
#             logp = log_probs[0, target_id].item()

#         return logp

#     def _image_occlusion_attributions(
#         self,
#         image: Image.Image,
#         question: str,
#         response: str,
#         grid_size: int = 16,
#         mask_value: int = 127,
#     ) -> np.ndarray:
#         """
#         Patch occlusion:
#           - divide image into grid_size x grid_size patches
#           - mask each patch, recompute log p(answer)
#           - score = base_logp - masked_logp

#         Returns a (grid_size, grid_size) map.
#         """
#         W, H = image.size
#         patch_w = W // grid_size
#         patch_h = H // grid_size

#         base_logp = self._forward_log_prob(image, question, response)

#         scores = np.zeros((grid_size, grid_size), dtype=np.float32)

#         for gy in range(grid_size):
#             for gx in range(grid_size):
#                 x0 = gx * patch_w
#                 y0 = gy * patch_h
#                 x1 = W if gx == grid_size - 1 else (gx + 1) * patch_w
#                 y1 = H if gy == grid_size - 1 else (gy + 1) * patch_h

#                 img_masked = image.copy()
#                 arr = np.array(img_masked)
#                 arr[y0:y1, x0:x1, :] = mask_value  # gray patch
#                 img_masked = Image.fromarray(arr)

#                 masked_logp = self._forward_log_prob(img_masked, question, response)
#                 scores[gy, gx] = base_logp - masked_logp

#         return scores  # higher = more important for this answer

#     # ------------------------------------------------------------------ #
#     # PER-CASE IMAGE ATTRIBUTIONS FOR VARIANTS
#     # ------------------------------------------------------------------ #
#     def compute_image_attributions_for_case(self, case: Dict, image_folder: str) -> Dict:
#         """
#         Compute occlusion attributions for all variants in a single conflict case.
#         Uses stored predictions & questions from the JSON (no re-inference).
#         """
#         image_folder = Path(image_folder)
#         results = {
#             "image_id": case["image_id"],
#             "pattern": case["pattern"],
#             "variants": {},
#         }

#         variant_to_subdir = {
#             "notext": "notext",
#             "correct": "correct",
#             "incorrect": "misleading",
#             "irrelevant": "irrelevant",
#         }

#         for variant_name, variant_data in case["variants"].items():
#             if variant_name not in variant_to_subdir:
#                 continue

#             image_path = image_folder / variant_to_subdir[variant_name] / variant_data["image"]
#             if not image_path.exists():
#                 print(f"Warning: Image not found: {image_path}")
#                 continue

#             image = Image.open(image_path).convert("RGB")

#             question = variant_data.get("question", "")
#             response = variant_data["predicted_answer"]

#             if not question:
#                 print(f"Warning: empty question for {case['image_id']} ({variant_name})")
#                 continue

#             # Occlusion map (grid_size x grid_size)
#             attr_map = self._image_occlusion_attributions(
#                 image=image,
#                 question=question,
#                 response=response,
#                 grid_size=16,   # you can tune this
#             )

#             total_image_importance = float(np.sum(np.abs(attr_map)))

#             results["variants"][variant_name] = {
#                 "prediction": response,
#                 "is_correct": variant_data["is_correct"],
#                 "image_attr_map": attr_map,
#                 "total_image_importance": total_image_importance,
#                 "image": image,
#                 "image_path": str(image_path),
#             }

#         return results

#     # ------------------------------------------------------------------ #
#     # VISUALIZATION
#     # ------------------------------------------------------------------ #
#     def _resize_attr_to_image(self, attr_map: np.ndarray, image: Image.Image) -> np.ndarray:
#         """
#         Resize attr_map to the original image size.
#         Accepts:
#         - (H, W)
#         - (C, H, W)
#         - (1, H, W)
#         and reduces to (H, W).
#         """
#         if isinstance(attr_map, torch.Tensor):
#             attr_map = attr_map.detach().cpu().numpy()

#         if attr_map.ndim == 4:
#             attr_map = attr_map[0]
#         if attr_map.ndim == 3:
#             attr_map = attr_map.mean(axis=0)

#         if attr_map.ndim != 2:
#             raise ValueError(f"Expected 2D attr_map in _resize_attr_to_image, got {attr_map.shape}")

#         img_array = np.array(image)
#         H_img, W_img = img_array.shape[:2]
#         H_pre, W_pre = attr_map.shape

#         scale_h = H_img / H_pre
#         scale_w = W_img / W_pre

#         attr_resized = zoom(attr_map, (scale_h, scale_w), order=1)
#         return attr_resized

#     def visualize_case_comparison(self, case_results: Dict, output_path: Path):
#         variants = list(case_results["variants"].keys())
#         num_variants = len(variants)
#         if num_variants == 0:
#             return

#         fig, axes = plt.subplots(num_variants, 3, figsize=(18, 6 * num_variants))
#         if num_variants == 1:
#             axes = axes.reshape(1, -1)

#         for idx, variant in enumerate(variants):
#             v_data = case_results["variants"][variant]
#             image = v_data["image"]
#             img_array = np.array(image)

#             # 1) resize occlusion map to image size
#             attr_resized = self._resize_attr_to_image(v_data["image_attr_map"], image)

#             # 2) take magnitude and keep only top 1% of pixels
#             attr_abs = np.abs(attr_resized)
#             attr_sparse = self._topk_mask(attr_abs, keep_ratio=0.01)

#             # ----- column 1: original image -----
#             axes[idx, 0].imshow(img_array)
#             axes[idx, 0].set_title(f"{variant} - Original", fontsize=10)
#             axes[idx, 0].axis("off")

#             # ----- column 2: sparse heatmap only -----
#             im1 = axes[idx, 1].imshow(
#                 attr_sparse,
#                 cmap="hot",
#                 vmin=0.0,
#                 vmax=1.0,
#                 interpolation="nearest",
#             )
#             axes[idx, 1].set_title(
#                 f"{variant} - Heatmap\nPred: {v_data['prediction']} "
#                 f"({'✓' if v_data['is_correct'] else '✗'})",
#                 fontsize=10,
#             )
#             axes[idx, 1].axis("off")
#             plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)

#             # ----- column 3: sparse overlay on the image -----
#             axes[idx, 2].imshow(img_array)
#             im2 = axes[idx, 2].imshow(
#                 attr_sparse,
#                 alpha=0.7,
#                 cmap="jet",
#                 vmin=0.0,
#                 vmax=1.0,
#                 interpolation="nearest",
#             )
#             axes[idx, 2].set_title(
#                 f"{variant} - Overlay\nTotal Occlusion (|·| sum): {v_data['total_image_importance']:.3f}",
#                 fontsize=10,
#             )
#             axes[idx, 2].axis("off")
#             plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)

#         plt.suptitle(
#             f"Image Occlusion Comparison - {case_results['image_id']}\nPattern: {case_results['pattern']}",
#             fontsize=14,
#             y=0.995,
#         )
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=300)
#         plt.close()

#         # ------------------------------------------------------------------
#         # EXTRA: delta maps between variants (option 2)
#         # ------------------------------------------------------------------
#         delta_pairs = [
#             ("incorrect", "notext"),
#             ("correct", "notext"),
#             ("irrelevant", "notext"),
#         ]

#         for v1, v2 in delta_pairs:
#             if v1 not in case_results["variants"] or v2 not in case_results["variants"]:
#                 continue

#             v1_data = case_results["variants"][v1]
#             v2_data = case_results["variants"][v2]

#             img = v1_data["image"]
#             attr1 = self._resize_attr_to_image(v1_data["image_attr_map"], img)
#             attr2 = self._resize_attr_to_image(v2_data["image_attr_map"], img)

#             delta_sparse = self._sparse_signed_delta(attr1, attr2, keep_ratio=0.01)

#             img_array = np.array(img)

#             fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 5))

#             im_d1 = axes_d[0].imshow(
#                 delta_sparse,
#                 cmap="seismic",
#                 vmin=-1.0,
#                 vmax=1.0,
#                 interpolation="nearest",
#             )
#             axes_d[0].set_title(f"{v1} - {v2} (Heatmap)", fontsize=10)
#             axes_d[0].axis("off")
#             plt.colorbar(im_d1, ax=axes_d[0], fraction=0.046)

#             axes_d[1].imshow(img_array)
#             im_d2 = axes_d[1].imshow(
#                 delta_sparse,
#                 alpha=0.7,
#                 cmap="seismic",
#                 vmin=-1.0,
#                 vmax=1.0,
#                 interpolation="nearest",
#             )
#             axes_d[1].set_title(f"{v1} - {v2} (Overlay)", fontsize=10)
#             axes_d[1].axis("off")
#             plt.colorbar(im_d2, ax=axes_d[1], fraction=0.046)

#             plt.suptitle(
#                 f"Δ Occlusion ({v1} - {v2}) - {case_results['image_id']}",
#                 fontsize=14,
#                 y=0.995,
#             )

#             delta_path = output_path.with_name(
#                 f"{output_path.stem}_delta_{v1}_vs_{v2}{output_path.suffix}"
#             )
#             plt.tight_layout()
#             plt.savefig(delta_path, dpi=300)
#             plt.close()

#     # ------------------------------------------------------------------ #
#     # PATTERN-LEVEL STATS
#     # ------------------------------------------------------------------ #
#     def _compute_pattern_statistics(self, results: List[Dict], pattern: str) -> Dict:
#         """
#         Simple statistics: per-variant average total image importance.
#         """
#         stats = {
#             "pattern": pattern,
#             "num_cases": len(results),
#             "variants": {},
#         }

#         all_variants = set()
#         for r in results:
#             all_variants.update(r["variants"].keys())

#         for variant in all_variants:
#             vals = []
#             for r in results:
#                 if variant in r["variants"]:
#                     vals.append(r["variants"][variant]["total_image_importance"])
#             if not vals:
#                 continue

#             stats["variants"][variant] = {
#                 "num_examples": len(vals),
#                 "avg_image_importance": float(np.mean(vals)),
#                 "std_image_importance": float(np.std(vals)),
#             }

#         return stats

#     def analyze_pattern_group(self, pattern: str, cases: List[Dict],
#                               image_folder: str, output_dir: Path) -> Dict:
#         """
#         For all cases in a pattern, compute occlusion map for each variant
#         and generate per-case visualizations + basic stats.
#         """
#         print(f"\n{'=' * 60}")
#         print(f"Analyzing pattern: {pattern}")
#         print(f"Number of cases: {len(cases)}")
#         print(f"{'=' * 60}")

#         pattern_dir = output_dir / pattern
#         pattern_dir.mkdir(parents=True, exist_ok=True)

#         spatial_viz_dir = pattern_dir / "spatial_visualizations"
#         spatial_viz_dir.mkdir(exist_ok=True)

#         all_results = []

#         for case in tqdm(cases, desc=f"Processing {pattern}"):
#             case_results = self.compute_image_attributions_for_case(case, image_folder)
#             all_results.append(case_results)

#             comparison_path = spatial_viz_dir / f"{case_results['image_id']}_comparison.png"
#             self.visualize_case_comparison(case_results, comparison_path)

#         stats = self._compute_pattern_statistics(all_results, pattern)

#         with open(pattern_dir / "statistics.json", "w") as f:
#             json.dump(stats, f, indent=2)

#         print(f"\nStatistics for {pattern}:")
#         print(f"  Total cases: {stats['num_cases']}")
#         for variant, v_stats in stats["variants"].items():
#             print(
#                 f"  {variant}: n={v_stats['num_examples']}, "
#                 f"avg Occlusion={v_stats['avg_image_importance']:.4f} ± {v_stats['std_image_importance']:.4f}"
#             )

#         return stats

#     # ------------------------------------------------------------------ #
#     # TOP-LEVEL DRIVER
#     # ------------------------------------------------------------------ #
#     def run_analysis(self, results_dir: str, image_folder: str, output_dir: str):
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         all_results = self.load_results(results_dir)
#         differing_cases = self.identify_differing_cases(all_results)

#         patterns = defaultdict(list)
#         for case in differing_cases:
#             patterns[case["pattern"]].append(case)

#         case_summary = {
#             pattern: [c["image_id"] for c in cases]
#             for pattern, cases in patterns.items()
#         }
#         with open(output_dir / "case_categorization.json", "w") as f:
#             json.dump(case_summary, f, indent=2)

#         priority_patterns = [
#             "fooled_by_incorrect",
#             "fooled_by_irrelevant",
#             "helped_by_correct",
#             "resisted_incorrect",
#         ]

#         all_stats = {}

#         for pattern in priority_patterns:
#             if pattern == "fooled_by_incorrect" or pattern == "helped_by_correct":
#                 continue
#             if pattern in patterns and patterns[pattern]:
#                 print(f"\nProcessing pattern: {pattern} ({len(patterns[pattern])} cases)")
#                 stats = self.analyze_pattern_group(pattern, patterns[pattern], image_folder, output_dir)
#                 all_stats[pattern] = stats

#         for pattern, cases in patterns.items():
#             if pattern not in priority_patterns and cases:
#                 print(f"\nProcessing pattern: {pattern} ({len(cases)} cases)")
#                 stats = self.analyze_pattern_group(pattern, cases, image_folder, output_dir)
#                 all_stats[pattern] = stats

#         summary = {}
#         for pattern, stats in all_stats.items():
#             summary[pattern] = {
#                 "num_cases": stats["num_cases"],
#                 "variants": stats["variants"],
#             }

#         with open(output_dir / "cross_pattern_summary.json", "w") as f:
#             json.dump(summary, f, indent=2)

#         print("\n" + "=" * 60)
#         print("✓ Image occlusion analysis complete!")
#         print(f"Results saved to: {output_dir}/")
#         print(f"  - Spatial visualizations: {output_dir}/[pattern]/spatial_visualizations/")
#         print(f"  - Pattern stats: {output_dir}/[pattern]/statistics.json")
#         print(f"  - Cross-pattern summary: {output_dir}/cross_pattern_summary.json")
#         print("=" * 60)


# def main():
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Image-based Occlusion Analysis for LLaVA / LLaVA-Next Variants"
#     )
#     parser.add_argument(
#         "--results_dir",
#         required=True,
#         help="Directory containing inference result JSON files",
#     )
#     parser.add_argument(
#         "--image_folder",
#         required=True,
#         help="Root folder containing image variant subfolders (notext, correct, misleading, irrelevant)",
#     )
#     parser.add_argument(
#         "--output_dir",
#         required=True,
#         help="Output directory for attribution analysis",
#     )
#     parser.add_argument(
#         "--model_type",
#         default="llava",
#         help="VLM type (default: llava)",
#     )
#     parser.add_argument(
#         "--model_id",
#         default=None,
#         help="Specific model ID (optional)",
#     )
#     parser.add_argument(
#         "--device",
#         default="auto",
#         help="Device to use (default: auto)",
#     )

#     args = parser.parse_args()

#     analyzer = ImageVariantAnalyzer(
#         model_type=args.model_type,
#         model_id=args.model_id,
#         device=args.device,
#     )

#     analyzer.run_analysis(
#         results_dir=args.results_dir,
#         image_folder=args.image_folder,
#         output_dir=args.output_dir,
#     )


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
IMPROVED Occlusion Attribution - Fixes Corner Artifacts

Key improvements:
1. Resize images to model's target size BEFORE occlusion (fixes preprocessing artifacts)
2. Use image mean for masking instead of fixed gray value
3. Add edge exclusion zone to remove unreliable corner measurements
4. Keep the working text detection capabilities
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn.functional import log_softmax

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from infere_vlms import get_evaluator


class ImprovedOcclusionAnalyzer:
    """
    Occlusion-based attribution with corner artifact fixes.
    """

    def __init__(self, model_type: str = "llava", model_id: str = None, device: str = "auto"):
        print(f"Loading {model_type} model...")
        self.evaluator = get_evaluator(model_type, model_id, device)
        self.model = self.evaluator.model
        self.processor = self.evaluator.processor
        self.device = self.evaluator.device
        
        # Get model's vision config
        self.vision_config = self._get_vision_config()
        print(f"Vision encoder expects images of size: {self.vision_config['image_size']}")

    def _get_vision_config(self) -> Dict:
        """Extract vision encoder configuration."""
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
                vision_cfg = self.model.config.vision_config
                image_size = getattr(vision_cfg, 'image_size', 336)
            else:
                # Default for LLaVA models
                image_size = 336
            
            return {
                'image_size': image_size,
            }
        except:
            return {'image_size': 336}

    def _build_mcq_prompt_from_json(self, item: Dict) -> str:
        if "options" in item:
            options = item["options"]        # {"A": "...", "B": "...", ...}
        else:
            options = {k: item[k] for k in ["A", "B", "C", "D"] if k in item}

        instruction = (
            "Answer the following multiple-choice question by selecting the correct option."
        )

        prompt = f"{instruction}\n\n"
        prompt += f"Question: {item['question']}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"

        return prompt


    def _prepare_inputs_with_response(self, image: Image.Image, question: str, response: str) -> Dict:
        """Prepare inputs with both question and response."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        return inputs

    # def _forward_log_prob(self, image: Image.Image, question: str, response: str) -> float:
    #     """Compute log p(target_token | question, image, response)."""
    #     self.model.eval()

    #     inputs = self._prepare_inputs_with_response(image, question, response)

    #     input_ids = inputs["input_ids"]
    #     resp_ids = self.processor.tokenizer.encode(response, add_special_tokens=False)
    #     if len(resp_ids) == 0:
    #         target_id = input_ids[0, -1].item()
    #     else:
    #         target_id = resp_ids[0]

    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #         logits = outputs.logits
    #         log_probs = log_softmax(logits[:, -1, :], dim=-1)
    #         logp = log_probs[0, target_id].item()

    #     return logp

    

    def _forward_log_prob(self, image: Image.Image, question: str, response: str) -> float:
        self.model.eval()
        inputs = self._prepare_inputs_with_response(image, question, response)

        input_ids = inputs["input_ids"][0].tolist()
        resp_ids = self.processor.tokenizer.encode(response, add_special_tokens=False)

        if not resp_ids:
            # fallback: use last token in sequence (meh, but at least explicit)
            target_pos = len(input_ids) - 1
            target_id = input_ids[target_pos]
        else:
            # Find first occurrence of the response in the full input_ids
            # (this is heuristic but much closer to the intent)
            target_id = resp_ids[0]

            # naive subsequence search
            pos = None
            for i in range(len(input_ids) - len(resp_ids) + 1):
                if input_ids[i : i + len(resp_ids)] == resp_ids:
                    pos = i
                    break

            if pos is None:
                # fall back to using last token position if subsequence not found
                target_pos = len(input_ids) - 2  # -2 because logits[t] predicts token t+1
            else:
                # logits at index (pos-1) predict token at position pos
                target_pos = max(pos - 1, 0)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, T, V)
            log_probs = log_softmax(logits[0, target_pos, :], dim=-1)
            logp = log_probs[target_id].item()

        return logp


    def _preprocess_image_for_model(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to match model's expected size.
        This is the KEY FIX for corner artifacts.
        """
        target_size = self.vision_config['image_size']
        
        # Resize to target size (this is what the processor does internally)
        # Use LANCZOS for high-quality downsampling
        image_resized = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image_resized

    def _get_adaptive_mask_value(self, image: Image.Image) -> int:
        """
        Get mask value based on image statistics.
        Using mean is better than fixed gray (127).
        """
        img_array = np.array(image)
        mean_value = int(img_array.mean())
        return mean_value

    # def _image_occlusion_attributions(
    #     self,
    #     image: Image.Image,
    #     question: str,
    #     response: str,
    #     grid_size: int = 16,
    #     exclude_edge_patches: int = 1,  # NEW: exclude N patches from edges
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Compute occlusion-based importance scores with corner artifact fixes.
        
    #     Args:
    #         image: Input image
    #         question: Question about image
    #         response: Model's predicted answer
    #         grid_size: Number of patches per dimension
    #         exclude_edge_patches: Number of edge patches to exclude (default: 1)
        
    #     Returns:
    #         scores: (grid_size, grid_size) raw occlusion scores
    #         mask: (grid_size, grid_size) boolean mask (True = valid, False = excluded edge)
    #     """
    #     # FIX 1: Resize to model's target size FIRST
    #     image_preprocessed = self._preprocess_image_for_model(image)
    #     W, H = image_preprocessed.size
        
    #     # FIX 2: Use adaptive mask value
    #     mask_value = self._get_adaptive_mask_value(image_preprocessed)
        
    #     patch_w = W // grid_size
    #     patch_h = H // grid_size

    #     base_logp = self._forward_log_prob(image_preprocessed, question, response)

    #     scores = np.zeros((grid_size, grid_size), dtype=np.float32)
    #     valid_mask = np.ones((grid_size, grid_size), dtype=bool)
        
    #     # Mark edge patches as invalid
    #     if exclude_edge_patches > 0:
    #         valid_mask[:exclude_edge_patches, :] = False  # Top
    #         valid_mask[-exclude_edge_patches:, :] = False  # Bottom
    #         valid_mask[:, :exclude_edge_patches] = False  # Left
    #         valid_mask[:, -exclude_edge_patches:] = False  # Right

    #     for gy in tqdm(range(grid_size), desc="Occlusion", leave=False):
    #         for gx in range(grid_size):
    #             y0 = gy * patch_h
    #             x0 = gx * patch_w
    #             y1 = (gy + 1) * patch_h if gy < grid_size - 1 else H
    #             x1 = (gx + 1) * patch_w if gx < grid_size - 1 else W

    #             img_masked = image_preprocessed.copy()
    #             arr = np.array(img_masked)
    #             arr[y0:y1, x0:x1, :] = mask_value
    #             img_masked = Image.fromarray(arr)

    #             masked_logp = self._forward_log_prob(img_masked, question, response)
    #             scores[gy, gx] = base_logp - masked_logp
                
    #         # Clear cache periodically
    #         if (gy + 1) % 4 == 0:
    #             torch.cuda.empty_cache()

    #     return scores, valid_mask

    def _image_occlusion_attributions(
            self,
            image: Image.Image,
            question: str,
            response: str,
            grid_size: int = 16,
            exclude_edge_patches: int = 1,  # exclude N patches from edges
        ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Occlusion in ORIGINAL image space.

        Returns:
            scores: (grid_size, grid_size) log-prob drops: base_logp - masked_logp
            valid_mask: (grid_size, grid_size) boolean (False for excluded edge patches)
            base_logp: scalar log p(answer) for the unmasked image
        """
        W, H = image.size

        # Adaptive mask from original image
        mask_value = self._get_adaptive_mask_value(image)

        patch_w = W // grid_size
        patch_h = H // grid_size

        # Baseline log-prob with original image
        base_logp = self._forward_log_prob(image, question, response)

        scores = np.zeros((grid_size, grid_size), dtype=np.float32)
        valid_mask = np.ones((grid_size, grid_size), dtype=bool)

        # Mark edges as invalid if requested
        if exclude_edge_patches > 0:
            valid_mask[:exclude_edge_patches, :] = False   # top
            valid_mask[-exclude_edge_patches:, :] = False  # bottom
            valid_mask[:, :exclude_edge_patches] = False   # left
            valid_mask[:, -exclude_edge_patches:] = False  # right

        for gy in tqdm(range(grid_size), desc="Occlusion", leave=False):
            for gx in range(grid_size):
                y0 = gy * patch_h
                x0 = gx * patch_w
                y1 = (gy + 1) * patch_h if gy < grid_size - 1 else H
                x1 = (gx + 1) * patch_w if gx < grid_size - 1 else W

                img_masked = image.copy()
                arr = np.array(img_masked)
                arr[y0:y1, x0:x1, :] = mask_value
                img_masked = Image.fromarray(arr)

                masked_logp = self._forward_log_prob(img_masked, question, response)
                scores[gy, gx] = base_logp - masked_logp

            # Safe CUDA cache clear
            if (gy + 1) % 4 == 0:
                if torch.cuda.is_available() and getattr(self.device, "type", "") == "cuda":
                    torch.cuda.empty_cache()

        return scores, valid_mask, float(base_logp)



    def _example_id_from_filename(self, fname: str) -> str:
        """Map variant filenames to shared example id."""
        fname = Path(fname).name
        if fname.startswith("image_"):
            core = fname[len("image_"):]
            ex_id = core.split(".")[0]
        else:
            ex_id = fname.split("_")[0]
        return ex_id

    def load_results(self, results_dir: str) -> Dict[str, List[Dict]]:
        """Load inference results from all variants."""
        results_dir = Path(results_dir)
        
        variant_files = {
            "notext": "llava-next_notext_results.json",
            "correct": "llava-next_correct_results.json",
            "incorrect": "llava-next_misleading_results.json",
            "irrelevant": "llava-next_irrelevant_results.json",
        }
        
        all_results = {}
        
        for variant, filename in variant_files.items():
            filepath = results_dir / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    all_results[variant] = json.load(f)
                print(f"Loaded {len(all_results[variant])} results from {filename}")
        
        return all_results

    def identify_differing_cases(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
        """Identify cases where predictions differ across variants."""
        print("\n" + "=" * 60)
        print("IDENTIFYING CASES WITH VARIANT DIFFERENCES")
        print("=" * 60)
        
        grouped = defaultdict(dict)
        
        for variant, results in all_results.items():
            for item in results:
                ex_id = self._example_id_from_filename(item["image"])
                grouped[ex_id][variant] = item
        
        differing_cases = []
        
        for ex_id, variants in grouped.items():
            predictions = {
                variant: data["predicted_answer"]
                for variant, data in variants.items()
            }
            unique_predictions = set(predictions.values())
            
            if len(unique_predictions) > 1:
                correctness = {
                    variant: data["is_correct"]
                    for variant, data in variants.items()
                }
                case = {
                    "image_id": ex_id,
                    "image_files": {v: d["image"] for v, d in variants.items()},
                    "predictions": predictions,
                    "correctness": correctness,
                    "variants": variants,
                    "pattern": self._categorize_pattern(predictions, correctness),
                }
                differing_cases.append(case)
        
        print(f"\nFound {len(differing_cases)} cases with variant differences")
        print(f"Total cases: {len(grouped)}")
        
        pattern_counts = defaultdict(int)
        for case in differing_cases:
            pattern_counts[case["pattern"]] += 1
        
        print("\nPattern Distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern:40s}: {count:4d} cases")
        
        return differing_cases

    def _categorize_pattern(self, predictions: Dict[str, str], correctness: Dict[str, bool]) -> str:
        """Categorize behavioral pattern."""
        nt = correctness.get("notext")
        cor = correctness.get("correct")
        inc = correctness.get("incorrect")
        irr = correctness.get("irrelevant")
        
        if nt is True and inc is False:
            return "fooled_by_incorrect"
        if nt is True and irr is False:
            return "fooled_by_irrelevant"
        if nt is False and cor is True:
            return "helped_by_correct"
        if inc is False and cor is True:
            return "resisted_incorrect"
        
        vals = [v for v in correctness.values() if isinstance(v, bool)]
        if vals and all(vals):
            return "all_correct"
        if vals and not any(vals):
            return "all_incorrect"
        
        return "mixed_behavior"

    def compute_occlusion_for_case(
            self,
            case: Dict,
            image_folder: str,
            exclude_edges: bool = True
        ) -> Dict:
        """Compute occlusion attributions for all variants in a case."""
        image_folder = Path(image_folder)

        results = {
            "image_id": case["image_id"],
            "pattern": case["pattern"],
            "variants": {},
        }

        variant_to_subdir = {
            "notext": "notext",
            "correct": "correct",
            "incorrect": "misleading",
            "irrelevant": "irrelevant",
        }

        for variant_name, variant_data in case["variants"].items():
            if variant_name not in variant_to_subdir:
                continue

            image_path = image_folder / variant_to_subdir[variant_name] / variant_data["image"]
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            # question = variant_data.get("question", "")
            question = self._build_mcq_prompt_from_json(variant_data)

            response = variant_data["predicted_answer"]

            if not question:
                print(f"Warning: empty question for {case['image_id']} ({variant_name})")
                continue

            # NEW: which token are we attributing to?
            token_id, token_str = self._get_mcq_token_info(response)
            choices = variant_data.get("choices", None)

            # Compute occlusion in ORIGINAL image space
            scores, valid_mask, base_logp = self._image_occlusion_attributions(
                image=image,
                question=question,
                response=response,
                grid_size=16,
                exclude_edge_patches=1 if exclude_edges else 0,
            )

            # Only sum valid (non-edge) patches
            total_importance = float(np.sum(np.abs(scores[valid_mask])))

            results["variants"][variant_name] = {
                "prediction": response,
                "is_correct": variant_data["is_correct"],
                "scores": scores,
                "valid_mask": valid_mask,
                "total_importance": total_importance,
                "base_logp": float(base_logp),
                "question": question,
                "image": image,
                "image_path": str(image_path),


                # NEW FIELDS
                "token_id": token_id,
                "token_str": token_str,
                "choices": choices,
            }

        return results


    def _resize_attr_to_image(self, attr_map: np.ndarray, image: Image.Image) -> np.ndarray:
        """Resize attribution map to original image size."""
        if isinstance(attr_map, torch.Tensor):
            attr_map = attr_map.detach().cpu().numpy()
        
        if attr_map.ndim != 2:
            raise ValueError(f"Expected 2D attr_map, got {attr_map.shape}")
        
        img_array = np.array(image)
        H_img, W_img = img_array.shape[:2]
        H_attr, W_attr = attr_map.shape
        
        scale_h = H_img / H_attr
        scale_w = W_img / W_attr
        
        attr_resized = zoom(attr_map, (scale_h, scale_w), order=1)
        return attr_resized

    def _get_mcq_token_info(self, response: str) -> Tuple[int, str]:
        """
        For MCQ, we attribute to the FIRST token of the model's response.
        Returns (token_id, token_str).
        """
        tokenizer = self.processor.tokenizer
        resp_ids = tokenizer.encode(response, add_special_tokens=False)

        if not resp_ids:
            # Fallback if response is empty for some reason
            tok_id = getattr(tokenizer, "eos_token_id", None)
            if tok_id is None:
                tok_id = getattr(tokenizer, "pad_token_id", 0)
        else:
            tok_id = resp_ids[0]

        tok_str = tokenizer.decode([tok_id])
        return int(tok_id), tok_str

    def _format_choices_for_title(self, choices) -> str:
        """
        Format MCQ choices for the figure title.

        Supports:
          - list[str]  -> ["cat", "dog", ...]
          - list[dict] -> [{"label": "A", "text": "cat"}, ...]
        """
        if not choices:
            return ""

        # Case 1: simple list of strings
        if isinstance(choices[0], str):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            parts = []
            for i, ch in enumerate(choices):
                if i >= len(letters):
                    break
                text = ch.strip()
                if len(text) > 40:
                    text = text[:37] + "..."
                parts.append(f"{letters[i]}: {text}")
            return " | ".join(parts)

        # Case 2: list of dicts with label/text
        if isinstance(choices[0], dict):
            parts = []
            for c in choices:
                label = c.get("label") or c.get("option") or ""
                text = c.get("text") or c.get("answer") or ""
                if len(text) > 40:
                    text = text[:37] + "..."
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            return " | ".join(parts)

        # Fallback: just stringify
        return " | ".join(str(c) for c in choices)


    def _topk_mask(self, attr: np.ndarray, keep_ratio: float = 1.0, valid_mask: np.ndarray = None) -> np.ndarray:
        """Keep only top-k% of attribution values, only considering valid regions."""
        if valid_mask is not None:
            # Only consider valid regions for thresholding
            valid_values = attr[valid_mask]
            if valid_values.size == 0:
                return np.zeros_like(attr)
            
            k = max(int(valid_values.size * keep_ratio), 1)
            thresh = np.partition(valid_values, -k)[-k]
        else:
            flat = attr.flatten()
            if flat.size == 0:
                return attr
            k = max(int(flat.size * keep_ratio), 1)
            thresh = np.partition(flat, -k)[-k]
        
        mask = attr >= thresh
        out = np.zeros_like(attr, dtype=np.float32)
        if np.any(mask):
            out[mask] = attr[mask] / attr[mask].max()
        
        return out

    # def visualize_case_comparison(self, case_results: Dict, output_path: Path):
    #     """Create visualization comparing occlusion across variants."""
    #     variants = list(case_results["variants"].keys())
    #     num_variants = len(variants)
    #     if num_variants == 0:
    #         return
        
    #     fig, axes = plt.subplots(num_variants, 3, figsize=(18, 6 * num_variants))
    #     if num_variants == 1:
    #         axes = axes.reshape(1, -1)
        
    #     for idx, variant in enumerate(variants):
    #         v_data = case_results["variants"][variant]
    #         image = v_data["image"]
    #         img_array = np.array(image)
            
    #         # Get scores and mask invalid edges
    #         scores = v_data["scores"]
    #         valid_mask = v_data["valid_mask"]
            
    #         # Use absolute values for visualization
    #         scores_abs = np.abs(scores)
            
    #         # Mask out invalid regions (set to 0)
    #         scores_abs[~valid_mask] = 0
            
    #         # Resize to image dimensions
    #         attr_resized = self._resize_attr_to_image(scores_abs, image)
            
    #         # Sparse version (top 2%)
    #         attr_sparse = self._topk_mask(attr_resized, keep_ratio=0.02)
            
    #         # Column 1: Original image
    #         axes[idx, 0].imshow(img_array)
    #         axes[idx, 0].set_title(f"{variant} - Original", fontsize=10)
    #         axes[idx, 0].axis("off")
            
    #         # Column 2: Heatmap (edges excluded)
    #         im1 = axes[idx, 1].imshow(
    #             attr_sparse,
    #             cmap="hot",
    #             vmin=0.0,
    #             vmax=1.0,
    #             interpolation="bilinear",
    #         )
    #         axes[idx, 1].set_title(
    #             f"{variant} - Occlusion Map\nPred: {v_data['prediction']} "
    #             f"({'✓' if v_data['is_correct'] else '✗'})",
    #             fontsize=10,
    #         )
    #         axes[idx, 1].axis("off")
    #         plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
            
    #         # Column 3: Overlay
    #         axes[idx, 2].imshow(img_array)
    #         im2 = axes[idx, 2].imshow(
    #             attr_sparse,
    #             alpha=0.6,
    #             cmap="jet",
    #             vmin=0.0,
    #             vmax=1.0,
    #             interpolation="bilinear",
    #         )
    #         axes[idx, 2].set_title(
    #             f"{variant} - Overlay\nTotal Impact: {v_data['total_importance']:.3f}",
    #             fontsize=10,
    #         )
    #         axes[idx, 2].axis("off")
    #         plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
        
    #     plt.suptitle(
    #         f"Improved Occlusion Analysis - {case_results['image_id']}\n"
    #         f"Pattern: {case_results['pattern']}",
    #         fontsize=14,
    #         y=0.995,
    #     )
    #     plt.tight_layout()
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    def visualize_case_comparison(self, case_results: Dict, output_path: Path):
        """Create visualization comparing occlusion across variants."""
        variants = list(case_results["variants"].keys())
        num_variants = len(variants)
        if num_variants == 0:
            return

        # Representative question (same across variants for MCQ)
        all_questions = [
            v.get("question", "")
            for v in case_results["variants"].values()
            if v.get("question")
        ]
        if all_questions:
            question_str = all_questions[0]
            max_q_len = 200
            if len(question_str) > max_q_len:
                question_str = question_str[:max_q_len] + "..."
        else:
            question_str = ""

        # Representative choices (same across variants)
        all_choices = [
            v.get("choices")
            for v in case_results["variants"].values()
            if v.get("choices") is not None
        ]
        choices_str = ""
        if all_choices:
            choices_str = self._format_choices_for_title(all_choices[0])

        fig, axes = plt.subplots(num_variants, 3, figsize=(18, 6 * num_variants))
        if num_variants == 1:
            axes = axes.reshape(1, -1)

        for idx, variant in enumerate(variants):
            v_data = case_results["variants"][variant]
            image = v_data["image"]
            img_array = np.array(image)

            scores = v_data["scores"]
            valid_mask = v_data["valid_mask"]
            base_logp = v_data.get("base_logp", None)

            token_str = v_data.get("token_str", None)
            token_id = v_data.get("token_id", None)


            scores_abs = np.abs(scores)
            scores_abs[~valid_mask] = 0

            attr_resized = self._resize_attr_to_image(scores_abs, image)
            attr_sparse = self._topk_mask(attr_resized, keep_ratio=1.0)

            # Column 1: original image
            axes[idx, 0].imshow(img_array)
            axes[idx, 0].set_title(f"{variant} - Original", fontsize=10)
            axes[idx, 0].axis("off")

            # Column 2: heatmap
            title_lines = [
                f"{variant} - Occlusion Map",
                f"Pred: {v_data['prediction']} ({'✓' if v_data['is_correct'] else '✗'})",
            ]

            # NEW: token info
            if token_str is not None:
                tok_line = f"Token='{token_str}'"
                if token_id is not None:
                    tok_line += f" (id={token_id})"
                title_lines.append(tok_line)

            if base_logp is not None:
                title_lines.append(f"log p(ans) = {base_logp:.3f}")
            if base_logp is not None:
                title_lines.append(f"log p(ans) = {base_logp:.3f}")

            im1 = axes[idx, 1].imshow(
                attr_sparse,
                cmap="hot",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            axes[idx, 1].set_title("\n".join(title_lines), fontsize=9)
            axes[idx, 1].axis("off")
            plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)

            # Column 3: overlay
            axes[idx, 2].imshow(img_array)
            im2 = axes[idx, 2].imshow(
                attr_sparse,
                alpha=0.6,
                cmap="jet",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            axes[idx, 2].set_title(
                f"{variant} - Overlay\nTotal |Δ log p|: {v_data['total_importance']:.3f}",
                fontsize=10,
            )
            axes[idx, 2].axis("off")
            plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)

        # 👇 Figure-level title: pattern + question + choices
        suptitle_lines = [
            f"Improved Occlusion Analysis - {case_results['image_id']}",
            f"Pattern: {case_results['pattern']}",
        ]
        if question_str:
            suptitle_lines.append(f"Q: {question_str}")
        if choices_str:
            suptitle_lines.append(f"Choices: {choices_str}")

        plt.suptitle("\n".join(suptitle_lines), fontsize=12, y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()



    def _compute_pattern_statistics(self, results: List[Dict], pattern: str) -> Dict:
        """Compute statistics across all cases in a pattern."""
        stats = {
            "pattern": pattern,
            "num_cases": len(results),
            "variants": {},
        }
        
        all_variants = set()
        for r in results:
            all_variants.update(r["variants"].keys())
        
        for variant in all_variants:
            vals = []
            for r in results:
                if variant in r["variants"]:
                    vals.append(r["variants"][variant]["total_importance"])
            
            if not vals:
                continue
            
            stats["variants"][variant] = {
                "num_examples": len(vals),
                "avg_importance": float(np.mean(vals)),
                "std_importance": float(np.std(vals)),
                "min_importance": float(np.min(vals)),
                "max_importance": float(np.max(vals)),
            }
        
        return stats

    def analyze_pattern_group(
        self,
        pattern: str,
        cases: List[Dict],
        image_folder: str,
        output_dir: Path,
        exclude_edges: bool = True,
        max_cases: int = None,   # NEW
    ) -> Dict:
        """Analyze all cases in a pattern group."""
        print(f"\n{'=' * 60}")
        print(f"Analyzing pattern: {pattern}")
        print(f"Number of cases: {len(cases)}")
        print(f"Edge exclusion: {'ON' if exclude_edges else 'OFF'}")
        print(f"{'=' * 60}")
        
        pattern_dir = output_dir / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        viz_dir = pattern_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        for idx, case in enumerate(tqdm(cases, desc=f"Processing {pattern}")):
            if max_cases is not None and idx >= max_cases:
                break
            case_results = self.compute_occlusion_for_case(
                case, image_folder, exclude_edges
            )
            all_results.append(case_results)
            
            viz_path = viz_dir / f"{case_results['image_id']}_occlusion.png"
            self.visualize_case_comparison(case_results, viz_path)
        
        stats = self._compute_pattern_statistics(all_results, pattern)
        
        with open(pattern_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nStatistics for {pattern}:")
        print(f"  Total cases: {stats['num_cases']}")
        for variant, v_stats in stats["variants"].items():
            print(
                f"  {variant}: n={v_stats['num_examples']}, "
                f"avg={v_stats['avg_importance']:.4f} ± {v_stats['std_importance']:.4f}"
            )
        
        return stats

    def run_analysis(
        self,
        results_dir: str,
        image_folder: str,
        output_dir: str,
        exclude_edges: bool = True
    ):
        """Run complete improved occlusion analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = self.load_results(results_dir)
        differing_cases = self.identify_differing_cases(all_results)
        
        patterns = defaultdict(list)
        for case in differing_cases:
            patterns[case["pattern"]].append(case)
        
        case_summary = {
            pattern: [c["image_id"] for c in cases]
            for pattern, cases in patterns.items()
        }
        with open(output_dir / "case_categorization.json", "w") as f:
            json.dump(case_summary, f, indent=2)
        
        priority_patterns = [
            "helped_by_correct",
            "fooled_by_irrelevant",
            "fooled_by_incorrect",
            "resisted_incorrect",
        ]
        
        all_stats = {}
        
        for pattern in priority_patterns:
            if pattern == "fooled_by_irrelevant":
                continue
            if pattern in patterns and patterns[pattern]:
                print(f"\nProcessing pattern: {pattern} ({len(patterns[pattern])} cases)")
                stats = self.analyze_pattern_group(
                    pattern, patterns[pattern], image_folder, output_dir, exclude_edges
                )
                all_stats[pattern] = stats
        
        for pattern, cases in patterns.items():
            if pattern == "fooled_by_irrelevant":
                continue
            if pattern not in priority_patterns and cases:
                print(f"\nProcessing pattern: {pattern} ({len(cases)} cases)")
                stats = self.analyze_pattern_group(
                    pattern, cases, image_folder, output_dir, exclude_edges
                )
                all_stats[pattern] = stats
        
        summary = {}
        for pattern, stats in all_stats.items():
            summary[pattern] = {
                "num_cases": stats["num_cases"],
                "variants": stats["variants"],
            }
        
        with open(output_dir / "cross_pattern_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✓ Improved occlusion analysis complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Visualizations: {output_dir}/[pattern]/visualizations/")
        print(f"  - Pattern stats: {output_dir}/[pattern]/statistics.json")
        print(f"  - Cross-pattern summary: {output_dir}/cross_pattern_summary.json")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Improved Occlusion Analysis (fixes corner artifacts)"
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory containing inference result JSON files",
    )
    parser.add_argument(
        "--image_folder",
        default="./filtered_data",
        help="Root folder containing image variant subfolders",
    )
    parser.add_argument(
        "--output_dir",
        default="./integrated_gradients/ablation_test",
        help="Output directory for attribution analysis",
    )
    parser.add_argument(
        "--model_type",
        default="llava-next",
        help="VLM type (default: llava)",
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help="Specific model ID (optional)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--no_exclude_edges",
        action="store_true",
        help="Disable edge exclusion (keep corner measurements)",
    )
    
    args = parser.parse_args()
    
    analyzer = ImprovedOcclusionAnalyzer(
        model_type=args.model_type,
        model_id=args.model_id,
        device=args.device,
    )
    
    analyzer.run_analysis(
        results_dir=args.results_dir,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        exclude_edges=not args.no_exclude_edges,
    )


if __name__ == "__main__":
    main()
