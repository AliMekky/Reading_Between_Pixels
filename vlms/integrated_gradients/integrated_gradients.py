import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn.functional import log_softmax
from captum.attr import LayerIntegratedGradients

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Your VLM evaluator wrapper (must expose .model, .processor, .device)
from reading_between_pixels.Reading_Between_Pixels.vlms.inference.infere_vlms import get_evaluator


class ImageVariantAnalyzer:
    """
    Phase 1: Integrated Gradients over IMAGE TOKENS for LLaVA / LLaVA-Next-style models.

    Key design:
      - We run LayerIntegratedGradients w.r.t. the vision->LM projector layer.
      - The IG *input* is still pixel_values (image tensor).
      - The IG *coordinates* are projector outputs (image tokens).
      - We aggregate per-token attributions into a patch grid per crop.
      - For multi-crop (LLaVA-Next), we:
          * reconstruct each crop image from pixel_values
          * visualize IG on EACH CROP IMAGE separately (no fake global mapping).
    """

    def __init__(self, model_type: str = "llava", model_id: str = None, device: str = "auto"):
        print(f"Loading {model_type} model...")
        self.evaluator = get_evaluator(model_type, model_id, device)
        self.model = self.evaluator.model
        self.processor = self.evaluator.processor
        self.device = self.evaluator.device

    # ------------------------------------------------------------------ #
    # BASIC UTILITIES
    # ------------------------------------------------------------------ #
    def _get_device(self) -> torch.device:
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def _topk_mask(self, attr: np.ndarray, keep_ratio: float = 0.02) -> np.ndarray:
        """
        Take absolute values of a 2D map and keep only the top (keep_ratio)
        fraction of pixels, normalized to [0,1]. All others are 0.
        Assumes attr is non-negative (e.g., already |IG| or magnitude).
        """
        flat = attr.flatten()
        if flat.size == 0:
            return attr

        k = max(int(flat.size * keep_ratio), 1)
        thresh = np.partition(flat, -k)[-k]

        mask = attr >= thresh
        out = np.zeros_like(attr, dtype=np.float32)
        if np.any(mask):
            out[mask] = attr[mask] / (attr[mask].max() + 1e-8)
        return out

    def _example_id_from_filename(self, fname: str) -> str:
        """
        Map variant-specific filenames to a shared example id.

        Examples:
          '0_3.jpg'      -> '0'
          '1_0.jpg'      -> '1'
          'image_0.jpg'  -> '0'
          'image_10.jpg' -> '10'
        """
        fname = Path(fname).name  # strip any path

        if fname.startswith("image_"):
            core = fname[len("image_"):]
            ex_id = core.split(".")[0]
        else:
            ex_id = fname.split("_")[0]

        return ex_id

    # ------------------------------------------------------------------ #
    # INPUT PREPARATION (CHAT TEMPLATE)
    # ------------------------------------------------------------------ #
    def _prepare_inputs_inference(self, image: Image.Image, prompt: str) -> Dict:
        """
        Original inference-style input builder (user-only, add_generation_prompt=True).
        Kept for completeness; not used directly in IG.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        return inputs

    def _prepare_inputs_with_response(self, image: Image.Image, question: str, response: str) -> Dict:
        """
        Prepare inputs including both:
          - user question (with image)
          - assistant response (text)

        Used for IG:
          - add_generation_prompt=False because we already include the answer.
          - apply_chat_template handles inserting the image token(s),
            so pixel_values and input_ids stay in sync.
        """
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
        return inputs

    # ------------------------------------------------------------------ #
    # LOADING RESULTS & PATTERN CATEGORIZATION
    # ------------------------------------------------------------------ #
    def load_results(self, results_dir: str) -> Dict[str, List[Dict]]:
        """
        Load inference results from all variants.
        """
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
            else:
                print(f"Warning: {filename} not found")

        return all_results

    def _categorize_pattern(self, predictions: Dict[str, str],
                            correctness: Dict[str, bool]) -> str:
        """
        Categorize behavioral pattern based on correctness of each variant.

        Variants:
          - 'notext'
          - 'correct'
          - 'incorrect'
          - 'irrelevant'
        """
        nt  = correctness.get("notext")
        cor = correctness.get("correct")
        inc = correctness.get("incorrect")
        irr = correctness.get("irrelevant")

        # Fooled by incorrect text:
        if nt is True and inc is False:
            return "fooled_by_incorrect"

        # Fooled by irrelevant text:
        if nt is True and irr is False:
            return "fooled_by_irrelevant"

        # Helped by correct text:
        if nt is False and cor is True:
            return "helped_by_correct"

        # Resisted incorrect text:
        if inc is False and cor is True:
            return "resisted_incorrect"

        vals = [v for v in correctness.values() if isinstance(v, bool)]
        if vals and all(vals):
            return "all_correct"

        if vals and not any(vals):
            return "all_incorrect"

        return "mixed_behavior"

    def identify_differing_cases(self, all_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Group results by logical example id and keep only cases where variants disagree.
        """
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
        print(f"Proportion: {len(differing_cases) / max(len(grouped), 1) * 100:.1f}%")

        pattern_counts = defaultdict(int)
        for case in differing_cases:
            pattern_counts[case["pattern"]] += 1

        print("\nPattern Distribution:")
        print("-" * 60)
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern:40s}: {count:4d} cases")

        return differing_cases

    # ------------------------------------------------------------------ #
    # PROJECTOR & VISION NORM HELPERS
    # ------------------------------------------------------------------ #
    def _get_projector_module(self):
        """
        Locate the vision→LM projector module on common LLaVA(-Next) variants.
        Returns an nn.Module to use as the attribution layer.
        """
        m = self.model

        candidate_paths = [
            "mm_projector",
            "multi_modal_projector",
            "model.mm_projector",
            "model.multi_modal_projector",
            "visual_projector",
            "model.visual_projector",
        ]

        for path in candidate_paths:
            obj = m
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                return obj

        raise AttributeError(
            "Could not find a vision→LM projector module on the model. "
            "Tried: mm_projector, multi_modal_projector, visual_projector (and nested under .model). "
            "Inspect `type(self.model)` and `dir(self.model)` to locate the correct module "
            "and extend `_get_projector_module` with the appropriate path."
        )

    def _get_vision_norm_stats(self):
        """
        Read image normalization mean/std from the processor if possible.
        Fallback to CLIP defaults.
        """
        mean = None
        std = None

        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            ip = getattr(self.processor, "feature_extractor", None)

        if ip is not None and hasattr(ip, "image_mean") and hasattr(ip, "image_std"):
            mean = torch.tensor(ip.image_mean).view(1, -1, 1, 1)
            std = torch.tensor(ip.image_std).view(1, -1, 1, 1)
        else:
            # CLIP / LLaVA defaults
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

        return mean.to(self.device), std.to(self.device)

    def _pixel_values_to_crops(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Convert pixel_values to a numpy array of crops suitable for visualization.

        Args:
            pixel_values:
              - LLaVA:      (1, 3, H, W)
              - LLaVA-Next: (1, N_crops, 3, H, W)

        Returns:
            crops: np.ndarray
              - shape (N_crops, H, W, 3), values in [0,1]
              - for single-crop models, N_crops == 1
        """
        mean, std = self._get_vision_norm_stats()

        pv = pixel_values.detach()

        if pv.ndim == 4:
            # (1, 3, H, W) → single crop
            pv = pv  # (1, 3, H, W)
            pv = pv  # (1, 3, H, W)
            N_crops = 1
        elif pv.ndim == 5:
            # (1, N_crops, 3, H, W)
            pv = pv.squeeze(0)  # (N_crops, 3, H, W)
            N_crops = pv.shape[0]
        else:
            raise ValueError(f"Unexpected pixel_values shape: {tuple(pv.shape)}")

        if mean.shape[1] != pv.shape[1]:
            raise ValueError(
                f"Channel mismatch between pixel_values ({pv.shape[1]}) "
                f"and mean/std ({mean.shape[1]})."
            )

        # Expand mean/std to match N_crops if needed
        mean_exp = mean
        std_exp = std
        if mean.shape[0] == 1 and pv.shape[0] > 1:
            mean_exp = mean.expand(pv.shape[0], -1, 1, 1)
            std_exp = std.expand(pv.shape[0], -1, 1, 1)

        crops = pv * std_exp + mean_exp          # (N_crops, 3, H, W)
        crops = crops.clamp(0.0, 1.0).cpu().numpy()
        crops = np.transpose(crops, (0, 2, 3, 1))  # (N_crops, H, W, 3)

        return crops.astype(np.float32)

    # ------------------------------------------------------------------ #
    # LAYER IG OVER IMAGE TOKENS
    # ------------------------------------------------------------------ #
    def _image_token_ig_attributions(
        self,
        image: Image.Image,
        question: str,
        response: str,
        steps: int = 32,
    ):
        """
        Run Layer Integrated Gradients w.r.t. the vision projector outputs.

        - The IG *input* is pixel_values (image tensor).
        - The hooked layer is the projector (mm_projector / multi_modal_projector / ...).
        - We get attributions in projector output space, which we interpret as
          per-image-token attributions and reshape into a patch grid.

        Returns:
          patch_attr: np.ndarray of shape (N_crops, H_patch, W_patch)
          crops_np:   np.ndarray of shape (N_crops, H_crop, W_crop, 3)
        """
        self.model.eval()

        # 1) Build inputs with question + fixed response
        inputs = self._prepare_inputs_with_response(image, question, response)

        input_ids = inputs["input_ids"]            # (1, T)
        attention_mask = inputs["attention_mask"]  # (1, T)
        pixel_values = inputs["pixel_values"]      # (1, C, H, W) or (1, N_crops, 3, H, W)

        extra_inputs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "attention_mask", "pixel_values"]
        }

        # 2) Rebuild crop images for visualization
        crops_np = self._pixel_values_to_crops(pixel_values)  # (N_crops, H_crop, W_crop, 3)

        # 3) Choose target token: last token in the sequence
        target_token_id = input_ids[0, -1].item()

        def forward_func(pixel_values_fwd, input_ids_fwd, attention_mask_fwd, extra_dict):
            outputs = self.model(
                input_ids=input_ids_fwd,
                attention_mask=attention_mask_fwd,
                pixel_values=pixel_values_fwd,
                **extra_dict,
            )
            logits = outputs.logits  # (B, T, V)
            log_probs = log_softmax(logits[:, -1, :], dim=-1)
            return log_probs[:, target_token_id]

        baseline = torch.zeros_like(pixel_values, device=self.device)

        projector_module = self._get_projector_module()

        lig = LayerIntegratedGradients(forward_func, projector_module)
        attributions = lig.attribute(
            pixel_values,
            baselines=baseline,
            additional_forward_args=(input_ids, attention_mask, extra_inputs),
            n_steps=steps,
            internal_batch_size=1,
        )

        if isinstance(attributions, torch.Tensor):
            attr = attributions.detach().cpu()
        else:
            attr = torch.as_tensor(attributions)

        # ---------------------------------------------------
        # Shape handling
        # ---------------------------------------------------
        # Common case seen in LLaVA-Next:
        #   attr: (N_crops, T_seq, D_proj)
        #
        # Other possible shapes:
        #   attr: (1, N_crops, T_seq, D_proj)
        #   attr: (T_seq, D_proj)   (single crop, no extra dims)
        # We'll normalize to (N_crops, T_seq, D_proj).
        # ---------------------------------------------------
        if attr.ndim == 3:
            # (N_crops, T_seq, D_proj) or (T_seq, D_proj)
            if attr.shape[0] > 1:
                N_crops, T_seq, D_proj = attr.shape
            else:
                # (1, T_seq, D_proj) → single crop
                N_crops = 1
                T_seq, D_proj = attr.shape[0], attr.shape[1]
                attr = attr.unsqueeze(0)  # (1, T_seq, D_proj)
        elif attr.ndim == 4:
            # (B, N_crops, T_seq, D_proj) with B=1
            B, N_crops, T_seq, D_proj = attr.shape
            if B != 1:
                raise ValueError(f"Unexpected batch size in projector attributions: {B}")
            attr = attr.squeeze(0)  # (N_crops, T_seq, D_proj)
        elif attr.ndim == 2:
            # (T_seq, D_proj) → single crop, no batch dim
            T_seq, D_proj = attr.shape
            N_crops = 1
            attr = attr.unsqueeze(0)  # (1, T_seq, D_proj)
        else:
            raise ValueError(
                f"Expected 2D/3D/4D attributions from projector, got {tuple(attr.shape)}"
            )

        # 1) aggregate over feature dimension → scalar per token
        token_scores = attr.sum(dim=-1)  # (N_crops, T_seq)

        # 2) map tokens to a square patch grid
        grid = int(round(T_seq ** 0.5))
        if grid * grid != T_seq:
            raise ValueError(
                f"T_seq={T_seq} is not a perfect square; cannot map tokens to a "
                f"square patch grid. Got attr shape {tuple(attr.shape)}."
            )

        patch_attr = token_scores.reshape(N_crops, grid, grid)  # (N_crops, H_patch, W_patch)

        return patch_attr.numpy(), crops_np

    # ------------------------------------------------------------------ #
    # PER-CASE: COMPUTE IMAGE ATTRIBUTIONS FOR VARIANTS
    # ------------------------------------------------------------------ #
    def compute_image_attributions_for_case(self, case: Dict, image_folder: str) -> Dict:
        """
        Compute image-token IG attributions for all variants in a single conflict case.
        Uses stored predictions & questions from the JSON (no re-inference).
        """
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

            question = variant_data.get("question", "")
            response = variant_data["predicted_answer"]

            if not question:
                print(f"Warning: empty question for {case['image_id']} ({variant_name})")
                continue

            patch_attr, crops_np = self._image_token_ig_attributions(
                image=image,
                question=question,
                response=response,
                steps=32,
            )

            total_image_importance = float(np.sum(np.abs(patch_attr)))

            results["variants"][variant_name] = {
                "prediction": response,
                "is_correct": variant_data["is_correct"],
                "image_attr_map": patch_attr,   # (N_crops, H_patch, W_patch)
                "image_crops": crops_np,        # (N_crops, H_crop, W_crop, 3)
                "total_image_importance": total_image_importance,
                "image": image,                 # original full image (PIL)
                "image_path": str(image_path),
            }

        return results

    # ------------------------------------------------------------------ #
    # VISUALIZATION (PER-CROP, PER-VARIANT)
    # ------------------------------------------------------------------ #
    def visualize_case_comparison(self, case_results: Dict, output_path: Path):
        """
        Visualization for a single case:
          - For each variant (row):
              * Column 0: original full image (for context)
              * Columns 1..C: each crop image with overlayed IG map
        """
        variants = list(case_results["variants"].keys())
        num_variants = len(variants)
        if num_variants == 0:
            return

        # determine max number of crops across variants
        max_crops = 0
        for v in variants:
            v_data = case_results["variants"][v]
            patch_attr = v_data["image_attr_map"]
            if patch_attr.ndim == 2:
                n_crops = 1
            else:
                n_crops = patch_attr.shape[0]
            max_crops = max(max_crops, n_crops)

        # columns: 1 (original full image) + one per crop
        n_cols = 1 + max_crops

        fig, axes = plt.subplots(num_variants, n_cols, figsize=(4 * n_cols, 4 * num_variants))
        if num_variants == 1:
            axes = axes.reshape(1, -1)

        for row_idx, variant in enumerate(variants):
            v_data = case_results["variants"][variant]
            full_image = v_data["image"]
            full_img_array = np.array(full_image)

            # column 0: original full image
            ax0 = axes[row_idx, 0]
            ax0.imshow(full_img_array)
            ax0.set_title(
                f"{variant} - Original\nPred: {v_data['prediction']} "
                f"({'✓' if v_data['is_correct'] else '✗'})",
                fontsize=10,
            )
            ax0.axis("off")

            # per-crop overlays
            patch_attr = v_data["image_attr_map"]   # (N_crops, H_patch, W_patch) or (H_patch, W_patch)
            crops_np = v_data["image_crops"]        # (N_crops, H_crop, W_crop, 3)

            if patch_attr.ndim == 2:
                patch_attr = patch_attr[None, ...]  # (1, H_patch, W_patch)
            if crops_np.ndim == 3:
                crops_np = crops_np[None, ...]      # (1, H_crop, W_crop, 3)

            N_crops = patch_attr.shape[0]

            for c in range(max_crops):
                ax = axes[row_idx, 1 + c]
                if c >= N_crops:
                    ax.axis("off")
                    continue

                crop_img = crops_np[c]               # (H_crop, W_crop, 3)
                patch_map = patch_attr[c]            # (H_patch, W_patch)

                H_crop, W_crop = crop_img.shape[:2]
                H_patch, W_patch = patch_map.shape

                scale_h = H_crop / H_patch
                scale_w = W_crop / W_patch

                patch_up = zoom(patch_map, (scale_h, scale_w), order=1)  # (H_crop, W_crop)
                attr_abs = np.abs(patch_up)
                attr_sparse = self._topk_mask(attr_abs, keep_ratio=0.01)  # [0,1] sparse

                ax.imshow(crop_img)
                im = ax.imshow(
                    attr_sparse,
                    alpha=0.7,
                    cmap="jet",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax.set_title(f"{variant} – crop {c}", fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(
            f"Image Token IG (per-crop) - {case_results['image_id']}\nPattern: {case_results['pattern']}",
            fontsize=14,
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    # ------------------------------------------------------------------ #
    # PATTERN-LEVEL STATS
    # ------------------------------------------------------------------ #
    def _compute_pattern_statistics(self, results: List[Dict], pattern: str) -> Dict:
        """
        Simple statistics: per-variant average total image importance.
        """
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
                    vals.append(r["variants"][variant]["total_image_importance"])
            if not vals:
                continue

            stats["variants"][variant] = {
                "num_examples": len(vals),
                "avg_image_importance": float(np.mean(vals)),
                "std_image_importance": float(np.std(vals)),
            }

        return stats

    def analyze_pattern_group(self, pattern: str, cases: List[Dict],
                              image_folder: str, output_dir: Path) -> Dict:
        """
        For all cases in a pattern, compute image-token IG for each variant
        and generate per-case visualizations + basic stats.
        """
        print(f"\n{'=' * 60}")
        print(f"Analyzing pattern: {pattern}")
        print(f"Number of cases: {len(cases)}")
        print(f"{'=' * 60}")

        pattern_dir = output_dir / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)

        spatial_viz_dir = pattern_dir / "spatial_visualizations"
        spatial_viz_dir.mkdir(exist_ok=True)

        all_results = []

        for case in tqdm(cases, desc=f"Processing {pattern}"):
            case_results = self.compute_image_attributions_for_case(case, image_folder)
            all_results.append(case_results)

            comparison_path = spatial_viz_dir / f"{case_results['image_id']}_comparison.png"
            self.visualize_case_comparison(case_results, comparison_path)

        stats = self._compute_pattern_statistics(all_results, pattern)

        with open(pattern_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nStatistics for {pattern}:")
        print(f"  Total cases: {stats['num_cases']}")
        for variant, v_stats in stats["variants"].items():
            print(
                f"  {variant}: n={v_stats['num_examples']}, "
                f"avg IG={v_stats['avg_image_importance']:.4f} ± {v_stats['std_image_importance']:.4f}"
            )

        return stats

    # ------------------------------------------------------------------ #
    # TOP-LEVEL DRIVER
    # ------------------------------------------------------------------ #
    def run_analysis(self, results_dir: str, image_folder: str, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load stored results (no re-inference)
        all_results = self.load_results(results_dir)

        # 2) Identify only the differing (conflict) cases
        differing_cases = self.identify_differing_cases(all_results)

        # 3) Group by pattern
        patterns = defaultdict(list)
        for case in differing_cases:
            patterns[case["pattern"]].append(case)

        # Save which images per pattern
        case_summary = {
            pattern: [c["image_id"] for c in cases]
            for pattern, cases in patterns.items()
        }
        with open(output_dir / "case_categorization.json", "w") as f:
            json.dump(case_summary, f, indent=2)

        priority_patterns = [
            "fooled_by_incorrect",
            "fooled_by_irrelevant",
            "helped_by_correct",
            "resisted_incorrect",
        ]

        all_stats = {}

        # 4) Analyze priority patterns first
        for pattern in priority_patterns:
            if pattern in patterns and patterns[pattern]:
                print(f"\nProcessing pattern: {pattern} ({len(patterns[pattern])} cases)")
                stats = self.analyze_pattern_group(pattern, patterns[pattern], image_folder, output_dir)
                all_stats[pattern] = stats

        # 5) Analyze other patterns (e.g., mixed_behavior, all_correct, all_incorrect)
        for pattern, cases in patterns.items():
            if pattern not in priority_patterns and cases:
                print(f"\nProcessing pattern: {pattern} ({len(cases)} cases)")
                stats = self.analyze_pattern_group(pattern, cases, image_folder, output_dir)
                all_stats[pattern] = stats

        # 6) Cross-pattern summary
        summary = {}
        for pattern, stats in all_stats.items():
            summary[pattern] = {
                "num_cases": stats["num_cases"],
                "variants": stats["variants"],
            }

        with open(output_dir / "cross_pattern_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("✓ Image-token IG analysis (Phase 1) complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Spatial visualizations (per crop): {output_dir}/[pattern]/spatial_visualizations/")
        print(f"  - Pattern stats: {output_dir}/[pattern]/statistics.json")
        print(f"  - Cross-pattern summary: {output_dir}/cross_pattern_summary.json")
        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Image-token Integrated Gradients Analysis for LLaVA / LLaVA-Next Variants (Phase 1)"
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory containing inference result JSON files",
    )
    parser.add_argument(
        "--image_folder",
        default="./filtered_data",
        help="Root folder containing image variant subfolders (notext, correct, misleading, irrelevant)",
    )
    parser.add_argument(
        "--output_dir",
        default="./integrated_gradients/ig_tokens",
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

    args = parser.parse_args()

    analyzer = ImageVariantAnalyzer(
        model_type=args.model_type,
        model_id=args.model_id,
        device=args.device,
    )

    analyzer.run_analysis(
        results_dir=args.results_dir,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
