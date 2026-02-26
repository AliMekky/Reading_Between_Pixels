#!/usr/bin/env python3
"""
Updated Occlusion Attribution for VLMs with HuggingFace Dataset Support

Changes made:
1. Uses CORRECT ANSWER token instead of predicted answer for attribution
2. Saves logit differences and probabilities in 2D arrays for each mask
3. Processes ALL images with detailed pattern categorization (all_correct, only_X_wrong, etc.)
4. Computes both logit differences and probability differences
5. Identifies which mask patches intersect with text bounding boxes
6. Loads from HuggingFace dataset instead of legacy folder structure
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn.functional import log_softmax, softmax

import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# HF dataset imports
from datasets import load_dataset, load_from_disk, DatasetDict

import sys
sys.path.append('/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels')
from vlms.inference.infere_vlms import get_evaluator



def sanitize_repo_id(repo_id: str) -> str:
    """Make a filesystem-safe name for caching."""
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(
    dataset_id: str,
    local_cache_root: str = "./hf_dataset_local_cache",
    split: str = "test"
):
    """Download or load cached HF dataset."""
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    if cache_dir.exists():
        print(f"✓ Loading dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"⬇️  Downloading '{dataset_id}'...")
    ds = load_dataset(dataset_id, split=split)

    try:
        ds.save_to_disk(str(cache_dir))
        print(f"✓ Saved to cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️  Cache save failed: {e}")

    return ds


class HFOcclusionAnalyzer:
    """
    Occlusion-based attribution using HuggingFace dataset.
    
    Key changes:
    - Uses CORRECT answer token (not predicted)
    - Saves full logit/prob differences per mask
    - Processes all images with detailed patterns
    - Identifies text bbox intersections
    """

    def __init__(self, model_type: str = "llava-next", model_id: str = None, device: str = "auto"):
        print(f"Loading {model_type} model...")
        self.evaluator = get_evaluator(model_type, model_id, device)
        self.model = self.evaluator.model
        self.processor = self.evaluator.processor
        self.device = self.evaluator.device
        
        # Initialize answer token IDs (A, B, C, D)
        self._initialize_answer_tokens()
        
        # Get model's vision config
        self.vision_config = self._get_vision_config()
        print(f"Vision encoder expects images of size: {self.vision_config['image_size']}")

    def _initialize_answer_tokens(self):
        """Precompute token IDs for answer choices A/B/C/D."""
        print("\n=== Initializing Answer Tokens ===")
        self.answer_token_ids = {}
        
        for letter in ['A', 'B', 'C', 'D']:
            # Get both variants (with and without space)
            token_ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            token_ids_space = self.processor.tokenizer.encode(f" {letter}", add_special_tokens=False)
            
            # Store all possible token IDs for this letter
            ids_set = set()
            if token_ids:
                ids_set.add(token_ids[0])
            if token_ids_space:
                ids_set.add(token_ids_space[-1])
            
            self.answer_token_ids[letter] = ids_set
            print(f"  Token IDs for '{letter}': {ids_set}")
        
        print(f"All answer tokens initialized\n")

    def _get_vision_config(self) -> Dict:
        """Extract vision encoder configuration."""
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
                vision_cfg = self.model.config.vision_config
                image_size = getattr(vision_cfg, 'image_size', 336)
            else:
                image_size = 336
            
            return {'image_size': image_size}
        except:
            return {'image_size': 336}

    def _build_mcq_prompt(self, question: str, options: Dict[str, str]) -> str:
        """Build MCQ prompt."""
        instruction = "Answer the following multiple-choice question by selecting the correct option."
        
        prompt = f"{instruction}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        
        return prompt

    def _prepare_inputs_for_generation(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for generation (to get logits)."""
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

        inputs = self.processor(
            images=image,
            text=formatted_prompt,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        return inputs

    def _get_answer_logits_and_probs(self, image: Image.Image, prompt: str) -> Dict[str, Dict]:
        """
        Generate response and extract logits/probs for all answer tokens.
        
        Returns dict: {
            'A': {'logit': float, 'prob': float, 'token_id': int},
            'B': {'logit': float, 'prob': float, 'token_id': int},
            'C': {'logit': float, 'prob': float, 'token_id': int},
            'D': {'logit': float, 'prob': float, 'token_id': int},
        }
        """
        self.model.eval()
        inputs = self._prepare_inputs_for_generation(image, prompt)
        
        with torch.no_grad():
            # Generate to get the first token logits
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # Only need first token
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Get logits for the first generated token
            first_token_logits = outputs.scores[0][0]  # Shape: (vocab_size,)
            
            # Compute log probs and probs
            log_probs = log_softmax(first_token_logits, dim=-1)
            probs = softmax(first_token_logits, dim=-1)
        
        # Extract values for each answer
        result = {}
        for letter in ['A', 'B', 'C', 'D']:
            token_ids = list(self.answer_token_ids[letter])
            
            if token_ids:
                # Use the first token ID (could average if multiple)
                token_id = token_ids[0]
                result[letter] = {
                    'logit': first_token_logits[token_id].item(),
                    'log_prob': log_probs[token_id].item(),
                    'prob': probs[token_id].item(),
                    'token_id': token_id,
                }
            else:
                # Fallback if no token found
                result[letter] = {
                    'logit': -float('inf'),
                    'log_prob': -float('inf'),
                    'prob': 0.0,
                    'token_id': None,
                }
        
        return result

    def _get_adaptive_mask_value(self, image: Image.Image) -> int:
        """Get mask value based on image mean."""
        img_array = np.array(image)
        mean_value = int(img_array.mean())
        return mean_value

    def _compute_bbox_mask_intersections(
        self,
        bbox_xyxy: Optional[Tuple[float, float, float, float]],
        image_size: Tuple[int, int],
        grid_size: int = 16,
        min_overlap_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Returns a (grid_size, grid_size) boolean mask where True means the patch overlaps
        the bbox by at least `min_overlap_ratio` of the bbox area.

        bbox_xyxy: (x1, y1, x2, y2)
        """
        W, H = image_size
        patch_w = W / grid_size
        patch_h = H / grid_size
        patch_area = patch_w * patch_h

        mask = np.zeros((grid_size, grid_size), dtype=bool)

        if bbox_xyxy is None:
            return mask

        x1, y1, x2, y2 = map(float, bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            return mask

        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area <= 0:
            return mask

        for gy in range(grid_size):
            for gx in range(grid_size):
                px1 = gx * patch_w
                py1 = gy * patch_h
                px2 = (gx + 1) * patch_w
                py2 = (gy + 1) * patch_h

                ox1 = max(x1, px1)
                oy1 = max(y1, py1)
                ox2 = min(x2, px2)
                oy2 = min(y2, py2)

                if ox1 < ox2 and oy1 < oy2:
                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    overlap_ratio = overlap_area / (patch_h * patch_w)  
                    if overlap_ratio >= min_overlap_ratio:
                        mask[gy, gx] = True

        return mask
    
    def _xywh_to_xyxy(self, x, y, w, h):
        return (float(x), float(y), float(x + w), float(y + h))

    def compute_occlusion_attribution(
        self,
        image: Image.Image,
        prompt: str,
        correct_answer: str,
        text_bbox_xyxy: Optional[Tuple[float,float,float,float]] = None,
        correct_object_bbox_xyxy: Optional[Tuple[float,float,float,float]] = None,
        misleading_object_bbox_xyxy: Optional[Tuple[float,float,float,float]] = None,
        grid_size: int = 16,
        exclude_edge_patches: int = 1,
    ) -> Dict:
        """
        Compute occlusion attribution for CORRECT answer.
        
        Returns:
            {
                'logit_diffs': (grid_size, grid_size) array of logit differences,
                'prob_diffs': (grid_size, grid_size) array of prob differences,
                'base_logits': dict with base logits for all answers,
                'base_probs': dict with base probs for all answers,
                'valid_mask': (grid_size, grid_size) boolean mask,
                'text_intersection_mask': (grid_size, grid_size) boolean mask,
                'correct_answer': str,
            }
        """
        W, H = image.size
        mask_value = self._get_adaptive_mask_value(image)
        
        patch_w = W // grid_size
        patch_h = H // grid_size
        
        # Get baseline logits/probs for all answers
        base_answer_data = self._get_answer_logits_and_probs(image, prompt)
        
        # Extract correct answer data
        base_logit = base_answer_data[correct_answer]['logit']
        base_prob = base_answer_data[correct_answer]['prob']

        # --- Compute intersection masks (NEW BLOCK) ---

        text_intersection_mask = self._compute_bbox_mask_intersections(
            text_bbox_xyxy, (W, H), grid_size
        )

        correct_object_intersection_mask = self._compute_bbox_mask_intersections(
            correct_object_bbox_xyxy, (W, H), grid_size
        )

        misleading_object_intersection_mask = self._compute_bbox_mask_intersections(
            misleading_object_bbox_xyxy, (W, H), grid_size
        )
        
        # Initialize arrays
        logit_diffs = np.zeros((grid_size, grid_size), dtype=np.float32)
        prob_diffs = np.zeros((grid_size, grid_size), dtype=np.float32)
        valid_mask = np.ones((grid_size, grid_size), dtype=bool)
        
        # Mark edges as invalid
        if exclude_edge_patches > 0:
            valid_mask[:exclude_edge_patches, :] = False
            valid_mask[-exclude_edge_patches:, :] = False
            valid_mask[:, :exclude_edge_patches] = False
            valid_mask[:, -exclude_edge_patches:] = False
        
        
        # Occlude each patch
        for gy in tqdm(range(grid_size), desc="Occlusion", leave=False):
            for gx in range(grid_size):
                y0 = gy * patch_h
                x0 = gx * patch_w
                y1 = (gy + 1) * patch_h if gy < grid_size - 1 else H
                x1 = (gx + 1) * patch_w if gx < grid_size - 1 else W
                
                # Mask patch
                img_masked = image.copy()
                arr = np.array(img_masked)
                arr[y0:y1, x0:x1, :] = mask_value
                img_masked = Image.fromarray(arr)

                if text_intersection_mask[gy, gx]:
                    mask_dir = Path(".") / "masked_images_text_only"
                    mask_dir.mkdir(exist_ok=True, parents=True)
                    img_masked.save(mask_dir / f"mask_{gy:02d}_{gx:02d}.png")
                # After: img_masked = Image.fromarray(arr)
                # Add:
                # img_masked.save(f"masked_images/mask_{gy}_{gx}.png")
                
                # Get logits/probs with masked image
                masked_answer_data = self._get_answer_logits_and_probs(img_masked, prompt)
                masked_logit = masked_answer_data[correct_answer]['logit']
                masked_prob = masked_answer_data[correct_answer]['prob']
                
                # Store differences (base - masked)
                logit_diffs[gy, gx] = base_logit - masked_logit
                prob_diffs[gy, gx] = base_prob - masked_prob
            
            # Clear cache periodically
            if (gy + 1) % 4 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return {
            'logit_diffs': logit_diffs,
            'prob_diffs': prob_diffs,
            'base_answer_data': base_answer_data,
            'valid_mask': valid_mask,
            'text_intersection_mask': text_intersection_mask,
            'correct_object_intersection_mask': correct_object_intersection_mask,
            'misleading_object_intersection_mask': misleading_object_intersection_mask,
            'correct_answer': correct_answer,
        }

    def load_hf_dataset(self, dataset_id: str, cache_dir: str = "./hf_dataset_local_cache") -> List[Dict]:
        """
        Load GUIC from HuggingFace and build a list of question dicts that include:
        - image_variants: PIL images for all variants
        - options / answer (MCQ letters)
        - variant_bboxes: for every variant, includes:
            * text_bbox_xyxy (None for notext)
            * correct_object_bbox_xyxy (always present)
            * misleading_object_bbox_xyxy (always present)
        - label_to_key: mapping from option letter -> which variant text it came from
            (useful if you shuffle options)

        NOTE: This assumes GUIC schema:
        sample["notext"]["image"]
        sample["correct_answer"]["image"], ["text"], ["bbox"], and ["x","y","w","h"]
        sample["misleading_groundable"]["image"], ["text"], ["bbox"], and ["x","y","w","h"]
        sample["misleading_ungroundable"]["image"], ["text"], ["bbox"]
        sample["irrelevant_word"]["image"], ["text"], ["bbox"]
        """
        print(f"Loading HuggingFace dataset: {dataset_id}")
        ds = get_or_download_hf_dataset(dataset_id, cache_dir, split="test")

        if isinstance(ds, DatasetDict):
            split_name = "test" if "test" in ds else list(ds.keys())[0]
            dataset = ds[split_name]
        else:
            dataset = ds

        print(f"Dataset size: {len(dataset)} samples")

        variants = [
            "notext",
            "correct_answer",
            "misleading_groundable",
            "misleading_ungroundable",
            "irrelevant_word",
        ]

        def xywh_to_xyxy(x, y, w, h):
            return (float(x), float(y), float(x + w), float(y + h))

        def to_xyxy(b):
            if b is None:
                return None
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            return None

        questions_data: List[Dict] = []

        print("Building questions list...")
        for idx in tqdm(range(len(dataset)), desc="Loading questions"):
            try:
                sample = dataset[idx]

                question_id = sample.get("question_id") or f"q_{idx}"
                question = sample.get("question", "")

                # ----------------------------
                # Build MCQ options (A-D)
                # ----------------------------
                candidates = [
                    ("correct_answer", sample["correct_answer"]["text"]),
                    ("misleading_groundable", sample["misleading_groundable"]["text"]),
                    ("misleading_ungroundable", sample["misleading_ungroundable"]["text"]),
                    ("irrelevant_word", sample["irrelevant_word"]["text"]),
                ]

                labels = ["A", "B", "C", "D"]

                # OPTIONAL: shuffle per-question deterministically (recommended)
                import random
                rng = random.Random(f"42_{question_id}")
                rng.shuffle(candidates)

                options = {labels[i]: candidates[i][1] for i in range(4)}
                label_to_key = {labels[i]: candidates[i][0] for i in range(4)}
                correct_label = labels[[k for k, _ in candidates].index("correct_answer")]
                answer = correct_label

                # ----------------------------
                # Object bboxes (xywh -> xyxy)
                # ----------------------------
                correct_obj_xyxy = xywh_to_xyxy(
                    sample["correct_answer"]["x"],
                    sample["correct_answer"]["y"],
                    sample["correct_answer"]["w"],
                    sample["correct_answer"]["h"],
                )

                misleading_obj_xyxy = xywh_to_xyxy(
                    sample["misleading_groundable"]["x"],
                    sample["misleading_groundable"]["y"],
                    sample["misleading_groundable"]["w"],
                    sample["misleading_groundable"]["h"],
                )

                # ----------------------------
                # Text bbox per variant (xyxy)
                # ----------------------------
                text_bbox_by_variant = {"notext": None}
                for v in ["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"]:
                    text_bbox_by_variant[v] = to_xyxy(sample[v].get("bbox", None))

                # Bundle all bbox types for every variant
                variant_bboxes = {}
                for v in variants:
                    variant_bboxes[v] = {
                        "text_bbox_xyxy": text_bbox_by_variant.get(v),
                        "correct_object_bbox_xyxy": correct_obj_xyxy,
                        "misleading_object_bbox_xyxy": misleading_obj_xyxy,
                    }

                # ----------------------------
                # Load images for all variants
                # ----------------------------
                image_variants = {}

                # notext image
                img0 = sample["notext"]["image"]
                if isinstance(img0, Image.Image):
                    image_variants["notext"] = img0.convert("RGB")

                # overlay variants
                for v in ["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"]:
                    imgv = sample[v]["image"]
                    if isinstance(imgv, Image.Image):
                        image_variants[v] = imgv.convert("RGB")

                if not image_variants:
                    continue

                questions_data.append({
                    "question_id": question_id,
                    "question": question,
                    "options": options,
                    "answer": answer,                 # correct letter after shuffling
                    "label_to_key": label_to_key,     # A/B/C/D -> which variant's text it is
                    "variant_bboxes": variant_bboxes, # all bbox types for every variant
                    "image_variants": image_variants,
                    # Optional extra metadata if you want it:
                    "caption": sample.get("caption", ""),
                    "image_id": sample.get("image_id", ""),
                    "seg_id": sample.get("seg_id", None),
                })

            except Exception as e:
                print(f"⚠️  Error loading sample {idx}: {e}")
                continue

        print(f"✓ Loaded {len(questions_data)} questions with image variants")
        return questions_data

    def categorize_pattern(self, variant_predictions: Dict[str, str], correct_answer: str) -> str:
        """
        Categorize pattern based on which variants are wrong.
        
        Returns detailed pattern like:
        - 'all_correct'
        - 'only_notext_wrong'
        - 'only_misleading_wrong'
        - 'notext_misleading_wrong'
        - 'all_wrong'
        etc.
        """
        wrong_variants = []
        for variant, prediction in variant_predictions.items():
            if prediction != correct_answer:
                wrong_variants.append(variant)
        
        if len(wrong_variants) == 0:
            return 'all_correct'
        elif len(wrong_variants) == 4:
            return 'all_wrong'
        else:
            # Sort for consistency
            wrong_variants_sorted = sorted(wrong_variants)
            return '_'.join(wrong_variants_sorted) + '_wrong'

    def process_single_question(
        self,
        question_data: Dict,
        grid_size: int = 16,
        exclude_edges: bool = True
    ) -> Dict:
        """Process a single question across all variants."""
        question_id = question_data['question_id']
        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data['answer']
        variant_bboxes = question_data['variant_bboxes']  # ← Changed
        
        prompt = self._build_mcq_prompt(question, options)
        
        results = {
            'question_id': question_id,
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'variant_bboxes': variant_bboxes,  # ← NEW: per-variant bboxes
            'variants': {},
        }
                
        # First pass: get predictions for pattern categorization
        variant_predictions = {}
        for variant_name, image in question_data['image_variants'].items():
            # Quick generation to get prediction
            with torch.no_grad():
                response = self.evaluator.process_single(
                    image, prompt, max_new_tokens=5, do_sample=False
                )
                predicted = self.evaluator.extract_answer(response)
                variant_predictions[variant_name] = predicted
        
        # Categorize pattern
        pattern = self.categorize_pattern(variant_predictions, correct_answer)
        results['pattern'] = pattern
        
        # Second pass: compute occlusion for each variant
        for variant_name, image in question_data['image_variants'].items():
            print(f"  Processing {variant_name}...")
            
            bb = variant_bboxes.get(variant_name, {})

            attribution_data = self.compute_occlusion_attribution(
                image=image,
                prompt=prompt,
                correct_answer=correct_answer,
                text_bbox_xyxy=bb.get("text_bbox_xyxy"),
                correct_object_bbox_xyxy=bb.get("correct_object_bbox_xyxy"),
                misleading_object_bbox_xyxy=bb.get("misleading_object_bbox_xyxy"),
                grid_size=grid_size,
                exclude_edge_patches=1 if exclude_edges else 0,
            )
                        
            # Add prediction info
            attribution_data['predicted_answer'] = variant_predictions[variant_name]
            attribution_data['is_correct'] = (variant_predictions[variant_name] == correct_answer)
            
            results['variants'][variant_name] = attribution_data
        
        return results

    def save_results(self, results: Dict, output_dir: Path):
        """Save results for a single question."""
        question_id = results['question_id']
        pattern = results['pattern']
        
        # Create pattern directory
        pattern_dir = output_dir / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays and metadata separately
        arrays_to_save = {}
        metadata = {
            'question_id': question_id,
            'question': results['question'],
            'options': results['options'],
            'correct_answer': results['correct_answer'],
            'variant_bboxes': results.get('variant_bboxes', {}),  # ← Save all bboxes
            'pattern': pattern,
            'variants': {},
        }
        
        for variant_name, variant_data in results['variants'].items():
            # Save arrays
            arrays_to_save[f'{variant_name}_logit_diffs'] = variant_data['logit_diffs']
            arrays_to_save[f'{variant_name}_prob_diffs'] = variant_data['prob_diffs']
            arrays_to_save[f'{variant_name}_valid_mask'] = variant_data['valid_mask']
            arrays_to_save[f'{variant_name}_text_intersection'] = variant_data['text_intersection_mask']
            arrays_to_save[f"{variant_name}_correct_obj_intersection"] = variant_data["correct_object_intersection_mask"]
            arrays_to_save[f"{variant_name}_misleading_obj_intersection"] = variant_data["misleading_object_intersection_mask"]
            
            # Save metadata
            metadata['variants'][variant_name] = {
                'predicted_answer': variant_data['predicted_answer'],
                'is_correct': variant_data['is_correct'],
                'correct_answer': variant_data['correct_answer'],
                'base_answer_data': variant_data['base_answer_data'],
            }
        
        # Save arrays as .npz
        np.savez_compressed(
            pattern_dir / f"{question_id}_arrays.npz",
            **arrays_to_save
        )
        
        # Save metadata as JSON
        with open(pattern_dir / f"{question_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def run_analysis(
        self,
        dataset_id: str,
        output_dir: str,
        cache_dir: str = "./hf_dataset_local_cache",
        grid_size: int = 16,
        exclude_edges: bool = True,
        max_samples: Optional[int] = None,
        start: int = 0,
        end: int = 1062,
    ):
        """Run complete occlusion analysis on HF dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print('here')
        questions_data = self.load_hf_dataset(dataset_id, cache_dir)
        print(len(questions_data))
        
        if max_samples:
            questions_data = questions_data[:max_samples]
        
        print(f"\n{'='*60}")
        print(f"Processing {len(questions_data)} questions")
        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Edge exclusion: {'ON' if exclude_edges else 'OFF'}")
        print(f"{'='*60}\n")
        
        # Process all questions
        pattern_counts = defaultdict(int)
        
        for idx, question_data in enumerate(tqdm(questions_data, desc="Processing questions")):
            print(f"\n[{idx+1}/{len(questions_data)}] Question: {question_data['question_id']}")
            print(start)
            print(end)
            if idx < start:
                continue
            if idx >= end:
                break
            
            try:
                results = self.process_single_question(
                    question_data,
                    grid_size=grid_size,
                    exclude_edges=exclude_edges
                )
                
                # Save results
                self.save_results(results, output_dir)
                
                # Track pattern
                pattern_counts[results['pattern']] += 1
                
                print(f"  Pattern: {results['pattern']}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save summary
        summary = {
            'total_processed': len(questions_data),
            'pattern_counts': dict(pattern_counts),
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✓ Occlusion analysis complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"\nPattern distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern:40s}: {count:4d}")
        print(f"{'='*60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Occlusion Analysis with HuggingFace Dataset"
    )
    parser.add_argument(
        "--dataset_id",
        default="AHAAM/GUIC",
        help="HuggingFace dataset ID (e.g., AHAAM/CIM)",
    )
    parser.add_argument(
        "--output_dir",
        default="./occlusion_results_GUIC",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache_dir",
        default="./hf_dataset_GUIC",
        help="Cache directory for HF dataset",
    )
    parser.add_argument(
        "--model_type",
        default="llava-next",
        help="VLM type",
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help="Specific model ID (optional)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=8,
        help="Grid size for occlusion patches",
    )
    parser.add_argument(
        "--no_exclude_edges",
        action="store_true",
        help="Disable edge exclusion",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from a specific index",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1062,
        help="End processing at a specific index",
    )

    
    args = parser.parse_args()
    
    analyzer = HFOcclusionAnalyzer(
        model_type=args.model_type,
        model_id=args.model_id,
        device=args.device,
    )
    
    analyzer.run_analysis(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        grid_size=args.grid_size,
        exclude_edges=not args.no_exclude_edges,
        max_samples=args.max_samples,
        start = args.start,
        end = args.end,
    )


if __name__ == "__main__":
    main()