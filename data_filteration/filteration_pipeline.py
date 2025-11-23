import json
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
import easyocr
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class ImageFiltrationPipeline:
    """
    Simple 3-criterion filtration pipeline:
    1. Answer length (≤ 2 words)
    2. Minimal existing text (EasyOCR - no system dependencies)
    3. Writable surfaces (SegFormer segmentation)
    """
    
    def __init__(self, 
                 questions_json,
                 images_folder,
                 output_folder="filtered_results",
                 use_gpu=True):
        """
        Initialize the pipeline.
        
        Args:
            questions_json: Path to questions JSON file
            images_folder: Path to folder containing images
            output_folder: Where to save results
            use_gpu: Whether to use GPU for OCR and segmentation
        """
        print("="*60)
        print("Initializing Image Filtration Pipeline")
        print("="*60)
        
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Load questions
        print(f"\n1. Loading questions from {questions_json}...")
        with open(questions_json, 'r') as f:
            self.questions = json.load(f)
        print(f"   ✓ Loaded {len(self.questions)} questions")
        
        # Thresholds (tunable)
        self.max_answer_words = 2
        self.max_text_chars = 50  # Increased for OCR (more accurate than edge detection)
        self.min_writable_ratio = 0.10
        self.max_non_writable_ratio = 0.70
        self.min_region_area_ratio = 0.05
        self.min_region_width = 100
        self.min_region_height = 50
        
        # Check GPU
        self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        print(f"\n2. Using device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize EasyOCR
        print("\n3. Loading EasyOCR model...")
        print("   (This may take a minute on first run - downloading models)")
        self.ocr_reader = easyocr.Reader(
            ['en'],  # English only
            gpu=use_gpu,
            verbose=False
        )
        print(f"   ✓ EasyOCR loaded on {self.device}")
        
        # Load SegFormer model
        print("\n4. Loading SegFormer segmentation model...")
        model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
        self.seg_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.seg_model.eval()
        self.seg_model = self.seg_model.to(self.device)
        print(f"   ✓ SegFormer loaded on {self.device}")
        
        # Define ADE20K writable/non-writable classes
        # self.writable_classes = {
        #     0: "wall",
        #     3: "floor",
        #     5: "ceiling",
        #     7: "door",
        #     8: "windowpane",
        #     14: "table",
        #     15: "cabinet",
        #     33: "desk",
        #     86: "board",
        #     101: "countertop"
        # }
        
        self.non_writable_classes = {
                # Natural elements
                2: "sky",
                4: "tree",
                9: "grass",
                13: "earth/ground",
                16: "mountain",
                17: "plant",
                21: "water",
                26: "sea",
                29: "field",
                34: "rock",
                46: "sand",
                60: "river",
                66: "flower",
                68: "hill",
                72: "palm",
                94: "land",
                104: "fountain",
                109: "swimming pool",
                113: "waterfall",
                128: "lake",
                
                # Living beings
                12: "person",
                126: "animal",
                
                # Transparent/reflective (hard to write on)
                # 8: "windowpane",  # Actually can write on glass, so removed
                # 49: "glass",       # Same - can write on glass
                
                # Fabric/soft materials (hard to write on)
                18: "curtain",
                57: "pillow",
                81: "towel",
                120: "food",
                131: "blanket",
                
                # Light sources (can't write on light)
                36: "lamp",
                82: "light",
                87: "streetlight",

                3: "floor",
                6: "road",
                11: "sidewalk",
                63: "blind",
                136: "traffic light",
                145: "shower",
                # Small objects (too small to write on)
                # Removed - cars, furniture, etc. are writeable
        }
                
        
        # self.writable_class_ids = list(self.writable_classes.keys())
        self.non_writable_class_ids = list(self.non_writable_classes.keys())
        
        print("\n5. Pipeline ready!")
        print("="*60)
    
    # =========================================================================
    # CRITERION 1: Answer Length Check
    # =========================================================================
    
    def check_answer_length(self, question_data):
        """
        Check if all answer options are ≤ 2 words.
        """
        options = {"A": question_data.get("A", ""),
                   "B": question_data.get("B", ""),
                   "C": question_data.get("C", ""),
                   "D": question_data.get("D", "")}
        word_counts = []
        for key, answer in options.items():
            words = answer.strip().split()
            word_counts.append(len(words))
        
        max_words = max(word_counts) if word_counts else 0
        passes = max_words <= self.max_answer_words
        
        return {
            "criterion": "answer_length",
            "passes": passes,
            "max_words": max_words,
            "word_counts": word_counts
        }
    
    # =========================================================================
    # CRITERION 2: OCR Text Detection (EasyOCR)
    # =========================================================================
    
    def check_minimal_text(self, image_path):
        """
        Check if image has minimal existing text using EasyOCR.
        GPU-accelerated, no system dependencies needed.
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            
            if img is None:
                return {
                    "criterion": "existing_text",
                    "passes": False,
                    "error": "Could not load image"
                }
            
            # Run OCR (GPU-accelerated)
            # Returns list of (bbox, text, confidence)
            results = self.ocr_reader.readtext(img)
            
            # Extract all detected text
            all_text = ' '.join([text for (bbox, text, conf) in results])
            
            # Count characters
            cleaned_text = all_text.strip()
            char_count = len(cleaned_text)
            
            passes = char_count <= self.max_text_chars
            
            return {
                "criterion": "existing_text",
                "passes": passes,
                "method": "easyocr",
                "char_count": char_count,
                "num_text_regions": len(results),
                "detected_text": cleaned_text[:100]  # First 100 chars
            }
            
        except Exception as e:
            return {
                "criterion": "existing_text",
                "passes": False,
                "error": str(e)
            }
    
    # =========================================================================
    # CRITERION 3: Segmentation (Writable Surfaces)
    # =========================================================================
    
    def check_writable_surfaces(self, image_path):
        """
        Check for writable surfaces using SegFormer segmentation.
        Strategy: Everything is writable EXCEPT non-writable surfaces.
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run SegFormer (GPU-accelerated)
            inputs = self.seg_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
                logits = outputs.logits
            
            # Upsample to original size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            
            # Get segmentation map
            segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # Calculate metrics
            total_pixels = segmentation.size
            
            # Non-writable surfaces (what we want to AVOID)
            non_writable_mask = np.isin(segmentation, self.non_writable_class_ids)
            non_writable_pixels = non_writable_mask.sum()
            non_writable_ratio = non_writable_pixels / total_pixels
            
            # Writable surfaces = everything else
            writable_mask = ~non_writable_mask  # Invert the mask
            writable_pixels = writable_mask.sum()
            writable_ratio = writable_pixels / total_pixels
            
            # Find writable regions
            regions = self._find_writable_regions(writable_mask)
            
            # Evaluate regions
            good_regions = []
            for region in regions:
                if self._is_good_region(region):
                    good_regions.append(region)
            
            # Pass conditions
            has_enough_writable = writable_ratio >= self.min_writable_ratio
            not_too_much_non_writable = non_writable_ratio <= self.max_non_writable_ratio
            has_good_regions = len(good_regions) >= 1
            has_large_region = (good_regions[0]['area_ratio'] >= self.min_region_area_ratio) if good_regions else False
            
            passes = (
                has_enough_writable and
                not_too_much_non_writable and
                has_good_regions and
                has_large_region
            )
            
            # Get distribution of non-writable classes (for debugging)
            unique_classes = np.unique(segmentation)
            non_writable_dist = {}
            for class_id in unique_classes:
                if class_id in self.non_writable_class_ids:
                    pixels = (segmentation == class_id).sum()
                    ratio = pixels / total_pixels
                    non_writable_dist[self.non_writable_classes[class_id]] = round(float(ratio), 3)
            
            non_writable_dist = dict(sorted(non_writable_dist.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "criterion": "segmentation",
                "passes": passes,
                "writable_ratio": round(float(writable_ratio), 3),
                "non_writable_ratio": round(float(non_writable_ratio), 3),
                "num_regions": len(regions),
                "num_good_regions": len(good_regions),
                "best_region": good_regions[0] if good_regions else None,
                "non_writable_classes_present": non_writable_dist  # What's blocking us
            }
            
        except Exception as e:
            return {
                "criterion": "segmentation",
                "passes": False,
                "error": str(e)
            }
    
    def _find_writable_regions(self, writable_mask):
        """Find connected writable regions."""
        mask_uint8 = (writable_mask * 255).astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )
        
        total_pixels = writable_mask.size
        regions = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            aspect_ratio = width / height if height > 0 else 0
            area_ratio = area / total_pixels
            
            regions.append({
                'region_id': i,
                'area': int(area),
                'area_ratio': float(area_ratio),
                'bbox': [int(x), int(y), int(width), int(height)],
                'width': int(width),
                'height': int(height),
                'aspect_ratio': float(aspect_ratio)
            })
        
        regions.sort(key=lambda r: r['area'], reverse=True)
        return regions
    
    def _is_good_region(self, region):
        """Check if region is suitable for text."""
        width_ok = region['width'] >= self.min_region_width
        height_ok = region['height'] >= self.min_region_height
        aspect_ok = 0.3 <= region['aspect_ratio'] <= 5.0
        
        return width_ok and height_ok and aspect_ok
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def filter_single_image(self, image_index):
        """Apply all 3 criteria to one image."""
        
        question_data = self.questions[image_index]
        image_name = question_data.get('image')
        image_path = self.images_folder / image_name
        
        result = {
            "image_index": image_index,
            "image_name": image_name,
            "question": question_data.get('question'),
            "checks": {}
        }
        
        # CRITERION 1: Answer Length (fastest, no image needed)
        answer_check = self.check_answer_length(question_data)
        result["checks"]["answer_length"] = answer_check
        
        if not answer_check["passes"]:
            result["overall_pass"] = False
            result["rejection_reason"] = "answer_length"
            return result
        
        # CRITERION 2: Existing Text (OCR - GPU accelerated)
        text_check = self.check_minimal_text(image_path)
        result["checks"]["existing_text"] = text_check
        
        if not text_check["passes"]:
            result["overall_pass"] = False
            result["rejection_reason"] = "existing_text"
            return result
        
        # CRITERION 3: Writable Surfaces (GPU-accelerated)
        surface_check = self.check_writable_surfaces(image_path)
        result["checks"]["segmentation"] = surface_check
        
        if not surface_check["passes"]:
            result["overall_pass"] = False
            result["rejection_reason"] = "segmentation"
            return result
        
        # ALL PASSED
        result["overall_pass"] = True
        result["rejection_reason"] = None
        
        return result
    
    def run_pipeline(self, start_idx=0, end_idx=None):
        """Run pipeline on all images."""
        
        if end_idx is None:
            end_idx = len(self.questions)
        
        print(f"\n{'='*60}")
        print(f"Running Pipeline on {end_idx - start_idx} images")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        all_results = []
        passed_images = []
        failed_images = []
        
        failure_reasons = {
            "answer_length": 0,
            "existing_text": 0,
            "segmentation": 0
        }
        
        # Process images with progress bar
        for idx in tqdm(range(start_idx, end_idx), desc="Processing"):
            try:
                result = self.filter_single_image(idx)
                all_results.append(result)
                
                if result["overall_pass"]:
                    passed_images.append(result)
                else:
                    failed_images.append(result)
                    reason = result.get("rejection_reason")
                    if reason in failure_reasons:
                        failure_reasons[reason] += 1
                        
            except Exception as e:
                print(f"\nError processing image {idx}: {e}")
                failed_images.append({
                    "image_index": idx,
                    "overall_pass": False,
                    "error": str(e)
                })
        
        # Generate summary
        summary = {
            "total_images": end_idx - start_idx,
            "passed": len(passed_images),
            "failed": len(failed_images),
            "pass_rate": round(len(passed_images) / (end_idx - start_idx) * 100, 2),
            "failure_reasons": failure_reasons,
            "thresholds": {
                "max_answer_words": self.max_answer_words,
                "max_text_chars": self.max_text_chars,
                "min_writable_ratio": self.min_writable_ratio,
                "max_non_writable_ratio": self.max_non_writable_ratio,
                "min_region_area_ratio": self.min_region_area_ratio
            }
        }
        
        # Save results
        self._save_json(all_results, "all_results.json")
        self._save_json(passed_images, "passed_images.json")
        self._save_json(failed_images, "failed_images.json")
        self._save_json(summary, "summary.json")
        
        # Print summary
        self._print_summary(summary)
        
        return {
            "all_results": all_results,
            "passed": passed_images,
            "failed": failed_images,
            "summary": summary
        }
    
    def _save_json(self, data, filename):
        """
        Save data to JSON file, handling numpy types.
        """
        output_path = self.output_folder / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            print(f"✓ Saved: {output_path}")
        except Exception as e:
            print(f"✗ Failed to save {filename}: {e}")
            # Try to save without pretty printing as fallback
            try:
                with open(output_path, 'w') as f:
                    json.dump(data, f, cls=NumpyEncoder)
                print(f"✓ Saved (without formatting): {output_path}")
            except:
                print(f"✗ Could not save {filename}")
    
    def _print_summary(self, summary):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print("FILTRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Images:     {summary['total_images']}")
        print(f"Passed:           {summary['passed']} ({summary['pass_rate']}%)")
        print(f"Failed:           {summary['failed']}")
        print(f"\nFailure Breakdown:")
        for reason, count in summary['failure_reasons'].items():
            print(f"  - {reason:20s}: {count}")
        print(f"{'='*60}\n")





# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ImageFiltrationPipeline(
        questions_json="./data/questions.json",
        images_folder="./data/images",
        output_folder="filtered_results_seed_bench",
        use_gpu=True  # Set to False if no GPU available
    )
    
    # Run on all images
    results = pipeline.run_pipeline()
    
    # Or run on subset for testing
    # results = pipeline.run_pipeline(start_idx=0, end_idx=100)