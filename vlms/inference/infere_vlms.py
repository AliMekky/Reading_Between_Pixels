"""
VLM MCQ Evaluator - Modular Version with HF dataset support
Supports: LLaVA, Qwen-VL, InternVL, etc.

New args:
  --hf_dataset <repo_id>      : Use a Hugging Face dataset (e.g. username/read-betw-pixels)
  --hf_cache_dir <path>       : Where to store the downloaded dataset locally (default: ./hf_dataset_local_cache)
  --variant <notext|correct|irrelevant|misleading> : which image variant to use from HF dataset
If --hf_dataset is given, it takes precedence over --image_folder/--questions.
"""

import torch
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
from transformers import AutoProcessor
from tqdm import tqdm
from abc import ABC, abstractmethod

# datasets imports (for HF dataset support)
from datasets import load_dataset, load_from_disk, DatasetDict

# -- existing conditional imports --
try:
    from transformers import LlavaForConditionalGeneration
    LLAVA_AVAILABLE = True
except Exception:
    LLAVA_AVAILABLE = False
    print("Warning: LLaVA not available.")


try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except Exception:
    QWEN_AVAILABLE = False
    print("Warning: Qwen2VL not available. Install with: pip install qwen-vl-utils torchvision")


try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except Exception:
    LLAVA_NEXT_AVAILABLE = False

try:
    from transformers import AutoModel
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    INTERNVL_AVAILABLE = True
except Exception:
    INTERNVL_AVAILABLE = False

# ---------- (your BaseVLMEvaluator and model-specific classes unchanged) ----------
# [Keep the full content of BaseVLMEvaluator, LlavaEvaluator, QwenVLEvaluator, LlavaNextEvaluator,
#  InternVLEvaluator and MODEL_REGISTRY here — omitted for brevity in this snippet]
#
# For the final script, include the classes exactly as in your original file.
# I'll assume they remain unchanged and appear above the new HF-related functions.
# ------------------------------------------------------------------------------

# ---------- HF dataset helper utilities ----------

def sanitize_repo_id(repo_id: str) -> str:
    """Make a filesystem-safe name for caching the dataset."""
    return repo_id.replace("/", "__").replace(" ", "_")

def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str = "./hf_dataset_local_cache",
                               split: str = None) -> DatasetDict:
    """
    Ensure HF dataset is available locally. If not, download and save it.

    Returns a DatasetDict or Dataset if the HF dataset only has one split.
    """
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    # If cached on disk already, load from disk
    if cache_dir.exists():
        print(f"Found local HF dataset cache at: {cache_dir}. Loading from disk...")
        ds = load_from_disk(str(cache_dir))
        print("Loaded dataset from disk.")
        return ds

    # Not cached -> download
    print(f"Downloading dataset '{dataset_id}' from Hugging Face...")
    if split is None:
        ds = load_dataset(dataset_id)
    else:
        ds = load_dataset(dataset_id, split=split)
    # Save to disk for future runs (if it's a DatasetDict or Dataset)
    print(f"Saving downloaded dataset to local cache: {cache_dir} ...")
    # load_dataset returns DatasetDict or Dataset; save_to_disk is available on both
    try:
        ds.save_to_disk(str(cache_dir))
        print("Saved dataset to disk.")
    except Exception as e:
        print("Warning: failed to save dataset to disk:", e)
    return ds

# ---------- New evaluation path: evaluate HF dataset directly ----------

def build_questions_from_hf_dataset(ds, variant: str = "notext"):
    """
    Convert a loaded HF dataset into a questions_data list similar to previously expected JSON.
    Each item will contain:
      - 'image_path' : local path string to the chosen variant image (sample[variant]['path'] or sample[variant])
      - 'question'   : question string
      - 'options'    : dict {'A':..., 'B':..., 'C':..., 'D':...}
      - 'answer'     : ground-truth label (if present)
      - preserve other metadata if present
    """
    items = []
    # If user passed a DatasetDict, pick 'test' or the first split
    if isinstance(ds, DatasetDict):
        # prefer 'test' split if present
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        dataset = ds[split_name]
    else:
        dataset = ds

    for sample in dataset:
        # sample may have fields: question, choices (list), answer (A/B/C/D) and image variants
        # get question
        question = sample.get("question") or sample.get("question_text") or ""
        choices_list = sample.get("choices") or sample.get("options") or []
        # build options dict A-D
        options = {}
        labels = ["A", "B", "C", "D"]
        for i, lbl in enumerate(labels):
            options[lbl] = choices_list[i] if i < len(choices_list) else ""

        # correct answer label in dataset
        ans = sample.get("answer") or sample.get("correct_answer") or sample.get("label") or None

        # choose image path from variant
        img_obj = sample.get(variant)
        img_path = None
        if isinstance(img_obj, dict) and "path" in img_obj:
            img_path = img_obj["path"]
        elif isinstance(img_obj, str):
            img_path = img_obj
        else:
            # sometimes image column may be nested or named differently; attempt common names
            # try notext/correct/...
            # fallback: try sample['notext'] if variant missing
            # In absence, skip
            img_path = None

        if img_path is None:
            # skip this sample but print a warning
            print(f"Warning: sample missing '{variant}' image path, skipping. sample id: {sample.get('question_id')}")
            continue

        items.append({
            "image_path": img_path,
            "question": question,
            "options": options,
            "answer": ans,
            # keep original sample for reference if needed
            "raw_sample": sample
        })
    return items

# ---------- New evaluator method to operate on built list ----------

def evaluate_from_questions_list(evaluator, questions_list: List[Dict],
                                 output_file: str = None,
                                 batch_size: int = 4,
                                 max_new_tokens: int = 50):
    """
    Evaluate given questions_list (items with 'image_path','question','options','answer').
    Returns results list (same format as evaluate_mcq_folder).
    """
    image_paths = []
    prompts = []
    for item in questions_list:
        image_paths.append(item["image_path"])
        prompt = evaluator.format_mcq_prompt(item["question"], item["options"])
        prompts.append(prompt)

    print(f"Processing {len(image_paths)} items from HF dataset...\n")
    responses = evaluator.process_batch(
        image_paths,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )

    results = []
    valid_idx = 0
    for item in questions_list:
        response = responses[valid_idx]
        valid_idx += 1
        predicted_answer = evaluator.extract_answer(response)
        result = {
            "image": item["image_path"],
            "question": item["question"],
            "options": item["options"],
            "correct_answer": item.get("answer", None),
            "predicted_answer": predicted_answer,
            "full_response": response,
            "is_correct": (predicted_answer == item.get("answer", None)) if item.get("answer", None) else None
        }
        results.append(result)

    # compute accuracy if known
    if any(r["correct_answer"] for r in results):
        known = [r for r in results if r["correct_answer"] is not None]
        correct = sum(1 for r in known if r["is_correct"])
        total = len(known)
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\n{'='*50}")
        print(f"Evaluation Complete!")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*50}\n")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {output_file}")

    return results

def safe_suffix(path: str, suffix: str) -> str:
    """Add suffix before file extension: results.json -> results_correct.json"""
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
    return f"{path}_{suffix}"


def compute_accuracy(results: List[Dict]) -> Optional[float]:
    """Compute accuracy in % if correct answers exist; otherwise None."""
    known = [r for r in results if r.get("correct_answer") not in (None, "", "N/A")]
    if not known:
        return None
    correct = sum(1 for r in known if r.get("is_correct") is True)
    total = len(known)
    return (correct / total) * 100 if total else None


## For Debugging: Save per-step logits
def save_generation_logits(
    gen_out,
    tokenizer,
    inputs,
    output_path: str,
    top_k: int = 10
):
    """
    Save per-step top-K logits and chosen-token logits to a JSON file.
    """
    records = []

    # Length of the prompt (to isolate generated tokens)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = gen_out.sequences[0][prompt_len:]

    for step, step_scores in enumerate(gen_out.scores):
        # step_scores: [1, vocab]
        scores = step_scores[0]

        # Top-K tokens
        topk_logits, topk_token_ids = torch.topk(scores, k=top_k, dim=-1)

        top_tokens = []
        for logit, tok_id in zip(topk_logits.tolist(), topk_token_ids.tolist()):
            top_tokens.append({
                "token_id": int(tok_id),
                "token": tokenizer.decode([tok_id]),
                "logit": float(logit),
            })

        # Chosen token (greedy or sampled)
        chosen_token_id = generated_ids[step].item()
        chosen_logit = scores[chosen_token_id].item()

        records.append({
            "step": step,
            "top_tokens": top_tokens,  # ← TOP-10 HERE
            "chosen_token": {
                "token_id": int(chosen_token_id),
                "token": tokenizer.decode([chosen_token_id]),
                "logit": float(chosen_logit),
            }
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

class BaseVLMEvaluator(ABC):
    """
    Abstract base class for VLM evaluators.
    All VLM-specific implementations should inherit from this.
    """
    
    def __init__(self, model_id: str, device: str = None):
        """
        Initialize the VLM evaluator.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing {self.__class__.__name__} on {self.device}...")
        print(f"Loading model: {model_id}")
        
        self.model = None
        self.processor = None
        self._load_model()
        
        print("Model loaded successfully!\n")
    
    @abstractmethod
    def _load_model(self):
        """Load model and processor. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _decode_output(self, output) -> str:
        """Decode model output. Must be implemented by subclasses."""
        pass
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        return Image.open(image_path).convert('RGB')
    
    def format_mcq_prompt(self, question: str, options: Dict[str, str], 
                          instruction: str = None) -> str:
        """
        Format MCQ question with options into a prompt.
        
        Args:
            question: The question text
            options: Dictionary of options, e.g., {'A': 'option1', 'B': 'option2', ...}
            instruction: Optional instruction prefix
            
        Returns:
            Formatted prompt string
        """
        if instruction is None:
            instruction = "Answer the following multiple-choice question by selecting the correct option."
        
        prompt = f"{instruction}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        
        return prompt
    
    def process_single(self, image_path: str, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False) -> str:
        """
        Process a single image with a prompt.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (False = greedy/deterministic)
            
        Returns:
            Generated response text
        """
        # Load image
        image = self.load_image(image_path)
        
        # Prepare inputs (model-specific)
        inputs = self._prepare_inputs(image, prompt)
        
        # Generate response
        with torch.inference_mode():
            gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            )

        # Decode output as before
        output_ids = gen_out.sequences
        response = self._decode_output(output_ids)


        # ---- SAVE LOGITS (no printing) ----
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
        # create a unique filename per example
            os.makedirs("logits_debug", exist_ok=True)
            debug_path = os.path.join(
            "logits_debug",
            f"logits_{os.path.basename(image_path)}.json"
            )


            save_generation_logits(
            gen_out=gen_out,
            tokenizer=tokenizer,
            inputs=inputs,
            output_path=debug_path
            )

            
        
        # For Qwen-VL, trim the input tokens from output (removed to save the same output for both for manual validation)
        # if isinstance(self, QwenVLEvaluator):
        #     generated_ids = [
        #         output_ids[len(input_ids):] 
        #         for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        #     ]
        #     response = self._decode_output(generated_ids[0])
        # else:
        #     # For other models, decode normally
        #     response = self._decode_output(output_ids)
        
        return response
    
    def process_batch(self, image_paths: List[str], prompts: List[str],
                     max_new_tokens: int = 200,
                     do_sample: bool = False,
                     batch_size: int = 4) -> List[str]:
        """
        Process multiple images with prompts in batches.
        
        Args:
            image_paths: List of image file paths
            prompts: List of prompts (must match length of image_paths)
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            batch_size: Number of images to process at once
            
        Returns:
            List of generated responses
        """
        assert len(image_paths) == len(prompts), "Number of images and prompts must match"
        
        all_responses = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_images = image_paths[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            # Process each in batch (can be optimized per model)
            for img_path, prompt in zip(batch_images, batch_prompts):
                response = self.process_single(img_path, prompt, max_new_tokens, do_sample)
                all_responses.append(response)
        
        return all_responses
    
    def evaluate_mcq_folder(self, image_folder: str, questions_file: str,
                           output_file: str = None,
                           batch_size: int = 4,
                           max_new_tokens: int = 50) -> List[Dict]:
        """
        Evaluate MCQ questions from a folder of images.
        
        Args:
            image_folder: Path to folder containing images
            questions_file: Path to JSON file with questions
            output_file: Optional path to save results as JSON
            batch_size: Batch size for processing
            max_new_tokens: Maximum tokens for each answer
            
        Returns:
            List of dictionaries with results
        """
        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        print(f"Loaded {len(questions_data)} questions from {questions_file}")
        
        # Prepare image paths and prompts
        image_paths = []
        prompts = []
        
        for item in questions_data:
            image_path = os.path.join(image_folder, item['image'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            image_paths.append(image_path)
            prompt = self.format_mcq_prompt(item['question'], item['options'])
            prompts.append(prompt)
        
        print(f"Processing {len(image_paths)} valid image-question pairs...\n")
        
        # Process all questions
        responses = self.process_batch(
            image_paths, 
            prompts, 
            max_new_tokens=max_new_tokens,
            batch_size=batch_size
        )
        
        # Compile results
        results = []
        valid_idx = 0
        
        for item in questions_data:
            image_path = os.path.join(image_folder, item['image'])
            if not os.path.exists(image_path):
                continue
            
            response = responses[valid_idx]
            valid_idx += 1
            
            # Extract predicted answer
            predicted_answer = self.extract_answer(response)
            
            result = {
                "image": item['image'],
                "question": item['question'],
                "options": item['options'],
                "correct_answer": item.get('answer', 'N/A'),
                "predicted_answer": predicted_answer,
                "full_response": response,
                "is_correct": predicted_answer == item.get('answer', None)
            }
            results.append(result)
        
        # Calculate accuracy
        if any('answer' in item for item in questions_data):
            correct = sum(1 for r in results if r['is_correct'])
            total = len(results)
            accuracy = correct / total * 100 if total > 0 else 0
            print(f"\n{'='*50}")
            print(f"Evaluation Complete!")
            print(f"Correct: {correct}/{total}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"{'='*50}\n")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return results
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer letter (A, B, C, or D) from model response.
        
        Args:
            response: Full model response
            
        Returns:
            Extracted answer letter or 'UNKNOWN'
        """
        import re
        
        # First, try to extract only the assistant's actual response
        assistant_response = response
        
        # Handle different assistant markers (case-insensitive)
        # Look for the LAST occurrence of assistant marker
        markers = ["ASSISTANT:", "Assistant:", "assistant:"]
        last_position = -1
        found_marker = None
        
        for marker in markers:
            pos = response.rfind(marker)  # rfind = find last occurrence
            if pos > last_position:
                last_position = pos
                found_marker = marker
        
        if found_marker:
            assistant_response = response[last_position + len(found_marker):].strip()
        
        # Convert to uppercase for pattern matching
        assistant_response_upper = assistant_response.upper()
        
        # Try to find answer patterns in order of specificity
        patterns = [
            r'ANSWER[:\s]+([ABCD])\b',  # Answer: A or Answer A
            r'^\s*([ABCD])\s*$',  # Just A on a line by itself
            r'^([ABCD])\b',  # A at the start
            r'\b([ABCD])\s*$',  # A at the end
            r'\b([ABCD])\b',  # Standalone A, B, C, or D (last resort)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, assistant_response_upper)
            if match:
                return match.group(1)
        
        return 'UNKNOWN'


class LlavaEvaluator(BaseVLMEvaluator):
    """
    LLaVA-specific evaluator implementation.
    """
    
    def _load_model(self):
        """Load LLaVA model and processor."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for LLaVA model."""
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
            conversation, add_generation_prompt=True
        )
        
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors='pt')
        inputs = inputs.to(self.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        """Decode LLaVA model output."""
        return self.processor.decode(output[0], skip_special_tokens=True)


class QwenVLEvaluator(BaseVLMEvaluator):
    """
    Qwen-VL-specific evaluator implementation.
    Requires: pip install qwen-vl-utils torchvision
    """
    
    def __init__(self, model_id: str, device: str = None):
        """Check if Qwen-VL dependencies are available."""
        if not QWEN_AVAILABLE:
            raise ImportError(
                "Qwen-VL requires additional dependencies. Install with:\n"
                "pip install qwen-vl-utils torchvision\n"
                "or\n"
                "pip install 'transformers[torch-vision]'"
            )
        super().__init__(model_id, device)
    
    def _load_model(self):
        """Load Qwen-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for Qwen-VL model using the correct format."""
        from qwen_vl_utils import process_vision_info
        
        # Qwen-VL requires this specific message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision info (images and videos)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Process with processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        """Decode Qwen-VL model output."""
        return self.processor.batch_decode(
            output, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]


class LlavaNextEvaluator(BaseVLMEvaluator):
    """
    LLaVA-NeXT (v1.6) evaluator implementation.
    """
    
    def __init__(self, model_id: str, device: str = None):
        """Check if LLaVA-NeXT is available."""
        if not LLAVA_NEXT_AVAILABLE:
            raise ImportError(
                "LLaVA-NeXT requires updated transformers. Install with:\n"
                "pip install -U transformers"
            )
        super().__init__(model_id, device)
    
    def _load_model(self):
        """Load LLaVA-NeXT model and processor."""
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for LLaVA-NeXT model."""
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
            conversation, add_generation_prompt=True
        )
        
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors='pt')
        inputs = inputs.to(self.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        """Decode LLaVA-NeXT model output."""
        return self.processor.decode(output[0], skip_special_tokens=True)
    
class InternVLEvaluator(BaseVLMEvaluator):
    """
    InternVL3.5 evaluator implementation.
    """
    
    def __init__(self, model_id: str, device: str = None):
        """Check if InternVL is available."""
        if not INTERNVL_AVAILABLE:
            raise ImportError(
                "InternVL requires: pip install torchvision einops timm"
            )
        super().__init__(model_id, device)
        
        # Load the helper functions from InternVL
        self._setup_image_processing()
    
    def _setup_image_processing(self):
        """Setup image processing functions from InternVL."""
        import math
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            return transform
        
        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio
        
        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height
            
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)
            
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
            
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images
        
        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values
        
        self.load_image_func = load_image
    
    def _load_model(self):
        """Load InternVL model."""
        from transformers import AutoTokenizer, AutoModel
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model = self.model.eval().to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        self.processor = None
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Not used for InternVL."""
        return {}
    
    def _decode_output(self, output) -> str:
        """Not used for InternVL."""
        return output
    
    def process_single(self, image_path: str, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False) -> str:
        """Process a single image with InternVL."""
        # Load PIL image
        image = self.load_image(image_path)
        
        # Preprocess using InternVL's function
        pixel_values = self.load_image_func(image, max_num=12).to(torch.bfloat16).to(self.device)
        
        # Generate
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
        
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )
        
        # Format for consistency
        full_response = f"User: {prompt}\nAssistant: {response}"
        
        return full_response

# Model registry for easy access
MODEL_REGISTRY = {
    'llava': {
        'class': LlavaEvaluator,
        'default_model': 'llava-hf/llava-1.5-7b-hf',
        'available_models': [
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-1.5-13b-hf',
        ],
        'available': True
    },
}

## try llava next or llama 3.2 model
## at least qwen 2.5 or 3
## internvl model 3 


# Add Qwen-VL only if dependencies are available
if QWEN_AVAILABLE:
    MODEL_REGISTRY['qwen-vl'] = {
        'class': QwenVLEvaluator,
        'default_model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'available_models': [
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
        ],
        'available': True
    }
else:
    MODEL_REGISTRY['qwen-vl'] = {
        'available': False,
        'error_message': 'Qwen-VL requires: pip install qwen-vl-utils torchvision'
    }


if LLAVA_NEXT_AVAILABLE:
    MODEL_REGISTRY['llava-next'] = {
        'class': LlavaNextEvaluator,
        'default_model': 'llava-hf/llava-v1.6-mistral-7b-hf',
        'available_models': [
            'llava-hf/llava-v1.6-mistral-7b-hf',
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
        ],
        'available': True
    }
else:
    MODEL_REGISTRY['llava-next'] = {
        'available': False,
        'error_message': 'LLaVA-NeXT requires: pip install -U transformers'
    }

# Add InternVL if available
if INTERNVL_AVAILABLE:
    MODEL_REGISTRY['internvl'] = {
        'class': InternVLEvaluator,
        'default_model': 'OpenGVLab/InternVL3_5-8B',
        'available_models': [
            'OpenGVLab/InternVL3_5-8B',
            'OpenGVLab/InternVL3_5-4B',
            'OpenGVLab/InternVL3_5-2B',
        ],
        'available': True
    }
else:
    MODEL_REGISTRY['internvl'] = {
        'available': False,
        'error_message': 'InternVL requires: pip install torchvision'
    }

def get_evaluator(model_type: str, model_id: str = None, device: str = None) -> BaseVLMEvaluator:
    """
    Factory function to get the appropriate evaluator.
    
    Args:
        model_type: Type of model ('llava', 'qwen-vl')
        model_id: Specific model ID (optional, uses default if not provided)
        device: Device to use
        
    Returns:
        Appropriate evaluator instance
    """
    if model_type not in MODEL_REGISTRY:
        available = [k for k, v in MODEL_REGISTRY.items() if v.get('available', False)]
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {available}"
        )
    
    model_info = MODEL_REGISTRY[model_type]
    
    # Check if model is available
    if not model_info.get('available', False):
        error_msg = model_info.get('error_message', 'Model not available')
        raise ImportError(f"Cannot use {model_type}: {error_msg}")
    
    evaluator_class = model_info['class']
    
    if model_id is None:
        model_id = model_info['default_model']
    
    return evaluator_class(model_id=model_id, device=device)


def create_sample_questions_file(output_path: str = "sample_questions.json"):
    """Create a sample questions JSON file for reference."""
    sample_data = [
        {
            "image": "image1.jpg",
            "question": "What is the main object in this image?",
            "options": {
                "A": "A cat",
                "B": "A dog",
                "C": "A bird",
                "D": "A fish"
            },
            "correct_answer": "A"
        },
        {
            "image": "image2.jpg",
            "question": "What color is dominant in this image?",
            "options": {
                "A": "Red",
                "B": "Blue",
                "C": "Green",
                "D": "Yellow"
            },
            "correct_answer": "B"
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample questions file created: {output_path}")


## OLD MAIN FUNCTION FOR the OLD DEFAULT FLOW WITHOUT HF DATASET SUPPORT
# def main():
#     """Main function to run MCQ evaluation from command line."""
#     parser = argparse.ArgumentParser(
#         description="VLM MCQ Evaluator - Evaluate multiple-choice questions with various vision-language models",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
#         Examples:
#         # Evaluate with LLaVA (default)
#         python script.py --model_type llava --image_folder ./images/ --questions questions.json
        
#         # Evaluate with Qwen-VL
#         python script.py --model_type qwen-vl --image_folder ./images/ --questions questions.json
        
#         # Use specific model
#         python script.py --model_type llava --model_id llava-hf/llava-1.5-13b-hf --image_folder ./images/ --questions questions.json
        
#         # List available models
#         python script.py --list_models
        
#         # Create sample questions file
#         python script.py --create_sample sample_questions.json
#                 """
#             )
    
#     parser.add_argument(
#         '--model_type',
#         type=str,
#         choices=list(MODEL_REGISTRY.keys()),
#         default='llava',
#         help='Type of VLM to use (default: llava)'
#     )
    
#     parser.add_argument(
#         '--model_id',
#         type=str,
#         help='Specific HuggingFace model ID (optional, uses default for model_type if not provided)'
#     )
    
#     parser.add_argument(
#         '--image_folder',
#         type=str,
#         help='Path to folder containing images'
#     )
    
#     parser.add_argument(
#         '--questions',
#         type=str,
#         help='Path to JSON file with questions'
#     )
    
#     parser.add_argument(
#         '--output',
#         type=str,
#         default='results.json',
#         help='Path to save results JSON (default: results.json)'
#     )
    
#     parser.add_argument(
#         '--batch_size',
#         type=int,
#         default=4,
#         help='Batch size for processing (default: 4)'
#     )
    
#     parser.add_argument(
#         '--max_tokens',
#         type=int,
#         default=50,
#         help='Maximum tokens to generate per answer (default: 50)'
#     )
    
#     parser.add_argument(
#         '--device',
#         type=str,
#         choices=['cuda', 'cpu', 'auto'],
#         default='auto',
#         help='Device to use (default: auto)'
#     )
    
#     parser.add_argument(
#         '--list_models',
#         action='store_true',
#         help='List all available models and exit'
#     )
    
#     parser.add_argument(
#         '--create_sample',
#         type=str,
#         metavar='OUTPUT_FILE',
#         help='Create a sample questions JSON file and exit'
#     )
    
#     args = parser.parse_args()
    
#     # Handle list models
#     if args.list_models:
#         print("\nAvailable Models:")
#         print("=" * 60)
#         for model_type, info in MODEL_REGISTRY.items():
#             if info.get('available', False):
#                 print(f"\n{model_type.upper()}: ✓ Available")
#                 print(f"  Default: {info['default_model']}")
#                 print(f"  Available models:")
#                 for model in info['available_models']:
#                     print(f"    - {model}")
#             else:
#                 print(f"\n{model_type.upper()}: ✗ Not Available")
#                 print(f"  Reason: {info.get('error_message', 'Unknown')}")
#         print("\n")
#         return
    
#     # Handle sample creation
#     if args.create_sample:
#         create_sample_questions_file(args.create_sample)
#         return
    
#     # Validate required arguments
#     if not args.image_folder or not args.questions:
#         parser.error("--image_folder and --questions are required (unless using --list_models or --create_sample)")
    
#     # Validate paths
#     if not os.path.exists(args.image_folder):
#         parser.error(f"Image folder not found: {args.image_folder}")
    
#     if not os.path.exists(args.questions):
#         parser.error(f"Questions file not found: {args.questions}")
    
#     # Set device
#     device = None if args.device == 'auto' else args.device
    
#     # Get evaluator
#     evaluator = get_evaluator(
#         model_type=args.model_type,
#         model_id=args.model_id,
#         device=device
#     )
    
#     # Run evaluation
#     results = evaluator.evaluate_mcq_folder(
#         image_folder=args.image_folder,
#         questions_file=args.questions,
#         output_file=args.output,
#         batch_size=args.batch_size,
#         max_new_tokens=args.max_tokens
#     )
    
#     print(f"\nEvaluation completed! {len(results)} questions processed.")


# if __name__ == "__main__":
#     main()



# ---------- Main: parse args and orchestrate flows ----------
def main():
    parser = argparse.ArgumentParser(
        description="VLM MCQ Evaluator with Hugging Face dataset support",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model_type', type=str, default='llava', help='Model type key from registry')
    parser.add_argument('--model_id', type=str, help='Specific model id (optional)')
    parser.add_argument('--image_folder', type=str, help='Path to folder containing images (legacy mode)')
    parser.add_argument('--questions', type=str, help='Path to JSON file with questions (legacy mode)')
    parser.add_argument('--hf_dataset', type=str, help='Hugging Face dataset id to use (e.g. USERNAME/dataset)')
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_dataset_local_cache', help='Local cache root for HF dataset')
    # parser.add_argument('--variant', type=str, choices=['notext','correct','irrelevant','misleading'], default='notext', help='Which image variant to evaluate from HF dataset')
    parser.add_argument('--output', type=str, default='results.json', help='Output results JSON')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--device', type=str, choices=['cuda','cpu','auto'], default='auto')
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--create_sample', type=str, help='Create sample questions JSON and exit')

    args = parser.parse_args()

    # list models handling (reuse your previous listing code)
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 60)
        for model_type, info in MODEL_REGISTRY.items():
            if info.get('available', False):
                print(f"\n{model_type.upper()}: ✓ Available")
                print(f"  Default: {info['default_model']}")
                print(f"  Available models:")
                for model in info['available_models']:
                    print(f"    - {model}")
            else:
                print(f"\n{model_type.upper()}: ✗ Not Available")
                print(f"  Reason: {info.get('error_message', 'Unknown')}")
        return

    if args.create_sample:
        create_sample_questions_file(args.create_sample)
        return

    # Validate model
    device = None if args.device == 'auto' else args.device
    evaluator = get_evaluator(model_type=args.model_type, model_id=args.model_id, device=device)

    # If HF dataset is provided, use that path (takes precedence)
    if args.hf_dataset:
        ds = get_or_download_hf_dataset(args.hf_dataset, local_cache_root=args.hf_cache_dir)

        # Evaluate a single variant OR all variants
        variants_to_run = ['notext', 'correct', 'irrelevant', 'misleading']

        summary = []
        all_results_by_variant = {}

        for v in variants_to_run:
            print(f"\n{'='*60}\nEvaluating HF dataset variant: {v}\n{'='*60}")

            questions_list = build_questions_from_hf_dataset(ds, variant=v)

            # Save each variant to its own file:
            # e.g., results.json -> results_correct.json
            variant_output = safe_suffix(args.output, v)

            results = evaluate_from_questions_list(
                evaluator,
                questions_list,
                output_file=variant_output,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens
            )

            all_results_by_variant[v] = results

            acc = compute_accuracy(results)
            summary.append({
                "variant": v,
                "num_samples": len(results),
                "accuracy_percent": None if acc is None else round(acc, 2),
                "output_file": variant_output
            })

        # Write a summary JSON next to outputs
        summary_path = safe_suffix(args.output, "summary")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved variant summary to: {summary_path}")
        print("\nVariant Summary:")
        for s in summary:
            if s["accuracy_percent"] is None:
                print(f"  - {s['variant']}: {s['num_samples']} samples (no GT answers) -> {s['output_file']}")
            else:
                print(f"  - {s['variant']}: {s['num_samples']} samples, acc={s['accuracy_percent']}% -> {s['output_file']}")

        return

    # Legacy mode: use image_folder + questions JSON
    if not args.image_folder or not args.questions:
        parser.error("--image_folder and --questions are required when --hf_dataset is not provided")

    # Basic path checks
    if not os.path.exists(args.image_folder):
        parser.error(f"Image folder not found: {args.image_folder}")
    if not os.path.exists(args.questions):
        parser.error(f"Questions file not found: {args.questions}")

    # Use existing evaluate_mcq_folder method from evaluator
    results = evaluator.evaluate_mcq_folder(
        image_folder=args.image_folder,
        questions_file=args.questions,
        output_file=args.output,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens
    )
    print(f"\nEvaluation completed! {len(results)} questions processed.")

if __name__ == "__main__":
    main()