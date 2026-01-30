"""
VLM MCQ Evaluator - HF Dataset Version
Evaluates all 4 image variants from Hugging Face dataset.
"""

import torch
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageFile
from transformers import AutoProcessor
from tqdm import tqdm
from abc import ABC, abstractmethod
from datasets import load_dataset, load_from_disk, DatasetDict
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Conditional imports for different models
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
    print("Warning: Qwen2VL not available.")

try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except Exception:
    LLAVA_NEXT_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    INTERNVL_AVAILABLE = True
except Exception:
    INTERNVL_AVAILABLE = False


# ==================== HF Dataset Utilities ====================

def sanitize_repo_id(repo_id: str) -> str:
    """Make a filesystem-safe name for caching the dataset."""
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(
    dataset_id: str, 
    local_cache_root: str = "./hf_dataset_local_cache",
    split: str = "test"
) -> DatasetDict:
    """Download or load cached HF dataset."""
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_repo_id(dataset_id)
    cache_dir = local_cache_root / safe_name

    if cache_dir.exists():
        print(f"Loading dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"Downloading dataset '{dataset_id}' from Hugging Face...")
    ds = load_dataset(dataset_id, split=split)
    
    try:
        ds.save_to_disk(str(cache_dir))
        print(f"Saved dataset to cache: {cache_dir}")
    except Exception as e:
        print(f"Warning: failed to save dataset to disk: {e}")
    
    return ds


def build_questions_from_hf_dataset(ds, variant: str = "notext") -> List[Dict]:
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

    # ← CHANGE THIS LINE FROM:
    # for sample in dataset:
    # ← TO:
    for idx in tqdm(range(len(dataset)), desc=f"Loading {variant} variant"):
        try:
            sample = dataset[idx]  # ← Access by index instead of iteration
            
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
            if img_obj is None:
                print(f"Warning: sample {idx} missing '{variant}' image, skipping. sample id: {sample.get('question_id')}")
                continue

            items.append({
                "image_id": sample.get("question_id", "unknown"),
                "image_input": img_obj, 
                "question": question,
                "options": options,
                "answer": ans,
                "raw_sample": sample
            })
        
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            continue
    
    print(f"✓ Loaded {len(items)}/{len(dataset)} samples for variant '{variant}'")
    return items

def evaluate_from_questions_list(
    evaluator,
    questions_list: List[Dict],
    output_file: str = None,
    batch_size: int = 4,
    max_new_tokens: int = 50,
    variant: str = None
) -> List[Dict]:
    """Evaluate a list of questions with the given evaluator."""
    image_inputs = []
    prompts = []
    
    for item in questions_list:
        image_inputs.append(item["image_input"])
        prompt = evaluator.format_mcq_prompt(item["question"], item["options"])
        prompts.append(prompt)

    print(f"Processing {len(image_inputs)} items...\n")
    
    responses = evaluator.process_batch(
        image_inputs,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        sample_ids=[item.get("image_id", f"unknown_{i}") for i, item in enumerate(questions_list)],
        variant=variant
    )

    results = []
    for idx, item in enumerate(questions_list):
        response = responses[idx]
        predicted_answer = evaluator.extract_answer(response)
        
        result = {
            "image_id": item["image_id"],
            "question": item["question"],
            "options": item["options"],
            "correct_answer": item.get("answer"),
            "predicted_answer": predicted_answer,
            "full_response": response,
            "is_correct": (predicted_answer == item.get("answer")) if item.get("answer") else None
        }
        results.append(result)

    # Compute accuracy
    known = [r for r in results if r["correct_answer"] is not None]
    if known:
        correct = sum(1 for r in known if r["is_correct"])
        total = len(known)
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\n{'='*50}")
        print(f"Evaluation Complete!")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*50}\n")

    if output_file:
        # Don't serialize PIL images to JSON
        json_safe_results = []
        for r in results:
            r_copy = r.copy()
            if "raw_sample" in r_copy:
                del r_copy["raw_sample"]
            json_safe_results.append(r_copy)
        
        with open(output_file, "w") as f:
            json.dump(json_safe_results, f, indent=2)
        print(f"Saved results to: {output_file}")

    return results


def safe_suffix(path: str, suffix: str) -> str:
    """Add suffix before file extension: results.json -> results_correct.json"""
    p = Path(path)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
    return f"{path}_{suffix}"


def compute_accuracy(results: List[Dict]) -> Optional[float]:
    """Compute accuracy in % if correct answers exist."""
    known = [r for r in results if r.get("correct_answer") not in (None, "", "N/A")]
    if not known:
        return None
    correct = sum(1 for r in known if r.get("is_correct") is True)
    total = len(known)
    return (correct / total) * 100 if total else None


def save_generation_logits(gen_out, tokenizer, inputs, output_path: str, top_k: int = 10):
    """Save per-step top-K logits to JSON."""
    records = []
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = gen_out.sequences[0][prompt_len:]

    for step, step_scores in enumerate(gen_out.scores):
        scores = step_scores[0]
        topk_logits, topk_token_ids = torch.topk(scores, k=top_k, dim=-1)

        top_tokens = []
        for logit, tok_id in zip(topk_logits.tolist(), topk_token_ids.tolist()):
            top_tokens.append({
                "token_id": int(tok_id),
                "token": tokenizer.decode([tok_id]),
                "logit": float(logit),
            })

        chosen_token_id = generated_ids[step].item()
        chosen_logit = scores[chosen_token_id].item()

        records.append({
            "step": step,
            "top_tokens": top_tokens,
            "chosen_token": {
                "token_id": int(chosen_token_id),
                "token": tokenizer.decode([chosen_token_id]),
                "logit": float(chosen_logit),
            }
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)


# ==================== Base Evaluator Class ====================

class BaseVLMEvaluator(ABC):
    """Abstract base class for VLM evaluators."""
    
    def __init__(self, model_id: str, device: str = None):
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
        """Load model and processor."""
        pass
    
    @abstractmethod
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs for the model."""
        pass
    
    @abstractmethod
    def _decode_output(self, output) -> str:
        """Decode model output."""
        pass
    
    def load_image(self, image_input) -> Image.Image:
        """
        Load image from various formats.
        Since HF dataset provides PIL.Image.Image, just ensure RGB.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")
    
    def format_mcq_prompt(self, question: str, options: Dict[str, str], 
                          instruction: str = None) -> str:
        """Format MCQ question with options into a prompt."""
        if instruction is None:
            instruction = "Answer the following multiple-choice question by selecting the correct option."
        
        prompt = f"{instruction}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        
        return prompt
    
    def process_single(self, image_input, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False, 
                      sid: str = None, variant: str = None) -> str:
        """Process a single image with a prompt."""
        image = self.load_image(image_input)
        inputs = self._prepare_inputs(image, prompt)
        
        with torch.inference_mode():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
            )

        output_ids = gen_out.sequences
        response = self._decode_output(output_ids)

        # Save logits for debugging
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and sid is not None:
            os.makedirs("logits_debug", exist_ok=True)
            debug_path = os.path.join("logits_debug", f"logits_{variant}_{self.model_id}_{sid}.json")
            save_generation_logits(gen_out, tokenizer, inputs, debug_path)
        
        return response
    
    def process_batch(self, image_inputs: List, prompts: List[str],
                     max_new_tokens: int = 200,
                     do_sample: bool = False,
                     batch_size: int = 4, 
                     sample_ids: List[str] = None, variant: str = None) -> List[str]:
        """Process multiple images with prompts in batches."""
        assert len(image_inputs) == len(prompts), "Number of images and prompts must match"
        
        all_responses = []
        
        for i in tqdm(range(0, len(image_inputs), batch_size), desc="Processing batches"):
            batch_images = image_inputs[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            sids = sample_ids[i:i+batch_size] if sample_ids else [None] * len(batch_images)
            
            for img_input, prompt, sid in zip(batch_images, batch_prompts, sids):
                response = self.process_single(img_input, prompt, max_new_tokens, do_sample, sid, variant=variant)
                all_responses.append(response)
        
        return all_responses
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer letter (A, B, C, or D) from model response."""
        import re
        
        assistant_response = response
        markers = ["ASSISTANT:", "Assistant:", "assistant:"]
        last_position = -1
        found_marker = None
        
        for marker in markers:
            pos = response.rfind(marker)
            if pos > last_position:
                last_position = pos
                found_marker = marker
        
        if found_marker:
            assistant_response = response[last_position + len(found_marker):].strip()
        
        assistant_response_upper = assistant_response.upper()
        
        patterns = [
            r'ANSWER[:\s]+([ABCD])\b',
            r'^\s*([ABCD])\s*$',
            r'^([ABCD])\b',
            r'\b([ABCD])\s*$',
            r'\b([ABCD])\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, assistant_response_upper)
            if match:
                return match.group(1)
        
        return 'UNKNOWN'


# ==================== Model-Specific Evaluators ====================

class LlavaEvaluator(BaseVLMEvaluator):
    """LLaVA evaluator implementation."""
    
    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
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
        return self.processor.decode(output[0], skip_special_tokens=True)


class QwenVLEvaluator(BaseVLMEvaluator):
    """Qwen-VL evaluator implementation."""
    
    def __init__(self, model_id: str, device: str = None):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen-VL requires: pip install qwen-vl-utils torchvision")
        super().__init__(model_id, device)
    
    def _load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        return inputs.to(self.device)
    
    def _decode_output(self, output) -> str:
        return self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class LlavaNextEvaluator(BaseVLMEvaluator):
    """LLaVA-NeXT (v1.6) evaluator implementation."""
    
    def __init__(self, model_id: str, device: str = None):
        if not LLAVA_NEXT_AVAILABLE:
            raise ImportError("LLaVA-NeXT requires: pip install -U transformers")
        super().__init__(model_id, device)
    
    def _load_model(self):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
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
        return self.processor.decode(output[0], skip_special_tokens=True)


class InternVLEvaluator(BaseVLMEvaluator):
    """InternVL3.5 evaluator implementation."""
    
    def __init__(self, model_id: str, device: str = None):
        if not INTERNVL_AVAILABLE:
            raise ImportError("InternVL requires: pip install torchvision einops timm")
        super().__init__(model_id, device)
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
    
    def process_single(self, image_input, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False, 
                      sid: str = None) -> str:
        """Process a single image with InternVL."""
        image = self.load_image(image_input)
        pixel_values = self.load_image_func(image, max_num=12).to(torch.bfloat16).to(self.device)
        
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
        
        full_response = f"User: {prompt}\nAssistant: {response}"
        return full_response

# ==================== Model Registry ====================

MODEL_REGISTRY = {
    'llava': {
        'class': LlavaEvaluator,
        'default_model': 'llava-hf/llava-1.5-7b-hf',
        'available_models': [
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-1.5-13b-hf',
        ],
        'available': LLAVA_AVAILABLE
    },
}

if QWEN_AVAILABLE:
    MODEL_REGISTRY['qwen-vl'] = {
        'class': QwenVLEvaluator,
        'default_model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'available_models': [
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-7B-Instruct',
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
        'error_message': 'InternVL requires: pip install torchvision einops timm'
    }


def get_evaluator(model_type: str, model_id: str = None, device: str = None) -> BaseVLMEvaluator:
    """Factory function to get the appropriate evaluator."""
    if model_type not in MODEL_REGISTRY:
        available = [k for k, v in MODEL_REGISTRY.items() if v.get('available', False)]
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_info = MODEL_REGISTRY[model_type]
    
    if not model_info.get('available', False):
        raise ImportError(f"Model {model_type} not available")
    
    evaluator_class = model_info['class']
    if model_id is None:
        model_id = model_info['default_model']
    
    return evaluator_class(model_id=model_id, device=device)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="VLM MCQ Evaluator for HF datasets")
    
    parser.add_argument('--model_type', type=str, default='llava')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--hf_dataset', type=str, required=True, help='HF dataset ID (e.g., AHAAM/CIM)')
    parser.add_argument('--hf_cache_dir', type=str, default='./hf_dataset_local_cache')
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--device', type=str, choices=['cuda','cpu','auto'], default='auto')
    parser.add_argument('--list_models', action='store_true')

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Models:")
        for model_type, info in MODEL_REGISTRY.items():
            status = "✓" if info.get('available') else "✗"
            print(f"  {model_type}: {status}")
        return

    device = None if args.device == 'auto' else args.device
    evaluator = get_evaluator(model_type=args.model_type, model_id=args.model_id, device=device)

    # Load HF dataset
    ds = get_or_download_hf_dataset(args.hf_dataset, local_cache_root=args.hf_cache_dir, split="test")

    # Evaluate all 4 variants
    variants = ['notext', 'correct', 'irrelevant', 'misleading']
    summary = []

    for variant in variants:
        print(f"\n{'='*60}\nEvaluating variant: {variant}\n{'='*60}")
        
        questions_list = build_questions_from_hf_dataset(ds, variant=variant)
        variant_output = safe_suffix(args.output, variant)
        
        results = evaluate_from_questions_list(
            evaluator,
            questions_list,
            output_file=variant_output,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
            variant = variant
        )
        
        acc = compute_accuracy(results)
        summary.append({
            "variant": variant,
            "num_samples": len(results),
            "accuracy_percent": round(acc, 2) if acc is not None else None,
            "output_file": variant_output
        })

    # Save summary
    summary_path = safe_suffix(args.output, "summary")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("All variants evaluated!")
    print(f"Summary saved to: {summary_path}")
    for s in summary:
        acc_str = f"{s['accuracy_percent']}%" if s['accuracy_percent'] is not None else "N/A"
        print(f"  {s['variant']}: {s['num_samples']} samples, accuracy={acc_str}")


if __name__ == "__main__":
    main()