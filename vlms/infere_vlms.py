"""
VLM MCQ Evaluator - Modular Version with Multiple Model Support
Supports LLaVA, Qwen-VL, and other vision-language models
"""

import torch
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from abc import ABC, abstractmethod

# Conditional imports for different models
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen2VL not available. Install with: pip install qwen-vl-utils torchvision")


try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except ImportError:
    LLAVA_NEXT_AVAILABLE = False

try:
    from transformers import AutoModel
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    INTERNVL_AVAILABLE = True
except ImportError:
    INTERNVL_AVAILABLE = False

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
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample
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
        response = self._decode_output(output_ids)
        
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


def main():
    """Main function to run MCQ evaluation from command line."""
    parser = argparse.ArgumentParser(
        description="VLM MCQ Evaluator - Evaluate multiple-choice questions with various vision-language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with LLaVA (default)
  python script.py --model_type llava --image_folder ./images/ --questions questions.json
  
  # Evaluate with Qwen-VL
  python script.py --model_type qwen-vl --image_folder ./images/ --questions questions.json
  
  # Use specific model
  python script.py --model_type llava --model_id llava-hf/llava-1.5-13b-hf --image_folder ./images/ --questions questions.json
  
  # List available models
  python script.py --list_models
  
  # Create sample questions file
  python script.py --create_sample sample_questions.json
        """
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default='llava',
        help='Type of VLM to use (default: llava)'
    )
    
    parser.add_argument(
        '--model_id',
        type=str,
        help='Specific HuggingFace model ID (optional, uses default for model_type if not provided)'
    )
    
    parser.add_argument(
        '--image_folder',
        type=str,
        help='Path to folder containing images'
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        help='Path to JSON file with questions'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Path to save results JSON (default: results.json)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for processing (default: 4)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate per answer (default: 50)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List all available models and exit'
    )
    
    parser.add_argument(
        '--create_sample',
        type=str,
        metavar='OUTPUT_FILE',
        help='Create a sample questions JSON file and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list models
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
        print("\n")
        return
    
    # Handle sample creation
    if args.create_sample:
        create_sample_questions_file(args.create_sample)
        return
    
    # Validate required arguments
    if not args.image_folder or not args.questions:
        parser.error("--image_folder and --questions are required (unless using --list_models or --create_sample)")
    
    # Validate paths
    if not os.path.exists(args.image_folder):
        parser.error(f"Image folder not found: {args.image_folder}")
    
    if not os.path.exists(args.questions):
        parser.error(f"Questions file not found: {args.questions}")
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    # Get evaluator
    evaluator = get_evaluator(
        model_type=args.model_type,
        model_id=args.model_id,
        device=device
    )
    
    # Run evaluation
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










# def extract_answer(self, response: str) -> str:
#         """
#         Extract the answer letter (A, B, C, or D) from model response.
        
#         Args:
#             response: Full model response
            
#         Returns:
#             Extracted answer letter or 'UNKNOWN'
#         """
#         import re
        
#         # First, try to extract only the assistant's actual response
#         assistant_response = response
        
#         # Handle different assistant markers (case-insensitive)
#         # Look for the LAST occurrence of assistant marker
#         markers = ["ASSISTANT:", "Assistant:", "assistant:"]
#         last_position = -1
#         found_marker = None
        
#         for marker in markers:
#             pos = response.rfind(marker)  # rfind = find last occurrence
#             if pos > last_position:
#                 last_position = pos
#                 found_marker = marker
        
#         if found_marker:
#             assistant_response = response[last_position + len(found_marker):].strip()
        
#         # Convert to uppercase for pattern matching
#         assistant_response_upper = assistant_response.upper()
        
#         # Try to find answer patterns in order of specificity
#         patterns = [
#             r'ANSWER[:\s]+([ABCD])\b',  # Answer: A or Answer A
#             r'^\s*([ABCD])\s*$',  # Just A on a line by itself
#             r'^([ABCD])\b',  # A at the start
#             r'\b([ABCD])\s*$',  # A at the end
#             r'\b([ABCD])\b',  # Standalone A, B, C, or D (last resort)
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, assistant_response_upper)
#             if match:
#                 return match.group(1)
        
#         return 'UNKNOWN'