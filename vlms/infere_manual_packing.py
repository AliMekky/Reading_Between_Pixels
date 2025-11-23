# """
# VLM MCQ Evaluator - WITH MANUAL VISION/TEXT PACKING

# This version manually packs vision and text to test if we get merged sequences.
# """

# import torch
# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict, Union, Optional
# from PIL import Image
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from tqdm import tqdm
# from abc import ABC, abstractmethod

# try:
#     from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
#     LLAVA_NEXT_AVAILABLE = True
# except ImportError:
#     LLAVA_NEXT_AVAILABLE = False


# class BaseVLMEvaluator(ABC):
#     """Abstract base class for VLM evaluators."""
    
#     def __init__(self, model_id: str, device: str = None, use_manual_packing: bool = False):
#         self.model_id = model_id
#         self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         self.use_manual_packing = use_manual_packing
        
#         print(f"Initializing {self.__class__.__name__} on {self.device}...")
#         print(f"Manual packing: {'ENABLED' if use_manual_packing else 'DISABLED'}")
#         print(f"Loading model: {model_id}")
        
#         self.model = None
#         self.processor = None
#         self._load_model()
        
#         print("Model loaded successfully!\n")
    
#     @abstractmethod
#     def _load_model(self):
#         pass
    
#     @abstractmethod
#     def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
#         pass
    
#     @abstractmethod
#     def _decode_output(self, output) -> str:
#         pass
    
#     def load_image(self, image_path: str) -> Image.Image:
#         return Image.open(image_path).convert('RGB')
    
#     def format_mcq_prompt(self, question: str, options: Dict[str, str], 
#                           instruction: str = None) -> str:
#         if instruction is None:
#             instruction = "Answer the following multiple-choice question by selecting the correct option."
        
#         prompt = f"{instruction}\n\n"
#         prompt += f"Question: {question}\n\n"
#         prompt += "Options:\n"
#         for key, value in options.items():
#             prompt += f"{key}) {value}\n"
#         prompt += "\nAnswer with only the letter (A, B, C, or D):"
        
#         return prompt
    
#     def extract_answer(self, response: str) -> str:
#         import re
        
#         assistant_response = response
#         markers = ["ASSISTANT:", "Assistant:", "assistant:"]
#         last_position = -1
#         found_marker = None
        
#         for marker in markers:
#             pos = response.rfind(marker)
#             if pos > last_position:
#                 last_position = pos
#                 found_marker = marker
        
#         if found_marker:
#             assistant_response = response[last_position + len(found_marker):].strip()
        
#         assistant_response_upper = assistant_response.upper()
        
#         patterns = [
#             r'ANSWER[:\s]+([ABCD])\b',
#             r'^\s*([ABCD])\s*$',
#             r'^([ABCD])\b',
#             r'\b([ABCD])\s*$',
#             r'\b([ABCD])\b',
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, assistant_response_upper)
#             if match:
#                 return match.group(1)
        
#         return 'UNKNOWN'
    
#     def process_single(self, image_path: str, prompt: str, 
#                       max_new_tokens: int = 200, 
#                       do_sample: bool = False) -> str:
#         """Process single image - will use manual packing if enabled."""
#         raise NotImplementedError("Subclass must implement process_single")


# class LlavaNextEvaluator(BaseVLMEvaluator):
#     """LLaVA-NeXT evaluator with manual packing support."""
    
#     def __init__(self, model_id: str, device: str = None, use_manual_packing: bool = False):
#         if not LLAVA_NEXT_AVAILABLE:
#             raise ImportError("LLaVA-NeXT requires: pip install -U transformers")
#         super().__init__(model_id, device, use_manual_packing)
    
#     def _load_model(self):
#         from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
#         self.model = LlavaNextForConditionalGeneration.from_pretrained(
#             self.model_id,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )
#         self.model.to(self.device)
#         self.model.eval()
        
#         self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
    
#     def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
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
#             conversation, add_generation_prompt=True
#         )
        
#         inputs = self.processor(images=image, text=formatted_prompt, return_tensors='pt')
#         inputs = inputs.to(self.device)
        
#         if 'pixel_values' in inputs:
#             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
#         return inputs
    
#     def _decode_output(self, output) -> str:
#         return self.processor.decode(output[0], skip_special_tokens=True)
    
#     def _manually_pack_vision_text(self, inputs: Dict) -> Dict:
#         """
#         Manually pack vision and text embeddings.
        
#         Returns:
#             Dictionary with inputs_embeds and attention_mask for language model
#         """
#         print("\n  === MANUAL PACKING ===")
        
#         # Step 1: Get vision features from model
#         with torch.no_grad():
#             outputs = self.model(
#                 **inputs,
#                 output_hidden_states=True,
#                 return_dict=True
#             )
#             vision_features = outputs.image_hidden_states  # [num_vision_tokens, hidden_dim]
        
#         print(f"  Vision features shape: {vision_features.shape}")
        
#         # Step 2: Get text embeddings
#         input_ids = inputs['input_ids']
#         text_embeds = self.model.get_input_embeddings()(input_ids)  # [1, seq_len, hidden_dim]
        
#         print(f"  Text embeddings shape: {text_embeds.shape}")
#         print(f"  Input_ids shape: {input_ids.shape}")
        
#         # Step 3: Find <image> token positions
#         image_token_id = self.model.config.image_token_index
#         image_mask = (input_ids == image_token_id)  # [1, seq_len]
#         num_image_tokens = image_mask.sum().item()
        
#         print(f"  Image token ID: {image_token_id}")
#         print(f"  Number of <image> tokens: {num_image_tokens}")
#         print(f"  Vision features available: {vision_features.shape[0]}")
        
#         # Step 4: Replace <image> token embeddings with vision features
#         if num_image_tokens != vision_features.shape[0]:
#             print(f"  ⚠️  WARNING: Mismatch between <image> tokens ({num_image_tokens}) and vision features ({vision_features.shape[0]})")
#             print(f"  This might cause issues!")
        
#         # Clone text embeddings
#         inputs_embeds = text_embeds.clone()
        
#         # Replace <image> positions with vision features
#         # image_mask is [1, seq_len], we need to replace those positions
#         inputs_embeds[image_mask] = vision_features.to(inputs_embeds.dtype)
        
#         print(f"  Final inputs_embeds shape: {inputs_embeds.shape}")
        
#         # Step 5: Create attention mask (all ones since we have real content everywhere)
#         attention_mask = inputs['attention_mask']
        
#         print(f"  Attention mask shape: {attention_mask.shape}")
#         print(f"  === PACKING COMPLETE ===\n")
        
#         return {
#             'inputs_embeds': inputs_embeds,
#             'attention_mask': attention_mask
#         }
    
#     def process_single(self, image_path: str, prompt: str, 
#                       max_new_tokens: int = 200, 
#                       do_sample: bool = False) -> str:
#         """Process single image with optional manual packing."""
        
#         # Load image and prepare inputs
#         image = self.load_image(image_path)
#         inputs = self._prepare_inputs(image, prompt)
        
#         if self.use_manual_packing:
#             print("Using MANUAL PACKING mode")
            
#             # Manually pack vision and text
#             packed_inputs = self._manually_pack_vision_text(inputs)
            
#             # Generate using FULL MODEL but pass inputs_embeds
#             # Remove input_ids since we're using inputs_embeds
#             with torch.inference_mode():
#                 output_ids = self.model.generate(
#                     inputs_embeds=packed_inputs['inputs_embeds'],
#                     attention_mask=packed_inputs['attention_mask'],
#                     max_new_tokens=max_new_tokens,
#                     do_sample=do_sample,
#                     # Don't pass input_ids when using inputs_embeds
#                 )
#         else:
#             print("Using STANDARD mode (model's automatic packing)")
            
#             # Generate normally
#             with torch.inference_mode():
#                 output_ids = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     do_sample=do_sample
#                 )
        
#         # Decode output
#         response = self._decode_output(output_ids)
        
#         return response
    
#     def evaluate_mcq_folder(self, image_folder: str, questions_file: str,
#                            output_file: str = None,
#                            max_new_tokens: int = 50) -> List[Dict]:
#         """Evaluate MCQ questions from a folder."""
        
#         # Load questions
#         with open(questions_file, 'r') as f:
#             questions_data = json.load(f)
        
#         print(f"Loaded {len(questions_data)} questions from {questions_file}")
        
#         results = []
        
#         for idx, item in enumerate(tqdm(questions_data, desc="Processing questions")):
#             image_path = os.path.join(image_folder, item['image'])
#             if not os.path.exists(image_path):
#                 print(f"Warning: Image not found: {image_path}")
#                 continue
            
#             prompt = self.format_mcq_prompt(item['question'], item['options'])
            
#             # Process
#             try:
#                 response = self.process_single(
#                     image_path,
#                     prompt,
#                     max_new_tokens=max_new_tokens
#                 )
                
#                 # Extract answer
#                 predicted_answer = self.extract_answer(response)
                
#                 result = {
#                     "image": item['image'],
#                     "question": item['question'],
#                     "options": item['options'],
#                     "correct_answer": item.get('answer', 'N/A'),
#                     "predicted_answer": predicted_answer,
#                     "full_response": response,
#                     "is_correct": predicted_answer == item.get('answer', None)
#                 }
#                 results.append(result)
                
#             except Exception as e:
#                 print(f"\nError processing {item['image']}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         # Calculate accuracy
#         if any('answer' in item for item in questions_data):
#             correct = sum(1 for r in results if r['is_correct'])
#             total = len(results)
#             accuracy = correct / total * 100 if total > 0 else 0
#             print(f"\n{'='*50}")
#             print(f"Evaluation Complete!")
#             print(f"Correct: {correct}/{total}")
#             print(f"Accuracy: {accuracy:.2f}%")
#             print(f"{'='*50}\n")
        
#         # Save results
#         if output_file:
#             with open(output_file, 'w') as f:
#                 json.dump(results, f, indent=2)
#             print(f"Results saved to: {output_file}")
        
#         return results


# def main():
#     """Main function."""
#     parser = argparse.ArgumentParser(
#         description="VLM MCQ Evaluator with Manual Packing Test"
#     )
    
#     parser.add_argument(
#         '--model_id',
#         type=str,
#         default='llava-hf/llava-v1.6-mistral-7b-hf',
#         help='HuggingFace model ID'
#     )
    
#     parser.add_argument(
#         '--image_folder',
#         type=str,
#         required=True,
#         help='Path to folder containing images'
#     )
    
#     parser.add_argument(
#         '--questions',
#         type=str,
#         required=True,
#         help='Path to JSON file with questions'
#     )
    
#     parser.add_argument(
#         '--output',
#         type=str,
#         default='results_manual_packing.json',
#         help='Path to save results'
#     )
    
#     parser.add_argument(
#         '--max_tokens',
#         type=int,
#         default=50,
#         help='Maximum tokens to generate'
#     )
    
#     parser.add_argument(
#         '--device',
#         type=str,
#         choices=['cuda', 'cpu', 'auto'],
#         default='auto',
#         help='Device to use'
#     )
    
#     parser.add_argument(
#         '--use_manual_packing',
#         action='store_true',
#         help='Use manual vision/text packing instead of model default'
#     )
    
#     args = parser.parse_args()
    
#     # Validate paths
#     if not os.path.exists(args.image_folder):
#         parser.error(f"Image folder not found: {args.image_folder}")
    
#     if not os.path.exists(args.questions):
#         parser.error(f"Questions file not found: {args.questions}")
    
#     # Set device
#     device = None if args.device == 'auto' else args.device
    
#     print("="*70)
#     print("VLM MCQ EVALUATOR - MANUAL PACKING TEST")
#     print("="*70)
#     print(f"Manual packing: {'ENABLED ✓' if args.use_manual_packing else 'DISABLED (standard mode)'}")
#     print("="*70 + "\n")
    
#     # Create evaluator
#     evaluator = LlavaNextEvaluator(
#         model_id=args.model_id,
#         device=device,
#         use_manual_packing=args.use_manual_packing
#     )
    
#     # Run evaluation
#     results = evaluator.evaluate_mcq_folder(
#         image_folder=args.image_folder,
#         questions_file=args.questions,
#         output_file=args.output,
#         max_new_tokens=args.max_tokens
#     )
    
#     print(f"\nEvaluation completed! {len(results)} questions processed.")


# if __name__ == "__main__":
#     main()


"""
VLM MCQ Evaluator - WITH MANUAL VISION/TEXT PACKING

This version manually packs vision and text to test if we get merged sequences.
"""

import torch
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from abc import ABC, abstractmethod

try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except ImportError:
    LLAVA_NEXT_AVAILABLE = False


class BaseVLMEvaluator(ABC):
    """Abstract base class for VLM evaluators."""
    
    def __init__(self, model_id: str, device: str = None, use_manual_packing: bool = False):
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_manual_packing = use_manual_packing
        
        print(f"Initializing {self.__class__.__name__} on {self.device}...")
        print(f"Manual packing: {'ENABLED' if use_manual_packing else 'DISABLED'}")
        print(f"Loading model: {model_id}")
        
        self.model = None
        self.processor = None
        self._load_model()
        
        print("Model loaded successfully!\n")
    
    @abstractmethod
    def _load_model(self):
        pass
    
    @abstractmethod
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        pass
    
    @abstractmethod
    def _decode_output(self, output) -> str:
        pass
    
    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert('RGB')
    
    def format_mcq_prompt(self, question: str, options: Dict[str, str], 
                          instruction: str = None) -> str:
        if instruction is None:
            instruction = "Answer the following multiple-choice question by selecting the correct option."
        
        prompt = f"{instruction}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        
        return prompt
    
    def extract_answer(self, response: str) -> str:
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
    
    def process_single(self, image_path: str, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False) -> str:
        """Process single image - will use manual packing if enabled."""
        raise NotImplementedError("Subclass must implement process_single")


class LlavaNextEvaluator(BaseVLMEvaluator):
    """LLaVA-NeXT evaluator with manual packing support."""
    
    def __init__(self, model_id: str, device: str = None, use_manual_packing: bool = False):
        if not LLAVA_NEXT_AVAILABLE:
            raise ImportError("LLaVA-NeXT requires: pip install -U transformers")
        super().__init__(model_id, device, use_manual_packing)
    
    def _load_model(self):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
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
    
    def _manually_pack_vision_text(self, inputs: Dict, verbose: bool = True):
        """
        Manually pack vision and text embeddings - IMPROVED VERSION.
        
        Args:
            inputs: Model inputs dictionary
            verbose: Print debugging info
            
        Returns:
            inputs_embeds: Packed embeddings [1, S_full, H]
            attention_mask: Attention mask [1, S_full]
            vision_span: (start_idx, end_idx) half-open range of vision positions
        """
        if verbose:
            print("\n  === MANUAL PACKING (IMPROVED) ===")
        
        # 1) Get projected vision features from the model's own path (stable)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            vision_features = out.image_hidden_states  # [N_vision, H]
        
        # 2) Text embeddings
        input_ids = inputs["input_ids"]        # [1, S_text]
        text_embeds = self.model.get_input_embeddings()(input_ids)  # [1, S_text, H]
        attn = inputs.get("attention_mask", torch.ones_like(input_ids))
        
        # 3) Locate <image> positions
        image_tok_id = self.model.config.image_token_index
        mask = (input_ids == image_tok_id)           # [1, S_text]
        num_placeholders = int(mask.sum().item())
        N_vision = int(vision_features.shape[0])
        H = int(text_embeds.shape[-1])
        
        if verbose:
            print(f"  vision_features: {tuple(vision_features.shape)}")
            print(f"  text_embeds    : {tuple(text_embeds.shape)}")
            print(f"  placeholders   : {num_placeholders}")
        
        if num_placeholders == N_vision and N_vision > 0:
            # Case A: already expanded → masked replacement is correct
            inputs_embeds = text_embeds.clone()
            inputs_embeds[mask] = vision_features.to(inputs_embeds.dtype)
            
            # Vision span is the (possibly non-contiguous) set of positions
            idxs = mask.nonzero(as_tuple=False)[:, 1].tolist()
            v_start, v_end = min(idxs), max(idxs) + 1
            
            if verbose and (len(idxs) != (v_end - v_start)):
                print("  ⚠ vision positions are not contiguous (multiple blocks).")
            
            attention_mask = attn
            vision_span = (v_start, v_end)
            
        elif num_placeholders == 1 and N_vision > 0:
            # Case B: NOT expanded → splice the entire vision block
            image_pos = int(mask.nonzero(as_tuple=False)[0, 1].item())
            before = text_embeds[:, :image_pos, :]                     # [1, A, H]
            after  = text_embeds[:, image_pos+1:, :]                   # [1, B, H]
            vfeat  = vision_features.unsqueeze(0).to(text_embeds.dtype)  # [1, N_vision, H]
            inputs_embeds = torch.cat([before, vfeat, after], dim=1)   # [1, A+N_vision+B, H]
            
            # Expand attention mask accordingly
            attn_before = attn[:, :image_pos]
            attn_after  = attn[:, image_pos+1:]
            attn_vis    = torch.ones(attn.shape[0], N_vision, device=attn.device, dtype=attn.dtype)
            attention_mask = torch.cat([attn_before, attn_vis, attn_after], dim=1)
            
            vision_span = (image_pos, image_pos + N_vision)
            
        else:
            raise ValueError(
                f"Cannot pack: placeholders={num_placeholders}, vision_features={N_vision}. "
                "Either processor must expand to N_vision placeholders, or there must be exactly one placeholder."
            )
        
        if verbose:
            S_full = inputs_embeds.shape[1]
            print(f"  packed inputs_embeds: {tuple(inputs_embeds.shape)}")
            print(f"  packed attention    : {tuple(attention_mask.shape)}")
            print(f"  vision_span         : [{vision_span[0]}:{vision_span[1]}) (len={vision_span[1]-vision_span[0]})")
            print(f"  === PACKING COMPLETE ===\n")
        
        return inputs_embeds, attention_mask, vision_span
    
    def process_single(self, image_path: str, prompt: str, 
                      max_new_tokens: int = 200, 
                      do_sample: bool = False) -> str:
        """Process single image with optional manual packing."""
        
        # Load image and prepare inputs
        image = self.load_image(image_path)
        inputs = self._prepare_inputs(image, prompt)
        
        if self.use_manual_packing:
            print("Using MANUAL PACKING mode")
            
            # Manually pack vision and text
            inputs_embeds, attention_mask, vision_span = self._manually_pack_vision_text(inputs)
            
            # Generate using FULL MODEL but pass inputs_embeds
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
        else:
            print("Using STANDARD mode (model's automatic packing)")
            
            # Generate normally
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample
                )
        
        # Decode output
        response = self._decode_output(output_ids)
        
        return response
    
    def evaluate_mcq_folder(self, image_folder: str, questions_file: str,
                           output_file: str = None,
                           max_new_tokens: int = 50) -> List[Dict]:
        """Evaluate MCQ questions from a folder."""
        
        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        print(f"Loaded {len(questions_data)} questions from {questions_file}")
        
        results = []
        
        for idx, item in enumerate(tqdm(questions_data, desc="Processing questions")):
            image_path = os.path.join(image_folder, item['image'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            prompt = self.format_mcq_prompt(item['question'], item['options'])
            
            # Process
            try:
                response = self.process_single(
                    image_path,
                    prompt,
                    max_new_tokens=max_new_tokens
                )
                
                # Extract answer
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
                
            except Exception as e:
                print(f"\nError processing {item['image']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
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
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="VLM MCQ Evaluator with Manual Packing Test"
    )
    
    parser.add_argument(
        '--model_id',
        type=str,
        default='llava-hf/llava-v1.6-mistral-7b-hf',
        help='HuggingFace model ID'
    )
    
    parser.add_argument(
        '--image_folder',
        type=str,
        required=True,
        help='Path to folder containing images'
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        required=True,
        help='Path to JSON file with questions'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results_manual_packing.json',
        help='Path to save results'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use'
    )
    
    parser.add_argument(
        '--use_manual_packing',
        action='store_true',
        help='Use manual vision/text packing instead of model default'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.image_folder):
        parser.error(f"Image folder not found: {args.image_folder}")
    
    if not os.path.exists(args.questions):
        parser.error(f"Questions file not found: {args.questions}")
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    print("="*70)
    print("VLM MCQ EVALUATOR - MANUAL PACKING TEST")
    print("="*70)
    print(f"Manual packing: {'ENABLED ✓' if args.use_manual_packing else 'DISABLED (standard mode)'}")
    print("="*70 + "\n")
    
    # Create evaluator
    evaluator = LlavaNextEvaluator(
        model_id=args.model_id,
        device=device,
        use_manual_packing=args.use_manual_packing
    )
    
    # Run evaluation
    results = evaluator.evaluate_mcq_folder(
        image_folder=args.image_folder,
        questions_file=args.questions,
        output_file=args.output,
        max_new_tokens=args.max_tokens
    )
    
    print(f"\nEvaluation completed! {len(results)} questions processed.")


if __name__ == "__main__":
    main()