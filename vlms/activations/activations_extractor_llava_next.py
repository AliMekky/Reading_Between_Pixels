"""
VLM MCQ Evaluator with Activation Extraction - Updated with Multiple Token Windows
Extracts from: vision tokens, after-vision tokens, decision tokens, and single token
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

# Conditional imports
try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except ImportError:
    LLAVA_NEXT_AVAILABLE = False
    print("Error: LLaVA-NeXT not available. Install with: pip install -U transformers")


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


class LlavaNextEvaluator(BaseVLMEvaluator):
    """LLaVA-NeXT evaluator with activation extraction from multiple token windows."""
    
    def __init__(self, model_id: str, device: str = None):
        if not LLAVA_NEXT_AVAILABLE:
            raise ImportError("LLaVA-NeXT requires: pip install -U transformers")
        super().__init__(model_id, device)
        self.answer_token_ids = None
        self._initialize_answer_tokens()
    
    def _initialize_answer_tokens(self):
        self.answer_token_ids = set()
        for letter in ['A', 'B', 'C', 'D']:
            token_ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if token_ids:
                self.answer_token_ids.add(token_ids[0])
            token_ids_space = self.processor.tokenizer.encode(f" {letter}", add_special_tokens=False)
            if token_ids_space:
                self.answer_token_ids.add(token_ids_space[-1])
    
    def _load_model(self):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        vision_config = self.model.vision_tower.config
        print(vision_config)
        
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
        out = self.processor.decode(inputs["input_ids"][0])

        # print the total number of tokens
        num_tokens = inputs["input_ids"].shape[1]
        print(f"  Total input tokens: {num_tokens}")
        # with open(f"decoded_input_ids_{num_tokens}.txt", "w", encoding="utf-8") as f:
        #     f.write(out)

        print(f"  length pixel values:\n{inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}\n")
    
        

        inputs = inputs.to(self.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def extract_activations_with_answer(
        self, 
        image_path: str, 
        prompt: str,
        max_new_tokens: int = 50,
        do_sample: bool = False
    ) -> Dict:
        """Extract activations from prefill phase with windows based on <image> tokens."""
        image = self.load_image(image_path)
        inputs = self._prepare_inputs(image, prompt)

        # We'll need input_ids on CPU to inspect token positions
        input_ids = inputs["input_ids"][0].detach().cpu()  # [seq_len]
        # Use tokenizer to get the image token id (should be 32000 for LLaVA-NeXT)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        print(f"  <image> token id: {image_token_id}")

        # Dynamically find all vision token positions
        vision_positions = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        print(f"  Found {len(vision_positions)} vision tokens")

        hidden_states_all_layers: Dict[int, torch.Tensor] = {}

        def hidden_state_hook(module, input, output, layer_idx):
            """
            Collect hidden states from each layer.
            output[0] has shape [batch_size, seq_len, hidden_dim]
            """
            hidden_states_all_layers[layer_idx] = output[0].detach().cpu()

        # Register hooks on all decoder layers
        hooks = []
        decoder_layers = self.model.model.language_model.layers

        print(f"  Registering hooks on {len(decoder_layers)} layers...")
        for idx, layer in enumerate(decoder_layers):
            hook = layer.register_forward_hook(
                lambda module, inp, out, idx=idx: hidden_state_hook(module, inp, out, idx)
            )
            hooks.append(hook)

        # Forward pass to get hidden states for the *prefill* (no generation)
        with torch.inference_mode():
            _ = self.model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=False,  # we use our own hooks
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not hidden_states_all_layers:
            raise RuntimeError("No hidden states were collected from hooks.")

        # Determine sequence length from one layer
        example_layer = next(iter(hidden_states_all_layers.values()))


        seq_len = hidden_states_all_layers[0].shape[0]
        print(seq_len)

        print(f"  Sequence length: {seq_len}")

        # Now generate the answer (no hooks needed)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences

        # outputs.hidden_states is a tuple of tuples
        # Structure: (step1, step2, ...) where each step has (layer0, layer1, ..., layerN)
        # We want the FIRST generated token (step 0)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            first_gen_token_states = outputs.hidden_states[0]  # First generation step
            
            # Extract from each layer for the decision token
            decision_token_hidden = []
            for layer_states in first_gen_token_states:
                # layer_states shape: [batch, 1, hidden_dim] (since it's one new token)
                decision_token_hidden.append(layer_states[0, -1, :].detach().cpu())
            


        response = self._decode_output(output_ids)

        results: Dict[str, Union[str, int, List[torch.Tensor]]] = {
            "response": response,
            "seq_len": seq_len,
            "predicted_answer": self.extract_answer(response),
        }
                # Add to results as a new window
        results["hidden_states_decision_token"] = decision_token_hidden

        # ----- Define dynamic windows -----
        # 1) single_token: last token in the input sequence
        single_token_positions = [seq_len - 1] if seq_len > 0 else []

        # 2) vision_tokens: all <image> positions
        vision_token_positions = vision_positions

        # 3) last_vision_token
        if vision_positions:
            last_vision_token_positions = [max(vision_positions)]
        else:
            last_vision_token_positions = []

        # 4) all_tokens: 0 .. seq_len-1
        all_token_positions = list(range(seq_len))

        windows: Dict[str, List[int]] = {
            "single_token": single_token_positions,
            "vision_tokens": vision_token_positions,
            "last_vision_token": last_vision_token_positions,
            "all_tokens": all_token_positions,
        }

        # ----- Extract and average hidden states for each window -----
        for window_name, token_positions in windows.items():
            hidden_averaged_per_layer: List[torch.Tensor] = []

            for layer_idx in sorted(hidden_states_all_layers.keys()):
                layer_hidden = hidden_states_all_layers[layer_idx]

                tokens_in_window: List[torch.Tensor] = []

                if not token_positions:
                    # No tokens in this window (e.g., no vision tokens)
                    averaged_hidden = torch.zeros(layer_hidden.shape[-1])
                    hidden_averaged_per_layer.append(averaged_hidden)
                    continue

                if layer_hidden.dim() == 3:
                    # [batch, seq, hidden]
                    for pos in token_positions:
                        if 0 <= pos < layer_hidden.shape[1]:
                            tokens_in_window.append(layer_hidden[0, pos, :])
                elif layer_hidden.dim() == 2:
                    # [seq, hidden]
                    for pos in token_positions:
                        if 0 <= pos < layer_hidden.shape[0]:
                            tokens_in_window.append(layer_hidden[pos, :])
                else:
                    raise ValueError(f"Unexpected hidden state shape: {layer_hidden.shape}")

                if tokens_in_window:
                    averaged_hidden = torch.stack(tokens_in_window, dim=0).mean(dim=0)
                else:
                    averaged_hidden = torch.zeros(layer_hidden.shape[-1])

                hidden_averaged_per_layer.append(averaged_hidden)

            results[f"hidden_states_{window_name}"] = hidden_averaged_per_layer
            print(f"    Window '{window_name}': averaged {len(token_positions)} tokens")

        # Default "main" representation: you can choose which to prefer downstream
        # For example: last_vision_token or single_token.
        # Here we choose the single last token as a simple baseline:
        results["hidden_states"] = results["hidden_states_single_token"]

        return results


    def evaluate_mcq_with_activations(
        self,
        image_variants: Dict[str, str],
        question: str,
        options: Dict[str, str],
        correct_answer: str = None,
        max_new_tokens: int = 50
    ) -> Dict:
        """Evaluate MCQ across all variants."""
        prompt = self.format_mcq_prompt(question, options)
        
        results = {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'variants': {}
        }
        
        for variant_name, image_path in image_variants.items():
            print(f"  Processing variant: {variant_name}...")
            
            variant_result = self.extract_activations_with_answer(
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            results['variants'][variant_name] = variant_result
            
            print(f"    Predicted: {variant_result['predicted_answer']}")
            print(f"    Sequence length: {variant_result.get('seq_len', 0)}")
        
        return results
    
    def save_activations(self, results: Dict, output_path: str):
        """Save activation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'question': results['question'],
            'options': results['options'],
            'correct_answer': results['correct_answer'],
            'variants': {}
        }
        
        tensors = {}
        
        for variant_name, variant_data in results['variants'].items():
            metadata['variants'][variant_name] = {
                'response': variant_data.get('response', ''),
                'predicted_answer': variant_data.get('predicted_answer', 'UNKNOWN'),
                'seq_len': variant_data.get('seq_len', 0)
            }
            
            # Save hidden states for all windows
            for key in variant_data.keys():
                if key.startswith('hidden_states_'):
                    if variant_data[key]:
                        tensor_key = f'{variant_name}_{key}'
                        tensors[tensor_key] = torch.stack(variant_data[key])
        
        torch.save(tensors, output_path.with_suffix('.pt'))
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved to: {output_path.with_suffix('.pt')} and .json\n")


def load_questions_from_filtered_data(data_folder: str) -> List[Dict]:
    """
    Load questions from filtered_data structure.
    Each variant has its own JSON file with different image names.
    """
    data_folder = Path(data_folder)
    
    # Define the variants we need
    variants = ['notext', 'irrelevant', 'correct', 'misleading']
    
    # Load all JSON files
    variant_data = {}
    for variant in variants:
        json_file = data_folder / f'filtered_questions_{variant}.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                variant_data[variant] = json.load(f)
        else:
            print(f"Warning: JSON file not found: {json_file}")
    
    if not variant_data:
        raise FileNotFoundError(f"No question JSON files found in {data_folder}")
    
    # Use the first available variant as base for question structure
    base_variant = list(variant_data.keys())[0]
    base_questions = variant_data[base_variant]
    
    questions_data = []
    
    for idx, base_item in enumerate(base_questions):
        question_dict = {
            'question_id': base_item.get('id', base_item.get('question_id', f'q{idx+1}')),
            'question': base_item['question'],
            'options': base_item['options'],
            'answer': base_item.get('answer', base_item.get('correct_answer', '')),
            'image_variants': {}
        }
        
        # For each variant, get the corresponding image
        for variant in variants:
            if variant in variant_data and idx < len(variant_data[variant]):
                variant_item = variant_data[variant][idx]
                image_name = variant_item.get('image', '')
                
                if image_name:
                    variant_image_path = data_folder / variant / image_name
                    if variant_image_path.exists():
                        question_dict['image_variants'][variant] = str(variant_image_path)
                    else:
                        print(f"Warning: Image not found: {variant_image_path}")
        
        # Only add if we have at least some variants
        if question_dict['image_variants']:
            questions_data.append(question_dict)
    
    return questions_data


def process_all_questions(
    evaluator: LlavaNextEvaluator,
    questions_data: List[Dict],
    output_dir: str,
    max_new_tokens: int = 50
):
    """Process all questions and save activations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_questions = len(questions_data)
    processed_questions = 0
    
    print(f"\n{'='*70}")
    print(f"Processing {total_questions} questions with activation extraction")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    for idx, item in enumerate(tqdm(questions_data, desc="Processing questions")):
        question_id = item.get('question_id', f'q{idx+1}')
        print(f"\n[Question {idx+1}/{total_questions}] ID: {question_id}")
        print(f"Question: {item['question'][:80]}...")
        if not item['image_variants']:
            print(f"  Skipping: No image variants found")
            continue
        
        try:
            results = evaluator.evaluate_mcq_with_activations(
                image_variants=item['image_variants'],
                question=item['question'],
                options=item['options'],
                correct_answer=item.get('answer'),
                max_new_tokens=max_new_tokens
            )
            
            output_file = output_path / f"{question_id}_activations"
            evaluator.save_activations(results, str(output_file))
            
            processed_questions += 1
            
        except Exception as e:
            print(f"  Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_questions}/{total_questions} questions")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract activations from LLaVA-NeXT with multiple token windows"
    )
    
    parser.add_argument(
        '--model_id',
        type=str,
        default='llava-hf/llava-v1.6-mistral-7b-hf',
        help='HuggingFace model ID'
    )
    
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to filtered_data folder'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='activations',
        help='Directory to save activation results'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate per answer'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='cuda',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        parser.error(f"Data folder not found: {args.data_folder}")
    
    device = None if args.device == 'auto' else args.device
    
    print("Loading questions from filtered_data structure...")
    questions_data = load_questions_from_filtered_data(args.data_folder)
    print(f"Loaded {len(questions_data)} questions\n")
    
    evaluator = LlavaNextEvaluator(model_id=args.model_id, device=device)
    
    process_all_questions(
        evaluator=evaluator,
        questions_data=questions_data,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()