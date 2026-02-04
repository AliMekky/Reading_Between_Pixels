"""
VLM MCQ Evaluator with Activation Extraction - Multi-Model Support
Supports: LLaVA-1.5, LLaVA-NeXT, Qwen2-VL
Note: InternVL not supported due to .chat() API limitations
"""

import torch
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from PIL import Image, ImageFile
from transformers import AutoProcessor
from tqdm import tqdm
from abc import ABC, abstractmethod
from datetime import datetime
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
    print("Warning: Qwen2-VL not available.")

try:
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    LLAVA_NEXT_AVAILABLE = True
except Exception:
    LLAVA_NEXT_AVAILABLE = False
    print("Warning: LLaVA-NeXT not available.")


# ==================== HF Dataset Utilities ====================

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


def load_questions_from_hf_dataset(dataset_id: str, cache_dir: str = "./hf_dataset_local_cache") -> List[Dict]:
    """Load questions from HuggingFace dataset."""
    print(f"Loading HuggingFace dataset: {dataset_id}")
    ds = get_or_download_hf_dataset(dataset_id, cache_dir, split="test")
    
    if isinstance(ds, DatasetDict):
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        dataset = ds[split_name]
    else:
        dataset = ds
    
    print(f"Dataset size: {len(dataset)} samples")
    
    variants = ['notext', 'correct', 'irrelevant', 'misleading']
    questions_data = []
    
    print("Building questions list...")
    for idx in tqdm(range(len(dataset)), desc="Loading questions"):
        try:
            sample = dataset[idx]
            
            question_id = sample.get("question_id") or f"q_{idx}"
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            
            options = {}
            labels = ["A", "B", "C", "D"]
            for i, lbl in enumerate(labels):
                options[lbl] = choices[i] if i < len(choices) else ""
            
            answer = sample.get("answer", "")
            
            image_variants = {}
            for variant in variants:
                img_obj = sample.get(variant)
                if img_obj is not None:
                    if isinstance(img_obj, Image.Image):
                        image_variants[variant] = img_obj
                    else:
                        print(f"⚠️  Unexpected image type for {question_id}/{variant}: {type(img_obj)}")
            
            if image_variants:
                questions_data.append({
                    'question_id': question_id,
                    'question': question,
                    'options': options,
                    'answer': answer,
                    'image_variants': image_variants
                })
        
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            continue
    
    print(f"✓ Loaded {len(questions_data)} questions with image variants\n")
    return questions_data


# ==================== Debug Logger ====================

class DebugLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file: Optional[str] = None, console: bool = True):
        self.console = console
        self.log_file = None
        
        if log_file:
            self.log_file = open(log_file, 'w', encoding='utf-8')
            self.write(f"Debug log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.write("="*80 + "\n\n")
    
    def write(self, message: str):
        """Write message to file and optionally console."""
        if self.console:
            print(message, end='')
        
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            self.write(f"\n\nDebug log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
            self.log_file = None
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()


# ==================== Base Architecture Config ====================

class VLMArchitectureConfig(ABC):
    """Abstract configuration for different VLM architectures."""
    
    @abstractmethod
    def get_decoder_layers(self, model):
        """Get the decoder layers to hook."""
        pass
    
    @abstractmethod
    def get_image_token_id(self, processor) -> int:
        """Get the image token ID."""
        pass
    
    @abstractmethod
    def extract_hidden_state_from_hook(self, output) -> torch.Tensor:
        """Extract hidden states from hook output."""
        pass
    
    @abstractmethod
    def supports_activation_extraction(self) -> bool:
        """Whether this architecture supports activation extraction."""
        pass


class LLaVANextArchConfig(VLMArchitectureConfig):
    """Configuration for LLaVA-NeXT models."""
    
    def get_decoder_layers(self, model):
        return model.model.language_model.layers
    
    def get_image_token_id(self, processor):
        return processor.tokenizer.convert_tokens_to_ids("<image>")
    
    def extract_hidden_state_from_hook(self, output):
        return output[0]
    
    def supports_activation_extraction(self) -> bool:
        return True


class LLaVA15ArchConfig(VLMArchitectureConfig):
    """Configuration for LLaVA-1.5 models."""
    
    def get_decoder_layers(self, model):
        # LLaVA-1.5 structure: model.language_model.layers
        # The language_model is a LlamaModel which has layers directly
        try:
            # LLaVA-1.5: model.language_model.layers (LlamaModel has layers directly)
            return model.language_model.layers
        except AttributeError:
            # Fallback: try with .model in between
            try:
                return model.language_model.model.layers
            except AttributeError:
                # Print debug info
                print(f"LLaVA-1.5 Model structure:")
                print(f"  model type: {type(model)}")
                print(f"  model.language_model type: {type(model.language_model)}")
                if hasattr(model.language_model, 'layers'):
                    print(f"  ✓ Has layers directly at model.language_model.layers")
                else:
                    print(f"  ✗ No layers attribute found")
                    print(f"  Available attributes: {[a for a in dir(model.language_model) if not a.startswith('_')][:20]}")
                raise AttributeError("Could not find decoder layers in LLaVA-1.5 model structure")
    
    def get_image_token_id(self, processor):
        return processor.tokenizer.convert_tokens_to_ids("<image>")
    
    def extract_hidden_state_from_hook(self, output):
        return output[0]
    
    def supports_activation_extraction(self) -> bool:
        return True


class Qwen2VLArchConfig(VLMArchitectureConfig):
    """Configuration for Qwen2-VL models."""
    
    def get_decoder_layers(self, model):
        # Qwen2-VL structure: model.model is Qwen2_5_VLModel
        # The actual language model layers are at: model.model.language_model.model.layers
        try:
            # Try the full path first
            return model.model.language_model.layers
        except AttributeError:
            # Fallback: try alternative paths
            try:
                return model.language_model.model.layers
            except AttributeError:
                # Last resort: print structure and raise informative error
                print(f"Model structure: {model}")
                print(f"Model.model type: {type(model.model)}")
                print(f"Model.model attributes: {dir(model.model)}")
                raise AttributeError("Could not find decoder layers in Qwen2-VL model structure")
    
    def get_image_token_id(self, processor):
        # Qwen2-VL uses multiple vision tokens:
        # <|vision_start|> - marks beginning of vision input
        # <|image_pad|> - represents each image patch/token (appears many times)
        # <|vision_end|> - marks end of vision input (if it exists)
        
        # For activation extraction, we want the image_pad tokens (the actual vision features)
        possible_tokens = [
            "<|image_pad|>",        # Primary: the actual vision tokens (appears many times)
            "<|vision_start|>",     # Alternative: marks start of vision
            "<|vision_end|>",       # Check if this exists
            "<image>",              # Fallback
            "<|vision_pad|>",       # Another possible name
        ]
        
        found_tokens = {}
        for token in possible_tokens:
            try:
                token_id = processor.tokenizer.convert_tokens_to_ids(token)
                # Check if it's a valid token (not unknown)
                if token_id != processor.tokenizer.unk_token_id and token_id is not None:
                    found_tokens[token] = token_id
                    print(f"  Found vision token: '{token}' (ID: {token_id})")
            except:
                continue
        
        # Prefer image_pad as it represents the actual vision features
        if "<|image_pad|>" in found_tokens:
            print(f"  → Using '<|image_pad|>' for vision token extraction")
            return found_tokens["<|image_pad|>"]
        elif "<|vision_start|>" in found_tokens:
            print(f"  → Using '<|vision_start|>' for vision token extraction")
            return found_tokens["<|vision_start|>"]
        elif found_tokens:
            # Use the first found token
            token_name = list(found_tokens.keys())[0]
            print(f"  → Using '{token_name}' for vision token extraction")
            return found_tokens[token_name]
        
        # If no vision token found, print available special tokens for debugging
        print(f"⚠️  Warning: Could not find any vision token for Qwen2-VL")
        print(f"  Special tokens: {processor.tokenizer.special_tokens_map}")
        if hasattr(processor.tokenizer, 'additional_special_tokens'):
            print(f"  Additional special tokens: {processor.tokenizer.additional_special_tokens}")
        
        # Return a fallback (will likely not find any vision tokens in the input)
        return processor.tokenizer.unk_token_id
    
    def extract_hidden_state_from_hook(self, output):
        return output[0]
    
    def supports_activation_extraction(self) -> bool:
        return True


# ==================== Base Evaluator ====================

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
    
    def load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return PIL Image directly."""
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        return Image.open(image_input).convert('RGB')
    
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
        
        raise ValueError(f"Unable to extract answer from response: {response}")


# ==================== Activation Extractor Base ====================

class BaseActivationExtractor(BaseVLMEvaluator):
    """Base class for activation extraction with architecture support."""
    
    def __init__(
        self, 
        model_id: str, 
        device: str = None, 
        debug_logger: Optional[DebugLogger] = None,
        arch_config: Optional[VLMArchitectureConfig] = None
    ):
        self.logger = debug_logger
        self.arch_config = arch_config
        super().__init__(model_id, device)
        self.answer_token_ids = None
        self._initialize_answer_tokens()
    
    def _debug_print(self, msg: str, level: int = 0):
        """Helper for debug printing with indentation."""
        if self.logger:
            indent = "  " * level
            self.logger.write(f"{indent}{msg}\n")
    
    def _initialize_answer_tokens(self):
        """Precompute token IDs for answer choices A/B/C/D."""
        self._debug_print("\n=== Initializing Answer Tokens ===")
        self.answer_token_ids = set()
        for letter in ['A', 'B', 'C', 'D']:
            token_ids = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if token_ids:
                self.answer_token_ids.add(token_ids[0])
                self._debug_print(f"Token for '{letter}': {token_ids[0]}", level=1)
            
            token_ids_space = self.processor.tokenizer.encode(f" {letter}", add_special_tokens=False)
            if token_ids_space:
                self.answer_token_ids.add(token_ids_space[-1])
                self._debug_print(f"Token for ' {letter}': {token_ids_space[-1]}", level=1)
        
        self._debug_print(f"All answer token IDs: {self.answer_token_ids}\n")
    
    def extract_activations_with_answer(
        self, 
        image_input: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 50,
        do_sample: bool = False
    ) -> Dict:
        """Extract activations with comprehensive debugging."""
        
        if not self.arch_config.supports_activation_extraction():
            raise NotImplementedError(f"{self.__class__.__name__} does not support activation extraction")
        
        self._debug_print("\n" + "="*80)
        self._debug_print("STARTING ACTIVATION EXTRACTION")
        self._debug_print(f"Architecture: {self.arch_config.__class__.__name__}")
        self._debug_print("="*80)
        
        ## STEP 1: Load image and prepare inputs
        self._debug_print("\n### STEP 1: Loading Image and Tokenizing ###")
        image = self.load_image(image_input)
        inputs = self._prepare_inputs(image, prompt)
        
        ## STEP 2: Find vision token positions
        self._debug_print("\n### STEP 2: Finding Vision Token Positions ###")
        input_ids = inputs["input_ids"][0].detach().cpu()
        seq_len = input_ids.shape[0]

        self._debug_print(f"Input sequence length: {seq_len}", level=1)

        # Get vision token id using architecture config
        image_token_id = self.arch_config.get_image_token_id(self.processor)
        self._debug_print(f"Primary vision token ID: {image_token_id}", level=1)

        # Find all positions
        vision_positions = (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        self._debug_print(f"Found {len(vision_positions)} vision tokens at positions: {vision_positions[:10]}{'...' if len(vision_positions) > 10 else ''}", level=1)

        # For Qwen2-VL, also check for vision_start and vision_end tokens
        try:
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            
            if vision_start_id != self.processor.tokenizer.unk_token_id:
                vision_start_positions = (input_ids == vision_start_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                if vision_start_positions:
                    self._debug_print(f"Found {len(vision_start_positions)} <|vision_start|> tokens at positions: {vision_start_positions}", level=1)
            
            if vision_end_id != self.processor.tokenizer.unk_token_id:
                vision_end_positions = (input_ids == vision_end_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                if vision_end_positions:
                    self._debug_print(f"Found {len(vision_end_positions)} <|vision_end|> tokens at positions: {vision_end_positions}", level=1)
        except:
            pass  # Not Qwen2-VL or tokens don't exist
        
        # Debug: Show tokens around vision positions
        if vision_positions and self.logger:
            self._debug_print("\n--- Context around vision tokens ---", level=1)
            for i, pos in enumerate(vision_positions[:5]):
                start = max(0, pos - 5)
                end = min(seq_len, pos + 6)
                context_ids = input_ids[start:end].tolist()
                context_tokens = [self.processor.tokenizer.decode([tid]) for tid in context_ids]
                
                self._debug_print(f"\nVision token #{i+1} at position {pos}:", level=2)
                self._debug_print(f"  Token IDs: {context_ids}", level=3)
                self._debug_print(f"  Tokens: {context_tokens}", level=3)
                
                decoded_context = self.processor.tokenizer.decode(input_ids[start:end])
                self._debug_print(f"  Decoded: '{decoded_context}'", level=3)
        
        ## STEP 3: Register hooks using architecture config
        self._debug_print("\n### STEP 3: Registering Hooks ###")
        hidden_states_all_layers: Dict[int, torch.Tensor] = {}

        def hidden_state_hook(module, input, output, layer_idx):
            """Collect hidden states from each layer using arch config."""
            hidden = self.arch_config.extract_hidden_state_from_hook(output)
            hidden_states_all_layers[layer_idx] = hidden.detach().cpu()
            if self.logger and layer_idx % 8 == 0:
                self._debug_print(f"  Captured layer {layer_idx}: shape {hidden.shape}", level=2)

        hooks = []
        # Get decoder layers using architecture config
        decoder_layers = self.arch_config.get_decoder_layers(self.model)
        self._debug_print(f"Registering hooks on {len(decoder_layers)} decoder layers...", level=1)
        
        for idx, layer in enumerate(decoder_layers):
            hook = layer.register_forward_hook(
                lambda module, inp, out, idx=idx: hidden_state_hook(module, inp, out, idx)
            )
            hooks.append(hook)

        ## STEP 4: Forward pass (prefill)
        self._debug_print("\n### STEP 4: Forward Pass (Prefill) ###")
        with torch.inference_mode():
            _ = self.model(
                **inputs,
                output_hidden_states=False,
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not hidden_states_all_layers:
            raise RuntimeError("No hidden states were collected from hooks.")

        # Debug: Check collected hidden states
        self._debug_print(f"Collected hidden states from {len(hidden_states_all_layers)} layers", level=1)
        example_layer = hidden_states_all_layers[0]
        self._debug_print(f"Example (layer 0) shape: {example_layer.shape}", level=1)
        
        # Determine actual sequence length from hidden states
        if example_layer.dim() == 3:
            actual_seq_len = example_layer.shape[1]
        elif example_layer.dim() == 2:
            actual_seq_len = example_layer.shape[0]
        else:
            raise ValueError(f"Unexpected hidden state dimensions: {example_layer.shape}")
        
        self._debug_print(f"Actual sequence length from hidden states: {actual_seq_len}", level=1)

        ## STEP 5: Generate answer
        self._debug_print("\n### STEP 5: Generating Answer ###")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences

        # Debug generated tokens
        generated_ids = output_ids[0, seq_len:].cpu().tolist()
        self._debug_print(f"Generated {len(generated_ids)} new tokens", level=1)
        self._debug_print(f"Generated token IDs: {generated_ids}", level=2)
        
        generated_tokens = [self.processor.tokenizer.decode([tid]) for tid in generated_ids]
        self._debug_print(f"Generated tokens: {generated_tokens}", level=2)

        # Check if first token is an answer token
        if generated_ids and len(generated_ids) > 0:
            first_token = generated_ids[0]
            if first_token in self.answer_token_ids:
                self._debug_print(f"✓ First token IS an answer token: {first_token}", level=2)
                # Decode to see which letter
                first_token_text = self.processor.tokenizer.decode([first_token])
                self._debug_print(f"  → Decoded as: '{first_token_text}'", level=3)
            else:
                self._debug_print(f"✗ First token is NOT an answer token: {first_token}", level=2)
                first_token_text = self.processor.tokenizer.decode([first_token])
                self._debug_print(f"  → Decoded as: '{first_token_text}'", level=3)
                
                full_generated = self.processor.tokenizer.decode(output_ids[0, seq_len:])
                self._debug_print(f"Full generated text: '{full_generated}'", level=2)
        
        # Extract decision token hidden states
        decision_token_hidden = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            first_gen_token_states = outputs.hidden_states[0]
            self._debug_print(f"Decision token hidden states: {len(first_gen_token_states)} layers", level=1)
            
            for layer_idx, layer_states in enumerate(first_gen_token_states):
                decision_token_hidden.append(layer_states[0, -1, :].detach().cpu())
                if self.logger and layer_idx % 8 == 0:
                    self._debug_print(f"  Layer {layer_idx} decision token shape: {layer_states[0, -1, :].shape}", level=2)

        response = self._decode_output(output_ids)
        self._debug_print(f"\nFull response: {response}", level=1)

        results: Dict[str, Union[str, int, List[torch.Tensor]]] = {
            "response": response,
            "seq_len": actual_seq_len,
            "predicted_answer": self.extract_answer(response),
        }
        results["hidden_states_decision_token"] = decision_token_hidden

        ## STEP 6: Define windows
        self._debug_print("\n### STEP 6: Defining Token Windows ###")
        
        last_text_token_positions = [actual_seq_len - 1] if actual_seq_len > 0 else []
        self._debug_print(f"last_text_token window: positions {last_text_token_positions}", level=1)
        
        vision_token_positions = vision_positions
        if len(vision_token_positions) > 10:
            self._debug_print(f"vision_tokens window: {len(vision_token_positions)} positions {vision_token_positions[:10]}...", level=1)
        else:
            self._debug_print(f"vision_tokens window: {len(vision_token_positions)} positions {vision_token_positions}", level=1)
        
        if vision_positions:
            last_vision_token_positions = [max(vision_positions)]
            self._debug_print(f"last_vision_token window: position {last_vision_token_positions}", level=1)
        else:
            last_vision_token_positions = []
            self._debug_print(f"last_vision_token window: EMPTY (no vision tokens found)", level=1)

        all_token_positions = list(range(actual_seq_len))
        self._debug_print(f"all_tokens window: {len(all_token_positions)} positions [0...{actual_seq_len-1}]", level=1)

        windows: Dict[str, List[int]] = {
            "last_text_token": last_text_token_positions,
            "vision_tokens": vision_token_positions,
            "last_vision_token": last_vision_token_positions,
            "all_tokens": all_token_positions,
        }

        ## STEP 7: Extract and average hidden states per window
        self._debug_print("\n### STEP 7: Extracting Averaged Hidden States per Window ###")
        
        for window_name, token_positions in windows.items():
            self._debug_print(f"\nProcessing window: {window_name}", level=1)
            self._debug_print(f"  Number of tokens: {len(token_positions)}", level=2)
            
            hidden_averaged_per_layer: List[torch.Tensor] = []

            for layer_idx in sorted(hidden_states_all_layers.keys()):
                layer_hidden = hidden_states_all_layers[layer_idx]

                tokens_in_window: List[torch.Tensor] = []

                if not token_positions:
                    averaged_hidden = torch.zeros(layer_hidden.shape[-1])
                    hidden_averaged_per_layer.append(averaged_hidden)
                    if self.logger and layer_idx == 0:
                        self._debug_print(f"  Layer {layer_idx}: Empty window, using zero vector", level=3)
                    continue

                if layer_hidden.dim() == 3:
                    for pos in token_positions:
                        if 0 <= pos < layer_hidden.shape[1]:
                            tokens_in_window.append(layer_hidden[0, pos, :])
                elif layer_hidden.dim() == 2:
                    for pos in token_positions:
                        if 0 <= pos < layer_hidden.shape[0]:
                            tokens_in_window.append(layer_hidden[pos, :])
                else:
                    raise ValueError(f"Unexpected hidden state shape: {layer_hidden.shape}")

                if tokens_in_window:
                    averaged_hidden = torch.stack(tokens_in_window, dim=0).mean(dim=0)
                    if self.logger and layer_idx % 8 == 0:
                        self._debug_print(f"  Layer {layer_idx}: Averaged {len(tokens_in_window)} tokens -> shape {averaged_hidden.shape}", level=3)
                else:
                    averaged_hidden = torch.zeros(layer_hidden.shape[-1])
                    if self.logger and layer_idx == 0:
                        self._debug_print(f"  Layer {layer_idx}: No valid tokens, using zero vector", level=3)

                hidden_averaged_per_layer.append(averaged_hidden)

            results[f"hidden_states_{window_name}"] = hidden_averaged_per_layer
            self._debug_print(f"  ✓ Window '{window_name}': {len(hidden_averaged_per_layer)} layers, {len(token_positions)} tokens averaged", level=2)

        results["hidden_states"] = results["hidden_states_last_text_token"]

        self._debug_print("\n" + "="*80)
        self._debug_print("ACTIVATION EXTRACTION COMPLETE")
        self._debug_print("="*80 + "\n")

        return results

    def evaluate_mcq_with_activations(
        self,
        image_variants: Dict[str, Union[str, Image.Image]],
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
        
        for variant_name, image_input in image_variants.items():
            self._debug_print(f"\n{'#'*80}")
            self._debug_print(f"Processing variant: {variant_name}")
            self._debug_print(f"{'#'*80}")
            
            print(f"\n  Processing variant: {variant_name}...")
            
            variant_result = self.extract_activations_with_answer(
                image_input=image_input,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            results['variants'][variant_name] = variant_result
            
            self._debug_print(f"\n{'='*40}")
            self._debug_print(f"VARIANT RESULT: {variant_name}")
            self._debug_print(f"  Predicted: {variant_result['predicted_answer']}")
            self._debug_print(f"  Correct: {correct_answer}")
            self._debug_print(f"  Match: {variant_result['predicted_answer'] == correct_answer}")
            self._debug_print(f"  Sequence length: {variant_result.get('seq_len', 0)}")
            self._debug_print(f"{'='*40}\n")
            
            print(f"    Predicted: {variant_result['predicted_answer']} | Correct: {correct_answer}")
        
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
            
            for key in variant_data.keys():
                if key.startswith('hidden_states_'):
                    if variant_data[key]:
                        tensor_key = f'{variant_name}_{key}'
                        tensors[tensor_key] = torch.stack(variant_data[key])
        
        torch.save(tensors, output_path.with_suffix('.pt'))
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._debug_print(f"  Saved to: {output_path.with_suffix('.pt')} and .json\n")


# ==================== Model-Specific Extractors ====================

class LlavaActivationExtractor(BaseActivationExtractor):
    """LLaVA-1.5 activation extractor."""
    
    def __init__(self, model_id: str, device: str = None, debug_logger: Optional[DebugLogger] = None):
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA requires: pip install transformers")
        arch_config = LLaVA15ArchConfig()
        super().__init__(model_id, device, debug_logger, arch_config)
    
    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        self._debug_print("\n=== Preparing Inputs ===")
        self._debug_print(f"Image size: {image.size}", level=1)
        self._debug_print(f"Prompt length: {len(prompt)} chars", level=1)
        
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
        
        self._debug_print(f"Formatted prompt length: {len(formatted_prompt)} chars", level=1)
        
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors='pt')
        
        num_tokens = inputs["input_ids"].shape[1]
        self._debug_print(f"Total input tokens: {num_tokens}", level=1)
        
        if 'pixel_values' in inputs:
            self._debug_print(f"Pixel values shape: {inputs['pixel_values'].shape}", level=1)
        
        decoded_input = self.processor.tokenizer.decode(inputs["input_ids"][0])
        self._debug_print(f"\n--- Decoded Input (first 500 chars) ---", level=1)
        self._debug_print(decoded_input[:500], level=2)
        if len(decoded_input) > 500:
            self._debug_print("...", level=2)
        
        inputs = inputs.to(self.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        return self.processor.decode(output[0], skip_special_tokens=True)


class LlavaNextActivationExtractor(BaseActivationExtractor):
    """LLaVA-NeXT activation extractor."""
    
    def __init__(self, model_id: str, device: str = None, debug_logger: Optional[DebugLogger] = None):
        if not LLAVA_NEXT_AVAILABLE:
            raise ImportError("LLaVA-NeXT requires: pip install -U transformers")
        arch_config = LLaVANextArchConfig()
        super().__init__(model_id, device, debug_logger, arch_config)
    
    def _load_model(self):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        self._debug_print("\n=== Preparing Inputs ===")
        self._debug_print(f"Image size: {image.size}", level=1)
        self._debug_print(f"Prompt length: {len(prompt)} chars", level=1)
        
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
        
        self._debug_print(f"Formatted prompt length: {len(formatted_prompt)} chars", level=1)
        
        inputs = self.processor(images=image, text=formatted_prompt, return_tensors='pt')
        
        num_tokens = inputs["input_ids"].shape[1]
        self._debug_print(f"Total input tokens: {num_tokens}", level=1)
        
        if 'pixel_values' in inputs:
            self._debug_print(f"Pixel values shape: {inputs['pixel_values'].shape}", level=1)
        
        if 'image_sizes' in inputs:
            self._debug_print(f"Image sizes: {inputs['image_sizes']}", level=1)
        
        decoded_input = self.processor.tokenizer.decode(inputs["input_ids"][0])
        self._debug_print(f"\n--- Decoded Input (first 500 chars) ---", level=1)
        self._debug_print(decoded_input[:500], level=2)
        if len(decoded_input) > 500:
            self._debug_print("...", level=2)
        
        inputs = inputs.to(self.device)
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        
        return inputs
    
    def _decode_output(self, output) -> str:
        return self.processor.decode(output[0], skip_special_tokens=True)


class QwenVLActivationExtractor(BaseActivationExtractor):
    """Qwen2-VL activation extractor."""
    
    def __init__(self, model_id: str, device: str = None, debug_logger: Optional[DebugLogger] = None):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen-VL requires: pip install qwen-vl-utils torchvision")
        arch_config = Qwen2VLArchConfig()
        super().__init__(model_id, device, debug_logger, arch_config)
        
    def _load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Debug: Print model structure to help understand layer paths
        if self.logger:
            self.logger.write("\n=== Qwen2-VL Model Structure Debug ===\n")
            self.logger.write(f"Model type: {type(self.model)}\n")
            self.logger.write(f"Model.model type: {type(self.model.model)}\n")
            
            # Try to find the language model
            if hasattr(self.model.model, 'language_model'):
                self.logger.write(f"Found language_model at: model.model.language_model\n")
                self.logger.write(f"Language model type: {type(self.model.model.language_model)}\n")
                
                if hasattr(self.model.model.language_model, 'model'):
                    self.logger.write(f"Found model.layers at: model.model.language_model.model\n")
                    if hasattr(self.model.model.language_model.model, 'layers'):
                        num_layers = len(self.model.model.language_model.model.layers)
                        self.logger.write(f"✓ Found {num_layers} decoder layers\n")
                    else:
                        self.logger.write(f"✗ No 'layers' attribute found\n")
            
            self.logger.write("=====================================\n\n")
    
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        self._debug_print("\n=== Preparing Inputs ===")
        self._debug_print(f"Image size: {image.size}", level=1)
        self._debug_print(f"Prompt length: {len(prompt)} chars", level=1)
        
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
        
        self._debug_print(f"Formatted prompt length: {len(text)} chars", level=1)
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        num_tokens = inputs["input_ids"].shape[1]
        self._debug_print(f"Total input tokens: {num_tokens}", level=1)
        
        if 'pixel_values' in inputs:
            self._debug_print(f"Pixel values shape: {inputs['pixel_values'].shape}", level=1)
        
        return inputs.to(self.device)
    
    def _decode_output(self, output) -> str:
        return self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ==================== Model Registry ====================

ACTIVATION_MODEL_REGISTRY = {
    'llava': {
        'class': LlavaActivationExtractor,
        'default_model': 'llava-hf/llava-1.5-7b-hf',
        'available_models': [
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-1.5-13b-hf',
        ],
        'available': LLAVA_AVAILABLE
    },
}

if LLAVA_NEXT_AVAILABLE:
    ACTIVATION_MODEL_REGISTRY['llava-next'] = {
        'class': LlavaNextActivationExtractor,
        'default_model': 'llava-hf/llava-v1.6-mistral-7b-hf',
        'available_models': [
            'llava-hf/llava-v1.6-mistral-7b-hf',
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
        ],
        'available': True
    }
else:
    ACTIVATION_MODEL_REGISTRY['llava-next'] = {
        'available': False,
        'error_message': 'LLaVA-NeXT requires: pip install -U transformers'
    }

if QWEN_AVAILABLE:
    ACTIVATION_MODEL_REGISTRY['qwen-vl'] = {
        'class': QwenVLActivationExtractor,
        'default_model': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'available_models': [
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-7B-Instruct',
        ],
        'available': True
    }
else:
    ACTIVATION_MODEL_REGISTRY['qwen-vl'] = {
        'available': False,
        'error_message': 'Qwen-VL requires: pip install qwen-vl-utils torchvision'
    }


def get_activation_extractor(
    model_type: str, 
    model_id: str = None, 
    device: str = None,
    debug_logger: Optional[DebugLogger] = None
) -> BaseActivationExtractor:
    """Factory function to get the appropriate activation extractor."""
    if model_type not in ACTIVATION_MODEL_REGISTRY:
        available = [k for k, v in ACTIVATION_MODEL_REGISTRY.items() if v.get('available', False)]
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_info = ACTIVATION_MODEL_REGISTRY[model_type]
    
    if not model_info.get('available', False):
        raise ImportError(f"Model {model_type} not available: {model_info.get('error_message', 'Unknown error')}")
    
    extractor_class = model_info['class']
    if model_id is None:
        model_id = model_info['default_model']
    
    return extractor_class(model_id=model_id, device=device, debug_logger=debug_logger)


# ==================== Processing Functions ====================

def process_all_questions(
    extractor: BaseActivationExtractor,
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
            results = extractor.evaluate_mcq_with_activations(
                image_variants=item['image_variants'],
                question=item['question'],
                options=item['options'],
                correct_answer=item.get('answer'),
                max_new_tokens=max_new_tokens
            )
            
            output_file = output_path / f"{question_id}_activations"
            extractor.save_activations(results, str(output_file))
            
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


# ==================== Main ====================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract activations from VLMs with multi-model support"
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='llava',
        help='Model type (llava, llava-next, qwen-vl)'
    )
    
    parser.add_argument(
        '--model_id',
        type=str,
        help='HuggingFace model ID (optional, uses default for model_type)'
    )
    
    parser.add_argument(
        '--hf_dataset',
        type=str,
        default="AHAAM/CIM",
        help='HuggingFace dataset ID (e.g., AHAAM/CIM)'
    )
    
    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        default='../inference/hf_cache/AHAAM__CIM/AHAAM__CIM',
        help='Local cache directory for HF dataset'
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
    
    parser.add_argument(
        '--debug_log',
        type=str,
        default='debug_log_llava.txt',
        help='Path to debug log file'
    )
    
    parser.add_argument(
        '--no_console_debug',
        action='store_true',
        help='Disable debug output to console (only write to file)'
    )
    
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List available model types and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Model Types for Activation Extraction:")
        print("="*60)
        for model_type, info in ACTIVATION_MODEL_REGISTRY.items():
            status = "✓ Available" if info.get('available') else "✗ Not Available"
            print(f"\n{model_type}: {status}")
            if info.get('available'):
                print(f"  Default model: {info['default_model']}")
                print(f"  Available models:")
                for m in info['available_models']:
                    print(f"    - {m}")
            else:
                print(f"  Error: {info.get('error_message', 'Unknown')}")
        return
    
    device = None if args.device == 'auto' else args.device
    
    # Create debug logger
    debug_log_path = Path(args.output_dir) / args.debug_log
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = DebugLogger(
        log_file=str(debug_log_path),
        console=not args.no_console_debug
    )
    
    print(f"Debug log will be saved to: {debug_log_path}")
    
    print("Loading questions from HuggingFace dataset...")
    questions_data = load_questions_from_hf_dataset(args.hf_dataset, args.hf_cache_dir)
    print(f"Loaded {len(questions_data)} questions\n")
    
    # Get activation extractor
    extractor = get_activation_extractor(
        model_type=args.model_type,
        model_id=args.model_id,
        device=device,
        debug_logger=logger
    )
    
    try:
        process_all_questions(
            extractor=extractor,
            questions_data=questions_data,
            output_dir=args.output_dir,
            max_new_tokens=args.max_tokens
        )
    finally:
        logger.close()
        print(f"\nDebug log saved to: {debug_log_path}")


if __name__ == "__main__":
    main()