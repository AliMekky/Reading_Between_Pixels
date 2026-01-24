"""
VLM Activation Extractor - LLaVA (No Manual Packing) + Encoder Verification + Q-only Tokenization

This script:
  • Uses LLaVA's internal multimodal merge (no manual inputs_embeds packing)
  • Captures per-layer hidden states of the merged (vision+text) sequence via hooks
  • Computes the vision span anchor and defines analysis windows
  • Verifies counts directly from the text tokenizer, vision tower, and decoder
  • Separately tokenizes ONLY (instruction + question + options + "Answer:") to confirm text token counts

Author: You
Date: Nov 2025
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from abc import ABC, abstractmethod

from transformers import AutoProcessor, LlavaForConditionalGeneration


# =========================
# Base evaluator (shared)
# =========================
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
        ...

    @abstractmethod
    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict:
        ...

    @abstractmethod
    def _decode_output(self, output) -> str:
        ...

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def format_mcq_prompt(
        self,
        question: str,
        options: Dict[str, str],
        instruction: str = None
    ) -> str:
        if instruction is None:
            instruction = "Answer the following multiple-choice question by selecting the correct option."

        prompt = f"{instruction}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in options.items():
            prompt += f"{key}) {value}\n"
        prompt += "\nAnswer:"
        return prompt

    def extract_answer(self, response: str) -> str:
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

        return "UNKNOWN"


# ======================================
# LLaVA evaluator (no manual packing)
# with robust image marker handling and
# encoder/decoder + Q-only verification
# ======================================
class LlavaAutoEvaluator(BaseVLMEvaluator):
    """
    LLaVA evaluator that extracts MERGED activations without manual packing.
    Adds encoder-side verification + question-only tokenization.
    """

    # ---------- BaseVLMEvaluator API ----------

    def _load_model(self):
        dtype = torch.float16 if "cuda" in self.device and torch.cuda.is_available() else torch.float32
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
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

        inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt")
        out = self.processor.decode(inputs["input_ids"][0])

        with open("decoded_input_ids.txt", "w", encoding="utf-8") as f:
            f.write(out)
        inputs = inputs.to(self.device)
        

        # Keep vision dtype in sync with model params
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(next(self.model.parameters()).dtype)

        return inputs

    def _decode_output(self, output_ids) -> str:
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    # ---------- helpers: decoder + markers + vision count ----------

    def _decoder_layers(self):
        """
        Get decoder layers robustly across wrappers.
        Usually: self.model.language_model.model.layers
        """
        lm = self.model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
        raise AttributeError("Could not locate decoder layers on language_model.")

    @torch.no_grad()
    def _candidate_image_token_ids(self) -> List[int]:
        """
        Collect token IDs that likely represent image markers across LLaVA variants.
        """
        tok = self.processor.tokenizer
        cand_strs = set()

        if hasattr(tok, "image_token"):
            cand_strs.add(tok.image_token)

        cand_strs.update(["<image>", "<image_1>", "<image_start>", "<image_end>"])

        for s in getattr(tok, "additional_special_tokens", []):
            if "image" in s.lower():
                cand_strs.add(s)

        ids = set()
        for s in cand_strs:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != -1 and tid != getattr(tok, "unk_token_id", -999):
                ids.add(tid)
        return sorted(ids)

    @torch.no_grad()
    def _find_image_runs_and_tokens(self, input_ids: torch.LongTensor) -> Tuple[List[Tuple[int, int]], List[str]]:
        """
        Return (runs, tokens) where:
          - runs: list of (start,end) for contiguous image-marker spans (half-open)
          - tokens: tokenizer strings for the whole text sequence
        """
        tok = self.processor.tokenizer
        ids = input_ids[0].tolist()  # [S]
        tokens = tok.convert_ids_to_tokens(ids, skip_special_tokens=False)
        cand_ids = set(self._candidate_image_token_ids())

        image_idxs = []
        for i, (tid, tstr) in enumerate(zip(ids, tokens)):
            if (tid in cand_ids) or ("image" in (tstr or "").lower()):
                image_idxs.append(i)

        runs = []
        if image_idxs:
            start = prev = image_idxs[0]
            for j in image_idxs[1:]:
                if j == prev + 1:
                    prev = j
                else:
                    runs.append((start, prev + 1))
                    start = prev = j
            runs.append((start, prev + 1))
        return runs, tokens

    @torch.no_grad()
    def _vision_tokens_count(self, pixel_values: torch.Tensor, prefer_cls_drop: bool = True) -> Tuple[int, int]:
        """
        Return (seq_vis_raw, N_vis_guess).
        Many CLIP-like towers output [CLS] + patches; start with N_vis = seq-1.
        """
        vt = self.model.vision_tower
        if isinstance(vt, (list, torch.nn.ModuleList)):
            vt = vt[0]
        vis = vt(pixel_values=pixel_values).last_hidden_state  # (b, seq_vis_raw, d_vis)
        seq_vis_raw = int(vis.shape[1])
        N_vis_guess = max(0, seq_vis_raw - 1) if prefer_cls_drop else seq_vis_raw
        return seq_vis_raw, N_vis_guess

    @torch.no_grad()
    def _observe_merged_len_once(self, inputs: Dict) -> int:
        """
        Run a quick forward with a single hook to read the decoder seq length after merge.
        """
        seq_len_box = {}

        def hook(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            seq_len_box["len"] = int(h.shape[1])

        layers = self._decoder_layers()
        h = layers[0].register_forward_hook(hook)
        _ = self.model(**inputs)
        h.remove()
        return seq_len_box["len"]

    # ---------- encoder/decoder + Q-only verification (exposed) ----------

    @torch.no_grad()
    def verify_from_encoders(
        self,
        inputs: Dict,
        dump_tokens_path: str = None,
        *,
        question_text: str = None,
        options: Dict[str, str] = None,
        instruction: str = "Answer the following multiple-choice question by selecting the correct option.",
    ):
        """
        Verifies text/vision/merged lengths and (optionally) compares with direct
        tokenization of the question+options-only block (no chat template).
        """
        # TEXT side
        input_ids = inputs["input_ids"]        # (1, S_text)
        S_text = int(input_ids.shape[1])
        runs, tokens = self._find_image_runs_and_tokens(input_ids)
        if not runs:
            raise RuntimeError("No image-marker tokens found in text.")

        if len(runs) > 1:
            print(f"⚠ Found {len(runs)} image-marker runs; assuming single image and using the FIRST run.")

        v_start, v_end_marker = runs[0]
        marker_len = v_end_marker - v_start

        # VISION side
        seq_vis_raw, N_vis = self._vision_tokens_count(inputs["pixel_values"], prefer_cls_drop=True)

        # DECODER (observed)
        merged_seq_len_obs = self._observe_merged_len_once(inputs)
        merged_seq_len_expected = S_text - marker_len + N_vis

        # Cross-check math
        text_tokens_in_merged_from_text = S_text - marker_len
        text_tokens_in_merged_from_obs = merged_seq_len_obs - N_vis

        print("\n=== ENCODER/DECODER CONSISTENCY CHECK ===")
        print(f"Text length S_text (pre-merge)            : {S_text}")
        print(f"Image marker run [start:end) (len)        : [{v_start}:{v_end_marker}) (len={marker_len})")
        print(f"Vision encoder seq (raw)/N_vis (guess)    : {seq_vis_raw} / {N_vis}")
        print(f"Merged seq len (observed via decoder)     : {merged_seq_len_obs}")
        print(f"Merged seq len (expected)                 : {merged_seq_len_expected}")
        ok_len = (merged_seq_len_obs == merged_seq_len_expected)
        print("→ merged length match?                    :", "✓ yes" if ok_len else "⚠ no")

        print(f"\nText tokens in merged (decoder observed)  : {text_tokens_in_merged_from_obs}")
        print(f"Text tokens in merged (from text side)    : {text_tokens_in_merged_from_text}")
        ok_text = (text_tokens_in_merged_from_obs == text_tokens_in_merged_from_text)
        print("→ text count match?                       :", "✓ yes" if ok_text else "⚠ no")

        # NEW: question+options-only tokenization
        if question_text and options:
            tok = self.processor.tokenizer
            raw_text = f"{instruction}\n\nQuestion: {question_text}\n\nOptions:\n"
            for key, val in options.items():
                raw_text += f"{key}) {val}\n"
            raw_text += "\nAnswer:"

            q_tokens = tok.encode(raw_text, add_special_tokens=False)
            q_token_strs = tok.convert_ids_to_tokens(q_tokens)
            print("\n=== QUESTION + OPTIONS TOKENIZATION (NO CHAT TEMPLATE) ===")
            print(f"Raw text token count                      : {len(q_tokens)}")
            print(f"→ Compare to text tokens in merged        : {text_tokens_in_merged_from_text}")
            if abs(len(q_tokens) - text_tokens_in_merged_from_text) <= 3:
                print("✓ Roughly same token count (difference ≤ 3).")
            else:
                print("⚠ Counts differ — chat template likely adds role/system tokens.")
            print("First 40 tokens:", q_token_strs[:40])

            if dump_tokens_path:
                with open(dump_tokens_path, "w") as f:
                    json.dump({
                        "S_text": S_text,
                        "marker_run": [v_start, v_end_marker],
                        "marker_len": marker_len,
                        "seq_vis_raw": seq_vis_raw,
                        "N_vis": N_vis,
                        "merged_seq_len_obs": merged_seq_len_obs,
                        "merged_seq_len_expected": merged_seq_len_expected,
                        "text_tokens_in_merged": text_tokens_in_merged_from_text,
                        "question_only_token_count": len(q_tokens),
                        "question_only_tokens": q_token_strs,
                    }, f, indent=2)
                print(f"• Wrote token dump to {dump_tokens_path}")
        else:
            print("\n(Skipping question+options-only tokenization: not provided.)")

        return {
            "S_text": S_text,
            "marker_start": v_start,
            "marker_end": v_end_marker,
            "marker_len": marker_len,
            "seq_vis_raw": seq_vis_raw,
            "N_vis": N_vis,
            "merged_seq_len_obs": merged_seq_len_obs,
            "merged_seq_len_expected": merged_seq_len_expected,
            "text_tokens_in_merged": text_tokens_in_merged_from_text,
            "ok_len": ok_len,
            "ok_text": ok_text,
        }

    # ---------- main extraction ----------

    @torch.no_grad()
    def extract_activations_with_answer(
        self,
        image_path: str,
        prompt: str,
        *,
        question_text: str = None,
        options: Dict[str, str] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        verify_encoders: bool = True,
        dump_tokens_path: str = None
    ) -> Dict:
        """
        Extract activations using LLaVA's native merge.
        Returns:
            Dictionary with activations, metadata, and response
        """
        image = self.load_image(image_path)
        inputs = self._prepare_inputs(image, prompt)

        # Optional verification (also does Q-only tokenization if provided)
        if verify_encoders:
            self.verify_from_encoders(
                inputs,
                dump_tokens_path=dump_tokens_path,
                question_text=question_text,
                options=options
            )

        # Find marker runs to anchor span
        runs, _ = self._find_image_runs_and_tokens(inputs["input_ids"])
        if not runs:
            raise RuntimeError("Could not locate image marker(s) in the tokenized prompt.")
        if len(runs) > 1:
            print(f"⚠ Found {len(runs)} image marker runs; assuming single image and using the first run.")

        # Estimate N_vis from vision tower (post-verify is fine)
        _, N_vis = self._vision_tokens_count(inputs["pixel_values"], prefer_cls_drop=True)
        v_start, v_end_marker = runs[0]
        v_end = v_start + N_vis
        print(f"\nAnchor & Span:")
        print(f"  marker_run      : [{v_start}, {v_end_marker})")
        print(f"  vision span est.: [{v_start}, {v_end}) len={N_vis}")

        # ---- capture merged hidden states via hooks ----
        hidden_states_all_layers = {}
        hooks = []

        def make_hook(idx):
            def hook(module, _in, out):
                h = out[0] if isinstance(out, tuple) else out  # (b, seq, hid)
                hidden_states_all_layers[idx] = h.detach().cpu()
            return hook

        decoder_layers = self._decoder_layers()
        print(f"  Registering hooks on {len(decoder_layers)} layers...")
        for idx, layer in enumerate(decoder_layers):
            hooks.append(layer.register_forward_hook(make_hook(idx)))

        # Trigger internal packing by calling the regular forward
        print("  Running forward pass (internal merge)...")
        _ = self.model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Sequence length as seen by the LM
        first_layer = hidden_states_all_layers[min(hidden_states_all_layers.keys())]
        seq_len = first_layer.shape[1]

        # Clamp span to seq_len just in case
        v_start = max(0, min(v_start, seq_len))
        v_end = max(v_start, min(v_end, seq_len))
        num_vision_tokens = v_end - v_start
        num_text_tokens = seq_len - num_vision_tokens

        print(f"  Merged seq len: {seq_len} | vision: {num_vision_tokens} | text: {num_text_tokens}")

        # Windows (matching LLaVA-NeXT windows)
        windows = {
            'single_token': range(seq_len - 1, seq_len) if seq_len > 0 else range(0),
            'vision_tokens': range(v_start, v_end),
            'last_vision_token': range(v_end - 1, v_end) if v_end > v_start else range(0),
            'all_tokens': range(seq_len),
        }

        # Average hidden states for each window × each layer
        print("  Averaging activations per window × per layer...")
        results = {
            'response': None,
            'seq_len': seq_len,
            'num_vision_tokens': num_vision_tokens,
            'num_text_tokens': num_text_tokens,
            'architecture_mode': 'merged',
            'predicted_answer': 'UNKNOWN',
        }

        for window_name, token_range in windows.items():
            hidden_averaged = []
            for layer_idx in sorted(hidden_states_all_layers.keys()):
                layer_hidden = hidden_states_all_layers[layer_idx][0]  # (seq, hid)
                idxs = [i for i in token_range if 0 <= i < layer_hidden.shape[0]]
                if idxs:
                    window_avg = layer_hidden[idxs].mean(dim=0)
                else:
                    window_avg = torch.zeros(layer_hidden.shape[-1])
                hidden_averaged.append(window_avg)
            results[f'hidden_states_{window_name}'] = hidden_averaged

        # Generate answer via standard path with hidden states for decision token
        print("  Generating answer...")
        with torch.inference_mode():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = gen_outputs.sequences
        
        response = self._decode_output(output_ids)
        results['response'] = response
        results['predicted_answer'] = self.extract_answer(response)
        
        # Extract decision token (first generated token) hidden states
        # gen_outputs.hidden_states is a tuple of tuples
        # Structure: (step1, step2, ...) where each step has (layer0, layer1, ..., layerN)
        if hasattr(gen_outputs, 'hidden_states') and gen_outputs.hidden_states:
            first_gen_token_states = gen_outputs.hidden_states[0]  # First generation step
            
            # Extract from each layer for the decision token
            decision_token_hidden = []
            for layer_states in first_gen_token_states:
                # layer_states shape: [batch, 1, hidden_dim] (since it's one new token)
                decision_token_hidden.append(layer_states[0, -1, :].detach().cpu())
            
            results["hidden_states_decision_token"] = decision_token_hidden
            print(f"    Extracted decision token hidden states across {len(decision_token_hidden)} layers")

        print("  ✓ Extraction complete!\n")
        return results


# ======================================
# Data loader (filtered_data structure)
# ======================================
def load_questions_from_filtered_data(data_folder: str) -> List[Dict]:
    """Load questions from filtered_data structure."""
    data_folder = Path(data_folder)

    variants = ['notext', 'correct', 'irrelevant', 'misleading']

    variant_data = {}
    for variant in variants:
        json_file = data_folder / f'questions_{variant}.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                variant_data[variant] = json.load(f)
        else:
            print(f"Warning: JSON file not found: {json_file}")

    if not variant_data:
        raise FileNotFoundError(f"No question JSON files found in {data_folder}")

    base_variant = list(variant_data.keys())[0]
    base_questions = variant_data[base_variant]

    questions_data = []

    for idx, base_item in enumerate(base_questions):
        try:
            options = base_item['options']
        except KeyError:
            options = {'A': base_item['A'], 'B': base_item['B'], 'C': base_item['C'], 'D': base_item['D']}
        question_dict = {
            'question_id': base_item['question_id'],
            'question': base_item['question'],
            'options': options,
            'answer': base_item["answer"],
            'image_variants': {}
        }

        for variant in variants:
            if variant in variant_data and idx < len(variant_data[variant]):
                # variant_item = variant_data[variant][idx]
                variant_item = next((item for item in variant_data[variant] if item['question_id'] == base_item['question_id']), None)
                if variant_item is None:
                    print(f"Warning: No matching question_id {base_item['question_id']} in variant {variant}")
                    continue
                try:
                    image_name = variant_item['image']
                except KeyError:
                    image_name = variant_item['question_id'] + '.jpg'

                if image_name:
                    variant_image_path = data_folder / variant / image_name
                    if variant_image_path.exists():
                        question_dict['image_variants'][variant] = str(variant_image_path)
                    else:
                        print(f"Warning: Image not found: {variant_image_path}")

        if question_dict['image_variants']:
            questions_data.append(question_dict)

    # --- persist loaded questions for inspection ---
    try:
        export_file = data_folder / "questions_data_export.json"
        with open(export_file, "w") as f:
            json.dump(questions_data, f, indent=2)
        print(f"Saved aggregated questions_data to: {export_file}")
    except Exception as e:
        print(f"Warning: failed to write aggregated questions data: {e}")

    try:
        per_dir = data_folder / "questions_data_items"
        per_dir.mkdir(parents=True, exist_ok=True)
        for idx, q in enumerate(questions_data):
            qid = q.get("question_id", f"q{idx+1}")
            out_path = per_dir / f"{qid}.json"
            with open(out_path, "w") as f:
                json.dump(q, f, indent=2)
        print(f"Wrote individual question files to: {per_dir}")
    except Exception as e:
        print(f"Warning: failed to write per-question files: {e}")

    return questions_data


# ======================================
# Saving logic
# ======================================
def save_activations(results: Dict, output_path: str):
    """Save activation results (metadata as JSON, window-averaged tensors as .pt)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'question': results.get('question'),
        'options': results.get('options'),
        'correct_answer': results.get('correct_answer'),
        'variants': {}
    }

    tensors = {}

    for variant_name, variant_data in results['variants'].items():
        metadata['variants'][variant_name] = {
            'response': variant_data.get('response', ''),
            'predicted_answer': variant_data.get('predicted_answer', 'UNKNOWN'),
            'seq_len': variant_data.get('seq_len', 0),
            'num_vision_tokens': variant_data.get('num_vision_tokens', 0),
            'num_text_tokens': variant_data.get('num_text_tokens', 0),
            'architecture_mode': variant_data.get('architecture_mode', 'merged')
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


# ======================================
# Batch processing
# ======================================
def process_all_questions(
    evaluator: LlavaAutoEvaluator,
    questions_data: List[Dict],
    output_dir: str,
    max_new_tokens: int = 50,
    verify_encoders: bool = True,
    dump_tokens: bool = False
):
    """Process all questions and save activations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_questions = len(questions_data)
    processed_questions = 0

    print(f"\n{'='*70}")
    print(f"Processing {total_questions} questions with MERGED activation extraction (no manual packing)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    for idx, item in enumerate(tqdm(questions_data, desc="Processing questions")):
        question_id = item.get('question_id', f'q{idx+1}')

        # Skip if already processed
        output_file = output_path / f"{question_id}_activations.pt"
        if output_file.exists():
            print(f"\n[Question {idx+1}/{total_questions}] ID: {question_id}")
            print(f"  Already processed, skipping...")
            processed_questions += 1
            continue

        print(f"\n[Question {idx+1}/{total_questions}] ID: {question_id}")
        print(f"Question: {item['question'][:80]}...")

        if not item['image_variants']:
            print(f"  Skipping: No image variants found")
            continue

        try:
            # Build prompt once
            prompt = evaluator.format_mcq_prompt(item['question'], item['options'])

            result_pack = {
                'question': item['question'],
                'options': item['options'],
                'correct_answer': item.get('answer'),
                'variants': {}
            }

            for variant_name, image_path in item['image_variants'].items():
                print(f"  Variant: {variant_name} -> {image_path}")

                dump_path = None
                if dump_tokens:
                    dump_path = str(output_path / f"{question_id}_{variant_name}_tokens.json")

                variant_result = evaluator.extract_activations_with_answer(
                    image_path=image_path,
                    prompt=prompt,
                    question_text=item['question'],          # pass to verifier
                    options=item['options'],                 # pass to verifier
                    max_new_tokens=max_new_tokens,
                    verify_encoders=verify_encoders,
                    dump_tokens_path=dump_path
                )
                result_pack['variants'][variant_name] = variant_result

                print(f"    Predicted: {variant_result['predicted_answer']}")
                print(f"    Total sequence: {variant_result['seq_len']} tokens")
                print(f"    Vision: {variant_result['num_vision_tokens']} tokens")
                print(f"    Text: {variant_result['num_text_tokens']} tokens")

            # Save
            output_file = output_path / f"{question_id}_activations"
            save_activations(result_pack, str(output_file))

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


# ======================================
# CLI
# ======================================
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract activations from LLaVA (merged mode, no manual packing) with encoder verification and Q-only tokenization"
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default='llava-hf/llava-1.5-7b-hf',
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
        default='activations_merged',
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
        '--no_verify_encoders',
        action='store_true',
        help='Disable encoder/decoder length verification'
    )

    parser.add_argument(
        '--dump_tokens',
        action='store_true',
        help='Dump token strings per (question,variant) to JSON for inspection'
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_folder):
        parser.error(f"Data folder not found: {args.data_folder}")

    device = None if args.device == 'auto' else args.device

    print("="*70)
    print("LLAVA ACTIVATION EXTRACTOR - MERGED (NO MANUAL PACKING) + VERIFICATION + Q-ONLY")
    print("="*70)
    print("\nFeatures:")
    print("  • Uses model's internal vision/text merge")
    print("  • MERGED architecture (vision + text in same sequence)")
    print("  • Vision tokens evolve across layers (captured via hooks)")
    print("  • Encoder/decoder consistency check (text vs vision vs merged)")
    print("  • Question+options-only tokenization check (no chat template)")
    print("="*70 + "\n")

    print("Loading questions from filtered_data structure...")
    questions_data = load_questions_from_filtered_data(args.data_folder)
    print(f"Loaded {len(questions_data)} questions\n")

    evaluator = LlavaAutoEvaluator(model_id=args.model_id, device=device)

    process_all_questions(
        evaluator=evaluator,
        questions_data=questions_data,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        verify_encoders=not args.no_verify_encoders,
        dump_tokens=args.dump_tokens
    )


if __name__ == "__main__":
    main()