#!/usr/bin/env python3
# cache_generated_token_attn_with_mapping_npz_qwen_two_models.py
"""
Generation-step attention caching for Qwen2-VL / Qwen2.5-VL
using TWO MODELS:

1) Prediction model:
   - loaded EXACTLY like your current inference-style path
   - used only to choose the greedy next token

2) Attention model:
   - loaded separately with eager attention
   - used only for manual prefill + one-step decode with output_attentions=True

Why this version exists
- Your current Qwen setup produces sensible next-token logits,
  but returns None attentions on the decode step.
- This script preserves your current prediction model loading,
  while extracting attentions from a separate model/backend that is more likely
  to return real tensors.

Saved NPZ fields
- attn                        : (L, S_prompt) float16
- meta                        : json string
- packed_mapping_summary      : json string
- packed_mapping_tokens       : json string
- image_placeholder_positions : int32 array
- prompt_input_ids            : int32 array
- attention_mask              : int32 array (if present)

Notes
- The saved attention corresponds to the token chosen by the prediction model.
- The attention itself is extracted from the separate attention model.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List, Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# -----------------------------
# Answer extraction
# -----------------------------
def extract_answer_letter(response: str) -> str:
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
        r"ANSWER[:\s]+([ABCD])\b",
        r"^\s*([ABCD])\s*$",
        r"^([ABCD])\b",
        r"\b([ABCD])\s*$",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, assistant_response_upper)
        if match:
            return match.group(1)

    return "UNKNOWN"


# -----------------------------
# Dataset utils
# -----------------------------
def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = local_cache_root / sanitize_repo_id(dataset_id)

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


def load_qid_whitelist(path: str) -> Set[str]:
    qids = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.add(s)
    return qids


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


# -----------------------------
# Prompt formatting
# -----------------------------
def format_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    instruction = "Answer the following multiple-choice question by selecting the correct option."
    prompt = f"{instruction}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for k in ["A", "B", "C", "D"]:
        if k in options:
            prompt += f"{k}) {options[k]}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def build_options_from_sample(sample: Dict, shuffle: bool = False, seed: int = 0) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
    labels = ["A", "B", "C", "D"]
    candidates = [
        {"key": "correct_answer", "text": sample["correct_answer"]["text"]},
        {"key": "misleading_groundable", "text": sample["misleading_groundable"]["text"]},
        {"key": "misleading_ungroundable", "text": sample["misleading_ungroundable"]["text"]},
        {"key": "irrelevant_word", "text": sample["irrelevant_word"]["text"]},
    ]

    qid = str(sample.get("question_id", ""))
    if shuffle:
        import random
        rng = random.Random(f"{seed}_{qid}")
        rng.shuffle(candidates)

    options = {labels[i]: candidates[i]["text"] for i in range(4)}
    correct_index = next(i for i, c in enumerate(candidates) if c["key"] == "correct_answer")
    correct_letter = labels[correct_index]

    option_meta = {
        "order": [c["key"] for c in candidates],
        "label_to_key": {labels[i]: candidates[i]["key"] for i in range(4)},
        "shuffle": bool(shuffle),
        "seed": int(seed),
    }
    return options, correct_letter, option_meta


# -----------------------------
# Qwen input preparation
# -----------------------------
def prepare_qwen_inputs(processor, image: Image.Image, prompt_text: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs, text, messages


# -----------------------------
# Debug helpers
# -----------------------------
def debug_print_prompt_and_tokens(processor, inputs, formatted_prompt: str, max_tokens_to_show: int = 120):
    tokenizer = processor.tokenizer
    input_ids = inputs["input_ids"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n=== FORMATTED PROMPT ===")
    print(formatted_prompt)
    print("========================")
    print(f"Prompt seq_len = {len(input_ids)}")
    print(f"First {min(max_tokens_to_show, len(toks))} prompt tokens:")
    print(toks[:max_tokens_to_show])
    print()


def debug_print_topk_next_tokens(logits: torch.Tensor, tokenizer, k: int = 10):
    scores = logits[0, -1]
    topk_vals, topk_ids = torch.topk(scores, k=k, dim=-1)

    print(f"Top-{k} next-token candidates from prefill:")
    for rank, (tid, val) in enumerate(zip(topk_ids.tolist(), topk_vals.tolist()), start=1):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"  {rank:02d}. id={tid:<6d} token={tok_str!r:<12} logit={val:.4f}")
    print()


# -----------------------------
# Image placeholder positions
# -----------------------------
def get_image_placeholder_positions(processor, input_ids: torch.Tensor) -> List[int]:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    print(f"Found {len(pos)} image placeholder positions at indices: {pos.tolist()}")
    return pos.tolist()


# -----------------------------
# Qwen image-token mapping
# -----------------------------
def build_qwen_image_token_map_for_one_image(
    *,
    image: Image.Image,
    image_grid_thw: Tuple[int, int, int],
    patch_size: int,
    merge_size: int,
) -> Dict[str, Any]:
    orig_w, orig_h = image.size
    T, H, W = image_grid_thw

    if T != 1:
        raise ValueError(f"This script assumes a still image with T=1, got T={T}")

    merged_h = H // merge_size
    merged_w = W // merge_size
    num_image_tokens = merged_h * merged_w

    resized_w = W * patch_size
    resized_h = H * patch_size

    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    tokens: List[Dict[str, Any]] = []
    token_idx = 0

    for row in range(merged_h):
        for col in range(merged_w):
            patch_row_start = row * merge_size
            patch_row_end = (row + 1) * merge_size
            patch_col_start = col * merge_size
            patch_col_end = (col + 1) * merge_size

            x0_r = patch_col_start * patch_size
            y0_r = patch_row_start * patch_size
            x1_r = patch_col_end * patch_size
            y1_r = patch_row_end * patch_size

            x0 = x0_r * scale_x
            y0 = y0_r * scale_y
            x1 = x1_r * scale_x
            y1 = y1_r * scale_y

            tokens.append(
                {
                    "token_idx": token_idx,
                    "kind": "merged_patch",
                    "row": row,
                    "col": col,
                    "patch_rows": (patch_row_start, patch_row_end - 1),
                    "patch_cols": (patch_col_start, patch_col_end - 1),
                    "resized_box_xyxy": (x0_r, y0_r, x1_r, y1_r),
                    "bbox": (y0, x0, y1, x1),
                }
            )
            token_idx += 1

    return {
        "summary": {
            "original_size_hw": (orig_h, orig_w),
            "resized_size_hw": (resized_h, resized_w),
            "image_grid_thw": (T, H, W),
            "patch_size": patch_size,
            "merge_size": merge_size,
            "merged_grid_hw": (merged_h, merged_w),
            "total_packed_image_tokens": num_image_tokens,
        },
        "tokens": tokens,
    }


# -----------------------------
# Prediction model: choose token
# -----------------------------
@torch.no_grad()
def choose_next_token_with_pred_model(pred_model, processor, pred_inputs, debug_topk: int = 10):
    out0 = pred_model(
        **pred_inputs,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )

    debug_print_topk_next_tokens(out0.logits, processor.tokenizer, k=debug_topk)

    next_id = out0.logits[:, -1].argmax(dim=-1, keepdim=True)

    token_id = int(next_id[0, 0].item())
    token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
    pred_letter = extract_answer_letter(token_text)

    print(f"Greedy next token from prediction model: id={token_id} text={token_text!r}")
    print(f"Parsed predicted answer letter from single token: {pred_letter!r}")
    print()

    return next_id, token_id, token_text, pred_letter


# -----------------------------
# Attention model: manual prefill + forced one-step decode
# -----------------------------
@torch.no_grad()
def attention_for_forced_token(attn_model, processor, attn_inputs, forced_next_id):
    """
    attn_inputs must already be on attn_model device.
    forced_next_id must already be on attn_model device.
    """
    S_prompt = int(attn_inputs["input_ids"].shape[1])

    out0 = attn_model(
        **attn_inputs,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = out0.past_key_values

    if "attention_mask" in attn_inputs:
        attn_mask2 = torch.cat(
            [attn_inputs["attention_mask"], torch.ones_like(forced_next_id)],
            dim=1,
        )
    else:
        attn_mask2 = None

    out1 = attn_model(
        input_ids=forced_next_id,
        attention_mask=attn_mask2,
        past_key_values=past,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

    step_attn = out1.attentions
    if step_attn is None:
        raise RuntimeError("Attention model returned no attentions.")

    print("Attention tensor shapes by layer:")
    for li, layer_attn in enumerate(step_attn):
        if layer_attn is None:
            print(f"  layer {li}: None")
        else:
            print(f"  layer {li}: {tuple(layer_attn.shape)}")
    print()

    layer_vecs = []
    S_so_far_ref = None

    for layer_attn in step_attn:
        if layer_attn is None:
            raise RuntimeError("Found None attention tensor in attention model.")
        S_so_far = int(layer_attn.shape[-1])
        if S_so_far_ref is None:
            S_so_far_ref = S_so_far
        v = layer_attn[0].mean(dim=0)[0]
        layer_vecs.append(v[:S_prompt])

    A = torch.stack(layer_vecs, dim=0)
    attn_np = A.to(torch.float16).cpu().numpy()

    return attn_np, int(S_so_far_ref)


# -----------------------------
# Save NPZ
# -----------------------------
def save_npz_sample(
    out_dir: str,
    qid: str,
    variant: str,
    *,
    attn_np: np.ndarray,
    meta: Dict[str, Any],
    mapping: Dict[str, Any],
    image_placeholder_positions: List[int],
    prompt_input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
    mode: str = "gen_token",
):
    sample_dir = Path(out_dir) / variant / qid
    sample_dir.mkdir(parents=True, exist_ok=True)

    npz_path = sample_dir / f"gen_attn_{mode}.npz"

    save_dict: Dict[str, Any] = {
        "attn": attn_np.astype(np.float16, copy=False),
        "meta": json.dumps(meta),
        "packed_mapping_summary": json.dumps(mapping["summary"]),
        "packed_mapping_tokens": json.dumps(mapping["tokens"]),
        "image_placeholder_positions": np.asarray(image_placeholder_positions, dtype=np.int32),
        "prompt_input_ids": prompt_input_ids.astype(np.int32),
    }
    if attention_mask is not None:
        save_dict["attention_mask"] = attention_mask.astype(np.int32)

    np.savez_compressed(str(npz_path), **save_dict)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="qwen-vl_attentions")
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--attn_device", type=str, default="cpu", choices=["cuda", "cpu"])

    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug_first_only", action="store_true")
    parser.add_argument("--debug_topk", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    pred_device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    attn_device = args.attn_device if torch.cuda.is_available() and args.attn_device == "cuda" else "cpu"

    pred_device_t = torch.device(pred_device)
    attn_device_t = torch.device(attn_device)

    print(f"Loading prediction model: {args.model_id} on {pred_device}")
    print("Prediction model loading is kept unchanged.")
    pred_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map="auto"
    )
    pred_model.eval()

    print(f"Loading attention model: {args.model_id} on {attn_device}")
    attn_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        device_map=None,
        attn_implementation="eager",
    ).to(attn_device_t)
    attn_model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    variant_root = out_root / args.variant
    variant_root.mkdir(parents=True, exist_ok=True)
    existing_qid_dirs = {d.name for d in variant_root.iterdir() if d.is_dir()}
    print(f"Found {len(existing_qid_dirs)} existing qid dirs in {variant_root}")

    mismatch_log = out_root / "mismatched_qids.txt"

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching gen-token attn [Qwen-2models] [{args.variant}]"):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        sample = ds[i]
        qid = str(sample.get("question_id", f"unknown_{i}"))
        if qid not in qids:
            continue
        if qid in existing_qid_dirs:
            continue

        if args.variant == "notext":
            img = sample["notext"]["image"]
        else:
            if args.variant not in sample:
                continue
            img = sample[args.variant]["image"]
        img = img.convert("RGB")

        options, correct_letter, option_meta = build_options_from_sample(
            sample, shuffle=args.shuffle_options, seed=args.seed
        )
        prompt_text = format_mcq_prompt(sample.get("question", ""), options)

        base_inputs, formatted_prompt, _ = prepare_qwen_inputs(processor, img, prompt_text)
        debug_print_prompt_and_tokens(processor, base_inputs, formatted_prompt)

        # prediction inputs
        pred_inputs = {k: v.to(pred_device_t) for k, v in base_inputs.items()}

        # choose token with prediction model
        next_id_pred, token_id, token_text, pred_letter = choose_next_token_with_pred_model(
            pred_model,
            processor,
            pred_inputs,
            debug_topk=args.debug_topk,
        )

        # placeholder positions from original/base inputs
        img_pos = get_image_placeholder_positions(processor, base_inputs["input_ids"])

        # mapping prerequisites from original/base inputs
        if "image_grid_thw" not in base_inputs:
            raise RuntimeError("Processor did not return image_grid_thw; Qwen mapping requires it.")

        grid = tuple(int(x) for x in base_inputs["image_grid_thw"][0].tolist())
        patch_size = int(processor.image_processor.patch_size)
        merge_size = int(processor.image_processor.merge_size)

        print(f"image_grid_thw: {grid}")
        print(f"patch_size: {patch_size}")
        print(f"merge_size: {merge_size}")
        print()

        packed_mapping = build_qwen_image_token_map_for_one_image(
            image=img,
            image_grid_thw=grid,
            patch_size=patch_size,
            merge_size=merge_size,
        )

        expected_img_tokens = int(packed_mapping["summary"]["total_packed_image_tokens"])
        mapping_mismatch = (len(img_pos) != expected_img_tokens)
        if mapping_mismatch:
            print(
                f"[{qid}] WARNING: token mapping mismatch | "
                f"len(image_placeholder_positions)={len(img_pos)} != expected={expected_img_tokens}"
            )
            append_line(mismatch_log, qid)

        # attention inputs on separate model/device
        attn_inputs = {k: v.to(attn_device_t) for k, v in base_inputs.items()}
        forced_next_id = next_id_pred.to(attn_device_t)

        try:
            attn_np, S_so_far_ref = attention_for_forced_token(
                attn_model,
                processor,
                attn_inputs,
                forced_next_id,
            )
        except Exception as e:
            print(f"[{qid}] ERROR during attention extraction: {e}")
            append_line(mismatch_log, qid)
            continue

        meta = {
            "question_id": qid,
            "variant": args.variant,
            "prompt_seq_len": int(base_inputs["input_ids"].shape[1]),
            "image_placeholder_count": int(len(img_pos)),
            "expected_packed_image_tokens": int(expected_img_tokens),
            "mapping_mismatch": bool(mapping_mismatch),
            "correct_letter": correct_letter,
            "option_meta": option_meta,
            "image_grid_thw": list(grid),
            "patch_size": patch_size,
            "merge_size": merge_size,
            "original_image_size_hw": [img.height, img.width],
            "formatted_prompt": formatted_prompt,
            "prediction_model_device": pred_device,
            "attention_model_device": attn_device,
            "attention_model_attn_impl": "eager",
            "num_layers": int(attn_np.shape[0]),
            "attn_seq_len_so_far": int(S_so_far_ref),
            "generated_token_id": int(token_id),
            "generated_token_text": token_text,
            "predicted_answer_letter": pred_letter,
            "max_new_tokens": 1,
        }

        prompt_input_ids = base_inputs["input_ids"][0].detach().cpu().numpy()
        attention_mask = base_inputs["attention_mask"][0].detach().cpu().numpy() if "attention_mask" in base_inputs else None

        save_npz_sample(
            args.out_dir,
            qid,
            args.variant,
            attn_np=attn_np,
            meta=meta,
            mapping=packed_mapping,
            image_placeholder_positions=img_pos,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            mode="gen_token",
        )

        if args.debug_first_only:
            print(f"[debug_first_only] Saved first sample qid={qid}. Exiting.")
            break

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Saved {kept} samples into: {args.out_dir}")
    print(f"Mismatches / failures logged to: {mismatch_log}")


if __name__ == "__main__":
    main()