#!/usr/bin/env python3
# cache_generated_token_attn_with_mapping_npz_qwen.py
"""
Generation-step attention caching for Qwen2-VL / Qwen2.5-VL, aligned with the Qwen IG script.

What it does
- Loads HF dataset samples filtered by question_id list
- Builds the same MCQ prompt as your evaluator
- Uses Qwen chat template with one image + text prompt
- Runs a manual 2-step decode:
    1) prefill on the prompt
    2) one greedy decode step with attentions
- Extracts per-layer head-averaged attention from the generated token -> prompt tokens only
- Builds Qwen image-token mapping from image_grid_thw / patch_size / merge_size
- Saves everything into one compressed NPZ per sample

Saved NPZ fields
- attn                     : (L, S_prompt) float16
- meta                     : json string
- packed_mapping_summary   : json string
- packed_mapping_tokens    : json string
- image_placeholder_positions : int32 array
- prompt_input_ids         : int32 array
- attention_mask           : int32 array (if present)

Notes
- Mapping is for Qwen merged image tokens (<|image_pad|> positions only).
- One image token corresponds to one merged patch block on the resized patch grid.
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
# Image placeholder positions
# -----------------------------
def get_image_placeholder_positions(processor, input_ids: torch.Tensor) -> List[int]:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
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
    """
    Builds mapping from Qwen image token index -> merged patch region.
    bbox format matches your other scripts: (y0, x0, y1, x1) in original image coordinates.
    """
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
# Generation-step attention extraction (manual 2-step version)
# -----------------------------
@torch.no_grad()
def next_token_attn_to_prompt(model, processor, inputs):
    S_prompt = int(inputs["input_ids"].shape[1])

    # 1) Prefill pass
    out0 = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = out0.past_key_values

    # greedy next token
    next_id = out0.logits[:, -1].argmax(dim=-1, keepdim=True)

    # 2) One decode step with attentions
    if "attention_mask" in inputs:
        attn_mask2 = torch.cat([inputs["attention_mask"], torch.ones_like(next_id)], dim=1)
    else:
        attn_mask2 = None

    out1 = model(
        input_ids=next_id,
        attention_mask=attn_mask2,
        past_key_values=past,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

    step_attn = out1.attentions  # tuple(L) of (B,H,1,S_so_far)

    layer_vecs = []
    S_so_far_ref = None
    for layer_attn in step_attn:
        if layer_attn is None:
            raise RuntimeError("Found None attention tensor.")
        S_so_far = int(layer_attn.shape[-1])
        if S_so_far_ref is None:
            S_so_far_ref = S_so_far
        v = layer_attn[0].mean(dim=0)[0]   # (S_so_far,)
        layer_vecs.append(v[:S_prompt])    # prompt-only slice

    A = torch.stack(layer_vecs, dim=0)  # (L, S_prompt)
    attn_np = A.to(torch.float16).cpu().numpy()

    gen_token_id = int(next_id[0, 0].item())
    tok = getattr(processor, "tokenizer", None)
    gen_token_text = tok.decode([gen_token_id], skip_special_tokens=False) if tok else str(gen_token_id)

    info = {
        "num_layers": int(A.shape[0]),
        "prompt_seq_len": int(S_prompt),
        "attn_seq_len_so_far": int(S_so_far_ref),
        "generated_token_id": gen_token_id,
        "generated_token_text": gen_token_text,
        "max_new_tokens": 1,
    }
    return attn_np, info


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="attn_cache_gen_qwen")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug_first_only", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device_t = torch.device(device)

    print(f"Loading model: {args.model_id} on {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
        device_map=None,
    ).to(device_t)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    variant_root = out_root / args.variant
    variant_root.mkdir(parents=True, exist_ok=True)
    existing_qid_dirs = {d.name for d in variant_root.iterdir() if d.is_dir()}
    print(f"Found {len(existing_qid_dirs)} existing qid dirs in {variant_root}")

    mismatch_log = out_root / "mismatched_qids.txt"

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching gen-token attn [Qwen] [{args.variant}]"):
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

        # choose image variant
        if args.variant == "notext":
            img = sample["notext"]["image"]
        else:
            if args.variant not in sample:
                continue
            img = sample[args.variant]["image"]
        img = img.convert("RGB")

        # build prompt
        options, correct_letter, option_meta = build_options_from_sample(
            sample, shuffle=args.shuffle_options, seed=args.seed
        )
        prompt_text = format_mcq_prompt(sample.get("question", ""), options)

        inputs, formatted_prompt, _ = prepare_qwen_inputs(processor, img, prompt_text)
        inputs = {k: v.to(device_t) for k, v in inputs.items()}

        # attention
        attn_np, gen_info = next_token_attn_to_prompt(model, processor, inputs)

        # placeholders
        input_ids = inputs["input_ids"]
        img_pos = get_image_placeholder_positions(processor, input_ids)

        # mapping prerequisites
        if "image_grid_thw" not in inputs:
            raise RuntimeError("Processor did not return image_grid_thw; Qwen mapping requires it.")

        grid = tuple(int(x) for x in inputs["image_grid_thw"][0].tolist())
        patch_size = int(processor.image_processor.patch_size)
        merge_size = int(processor.image_processor.merge_size)

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

        meta = {
            "question_id": qid,
            "variant": args.variant,
            "prompt_seq_len": int(input_ids.shape[1]),
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
            **gen_info,
        }

        prompt_input_ids = input_ids[0].detach().cpu().numpy()
        attention_mask = inputs["attention_mask"][0].detach().cpu().numpy() if "attention_mask" in inputs else None

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
    print(f"Mismatches (if any) logged to: {mismatch_log}")


if __name__ == "__main__":
    main()