# cache_aggregated_attn.py
"""
Cache head-averaged attention maps (per layer) for Llava-NeXT on a HF dataset,
only for samples whose question_id appears in no_overlap_question_ids.txt.

What is saved per sample:
- attn_avg: (K, S, S) float16  where K=last_k_layers (or all layers), S=seq_len
- meta.json: includes qid, variant, seq_len, image placeholder positions, etc.

Notes:
- This captures attentions for the PROMPT (prefill) forward pass, not generation steps.
- Attentions come from the LLM decoder self-attention (text+image placeholders in one sequence).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


# -----------------------------
# Dataset utils (same idea as yours)
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


# -----------------------------
# Prompt formatting (simple + stable)
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


def build_options_from_sample(sample: Dict, shuffle: bool = False, seed: int = 0) -> Tuple[Dict[str, str], str]:
    """
    Mirrors your dataset structure:
      correct_answer.text
      misleading_groundable.text
      misleading_ungroundable.text
      irrelevant_word.text

    If you want exact same shuffling logic as your evaluator, you can copy it here.
    For caching attentions, stable order is usually better, so shuffle=False by default.
    """
    labels = ["A", "B", "C", "D"]
    candidates = [
        ("correct_answer", sample["correct_answer"]["text"]),
        ("misleading_groundable", sample["misleading_groundable"]["text"]),
        ("misleading_ungroundable", sample["misleading_ungroundable"]["text"]),
        ("irrelevant_word", sample["irrelevant_word"]["text"]),
    ]

    if shuffle:
        import random
        rng = random.Random(f"{seed}_{sample.get('question_id','')}")
        rng.shuffle(candidates)

    options = {labels[i]: txt for i, (_, txt) in enumerate(candidates)}
    correct_idx = next(i for i, (k, _) in enumerate(candidates) if k == "correct_answer")
    correct_letter = labels[correct_idx]
    return options, correct_letter


# -----------------------------
# Attention caching
# -----------------------------
@torch.no_grad()
def forward_get_head_avg_attn(
    model: LlavaNextForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
    *,
    last_k_layers: Optional[int] = None,
) -> np.ndarray:
    """
    Run a single forward pass with output_attentions=True and return head-averaged attention.

    Returns:
        attn_avg: np.ndarray of shape (K, S, S), dtype float16
          K = last_k_layers if specified else num_layers
          S = sequence length
    """
    out = model(
        **inputs,
        output_attentions=True,
        return_dict=True,
        use_cache=False,  # important: we want full attention matrices, not cached incremental behavior
    )

    # out.attentions is typically a tuple length L
    # each: (B, H, S, S)
    attn_layers = out.attentions
    if attn_layers is None:
        raise RuntimeError("Model returned no attentions. Make sure output_attentions=True is supported.")

    if last_k_layers is not None:
        attn_layers = attn_layers[-last_k_layers:]

    # head-average per layer -> (B, S, S), then drop batch -> (S, S)
    attn_avg = []
    for a in attn_layers:
        # a: (B, H, S, S)
        a_mean = a.mean(dim=1)      # (B, S, S)
        a0 = a_mean[0]              # (S, S)
        attn_avg.append(a0)

    attn_avg = torch.stack(attn_avg, dim=0)  # (K, S, S)

    # store compactly
    return attn_avg.to(torch.float16).cpu().numpy()


def get_image_placeholder_positions(model, input_ids: torch.Tensor) -> List[int]:
    """
    Positions in the full sequence that correspond to <image> placeholders (image_token_id).
    """
    image_token_id = model.config.image_token_id
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return pos.tolist()


def save_sample(out_dir: str, qid: str, variant: str, attn_avg: np.ndarray, meta: Dict):
    sample_dir = Path(out_dir) / variant / qid
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save attention tensor
    # (K,S,S) float16 compressed
    np.savez_compressed(sample_dir / "attn_avg_layers.npz", attn_avg=attn_avg)

    # Save metadata
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--hf_dataset", type=str, required=True)
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_dataset_local_cache")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="notext",
                        help="Which dataset image variant to use (notext/correct_answer/...).")
    parser.add_argument("--qid_file", type=str, default="no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="attn_cache")
    parser.add_argument("--last_k_layers", type=int, default=0,
                        help="If >0, save only last K layers. If 0, save all layers.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, stop after this many cached samples (debug).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    print(f"Loading model: {args.model_id} on {device}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    last_k = args.last_k_layers if args.last_k_layers > 0 else None

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching attentions ({args.variant})"):
        sample = ds[i]
        qid = sample.get("question_id", f"unknown_{i}")
        if qid not in qids:
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
        options, correct_letter = build_options_from_sample(sample, shuffle=False)
        prompt_text = format_mcq_prompt(sample.get("question", ""), options)

        # IMPORTANT: LlavaNextProcessor expects <image> inside prompt
        # Here we keep it minimal and stable:
        # If you want chat template usage, you can add it, but this is usually fine for attention caching.
        prompt = f"[INST] <image>\n{prompt_text} [/INST]"

        inputs = processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)

        # forward -> head-avg attn
        attn_avg = forward_get_head_avg_attn(model, inputs, last_k_layers=last_k)

        # metadata
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])
        img_pos = get_image_placeholder_positions(model, input_ids)

        meta = {
            "question_id": qid,
            "variant": args.variant,
            "seq_len": seq_len,
            "num_layers_saved": int(attn_avg.shape[0]),
            "image_placeholder_count": len(img_pos),
            "image_placeholder_positions": img_pos,
            "correct_letter": correct_letter,  # optional
        }

        save_sample(args.out_dir, qid, args.variant, attn_avg, meta)

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Cached {kept} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()