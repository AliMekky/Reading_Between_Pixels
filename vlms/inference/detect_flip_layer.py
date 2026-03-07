# layerwise_next_token_flip_analysis_llava_next.py
"""
Layerwise next-token prediction + flip segmentation for Llava-NeXT.

For each (image, question):
- Runs a single forward pass on the PROMPT (no generation loop)
- Extracts per-layer hidden states at the last prompt position
- Projects each layer state through final norm + lm_head to get per-layer next-token logits
- Computes:
    - per-layer predicted token (argmax)
    - logit margin = top1_logit - top2_logit
    - entropy of next-token distribution
- Converts predicted token -> label in {A,B,C,D,UNKNOWN} (robust to ' A' tokens)
- Produces flip segments like: A:0->12, B:12->15, A:15->32

Notes:
- "layer index" here refers to transformer block index: 0..(num_layers-1).
- hidden_states[0] (embeddings) is ignored for segmentation by default.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


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
    qids: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.add(s)
    return qids


# -----------------------------
# Prompt formatting (match evaluator)
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
# Model component access (Llava-NeXT)
# -----------------------------
def get_lm_and_head_and_norm(model):
    """
    Returns (lm, lm_head, final_norm_or_identity)

    - Finds lm_head robustly
    - Finds final norm via common attribute paths OR by scanning named_modules
    - Falls back to nn.Identity() if not found (still works; just noisier early layers)
    """
    import torch

    if not hasattr(model, "language_model"):
        raise RuntimeError("Expected model.language_model for LlavaNextForConditionalGeneration")

    lm = model.language_model

    # ---- lm_head ----
    lm_head = None
    if hasattr(lm, "lm_head"):
        lm_head = lm.lm_head
    elif hasattr(model, "lm_head"):
        lm_head = model.lm_head
    else:
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Could not locate lm_head / output embeddings.")

    # ---- final norm: try common paths first ----
    candidates = []

    def try_get(obj, path: str):
        cur = obj
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    common_paths = [
        "model.norm",                 # Llama/Mistral/RMSNorm
        "model.final_layernorm",      # some architectures
        "transformer.ln_f",           # GPT-style
        "gpt_neox.final_layer_norm",  # GPT-NeoX style
        "model.decoder.norm",         # some seq2seq-ish wrappers
        "decoder.norm",
    ]

    for p in common_paths:
        m = try_get(lm, p)
        if m is not None:
            candidates.append((p, m))

    # If not found, scan named_modules for something that looks like the final norm
    # (LayerNorm or RMSNorm-like modules often contain 'norm' in their name)
    if not candidates:
        for name, mod in lm.named_modules():
            # common final norms often named exactly like these
            if name.endswith(".norm") or name.endswith("ln_f") or name.endswith("final_layernorm"):
                candidates.append((name, mod))

    # Validate candidate types: LayerNorm or RMSNorm-like (has weight + forward)
    def is_norm_module(m):
        if isinstance(m, torch.nn.LayerNorm):
            return True
        # RMSNorm classes vary; heuristic:
        return hasattr(m, "weight") and callable(getattr(m, "forward", None)) and "norm" in m.__class__.__name__.lower()

    candidates = [(n, m) for (n, m) in candidates if is_norm_module(m)]

    if candidates:
        # Prefer the “most final-looking” by longest name (often deeper)
        candidates.sort(key=lambda x: len(x[0]))
        final_norm_name, final_norm = candidates[-1]
        # print(f"[info] using final norm: {final_norm_name} ({final_norm.__class__.__name__})")
        return lm, lm_head, final_norm

    # ---- fallback: no norm found ----
    # print("[warn] Could not locate final norm; using Identity(). Early-layer logits may be noisier.")
    return lm, lm_head, torch.nn.Identity()
# -----------------------------
# Label extraction from a single token
# -----------------------------
def token_to_abcd_label(token_text: str) -> str:
    """
    Robustly map decoded single token to {A,B,C,D,UNKNOWN}.
    Handles tokens like ' A', 'A', '\nA', etc.
    """
    s = token_text.strip()
    if not s:
        return "UNKNOWN"
    c = s[0].upper()
    return c if c in ("A", "B", "C", "D") else "UNKNOWN"


# -----------------------------
# Flip segmentation
# -----------------------------
def segment_flips(labels: List[str]) -> Tuple[str, Optional[int], List[Tuple[str, int, int]]]:
    """
    labels: per-layer labels for layers 0..L-1
    Returns:
      - pretty string like "A:0->12, B:12->15, A:15->32"
      - first_flip_layer (int) where a change starts, else None
      - segments list [(label,start,end_exclusive), ...]
    """
    if not labels:
        return "", None, []
    segs: List[Tuple[str, int, int]] = []
    start = 0
    cur = labels[0]
    first_flip: Optional[int] = None

    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append((cur, start, i))
            if first_flip is None:
                first_flip = i
            start = i
            cur = labels[i]
    segs.append((cur, start, len(labels)))

    pretty = ", ".join([f"{lab}:{a}->{b}" for (lab, a, b) in segs])
    return pretty, first_flip, segs


# -----------------------------
# Core analysis
# -----------------------------
@torch.no_grad()
def layerwise_next_token_analysis(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    inputs: Dict[str, torch.Tensor],
    *,
    use_fp16_entropy: bool = False,
) -> Dict[str, Any]:
    """
    Runs a single forward pass over the prompt and extracts per-layer next-token predictions.
    """
    tokenizer = processor.tokenizer

    lm, lm_head, final_norm = get_lm_and_head_and_norm(model)

    # Forward pass with hidden states
    out = model(
        **inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden states location varies; prefer out.hidden_states if present
    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is None and hasattr(out, "language_model_outputs"):
        hidden_states = out.language_model_outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("No hidden_states found in model output (need output_hidden_states=True).")

    # hidden_states: tuple length (num_layers + 1), with [0]=embeddings, [i+1]=after block i
    num_layers = len(hidden_states) - 1
    last_pos = inputs["input_ids"].shape[1] - 1  # last prompt token position

    per_layer = []
    labels = []

    # For each transformer block layer
    for layer_idx in range(num_layers):
        h = hidden_states[layer_idx + 1]  # (B, S, D)
        x = h[:, last_pos, :]             # (B, D)

        # Apply final norm before unembedding (logit-lens style)
        x = final_norm(x)

        # Project to vocab logits
        logits = lm_head(x)  # (B, V)

        # top1/top2 + margin
        top2 = torch.topk(logits, k=2, dim=-1)
        top1_logit = float(top2.values[0, 0].item())
        top2_logit = float(top2.values[0, 1].item())
        top1_id = int(top2.indices[0, 0].item())
        top2_id = int(top2.indices[0, 1].item())
        margin = top1_logit - top2_logit

        # entropy
        # Full entropy over vocab; computed efficiently via log_softmax.
        if use_fp16_entropy:
            log_probs = F.log_softmax(logits.to(torch.float16), dim=-1).to(torch.float32)
        else:
            log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum(dim=-1)[0].item())

        tok1 = tokenizer.decode([top1_id], skip_special_tokens=False)
        lab = token_to_abcd_label(tok1)

        labels.append(lab)
        per_layer.append({
            "layer": layer_idx,
            "top1": {"token_id": top1_id, "token": tok1, "logit": top1_logit, "label": lab},
            "top2": {"token_id": top2_id, "token": tokenizer.decode([top2_id], skip_special_tokens=False), "logit": top2_logit},
            "margin": float(margin),
            "entropy": float(entropy),
        })

    pretty, first_flip, segs = segment_flips(labels)

    # Also compute the model's actual greedy next token from final logits (sanity)
    final_logits = out.logits[:, -1, :]  # (B,V)
    final_id = int(final_logits.argmax(dim=-1)[0].item())
    final_tok = tokenizer.decode([final_id], skip_special_tokens=False)
    final_lab = token_to_abcd_label(final_tok)

    return {
        "num_layers": num_layers,
        "last_prompt_pos": int(last_pos),
        "final_greedy": {"token_id": final_id, "token": final_tok, "label": final_lab},
        "flip_segments": pretty,
        "first_flip_layer": first_flip,
        "segments": [{"label": a, "start": b, "end": c} for (a, b, c) in segs],
        "per_layer": per_layer,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--hf_dataset", type=str, required=True)
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="layerwise_flip_cache")

    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="0 = no limit")

    # Keep consistent with evaluator
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Numerics
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use_fp16_entropy", action="store_true", help="Compute log_softmax in fp16 then cast to fp32 (faster, less stable).")
    parser.add_argument("--save_full_trace", action="store_true", help="If not set, drop per_layer to keep JSON small.")

    args = parser.parse_args()

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_t = torch.device(device)

    print(f"Loading model: {args.model_id} on {device} (attn_implementation={args.attn_implementation})")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device_t)
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    out_root = Path(args.out_dir) / args.variant
    out_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Layerwise flip analysis [{args.variant}]"):
        sample = ds[i]
        qid = str(sample.get("question_id", f"unknown_{i}"))
        if qid not in qids:
            continue
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        out_path = out_root / f"{qid}.json"
        if out_path.exists():
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

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=img, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)

        analysis = layerwise_next_token_analysis(
            model, processor, inputs, use_fp16_entropy=args.use_fp16_entropy
        )

        record = {
            "question_id": qid,
            "variant": args.variant,
            "correct_letter": correct_letter,
            "option_meta": option_meta,
            "flip_segments": analysis["flip_segments"],
            "first_flip_layer": analysis["first_flip_layer"],
            "final_greedy": analysis["final_greedy"],
            "num_layers": analysis["num_layers"],
        }
        if args.save_full_trace:
            record["segments"] = analysis["segments"]
            record["per_layer"] = analysis["per_layer"]

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Wrote {kept} JSON files into: {out_root}")


if __name__ == "__main__":
    main()