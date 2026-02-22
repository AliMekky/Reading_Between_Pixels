import sys
from pathlib import Path

# Add project root to PYTHONPATH dynamically
project_root = Path(__file__).resolve().parents[4]  # adjust depth if needed
sys.path.append(str(project_root))


#!/usr/bin/env python3
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from reading_between_pixels.Reading_Between_Pixels.vlms.inference.infere_vlms import get_evaluator


# -----------------------------
# utilities
# -----------------------------
def hdr(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def kv(k: str, v: Any):
    print(f"{k:<44}: {v}")


def warn(msg: str):
    print(f"⚠️  {msg}")


def ok(msg: str):
    print(f"✅ {msg}")


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for t in x:
            if torch.is_tensor(t):
                return t
    if isinstance(x, dict):
        for t in x.values():
            if torch.is_tensor(t):
                return t
    return None


# -----------------------------
# build inputs (same template style)
# -----------------------------
def build_inputs(processor: Any, device: torch.device, image: Image.Image, question: str) -> Dict[str, torch.Tensor]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }
    ]
    formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted, return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


# -----------------------------
# find LLM "first layer" robustly
# -----------------------------
def find_first_decoder_layer(model: Any) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    """
    Try to locate the first transformer block in the *language model*.
    Works across many HF-style LLaMA/decoder implementations.
    Returns (module_name, module).
    """

    # Common candidate roots that contain decoder stacks
    candidate_roots = []
    for name in ["language_model", "model", "llm", "lm", "transformer"]:
        if hasattr(model, name):
            candidate_roots.append((name, getattr(model, name)))
    # also try nested model.model
    if hasattr(model, "model") and hasattr(model.model, "model"):
        candidate_roots.append(("model.model", model.model))

    # 1) direct known paths (fast path)
    direct_paths = [
        "language_model.model.layers.0",
        "language_model.layers.0",
        "model.layers.0",
        "model.model.layers.0",
        "lm.model.layers.0",
        "transformer.h.0",          # GPT-like
        "transformer.blocks.0",     # some variants
        "decoder.layers.0",
    ]

    def get_by_path(root, path: str):
        obj = root
        for part in path.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    for p in direct_paths:
        m = get_by_path(model, p)
        if m is not None:
            return p, m

    # 2) heuristic: find a ModuleList called "layers" and take [0]
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            return f"{name}.0", mod[0]

    # 3) heuristic: find a ModuleList called "h" (GPT-style)
    for name, mod in model.named_modules():
        if name.endswith("h") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            return f"{name}.0", mod[0]

    return None, None


def find_embed_tokens(model: Any) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    """
    Attempt to locate token embedding module for text (embed_tokens / tok_embeddings).
    """
    candidates = []
    for name, mod in model.named_modules():
        ln = name.lower()
        if ln.endswith("embed_tokens") or ln.endswith("tok_embeddings") or ln.endswith("wte"):
            candidates.append((name, mod))
    # prefer shortest name (closer to root)
    if candidates:
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0]
    return None, None


# -----------------------------
# main logic: hook LLM inputs
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Debug what the LLM decoder actually receives (seq_len etc.).")
    ap.add_argument("--model_type", default="llava-next")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", default="../../scenetap/data/images/128.jpg")
    ap.add_argument("--question", default="Describe the image.")
    ap.add_argument("--print_modules", action="store_true", help="Print likely LLM-related module names.")
    args = ap.parse_args()

    hdr("LOAD MODEL")
    evaluator = get_evaluator(args.model_type, args.model_id, args.device)
    model = evaluator.model
    processor = evaluator.processor
    device = evaluator.device

    kv("model_type", args.model_type)
    kv("model_id", args.model_id)
    kv("device", device)
    kv("model python type", type(model))

    hdr("BUILD INPUTS")
    image = Image.open(args.image).convert("RGB")
    inputs = build_inputs(processor, device, image, args.question)

    if "input_ids" in inputs:
        kv("input_ids.shape", tuple(inputs["input_ids"].shape))
        text_len = int(inputs["input_ids"].shape[-1])
        kv("text token length (input_ids)", text_len)
    else:
        warn("No input_ids found in inputs dict.")
        text_len = None

    if "pixel_values" in inputs:
        kv("pixel_values.shape", tuple(inputs["pixel_values"].shape))
    else:
        warn("No pixel_values found in inputs dict.")

    if args.print_modules:
        hdr("CANDIDATE LLM MODULE NAMES (keyword scan)")
        keys = ["language_model", "llama", "decoder", "transformer", "embed_tokens", "tok_embeddings", ".layers", ".h"]
        for name, _ in model.named_modules():
            ln = name.lower()
            if any(k in ln for k in keys):
                print(name)

    # Locate embed tokens + first decoder layer
    hdr("LOCATE LLM ENTRY POINTS")
    emb_name, emb_mod = find_embed_tokens(model)
    kv("embed_tokens module", emb_name if emb_mod is not None else None)

    layer0_name, layer0_mod = find_first_decoder_layer(model)
    kv("decoder first layer", layer0_name if layer0_mod is not None else None)

    if layer0_mod is None:
        warn("Could not locate decoder first layer automatically. Use --print_modules and hook manually.")
        return

    # Hook outputs
    captured: Dict[str, torch.Tensor] = {}

    def embed_hook(_m, _inp, out):
        t = first_tensor(out)
        if t is not None:
            captured["embed_tokens_out"] = t.detach()

    def layer0_pre_hook(_m, inp):
        # For most decoders, first arg is hidden_states: (B, T, D)
        if inp and torch.is_tensor(inp[0]) and inp[0].ndim == 3:
            captured["decoder_layer0_in"] = inp[0].detach()

    def layer0_hook(_m, _inp, out):
        t = first_tensor(out)
        if t is not None and t.ndim == 3:
            captured["decoder_layer0_out"] = t.detach()

    handles = []
    try:
        if emb_mod is not None:
            handles.append(emb_mod.register_forward_hook(embed_hook))
        handles.append(layer0_mod.register_forward_pre_hook(layer0_pre_hook))
        handles.append(layer0_mod.register_forward_hook(layer0_hook))

        hdr("RUN FORWARD (NO GRAD)")
        model.eval()
        with torch.no_grad():
            # Many HF models accept output_hidden_states; if not, they ignore or error.
            # We'll try it cautiously.
            try:
                _ = model(**inputs, output_hidden_states=True, return_dict=True)
                ok("Forward pass succeeded with output_hidden_states=True")
            except TypeError:
                _ = model(**inputs)
                warn("Model forward did not accept output_hidden_states/return_dict; ran plain forward.")

    finally:
        for h in handles:
            h.remove()

    hdr("RESULTS: WHAT THE DECODER SEES")
    if "decoder_layer0_in" in captured:
        t = captured["decoder_layer0_in"]
        kv("decoder_layer0_in.shape", tuple(t.shape))
        B, T_total, D = t.shape
        kv("B (batch)", B)
        kv("T_total at decoder input", T_total)
        kv("D (hidden size)", D)

        if text_len is not None:
            # This is only a *diagnostic* difference: input_ids already includes the image placeholder token(s),
            # while decoder input may replace those with many image embeddings (or fewer, if pooled).
            delta = T_total - text_len
            kv("T_total - len(input_ids)", delta)

            if delta > 0:
                ok("Decoder input is longer than input_ids → image placeholder expanded into multiple embeddings.")
            elif delta == 0:
                warn("Decoder input length equals input_ids length. Either: images not inserted as extra embeddings, "
                     "or input_ids already includes all image tokens (rare), or model uses a different pathway.")
            else:
                warn("Decoder input shorter than input_ids (unusual). Possibly truncation/padding differences.")

    else:
        warn("Did not capture decoder_layer0_in. The first layer signature may differ (not taking hidden_states as arg0).")

    if "embed_tokens_out" in captured:
        kv("embed_tokens_out.shape", tuple(captured["embed_tokens_out"].shape))
        warn("Note: embed_tokens_out is text embedding output, not multimodal embeddings.")
    else:
        warn("Did not capture embed_tokens_out (embedding module not found or not used directly).")

    if "decoder_layer0_out" in captured:
        kv("decoder_layer0_out.shape", tuple(captured["decoder_layer0_out"].shape))

    hdr("INTERPRETATION TIP")
    print(
        "- Compare decoder_layer0_in T_total across images.\n"
        "- If projector stays (N_crops, 576, 4096) but decoder T_total varies, pooling/packing is happening\n"
        "  between projector and decoder (resampler/merger or multimodal packing).\n"
        "- If decoder T_total is *consistently* much smaller than (text_len + N_crops*576),\n"
        "  that’s strong evidence of pooling/compression before insertion into the LLM."
    )


if __name__ == "__main__":
    main()
