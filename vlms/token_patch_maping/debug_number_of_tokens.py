import sys
from pathlib import Path

# Add project root to PYTHONPATH dynamically
project_root = Path(__file__).resolve().parents[4]  # adjust depth if needed
sys.path.append(str(project_root))


import re
import torch
from PIL import Image

from reading_between_pixels.Reading_Between_Pixels.vlms.inference.infere_vlms import get_evaluator


KEYWORDS = [
    "vision", "tower", "encoder",
    "projector", "mm_projector", "multi_modal_projector",
    "resampler", "perceiver", "pool", "merge", "aggreg", "compress",
    "qformer", "adapter"
]

def should_hook(name: str) -> bool:
    ln = name.lower()
    return any(k in ln for k in KEYWORDS)

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

def build_inputs(processor, device, image: Image.Image, question: str):
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": question}, {"type": "image"}],
    }]
    formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=formatted, return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

def hook_shapes(model, inputs):
    events = []
    handles = []

    def make_hook(name):
        def hook_fn(mod, inp, out):
            t = first_tensor(out)
            if t is None:
                return
            # record only interesting rank tensors (avoid scalars)
            events.append((name, tuple(t.shape), str(t.dtype), str(t.device)))
        return hook_fn

    # Register hooks on relevant modules
    for name, m in model.named_modules():
        if should_hook(name):
            handles.append(m.register_forward_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Deduplicate while preserving order (many modules called multiple times)
    seen = set()
    uniq = []
    for e in events:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq

def main():
    evaluator = get_evaluator("llava-next", None, "cuda")
    model = evaluator.model
    processor = evaluator.processor
    device = evaluator.device

    img = Image.open("../../scenetap/data/images/128.jpg").convert("RGB")
    inputs = build_inputs(processor, device, img, "Describe the image.")

    print("pixel_values:", tuple(inputs["pixel_values"].shape))
    print("input_ids:", tuple(inputs["input_ids"].shape))

    shapes = hook_shapes(model, inputs)

    print("\n=== Hooked shapes (first occurrence of each module/shape) ===")
    for name, shape, dtype, dev in shapes:
        print(f"{name:70s} -> {shape}")

if __name__ == "__main__":
    main()
