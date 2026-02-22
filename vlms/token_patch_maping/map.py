import sys
from pathlib import Path

# Add project root to PYTHONPATH dynamically
project_root = Path(__file__).resolve().parents[4]  # adjust depth if needed
sys.path.append(str(project_root))

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reading_between_pixels.Reading_Between_Pixels.vlms.inference.infere_vlms import get_evaluator


# ============================================================
# PRETTY DEBUG PRINTS
# ============================================================

def hdr(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def kv(k: str, v: Any):
    print(f"{k:<42}: {v}")


def warn(msg: str):
    print(f"⚠️  {msg}")


def ok(msg: str):
    print(f"✅ {msg}")


# ============================================================
# MODEL INTROSPECTION
# ============================================================

def find_module_by_paths(root: Any, paths: List[str]) -> Optional[Any]:
    for path in paths:
        obj = root
        good = True
        for part in path.split("."):
            if not hasattr(obj, part):
                good = False
                break
            obj = getattr(obj, part)
        if good:
            return obj
    return None


def unwrap_vision_tower(model: Any) -> Optional[Any]:
    # Common in LLaVA forks
    if hasattr(model, "get_vision_tower") and callable(model.get_vision_tower):
        vt = model.get_vision_tower()
    else:
        vt = getattr(model, "vision_tower", None)
        if vt is None and hasattr(model, "model"):
            vt = getattr(model.model, "vision_tower", None)

    # Some implementations wrap in list/tuple
    if isinstance(vt, (list, tuple)) and len(vt) > 0:
        vt = vt[0]
    return vt


def infer_patch_size(model: Any) -> Tuple[int, str]:
    """
    Returns (patch_size, how_found)
    """
    vt = unwrap_vision_tower(model)
    if vt is None:
        raise RuntimeError("Could not locate vision tower on model.")

    cfg = getattr(vt, "config", None)

    # Common: cfg.patch_size
    if cfg is not None and hasattr(cfg, "patch_size"):
        return int(cfg.patch_size), "vision_tower.config.patch_size"

    # Sometimes nested
    if cfg is not None and hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "patch_size"):
        return int(cfg.vision_config.patch_size), "vision_tower.config.vision_config.patch_size"

    raise RuntimeError("Could not infer patch_size from vision tower config; extend infer_patch_size().")


def get_norm_stats_from_processor(processor: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Returns mean, std shaped (1,3,1,1) and how_found.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        ip = getattr(processor, "feature_extractor", None)

    if ip is not None and hasattr(ip, "image_mean") and hasattr(ip, "image_std"):
        mean = torch.tensor(ip.image_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(ip.image_std, device=device).view(1, -1, 1, 1)
        return mean, std, "processor.image_processor.image_mean/std"

    # CLIP defaults
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return mean, std, "CLIP default mean/std"


def find_projector_module(model: Any) -> Tuple[Any, str]:
    """
    Returns (module, path_description)
    """
    candidate_paths = [
        "mm_projector",
        "multi_modal_projector",
        "visual_projector",
        "model.mm_projector",
        "model.multi_modal_projector",
        "model.visual_projector",
    ]
    mod = find_module_by_paths(model, candidate_paths)
    if mod is not None:
        # Find which one
        for p in candidate_paths:
            if find_module_by_paths(model, [p]) is mod:
                return mod, p

    # Fallback: fuzzy search in named_modules
    for name, m in model.named_modules():
        if any(x in name.lower() for x in ["mm_projector", "multi_modal_projector", "visual_projector", "projector"]):
            return m, f"fuzzy named_modules match: {name}"

    raise RuntimeError("Could not find projector module. Extend find_projector_module().")


# ============================================================
# SHAPE HELPERS
# ============================================================

def describe_pixel_values(pv: torch.Tensor) -> Tuple[int, int, int]:
    """
    Returns (N_crops, H, W) from pixel_values.
    pv is either:
      (1,3,H,W) or (1,N,3,H,W)
    """
    if pv.ndim == 4:
        _, _, H, W = pv.shape
        return 1, H, W
    if pv.ndim == 5:
        _, N, _, H, W = pv.shape
        return N, H, W
    raise RuntimeError(f"Unexpected pixel_values.ndim={pv.ndim}, shape={tuple(pv.shape)}")


def pixel_values_to_crops_uint8(
    pixel_values: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """
    Convert normalized pixel_values -> uint8 crops [0..255]
    Returns crops: (N_crops, H, W, 3)
    """
    pv = pixel_values.detach()

    if pv.ndim == 4:
        # (1,3,H,W) -> (1,3,H,W)
        pv = pv
    elif pv.ndim == 5:
        # (1,N,3,H,W) -> (N,3,H,W)
        pv = pv.squeeze(0)
    else:
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pv.shape)}")

    # Expand mean/std if multiple crops
    if pv.ndim == 4:
        # (N,3,H,W)
        if mean.shape[0] == 1:
            mean_exp = mean.expand(pv.shape[0], -1, 1, 1)
            std_exp = std.expand(pv.shape[0], -1, 1, 1)
        else:
            mean_exp, std_exp = mean, std
    else:
        # (1,3,H,W)
        mean_exp, std_exp = mean, std

    # If pv is (1,3,H,W), treat as N=1 for consistent output
    if pv.ndim == 4:
        # N crops
        denorm = (pv * std_exp + mean_exp).clamp(0, 1)
        denorm = denorm.cpu().numpy()  # (N,3,H,W)
        denorm = np.transpose(denorm, (0, 2, 3, 1))  # (N,H,W,3)
    else:
        denorm = (pv * std_exp + mean_exp).clamp(0, 1).cpu().numpy()  # (1,3,H,W)
        denorm = np.transpose(denorm, (0, 2, 3, 1))

    crops_u8 = (denorm * 255.0 + 0.5).astype(np.uint8)
    return crops_u8


# ============================================================
# PROJECTOR OUTPUT CAPTURE
# ============================================================

def capture_projector_output(model: Any, projector: Any, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Forward model with a hook on projector to capture its output tensor.
    Returns the captured projector output (first call only).
    """
    captured: List[torch.Tensor] = []

    def hook_fn(_mod, _inp, out):
        # out can be tensor or tuple
        if isinstance(out, (tuple, list)):
            out0 = out[0]
        else:
            out0 = out
        if torch.is_tensor(out0):
            captured.append(out0.detach())
        else:
            warn(f"Projector hook output is not a tensor: {type(out0)}")

    handle = projector.register_forward_hook(hook_fn)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Projector hook did not capture any output. Projector may not run in this forward path.")

    return captured[0]


def interpret_projector_shape(proj_out: torch.Tensor) -> Tuple[int, int, Optional[int], str]:
    """
    Try to interpret projector output token count:
      Returns (N_crops, T_tokens, D, description)

    Common cases:
      - (B, T, D)
      - (B, N, T, D)
      - (N, T, D)
    """
    shp = tuple(proj_out.shape)

    if proj_out.ndim == 4:
        B, N, T, D = shp
        if B != 1:
            warn(f"Batch size in projector output is {B}; expected 1 for this debug script.")
        return N, T, D, "projector_out shape (B, N_crops, T_tokens, D)"

    if proj_out.ndim == 3:
        a, b, c = shp
        # Could be (B,T,D) or (N,T,D)
        # Heuristic: if a==1 -> (B,T,D)
        if a == 1:
            B, T, D = shp
            return 1, T, D, "projector_out shape (B=1, T_tokens, D)"
        else:
            N, T, D = shp
            return N, T, D, "projector_out shape (N_crops, T_tokens, D)"

    if proj_out.ndim == 2:
        T, D = shp
        return 1, T, D, "projector_out shape (T_tokens, D) (no batch dim)"

    raise RuntimeError(f"Unexpected projector output shape: {shp}")


# ============================================================
# TOKEN ↔ PATCH MAPPING
# ============================================================

def patch_grid_from_hw(H: int, W: int, patch_size: int) -> Tuple[int, int, int]:
    grid_h = H // patch_size
    grid_w = W // patch_size
    n_patches = grid_h * grid_w
    return grid_h, grid_w, n_patches


def detect_cls_and_offset(T_tokens: int, n_patches: int) -> Tuple[bool, int, str]:
    """
    Returns (has_cls, offset, reasoning)
    where offset = number of leading non-patch tokens to skip to get patch index.
    """
    if T_tokens == n_patches:
        return False, 0, "T_tokens == n_patches -> patch-only tokens (no CLS)"
    if T_tokens == n_patches + 1:
        return True, 1, "T_tokens == n_patches + 1 -> likely CLS at index 0"
    if T_tokens > n_patches + 1:
        return False, 0, f"T_tokens ({T_tokens}) > n_patches+1 ({n_patches+1}) -> extra non-patch tokens exist"
    # T_tokens < n_patches (token compression or pooling)
    return False, 0, f"T_tokens ({T_tokens}) < n_patches ({n_patches}) -> token compression/pooling likely"


def projector_token_to_patch_index(token_index: int, offset: int, n_patches: int) -> Optional[int]:
    """
    Convert projector token index -> patch index, if possible.
    """
    patch_index = token_index - offset
    if patch_index < 0 or patch_index >= n_patches:
        return None
    return patch_index


def patch_index_to_rowcol(patch_index: int, grid_w: int) -> Tuple[int, int]:
    row = patch_index // grid_w
    col = patch_index % grid_w
    return row, col


def patch_index_to_pixel_box(patch_index: int, grid_w: int, patch_size: int) -> Tuple[int, int, int, int]:
    row, col = patch_index_to_rowcol(patch_index, grid_w)
    x0 = col * patch_size
    y0 = row * patch_size
    x1 = (col + 1) * patch_size
    y1 = (row + 1) * patch_size
    return x0, y0, x1, y1


# ============================================================
# OVERLAY VISUALIZATION
# ============================================================

def load_font(font_size: int) -> ImageFont.ImageFont:
    # Default PIL font; optionally you can point to a .ttf
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def draw_patch_overlay(
    crop_img_u8: np.ndarray,
    patch_size: int,
    grid_h: int,
    grid_w: int,
    label_stride: int,
    font_size: int,
    draw_grid: bool,
    draw_boxes: bool,
) -> Image.Image:
    """
    Draw grid and patch indices on a crop image.
    label_stride=1 draws every patch index, 2 draws every other patch, etc.
    """
    img = Image.fromarray(crop_img_u8, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    H, W = crop_img_u8.shape[0], crop_img_u8.shape[1]

    # Optional grid lines
    if draw_grid:
        for r in range(grid_h + 1):
            y = r * patch_size
            draw.line([(0, y), (W, y)], width=1)
        for c in range(grid_w + 1):
            x = c * patch_size
            draw.line([(x, 0), (x, H)], width=1)

    # Optional boxes + labels
    for r in range(0, grid_h, label_stride):
        for c in range(0, grid_w, label_stride):
            patch_index = r * grid_w + c
            x0 = c * patch_size
            y0 = r * patch_size
            x1 = (c + 1) * patch_size
            y1 = (r + 1) * patch_size

            if draw_boxes:
                draw.rectangle([x0, y0, x1, y1], width=1)

            # Draw index near center
            cx = x0 + patch_size // 2
            cy = y0 + patch_size // 2
            text = str(patch_index)

            # Outline for readability: draw black then white
            draw.text((cx + 1, cy + 1), text, font=font)
            draw.text((cx, cy), text, font=font)

    return img


# ============================================================
# INPUT BUILDING (CHAT TEMPLATE)
# ============================================================

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
    out = {}
    for k, v in inputs.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Full debug: projector tokens ↔ patch grid ↔ pixel boxes + overlays.")
    ap.add_argument("--model_type", default="llava-next")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image", default="../../scenetap/data/images/128.jpg")
    ap.add_argument("--question", default="Describe the image.")
    ap.add_argument("--out_dir", default="./debug_patch_map_out")

    # overlay controls
    ap.add_argument("--label_stride", type=int, default=2, help="Draw label for every k-th patch (1=all).")
    ap.add_argument("--font_size", type=int, default=12)
    ap.add_argument("--draw_grid", action="store_true")
    ap.add_argument("--draw_boxes", action="store_true")

    # mapping verification controls
    ap.add_argument("--sample_tokens", type=int, default=20, help="How many projector tokens to sample for token→patch→box print.")
    ap.add_argument("--dump_json", action="store_true", help="Dump mapping metadata to JSON.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hdr("LOADING MODEL")
    evaluator = get_evaluator(args.model_type, args.model_id, args.device)
    model = evaluator.model
    processor = evaluator.processor
    device = evaluator.device

    kv("model_type arg", args.model_type)
    kv("model_id arg", args.model_id)
    kv("model python type", type(model))
    kv("device", device)
    kv("processor python type", type(processor))

    hdr("PROCESSOR IMAGE STATS")
    mean, std, how_norm = get_norm_stats_from_processor(processor, device)
    kv("norm stats source", how_norm)
    kv("mean", mean.flatten().tolist())
    kv("std", std.flatten().tolist())

    ip = getattr(processor, "image_processor", None) or getattr(processor, "feature_extractor", None)
    if ip is not None:
        kv("image_processor type", type(ip))
        if hasattr(ip, "size"):
            kv("image_processor.size", getattr(ip, "size"))
        if hasattr(ip, "crop_size"):
            kv("image_processor.crop_size", getattr(ip, "crop_size"))

    hdr("LOADING IMAGE + BUILDING INPUTS")
    image = Image.open(args.image).convert("RGB")
    kv("image path", str(Path(args.image).resolve()))
    kv("image size (PIL)", image.size)

    inputs = build_inputs(processor, device, image, args.question)
    if "pixel_values" not in inputs:
        raise RuntimeError("Processor did not produce pixel_values. Check your model/processor.")

    pv = inputs["pixel_values"]
    kv("pixel_values.shape", tuple(pv.shape))
    kv("pixel_values.dtype", pv.dtype)

    N_crops, H, W = describe_pixel_values(pv)
    kv("N_crops (from pixel_values)", N_crops)
    kv("crop tensor H,W", (H, W))

    hdr("INFER PATCH SIZE + PATCH GRID")
    patch_size, how_patch = infer_patch_size(model)
    kv("patch_size", patch_size)
    kv("patch size source", how_patch)

    grid_h, grid_w, n_patches = patch_grid_from_hw(H, W, patch_size)
    kv("grid_h", grid_h)
    kv("grid_w", grid_w)
    kv("n_patches", n_patches)

    if H % patch_size != 0 or W % patch_size != 0:
        warn(f"H or W not divisible by patch_size. H%P={H%patch_size}, W%P={W%patch_size}. Mapping may be off.")

    hdr("FIND + HOOK PROJECTOR, CAPTURE TOKEN COUNT")
    projector, proj_path = find_projector_module(model)
    kv("projector found at", proj_path)
    kv("projector type", type(projector))

    proj_out = capture_projector_output(model, projector, inputs)
    kv("captured projector_out.shape", tuple(proj_out.shape))

    Np, T_tokens, D, desc = interpret_projector_shape(proj_out)
    kv("interpreted projector shape", desc)
    kv("N_crops (projector space)", Np)
    kv("T_tokens per crop (projector)", T_tokens)
    kv("D_proj", D)

    if Np not in (1, N_crops):
        warn(f"Mismatch: pixel_values indicates N_crops={N_crops}, projector indicates Np={Np}. Might still be OK depending on model internals.")

    hdr("AUTOMATIC COMPARISON: PROJECTOR TOKENS vs PATCH GRID")
    has_cls, offset, reasoning = detect_cls_and_offset(T_tokens, n_patches)
    kv("CLS detected?", has_cls)
    kv("token offset to patches", offset)
    kv("reasoning", reasoning)

    if T_tokens == n_patches:
        ok("Projector tokens match patch grid exactly (patch-only).")
    elif T_tokens == n_patches + 1:
        ok("Projector tokens match patch grid + 1 (likely CLS token at index 0).")
    elif T_tokens > n_patches + 1:
        warn("Projector has MORE tokens than patches (+ maybe CLS). There are extra non-patch tokens; mapping is ambiguous.")
    else:
        warn("Projector has FEWER tokens than patches. Likely token pooling/compression; 1:1 mapping not possible.")

    hdr("TOKEN→PATCH INDEX VERIFICATION (COUNT-BASED)")
    # Mapping is only fully valid if tokens are exactly patches or patches+CLS
    mapping_possible = (T_tokens == n_patches) or (T_tokens == n_patches + 1)
    kv("1:1 mapping possible?", mapping_possible)

    if mapping_possible:
        # Check that last token maps to last patch (in patch-only mode)
        last_token = T_tokens - 1
        last_patch = projector_token_to_patch_index(last_token, offset, n_patches)
        kv("last token index", last_token)
        kv("maps to patch index", last_patch)
        if last_patch == n_patches - 1:
            ok("Sanity check: last token maps to last patch index.")
        else:
            warn("Sanity check failed: last token does not map to last patch index (unexpected ordering).")
    else:
        warn("Skipping strict verification: 1:1 mapping not supported by these token counts.")

    hdr("RECONSTRUCT CROPS + WRITE OVERLAY IMAGES")
    crops_u8 = pixel_values_to_crops_uint8(pv, mean, std)  # (N, H, W, 3)
    kv("reconstructed crops_u8.shape", crops_u8.shape)

    overlay_paths = []
    for ci in range(crops_u8.shape[0]):
        overlay = draw_patch_overlay(
            crop_img_u8=crops_u8[ci],
            patch_size=patch_size,
            grid_h=grid_h,
            grid_w=grid_w,
            label_stride=max(1, args.label_stride),
            font_size=args.font_size,
            draw_grid=args.draw_grid,
            draw_boxes=args.draw_boxes,
        )
        op = out_dir / f"crop_{ci:02d}_patch_overlay_stride{args.label_stride}.png"
        overlay.save(op)
        overlay_paths.append(str(op))
        print(f"Wrote overlay: {op}")

    hdr("SAMPLED PROJECTOR TOKEN → PATCH → PIXEL BOX PRINTS")
    if mapping_possible:
        # Sample token indices evenly from the valid patch-token range
        # If has_cls, token 0 is CLS and has no patch mapping
        valid_token_start = 1 if has_cls else 0
        valid_token_end = T_tokens - 1

        if valid_token_end < valid_token_start:
            warn("No valid patch tokens to sample.")
        else:
            sample_n = max(1, args.sample_tokens)
            token_samples = np.linspace(valid_token_start, valid_token_end, num=min(sample_n, (valid_token_end - valid_token_start + 1)), dtype=int)

            for ti in token_samples:
                pi = projector_token_to_patch_index(int(ti), offset, n_patches)
                if pi is None:
                    print(f"token_index={int(ti)} -> (no patch mapping)")
                    continue
                x0, y0, x1, y1 = patch_index_to_pixel_box(pi, grid_w, patch_size)
                r, c = patch_index_to_rowcol(pi, grid_w)
                print(f"token_index={int(ti):4d} -> patch_index={pi:4d} -> (row={r:3d}, col={c:3d}) -> box=({x0},{y0},{x1},{y1})")
    else:
        warn("Skipping sampled mapping prints because 1:1 mapping isn't supported by token counts.")

    hdr("DUMP JSON (OPTIONAL)")
    meta = {
        "image_path": str(Path(args.image).resolve()),
        "model_type": args.model_type,
        "model_id": args.model_id,
        "pixel_values_shape": list(pv.shape),
        "N_crops_pixel_values": int(N_crops),
        "crop_hw": [int(H), int(W)],
        "patch_size": int(patch_size),
        "grid_hw": [int(grid_h), int(grid_w)],
        "n_patches": int(n_patches),
        "projector_path": proj_path,
        "projector_out_shape": list(proj_out.shape),
        "N_crops_projector": int(Np),
        "T_tokens_projector": int(T_tokens),
        "D_proj": int(D) if D is not None else None,
        "cls_detected": bool(has_cls),
        "patch_offset": int(offset),
        "mapping_possible": bool(mapping_possible),
        "overlay_paths": overlay_paths,
        "notes": {
            "comparison_reasoning": reasoning,
        }
    }

    if args.dump_json:
        jp = out_dir / "debug_patch_map_metadata.json"
        with open(jp, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote JSON: {jp}")
    else:
        print("JSON dump disabled (use --dump_json to enable).")

    hdr("DONE")
    ok(f"Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
