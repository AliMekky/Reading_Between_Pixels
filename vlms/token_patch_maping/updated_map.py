"""
End-to-end Llava-NeXT image-token -> original-image-region mapping script.

What it does:
1) Loads your LlavaNext model + processor.
2) Preprocesses an image to get pixel_values (B, V, 3, 336, 336) and image_sizes (B, 2).
3) Computes the *exact* image token sequence length the model will insert into the LLM input
   under "spatial_unpad" packing (base tile + merged/unpadded mosaic + newline tokens).
4) Builds a per-token mapping:
   - base_patch tokens: which (row,col) in 24x24 and bbox in original pixels
   - mosaic_patch tokens: which (row,col) in unpadded mosaic grid and bbox in original pixels
   - newline tokens: row separators (no bbox)
5) Optionally verifies counts against the number of <image> placeholder tokens in input_ids.

Notes / caveats:
- The mapping is for the *packed image tokens inserted into the LLM sequence* (i.e., after merging + unpadding).
- The mapping uses the same unpad logic as the model code you pasted (aspect-ratio-based slicing).
- The base-view bbox mapping assumes the same pad/resize behavior as unpad logic; if your image processor uses
  a different crop policy, base bboxes can be slightly off at pixel precision (the mosaic logic is the main one).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution


# -----------------------------
# Geometry helpers (match model)
# -----------------------------

def get_anyres_image_grid_shape(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> Tuple[int, int]:
    """
    Matches your pasted get_anyres_image_grid_shape(), but accepts (H,W) tuple.
    Returns (num_tile_h, num_tile_w).
    """
    if not isinstance(image_size_hw, (list, tuple)):
        raise TypeError("image_size_hw must be (H,W) tuple/list")
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return best_h // tile_size, best_w // tile_size


def image_size_to_num_tiles_plus_base(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> int:
    """
    Matches your pasted image_size_to_num_patches(), but in the naming used by this script:
    it returns how many 336x336 tiles are expected INCLUDING the base tile (+1).
    """
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    num_tiles = 0
    for _ in range(0, best_h, tile_size):
        for _ in range(0, best_w, tile_size):
            num_tiles += 1
    num_tiles += 1  # add the base tile
    return num_tiles


def unpad_params(current_h: int, current_w: int, original_h: int, original_w: int) -> Dict[str, Any]:
    """
    Computes how unpad_image() slices the grid, in "grid units" (here: patch units).
    """
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h

    if orig_ar > curr_ar:
        # slice height
        scale = current_w / original_w
        new_h = int(round(original_h * scale, 7))
        pad = (current_h - new_h) // 2
        return {"mode": "slice_h", "scale": scale, "pad": pad, "new_h": new_h, "new_w": current_w}
    else:
        # slice width
        scale = current_h / original_h
        new_w = int(round(original_w * scale, 7))
        pad = (current_w - new_w) // 2
        return {"mode": "slice_w", "scale": scale, "pad": pad, "new_h": current_h, "new_w": new_w}


def grid_patch_bbox_to_original_pixels(
    r0: float, c0: float, r1: float, c1: float,
    current_h: int, current_w: int,
    original_h: int, original_w: int,
) -> tuple[float, float, float, float]:
    """
    Map a bbox defined in the *UNPADDED* grid coordinate system back to original pixels.
    This matches the fact that unpad_image() already removed padding before flattening.
    """
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h

    if orig_ar > curr_ar:
        # width is the limiting dimension; height was padded then removed
        scale = current_w / original_w
        y0 = r0 / scale
        y1 = r1 / scale
        x0 = c0 / scale
        x1 = c1 / scale
    else:
        # height is limiting; width was padded then removed
        scale = current_h / original_h
        y0 = r0 / scale
        y1 = r1 / scale
        x0 = c0 / scale
        x1 = c1 / scale

    # clamp to original bounds
    y0 = max(0.0, min(float(original_h), y0))
    y1 = max(0.0, min(float(original_h), y1))
    x0 = max(0.0, min(float(original_w), x0))
    x1 = max(0.0, min(float(original_w), x1))
    return (y0, x0, y1, x1)


# --------------------------------------------
# Build mapping for packed image token sequence
# --------------------------------------------

def build_packed_image_token_map_for_one_image(
    *,
    model: LlavaNextForConditionalGeneration,
    original_size_hw: Tuple[int, int],      # (H, W) from inputs["image_sizes"][0]
    num_views_provided: int,                # inputs["pixel_values"].shape[1]
    include_newline_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Builds mapping for the packed image tokens that will be inserted into the LLM.
    This matches your pasted get_image_features() + pack_image_features() behavior.

    Returns:
      {
        "summary": {...},
        "tokens": [ {token_idx, kind, row, col, bbox_y0x0y1x1, ...}, ... ],
      }
    """
    cfg = model.config
    vision_cfg = cfg.vision_config

    # These match your printed CLIP tower: image_size=336, patch_size=14 typically
    tile_size = int(vision_cfg.image_size)
    patch_size = int(vision_cfg.patch_size)
    patches_per_side = tile_size // patch_size  # 24
    per_tile_patch_tokens = patches_per_side * patches_per_side  # 576

    # In your pasted code, CLS is removed under "default"
    # We'll treat "default" as the normal case.
    cls_removed = (cfg.vision_feature_select_strategy == "default")
    per_tile_tokens_after_select = per_tile_patch_tokens if cls_removed else (per_tile_patch_tokens + 1)

    # How many tiles does the model think it should use (it will slice pixel_values to this)
    expected_tiles_plus_base = image_size_to_num_tiles_plus_base(original_size_hw, cfg.image_grid_pinpoints, tile_size)

    # base tile tokens are kept intact
    base_tokens = per_tile_tokens_after_select

    # Number of "non-base" tiles used in mosaic
    used_tiles_plus_base = min(num_views_provided, expected_tiles_plus_base)
    non_base_tiles = max(0, used_tiles_plus_base - 1)

    # Determine mosaic layout in tiles (num_tile_h x num_tile_w)
    # This is how it reshapes in pack_image_features()
    num_tile_h, num_tile_w = get_anyres_image_grid_shape(original_size_hw, cfg.image_grid_pinpoints, tile_size)

    # Mosaic pre-unpad grid size in patch units:
    mosaic_h = num_tile_h * patches_per_side
    mosaic_w = num_tile_w * patches_per_side

    # If for any reason non_base_tiles doesn't match num_tile_h*num_tile_w, the view() in the model would break.
    # In practice processor + image_sizes are consistent, but we keep this note for debugging.
    expected_non_base_tiles_for_layout = num_tile_h * num_tile_w

    # Apply unpadding to get final mosaic size
    orig_h, orig_w = original_size_hw
    params = unpad_params(mosaic_h, mosaic_w, orig_h, orig_w)
    if params["mode"] == "slice_h":
        unpadded_h, unpadded_w = params["new_h"], mosaic_w
    else:
        unpadded_h, unpadded_w = mosaic_h, params["new_w"]

    row_width_with_newline = unpadded_w + (1 if include_newline_tokens else 0)
    mosaic_tokens = unpadded_h * row_width_with_newline

    total_packed_image_tokens = base_tokens + mosaic_tokens

    tokens: List[Dict[str, Any]] = []

    # ---- Base tokens: 24x24 grid (global view) ----
    # If CLS is present, token 0 is CLS and has no bbox.
    for t in range(base_tokens):
        if not cls_removed and t == 0:
            tokens.append({
                "token_idx": t,
                "kind": "base_cls",
                "row": None,
                "col": None,
                "bbox_y0x0y1x1": None,
                "note": "CLS token (global summary), not a spatial patch."
            })
            continue

        patch_t = t if cls_removed else (t - 1)
        r = patch_t // patches_per_side
        c = patch_t % patches_per_side

        bbox = grid_patch_bbox_to_original_pixels(
            r0=float(r), c0=float(c),
            r1=float(r + 1), c1=float(c + 1),
            current_h=patches_per_side,  # 24
            current_w=patches_per_side,  # 24
            original_h=orig_h,
            original_w=orig_w,
        )
        tokens.append({
            "token_idx": t,
            "kind": "base_patch",
            "row": int(r),
            "col": int(c),
            "bbox_y0x0y1x1": bbox,
            "note": "Base/global 24x24 patch grid."
        })

    # ---- Mosaic tokens: merged tiles (unpadded), row-major, newline at end of each row ----
    base_offset = base_tokens
    for k in range(mosaic_tokens):
        token_idx = base_offset + k
        row = k // row_width_with_newline
        col = k % row_width_with_newline

        if include_newline_tokens and col == unpadded_w:
            tokens.append({
                "token_idx": token_idx,
                "kind": "newline",
                "row": int(row),
                "col": None,
                "bbox_y0x0y1x1": None,
                "note": "Row separator token."
            })
            continue

        bbox = grid_patch_bbox_to_original_pixels(
            r0=float(row), c0=float(col),
            r1=float(row + 1), c1=float(col + 1),
            current_h=mosaic_h,
            current_w=mosaic_w,
            original_h=orig_h,
            original_w=orig_w,
        )
        tokens.append({
            "token_idx": token_idx,
            "kind": "mosaic_patch",
            "row": int(row),
            "col": int(col),
            "bbox_y0x0y1x1": bbox,
            "note": "Merged tiles (after unpad), flattened row-major."
        })

    return {
        "summary": {
            "original_size_hw": original_size_hw,
            "tile_size": tile_size,
            "patch_size": patch_size,
            "patches_per_side": patches_per_side,
            "vision_feature_select_strategy": cfg.vision_feature_select_strategy,
            "cls_removed": cls_removed,
            "num_views_provided": num_views_provided,
            "expected_tiles_plus_base_from_image_sizes": expected_tiles_plus_base,
            "used_tiles_plus_base_after_slicing": used_tiles_plus_base,
            "non_base_tiles_used": non_base_tiles,
            "tile_layout_hw": (num_tile_h, num_tile_w),
            "expected_non_base_tiles_for_layout": expected_non_base_tiles_for_layout,
            "mosaic_pre_unpad_hw_in_patches": (mosaic_h, mosaic_w),
            "mosaic_unpadded_hw_in_patches": (unpadded_h, unpadded_w),
            "newline_tokens_enabled": include_newline_tokens,
            "row_width_with_newline": row_width_with_newline,
            "base_tokens": base_tokens,
            "mosaic_tokens": mosaic_tokens,
            "total_packed_image_tokens": total_packed_image_tokens,
            "warning": (
                None if non_base_tiles == expected_non_base_tiles_for_layout
                else "non_base_tiles_used != tile_layout tiles; processor/image_sizes mismatch may exist."
            ),
        },
        "tokens": tokens,
    }


# -----------------------------
# Main: load model, run, map
# -----------------------------

def main():
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"  # change if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = LlavaNextProcessor.from_pretrained(model_name)

    # --- Load an image (change path) ---
    image_path = "/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/scenetap/data/images/128.jpg"
    img = Image.open(image_path).convert("RGB")

    # --- Build prompt (must contain exactly N <image> tokens equal to packed image tokens, or processor/model will error) ---
    # IMPORTANT: the HF LlavaNext pipeline typically creates the right number of image placeholder tokens for you
    # when you use the processor correctly with "<image>" in the prompt, depending on the processor config.
    prompt = "[INST] <image>\nDescribe the image. [/INST]"

    inputs = processor(images=img, text=prompt, return_tensors="pt")

    # Move tensors to model device(s) in a simple way (works for single GPU / CPU; device_map='auto' is handled by model)
    # We'll only use these on CPU for mapping math anyway.
    pixel_values = inputs["pixel_values"]
    image_sizes = inputs.get("image_sizes", None)
    input_ids = inputs.get("input_ids", None)

    if image_sizes is None:
        raise ValueError("processor did not return image_sizes. This mapping script needs image_sizes (original H,W).")

    # Inputs typically:
    # pixel_values: (B, V, 3, 336, 336)
    # image_sizes:  (B, 2) as (H, W)
    B = pixel_values.shape[0]
    V = pixel_values.shape[1]
    if B != 1:
        raise ValueError(f"This demo expects batch_size=1 for simplicity. Got B={B}.")

    orig_h = int(image_sizes[0, 0].item())
    orig_w = int(image_sizes[0, 1].item())

    # Build mapping for this one image
    mapping = build_packed_image_token_map_for_one_image(
        model=model,
        original_size_hw=(orig_h, orig_w),
        num_views_provided=V,
        include_newline_tokens=True,
    )

    print("\n=== SHAPES ===")
    print(f"pixel_values.shape: {tuple(pixel_values.shape)}  (B, V, 3, 336, 336)")
    print(f"image_sizes[0]: (H,W)=({orig_h},{orig_w})")
    if input_ids is not None:
        print(f"input_ids.shape: {tuple(input_ids.shape)}")
        # How many image placeholder tokens are in the prompt?
        image_token_id = model.config.image_token_id
        num_placeholders = int((input_ids == image_token_id).sum().item())
        print(f"num <image> placeholder tokens in input_ids: {num_placeholders}")
        print(f"expected packed image tokens (from mapping): {mapping['summary']['total_packed_image_tokens']}")
        if num_placeholders != mapping["summary"]["total_packed_image_tokens"]:
            print("WARNING: placeholder count != packed image tokens. "
                  "If you run a forward pass, model will raise mismatch error.")
        else:
            print("OK: placeholder count matches packed image token count.")

    print("\n=== MAPPING SUMMARY ===")
    for k, v in mapping["summary"].items():
        print(f"{k}: {v}")

    # Example: print first 10 tokens + a few around the base/mosaic boundary
    toks = mapping["tokens"]
    base_tokens = mapping["summary"]["base_tokens"]
    print("\n=== SAMPLE TOKEN -> REGION MAPPING ===")
    sample_indices = list(range(10)) + [base_tokens - 1, base_tokens, base_tokens + 1, base_tokens + 50]
    sample_indices = [i for i in sample_indices if 0 <= i < len(toks)]
    for i in sample_indices:
        t = toks[i]
        print(
            f"token_idx={t['token_idx']:>5}  kind={t['kind']:<12} "
            f"row={str(t.get('row')):<4} col={str(t.get('col')):<4} "
            f"bbox(y0,x0,y1,x1)={t.get('bbox_y0x0y1x1')}"
        )

    # If you want: build a dict for fast lookup
    # token_to_bbox = {t["token_idx"]: t["bbox_y0x0y1x1"] for t in toks if t["bbox_y0x0y1x1"] is not None}


if __name__ == "__main__":
    main()