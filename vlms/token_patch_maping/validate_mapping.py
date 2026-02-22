"""
Validate Llava-NeXT image-token -> image-region mapping by VISUALIZATION + SIMPLE CHECKS.

What this script does
1) Loads LlavaNext model + processor (your same stack).
2) Generates 3 synthetic test images with obvious structure:
   - square (256x256), wide (256x1024), tall (1024x256)
   Each has a grid + colored quadrants so misalignment is immediately visible.
3) Runs processor to get pixel_values + image_sizes + input_ids.
4) Computes the packed image token layout exactly like your model code:
   base 24x24 + mosaic (tiles merged) + unpad + newline tokens
5) Produces validation artifacts per test image:
   - overlay_boxes_*.png: draws rectangles for "corner" tokens and some random mosaic tokens
   - mosaic_heatmap_*.png: heatmap showing mosaic token coverage over the image
6) Prints numeric checks:
   - placeholder count matches packed token count
   - bboxes stay inside image
   - corners land in the correct quadrants (rough sanity check)

Run:
  python validate_mapping.py

Edit:
  MODEL_NAME if needed.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution



def build_image_token_llm_index_mapping(model, input_ids):
    """
    Build a bidirectional index mapping between packed image tokens and positions in the full LLM input sequence.

    The Llava-NeXT processor inserts the special <image> placeholder token (config.image_token_id) into `input_ids`
    exactly once per packed image token. This function finds all placeholder positions and assigns them a contiguous
    image-token index (0..N-1) in the order they appear in the sequence.

    Args:
        model: A LlavaNextForConditionalGeneration (or compatible) model instance. Used to read `config.image_token_id`.
        input_ids (torch.LongTensor): Token IDs of shape (batch_size, seq_len). Only batch index 0 is used.

    Returns:
        tuple[dict[int, int], dict[int, int]]:
            - image_to_llm: maps image_token_idx -> llm_sequence_idx
            - llm_to_image: maps llm_sequence_idx -> image_token_idx

    Notes:
        - image_token_idx is indexed within the packed image token stream (base + mosaic + newline tokens).
        - llm_sequence_idx is indexed within the full model input sequence (text + image placeholders).
    """

    image_token_id = model.config.image_token_id

    # Get all positions in the LLM sequence that correspond to <image> placeholders
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    image_to_llm = {}
    llm_to_image = {}

    for image_idx, llm_idx in enumerate(image_positions.tolist()):
        image_to_llm[image_idx] = llm_idx
        llm_to_image[llm_idx] = image_idx

    return image_to_llm, llm_to_image


# -----------------------------
# Geometry helpers (match model)
# -----------------------------

def get_anyres_image_grid_shape(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> Tuple[int, int]:
    """
    Compute the (tile_rows, tile_cols) layout for any-resolution preprocessing.

    Llava-NeXT selects the closest supported resolution from `grid_pinpoints` given the original image size,
    then conceptually divides that chosen resolution into tiles of size `tile_size` x `tile_size`.

    Args:
        image_size_hw (tuple[int, int]): Original image size as (H, W).
        grid_pinpoints (list[tuple[int, int]]): Candidate resolutions used by Llava-NeXT anyres preprocessing.
        tile_size (int): Tile edge length in pixels (typically vision_config.image_size, e.g., 336).

    Returns:
        tuple[int, int]: (num_tile_rows, num_tile_cols) for the selected best resolution.

    Notes:
        This matches the model-side logic used to infer how many non-base tiles are expected.
    """
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return best_h // tile_size, best_w // tile_size


def image_size_to_num_tiles_plus_base(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> int:
    """
    Compute the number of vision "views" produced by any-resolution preprocessing, including the base view.

    Llava-NeXT represents an image as:
      - 1 base (global) view, plus
      - a grid of additional tiles covering the selected best resolution.

    Args:
        image_size_hw (tuple[int, int]): Original image size as (H, W).
        grid_pinpoints (list[tuple[int, int]]): Candidate resolutions used by anyres preprocessing.
        tile_size (int): Tile edge length in pixels (e.g., 336).

    Returns:
        int: Total number of views = (num_tiles_in_grid + 1 base view).

    Notes:
        This mirrors `image_size_to_num_patches` semantics in the HF LlavaNext code, but expressed as tiles/views.
    """
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    num_tiles = 0
    for _ in range(0, best_h, tile_size):
        for _ in range(0, best_w, tile_size):
            num_tiles += 1
    num_tiles += 1  # base tile
    return num_tiles


def unpad_params(current_h: int, current_w: int, original_h: int, original_w: int) -> Dict[str, Any]:
    """
    Compute how Llava-NeXT "unpads" a resized/padded mosaic grid to match the original aspect ratio.

    The mosaic grid is constructed in patch-space. Depending on aspect ratios, the model removes padding either
    along height or width so the final grid corresponds to the true image content region.

    Args:
        current_h (int): Current mosaic height (in patch units, not pixels).
        current_w (int): Current mosaic width (in patch units, not pixels).
        original_h (int): Original image height in pixels.
        original_w (int): Original image width in pixels.

    Returns:
        dict[str, Any]: A dictionary describing the unpadding decision:
            - mode: "slice_h" or "slice_w"
            - scale: scaling factor used during aspect-ratio match
            - pad: number of patch units removed from each side (symmetric)
            - new_h/new_w: resulting unpadded dimensions (in patch units)

    Notes:
        This matches the behavior of the HF `unpad_image` logic, but expressed at the mosaic-grid level.
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



def mean_rgb_in_bbox(img: Image.Image, bbox) -> Tuple[float, float, float]:
    """
    Compute the mean RGB value inside a bounding box.

    Args:
        img (PIL.Image.Image): Source image (RGB).
        bbox (tuple[float, float, float, float]): Bounding box in (y0, x0, y1, x1) pixel coordinates.
            Coordinates may be floating-point.

    Returns:
        tuple[float, float, float]: Mean RGB values (R, G, B) as floats.

    Notes:
        - The bbox is clamped to image bounds.
        - If the bbox is empty after clamping/rounding, returns (0.0, 0.0, 0.0).
    """
    y0, x0, y1, x1 = bbox
    y0i, x0i = int(np.floor(y0)), int(np.floor(x0))
    y1i, x1i = int(np.ceil(y1)), int(np.ceil(x1))
    y0i, x0i = max(0, y0i), max(0, x0i)
    y1i, x1i = min(img.height, y1i), min(img.width, x1i)
    if y1i <= y0i or x1i <= x0i:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(img)[y0i:y1i, x0i:x1i, :]
    m = arr.reshape(-1, 3).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def expected_quadrant_from_bbox(img: Image.Image, bbox) -> str:
    """
    Determine which quadrant of the image a bbox center falls into.

    Quadrants are defined by splitting the image at (H/2, W/2):
      - TL: top-left
      - TR: top-right
      - BL: bottom-left
      - BR: bottom-right

    Args:
        img (PIL.Image.Image): Reference image (used for height/width).
        bbox (tuple[float, float, float, float]): Bounding box in (y0, x0, y1, x1) pixel coordinates.

    Returns:
        str: One of {"TL", "TR", "BL", "BR"}.
    """
    H, W = img.height, img.width
    y0, x0, y1, x1 = bbox
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    top = cy < H / 2.0
    left = cx < W / 2.0
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def predicted_quadrant_from_rgb(mean_rgb: Tuple[float, float, float]) -> str:
    """
    Predict which synthetic test-image quadrant a region belongs to based on mean RGB dominance.

    The synthetic images are constructed so that:
      - TL is reddish (R dominant)
      - TR is greenish (G dominant)
      - BL is bluish (B dominant)
      - BR is yellowish (R and G high, B low)

    Args:
        mean_rgb (tuple[float, float, float]): Mean RGB values (R, G, B).

    Returns:
        str: One of {"TL", "TR", "BL", "BR"}.

    Notes:
        This is a heuristic intended for validation only (not for real images).
    """
    r, g, b = mean_rgb
    if r > g and r > b:
        return "TL"
    if g > r and g > b:
        return "TR"
    if b > r and b > g:
        return "BL"
    # otherwise likely yellow-ish
    return "BR"


def bbox_from_unpadded_grid_to_original(
    r0: float, c0: float, r1: float, c1: float,
    current_h: int, current_w: int,
    original_h: int, original_w: int,
) -> Tuple[float, float, float, float]:
    """
    Convert a cell bbox from an unpadded patch grid into original-image pixel coordinates.

    This function assumes (r, c) coordinates are expressed in the *unpadded* grid coordinate system.
    It projects grid coordinates back to original pixels using the same scale logic as model-side unpadding.

    Args:
        r0, c0, r1, c1 (float): Grid-space bbox coordinates (top-left and bottom-right corners).
        current_h (int): Unpadded grid height in patch units.
        current_w (int): Unpadded grid width in patch units.
        original_h (int): Original image height in pixels.
        original_w (int): Original image width in pixels.

    Returns:
        tuple[float, float, float, float]: Pixel-space bbox as (y0, x0, y1, x1), clamped to image bounds.

    Notes:
        - This does not re-add padding; it maps the unpadded grid directly onto the original image content region.
        - Returned coordinates may be fractional; drawing code typically rounds them.
    """
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h
    if orig_ar > curr_ar:
        scale = current_w / original_w
    else:
        scale = current_h / original_h

    y0, y1 = r0 / scale, r1 / scale
    x0, x1 = c0 / scale, c1 / scale

    # clamp
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
    original_size_hw: Tuple[int, int],      # (H, W)
    num_views_provided: int,                # pixel_values.shape[1]
    include_newline_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct the packed image-token stream (base + mosaic + newline) and map each token to an image-region bbox.

    Llava-NeXT encodes an image as:
      1) A base (global) view token grid (typically 24x24 patches for 336/14).
      2) Additional tiled views merged into a larger "mosaic" patch grid.
      3) Aspect-ratio unpadding applied to the mosaic grid.
      4) Optional newline tokens inserted at the end of each mosaic row.
      5) The resulting packed tokens are inserted into the LLM sequence via <image> placeholders.

    This function replicates that geometry and returns:
      - A per-token mapping (token_idx -> kind/row/col/bbox),
      - A summary of derived dimensions and token counts.

    Args:
        model (LlavaNextForConditionalGeneration): Loaded model instance. Used to read vision/text config.
        original_size_hw (tuple[int, int]): Original image size as (H, W) in pixels.
        num_views_provided (int): Number of views actually provided to the model (pixel_values.shape[1]).
        include_newline_tokens (bool): If True, adds 1 newline token per mosaic row.

    Returns:
        dict[str, Any]:
            - "summary": dict with key geometry, token counts, and derived grid sizes.
            - "tokens": list of per-token dicts, each containing:
                - token_idx (int): index within packed image tokens
                - kind (str): "base_patch" | "base_cls" | "mosaic_patch" | "newline"
                - row/col: grid indices where applicable
                - bbox: (y0,x0,y1,x1) in original pixel coordinates, or None for non-spatial tokens.

    Notes:
        - The base grid always maps to the full image extent in this approximation.
        - The mosaic grid uses anyres tiling + unpadding to match the original aspect ratio.
        - This mapping is geometric; it does not attempt to invert the learned multimodal projector or vision features.
    """
    cfg = model.config
    vision_cfg = cfg.vision_config

    tile_size = int(vision_cfg.image_size)         # 336
    patch_size = int(vision_cfg.patch_size)        # 14
    patches_per_side = tile_size // patch_size     # 24
    per_tile_patch_tokens = patches_per_side * patches_per_side  # 576

    cls_removed = (cfg.vision_feature_select_strategy == "default")
    per_tile_tokens_after_select = per_tile_patch_tokens if cls_removed else (per_tile_patch_tokens + 1)

    expected_tiles_plus_base = image_size_to_num_tiles_plus_base(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    used_tiles_plus_base = min(num_views_provided, expected_tiles_plus_base)
    non_base_tiles = max(0, used_tiles_plus_base - 1)

    num_tile_h, num_tile_w = get_anyres_image_grid_shape(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    expected_non_base_tiles_for_layout = num_tile_h * num_tile_w

    mosaic_h = num_tile_h * patches_per_side
    mosaic_w = num_tile_w * patches_per_side

    orig_h, orig_w = original_size_hw
    params = unpad_params(mosaic_h, mosaic_w, orig_h, orig_w)
    if params["mode"] == "slice_h":
        unpadded_h, unpadded_w = params["new_h"], mosaic_w
    else:
        unpadded_h, unpadded_w = mosaic_h, params["new_w"]

    row_width_with_newline = unpadded_w + (1 if include_newline_tokens else 0)

    base_tokens = per_tile_tokens_after_select
    mosaic_tokens = unpadded_h * row_width_with_newline
    total_packed_image_tokens = base_tokens + mosaic_tokens

    tokens: List[Dict[str, Any]] = []

    # Base tokens (24x24)
    for t in range(base_tokens):
        if not cls_removed and t == 0:
            tokens.append({"token_idx": t, "kind": "base_cls", "row": None, "col": None, "bbox": None})
            continue
        patch_t = t if cls_removed else (t - 1)
        r = patch_t // patches_per_side
        c = patch_t % patches_per_side
        bbox = bbox_from_unpadded_grid_to_original(
            r, c, r + 1, c + 1,
            patches_per_side, patches_per_side,
            orig_h, orig_w
        )
        tokens.append({"token_idx": t, "kind": "base_patch", "row": int(r), "col": int(c), "bbox": bbox})

    # Mosaic tokens (unpadded) + newline
    base_offset = base_tokens
    for k in range(mosaic_tokens):
        token_idx = base_offset + k
        row = k // row_width_with_newline
        col = k % row_width_with_newline
        if include_newline_tokens and col == unpadded_w:
            tokens.append({"token_idx": token_idx, "kind": "newline", "row": int(row), "col": None, "bbox": None})
            continue
        bbox = bbox_from_unpadded_grid_to_original(
            row, col, row + 1, col + 1,
            unpadded_h, unpadded_w,
            orig_h, orig_w
        )
        tokens.append({"token_idx": token_idx, "kind": "mosaic_patch", "row": int(row), "col": int(col), "bbox": bbox})

    return {
        "summary": {
            "original_size_hw": original_size_hw,
            "tile_size": tile_size,
            "patch_size": patch_size,
            "patches_per_side": patches_per_side,
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
        },
        "tokens": tokens,
    }
def draw_patch_grid_overlay(
    img: Image.Image,
    grid_h: int,
    grid_w: int,
    out_path: str,
    line_width: int = 1,
):
    """
    Draw the model's patch grid (grid_h x grid_w) over an image and save to disk.

    This is primarily used to validate that token bboxes align with the *model-derived* patch grid
    (rather than an arbitrary synthetic grid spacing).

    Args:
        img (PIL.Image.Image): Input image.
        grid_h (int): Number of patch rows.
        grid_w (int): Number of patch columns.
        out_path (str): Output filepath for the rendered image.
        line_width (int): Line thickness for grid lines.

    Returns:
        None
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)

    H, W = im.height, im.width
    dx = W / grid_w
    dy = H / grid_h

    # vertical lines
    for c in range(grid_w + 1):
        x = c * dx
        draw.line([x, 0, x, H], fill=(255, 255, 255), width=line_width)

    # horizontal lines
    for r in range(grid_h + 1):
        y = r * dy
        draw.line([0, y, W, y], fill=(255, 255, 255), width=line_width)

    im.save(out_path)
    print("Saved:", out_path)


def draw_boxes_on_image(
    img: Image.Image,
    tokens: List[Dict[str, Any]],
    token_idxs: List[int],
    out_path: str,
    color=(255, 0, 0),
    width=2,
):
    """
    Draw a set of token bounding boxes (by token index) on an image and save to disk.

    Args:
        img (PIL.Image.Image): Input image to draw on.
        tokens (list[dict]): Token mapping list returned by `build_packed_image_token_map_for_one_image`.
        token_idxs (list[int]): Token indices to draw.
        out_path (str): Output filepath for the rendered image.
        color (tuple[int,int,int]): RGB color for box outlines and labels.
        width (int): Line thickness for box outlines.

    Returns:
        None

    Notes:
        Tokens with `bbox=None` (e.g., newline tokens) are skipped.
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx in token_idxs:
        if idx < 0 or idx >= len(tokens):
            continue
        t = tokens[idx]
        bbox = t.get("bbox")
        if bbox is None:
            continue
        y0, x0, y1, x1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        draw.text((x0 + 2, y0 + 2), f"{idx}:{t['kind']}", fill=color, font=font)

    im.save(out_path)
    print("Saved:", out_path)

# -----------------------------
# Visualization
# -----------------------------

def make_test_image(h: int, w: int, grid: int = 32) -> Image.Image:
    """
    Create a synthetic validation image with strong visual structure.

    The image contains:
      - Four colored quadrants (TL red, TR green, BL blue, BR yellow),
      - A black grid at fixed pixel spacing,
      - Text labels indicating quadrant locations and the image size.

    This makes geometric misalignment easy to spot in overlay visualizations.

    Args:
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        grid (int): Spacing (in pixels) between grid lines.

    Returns:
        PIL.Image.Image: Generated RGB image.
    """
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # quadrants
    draw.rectangle([0, 0, w//2, h//2], fill=(255, 120, 120))       # TL
    draw.rectangle([w//2, 0, w, h//2], fill=(120, 255, 120))       # TR
    draw.rectangle([0, h//2, w//2, h], fill=(120, 120, 255))       # BL
    draw.rectangle([w//2, h//2, w, h], fill=(255, 255, 120))       # BR

    # grid lines
    for x in range(0, w, grid):
        draw.line([x, 0, x, h], fill=(0, 0, 0), width=1)
    for y in range(0, h, grid):
        draw.line([0, y, w, y], fill=(0, 0, 0), width=1)

    # labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((5, 5), f"{h}x{w} TL", fill=(0, 0, 0), font=font)
    draw.text((w//2 + 5, 5), "TR", fill=(0, 0, 0), font=font)
    draw.text((5, h//2 + 5), "BL", fill=(0, 0, 0), font=font)
    draw.text((w//2 + 5, h//2 + 5), "BR", fill=(0, 0, 0), font=font)

    return img


def draw_boxes(img: Image.Image, tokens: List[Dict[str, Any]], token_idxs: List[int], out_path: str):
    """
    Draw token bboxes on an image using a fixed red style and save to disk.

    This is a lightweight alternative to `draw_boxes_on_image` that uses a fixed styling.

    Args:
        img (PIL.Image.Image): Image to draw on.
        tokens (list[dict]): Token mapping list.
        token_idxs (list[int]): Token indices to draw.
        out_path (str): Output filepath.

    Returns:
        None
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx in token_idxs:
        if idx < 0 or idx >= len(tokens):
            continue
        t = tokens[idx]
        bbox = t["bbox"]
        if bbox is None:
            continue
        y0, x0, y1, x1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0 + 2, y0 + 2), f"{idx}:{t['kind']}", fill="red", font=font)

    im.save(out_path)


def mosaic_heatmap(img: Image.Image, tokens: List[Dict[str, Any]], out_path: str):
    """
    Render a coverage heatmap showing where mosaic patch tokens land on the original image.

    Each mosaic patch token contributes +1 to the pixels covered by its bbox. The accumulated mask is displayed
    semi-transparently over the image to visualize coverage and potential drift.

    Args:
        img (PIL.Image.Image): Original image.
        tokens (list[dict]): Token mapping list.
        out_path (str): Output filepath for the plot image.

    Returns:
        None

    Notes:
        - Only tokens with kind == "mosaic_patch" contribute.
        - Newline tokens and base tokens are ignored.
    """
    H, W = img.height, img.width
    mask = np.zeros((H, W), dtype=np.float32)

    for t in tokens:
        if t["kind"] != "mosaic_patch":
            continue
        y0, x0, y1, x1 = t["bbox"]
        y0i, x0i = int(np.floor(y0)), int(np.floor(x0))
        y1i, x1i = int(np.ceil(y1)), int(np.ceil(x1))
        y0i, x0i = max(0, y0i), max(0, x0i)
        y1i, x1i = min(H, y1i), min(W, x1i)
        if y1i > y0i and x1i > x0i:
            mask[y0i:y1i, x0i:x1i] += 1.0

    plt.figure()
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.title("Mosaic patch coverage heatmap")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------
# Checks
# -----------------------------

def bbox_inside_image(bbox, H, W) -> bool:
    """
    Check whether a bbox lies fully within image bounds.

    Args:
        bbox (tuple[float,float,float,float]): Bounding box as (y0,x0,y1,x1).
        H (int): Image height in pixels.
        W (int): Image width in pixels.

    Returns:
        bool: True if bbox coordinates are within [0..H] and [0..W] bounds.
    """
    y0, x0, y1, x1 = bbox
    return (0 <= x0 <= W and 0 <= x1 <= W and 0 <= y0 <= H and 0 <= y1 <= H)


def quadrant_of_bbox_center(bbox, H, W) -> str:
    """
    Determine which image quadrant a bbox center falls into.

    Args:
        bbox (tuple[float,float,float,float]): Bounding box as (y0,x0,y1,x1).
        H (int): Image height in pixels.
        W (int): Image width in pixels.

    Returns:
        str: One of {"TL", "TR", "BL", "BR"}.
    """
    y0, x0, y1, x1 = bbox
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    top = cy < H / 2.0
    left = cx < W / 2.0
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def run_one_case(model, processor, img: Image.Image, case_name: str, out_dir: str,
                 query_bbox: Optional[Tuple[float, float, float, float]] = None):
    """
    Run the full validation pipeline for a single test image.

    Steps:
      1) Process (image, prompt) with the LlavaNextProcessor to obtain:
         - pixel_values (views/tiles)
         - image_sizes (original H,W)
         - input_ids (text + <image> placeholders)
      2) Reconstruct packed image token mapping (base + mosaic + newline) and bbox projections.
      3) Verify placeholder count matches packed image token count.
      4) Save visualization artifacts:
         - model patch grid overlay
         - mosaic coverage heatmap
      5) Optionally validate the inverse mapping:
         - select tokens corresponding to `query_bbox`
         - draw query bbox and selected token bboxes

    Args:
        model: LlavaNextForConditionalGeneration model instance.
        processor: LlavaNextProcessor (or compatible) processor instance.
        img (PIL.Image.Image): Input image.
        case_name (str): Case identifier used in filenames.
        out_dir (str): Directory to write output images into.
        query_bbox (Optional[tuple[float,float,float,float]]): Query bbox in pixel coords (y0,x0,y1,x1).
            If provided, inverse-mapping overlays are created.

    Returns:
        None
    """

    prompt = "[INST] <image>\nDescribe the image. [/INST]"
    inputs = processor(images=img, text=prompt, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    image_sizes = inputs["image_sizes"]
    input_ids = inputs["input_ids"]

    B, V = pixel_values.shape[0], pixel_values.shape[1]
    assert B == 1, "This validation expects batch=1."

    orig_h = int(image_sizes[0, 0].item())
    orig_w = int(image_sizes[0, 1].item())

    mapping = build_packed_image_token_map_for_one_image(
        model=model,
        original_size_hw=(orig_h, orig_w),
        num_views_provided=V,
        include_newline_tokens=True,
    )
    summary = mapping["summary"]
    tokens = mapping["tokens"]

    image_to_llm, llm_to_image = build_image_token_llm_index_mapping(model, input_ids)

    print("\nFirst 10 image-token → LLM positions:")
    for i in range(10):
        print(f"image_token {i} -> llm_position {image_to_llm[i]}")

    # Existing prints (keep yours if you want)
    image_token_id = model.config.image_token_id
    n_placeholders = int((input_ids == image_token_id).sum().item())
    expected = int(summary["total_packed_image_tokens"])

    print(f"\n=== CASE: {case_name} ===")
    print(f"pixel_values.shape: {tuple(pixel_values.shape)}")
    print(f"image_sizes: (H,W)=({orig_h},{orig_w})")
    print(f"<image> placeholders: {n_placeholders} | expected packed image tokens: {expected}")
    print("tile_layout_hw:", summary["tile_layout_hw"])
    print("mosaic_unpadded_hw_in_patches:", summary["mosaic_unpadded_hw_in_patches"])
    print("base_tokens:", summary["base_tokens"], "mosaic_tokens:", summary["mosaic_tokens"], "total:", expected)

    os.makedirs(out_dir, exist_ok=True)

    # --- A) model patch grid image (you already added this earlier) ---
    mosaic_h, mosaic_w = summary["mosaic_unpadded_hw_in_patches"]
    grid_img_path = os.path.join(out_dir, f"model_grid_{case_name}.png")
    draw_patch_grid_overlay(img, mosaic_h, mosaic_w, grid_img_path, line_width=1)

    # --- existing mosaic heatmap / overlay boxes (keep your existing calls if you like) ---
    heat_path = os.path.join(out_dir, f"mosaic_heatmap_{case_name}.png")
    mosaic_heatmap(img, tokens, heat_path)

    # --- NEW: inverse mapping validation for a bbox ---
    if query_bbox is not None:
        # clamp bbox to image bounds
        qy0, qx0, qy1, qx1 = query_bbox
        qy0 = max(0.0, min(float(img.height), qy0))
        qy1 = max(0.0, min(float(img.height), qy1))
        qx0 = max(0.0, min(float(img.width), qx0))
        qx1 = max(0.0, min(float(img.width), qx1))
        query_bbox = (qy0, qx0, qy1, qx1)

        # get overlapping tokens (mosaic first; optionally also base)
        hits_mosaic = tokens_for_bbox_center(
            tokens,
            query_bbox,
            kinds=("mosaic_patch",),
            # min_intersection_area=1.0,
            # min_iou=0.0,
            # top_k=None,
        )
        hits_base = tokens_for_bbox_center(
            tokens,
            query_bbox,
            kinds=("base_patch",),
            # min_intersection_area=1.0,
            # min_iou=0.0,
            # top_k=None,
        )

        print(f"Query bbox (y0,x0,y1,x1): {tuple(round(x,2) for x in query_bbox)}")
        print(f"Overlapping mosaic tokens: {len(hits_mosaic)} (showing top 10 by overlap)")
        print([h["token_idx"] for h in hits_mosaic[:10]])
        print(f"Overlapping base tokens:   {len(hits_base)} (showing top 10 by overlap)")
        print([h["token_idx"] for h in hits_base[:10]])

        # Draw query + top token bboxes (mosaic)
        out_inv_mosaic = os.path.join(out_dir, f"bbox_inverse_mosaic_{case_name}.png")
        draw_query_and_token_bboxes(img, query_bbox, hits_mosaic, out_inv_mosaic)

        # Draw query + top token bboxes (base)
        out_inv_base = os.path.join(out_dir, f"bbox_inverse_base_{case_name}.png")
        draw_query_and_token_bboxes(img, query_bbox, hits_base, out_inv_base)



#### TOKENS FOR BOX MAPPING
from typing import Optional, Tuple, List, Dict, Any


# -----------------------------
# Inverse mapping helpers
# -----------------------------
def _area(b):
    """
    Compute the area of a bbox.

    Args:
        b (tuple[float,float,float,float]): Bounding box as (y0,x0,y1,x1).

    Returns:
        float: Area in pixel^2 (0 if bbox is degenerate).
    """
    y0, x0, y1, x1 = b
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)

def _intersect(a, b):
    """
    Compute the intersection bbox of two bboxes.

    Args:
        a (tuple[float,float,float,float]): First bbox (y0,x0,y1,x1).
        b (tuple[float,float,float,float]): Second bbox (y0,x0,y1,x1).

    Returns:
        tuple[float,float,float,float]: Intersection bbox (y0,x0,y1,x1). May be degenerate.
    """
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    y0 = max(ay0, by0)
    x0 = max(ax0, bx0)
    y1 = min(ay1, by1)
    x1 = min(ax1, bx1)
    return (y0, x0, y1, x1)

def _iou(a, b):
    """
    Compute intersection-over-union (IoU) between two bboxes.

    Args:
        a (tuple[float,float,float,float]): First bbox (y0,x0,y1,x1).
        b (tuple[float,float,float,float]): Second bbox (y0,x0,y1,x1).

    Returns:
        float: IoU in [0, 1]. Returns 0 if there is no overlap or union is zero.
    """
    inter = _intersect(a, b)
    ia = _area(inter)
    if ia <= 0:
        return 0.0
    ua = _area(a) + _area(b) - ia
    return ia / ua if ua > 0 else 0.0

def tokens_for_bbox(
    mapping_tokens: List[Dict[str, Any]],
    query_bbox: Tuple[float, float, float, float],  # (y0,x0,y1,x1)
    *,
    kinds: Tuple[str, ...] = ("mosaic_patch",),
    min_intersection_area: float = 1.0,
    min_iou: float = 0.0,
    top_k: Optional[int] = 50,
):
    """
    Find tokens whose mapped bboxes overlap a query bbox, using intersection area and/or IoU thresholds.

    Args:
        mapping_tokens (list[dict]): Token mapping list (output of `build_packed_image_token_map_for_one_image`).
        query_bbox (tuple[float,float,float,float]): Query bbox in pixel coords (y0,x0,y1,x1).
        kinds (tuple[str,...]): Token kinds to consider (e.g., ("mosaic_patch",) or ("base_patch","mosaic_patch")).
        min_intersection_area (float): Minimum intersection area (in pixel^2) to include a token.
        min_iou (float): Minimum IoU to include a token.
        top_k (Optional[int]): If set, return only the top_k tokens ranked by (intersection_area, iou).

    Returns:
        list[dict]: List of token hit dicts with fields:
            token_idx, kind, row, col, bbox, intersection_area, iou.

    Notes:
        - This selection is overlap-based; bbox edges may cut through patch cells.
        - Use top_k=None if you want complete coverage for visualization.
    """
    out = []
    for t in mapping_tokens:
        if t.get("kind") not in kinds:
            continue
        tb = t.get("bbox")
        if tb is None:
            continue
        inter = _intersect(query_bbox, tb)
        ia = _area(inter)
        if ia < min_intersection_area:
            continue
        iou_val = _iou(query_bbox, tb)
        if iou_val < min_iou:
            continue
        out.append({
            "token_idx": t["token_idx"],
            "kind": t["kind"],
            "row": t.get("row"),
            "col": t.get("col"),
            "bbox": tb,
            "intersection_area": ia,
            "iou": iou_val,
        })
    out.sort(key=lambda x: (x["intersection_area"], x["iou"]), reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out

def tokens_for_bbox_center(mapping_tokens, query_bbox, kinds=("mosaic_patch",)):
    """
    Find tokens whose bbox center lies inside a query bbox.

    This is often a more visually intuitive selection than IoU for patch grids, because it produces a
    dense “filled” set of patch cells inside the query region.

    Args:
        mapping_tokens (list[dict]): Token mapping list.
        query_bbox (tuple[float,float,float,float]): Query bbox in pixel coords (y0,x0,y1,x1).
        kinds (tuple[str,...]): Token kinds to consider.

    Returns:
        list[dict]: Subset of `mapping_tokens` whose bbox centers fall within the query bbox.
    """
    qy0,qx0,qy1,qx1 = query_bbox
    out=[]
    for t in mapping_tokens:
        if t.get("kind") not in kinds: 
            continue
        b=t.get("bbox")
        if b is None: 
            continue
        y0,x0,y1,x1=b
        cy=(y0+y1)/2
        cx=(x0+x1)/2
        if (qy0 <= cy <= qy1) and (qx0 <= cx <= qx1):
            out.append(t)
    return out


# -----------------------------
# Drawing helper: query bbox + token bboxes
# -----------------------------
def draw_query_and_token_bboxes(
    img: Image.Image,
    query_bbox: Tuple[float, float, float, float],
    token_hits: List[Dict[str, Any]],
    out_path: str,
    *,
    query_color=(0, 0, 0),      # black
    token_color=(255, 0, 0),    # red
    query_width=4,
    token_width=2,
):
    """
    Draw a query bbox plus a set of token bboxes on an image and save the result.

    Args:
        img (PIL.Image.Image): Image to draw on.
        query_bbox (tuple[float,float,float,float]): Query bbox (y0,x0,y1,x1) in pixel coords.
        token_hits (list[dict]): Token dicts with a "bbox" field (typically produced by tokens_for_bbox*).
        out_path (str): Output filepath.
        query_color (tuple[int,int,int]): RGB color for the query bbox outline and label.
        token_color (tuple[int,int,int]): RGB color for token bbox outlines and token index labels.
        query_width (int): Line thickness for the query bbox.
        token_width (int): Line thickness for token bboxes.

    Returns:
        None

    Notes:
        Tokens without a bbox (e.g., newline tokens) should not be passed here.
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw query bbox
    qy0, qx0, qy1, qx1 = query_bbox
    draw.rectangle([qx0, qy0, qx1, qy1], outline=query_color, width=query_width)
    draw.text((qx0 + 2, qy0 + 2), "QUERY", fill=query_color, font=font)

    # Draw token bboxes
    for h in token_hits:
        y0, x0, y1, x1 = h["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=token_color, width=token_width)
        draw.text((x0 + 1, y0 + 1), str(h["token_idx"]), fill=token_color, font=font)

    im.save(out_path)
    print("Saved:", out_path)




# -----------------------------
# UPDATE main() to pass example bbox
# -----------------------------
def main():
    """
    Entry point that loads the model/processor, generates synthetic test images, and runs validation per case.

    Creates three canonical test cases:
      - square_256x256
      - wide_256x1024
      - tall_1024x256

    For each case, it runs `run_one_case(...)` and writes visualization outputs under `out_dir`.

    Returns:
        None
    """
    MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"  # change if needed
    out_dir = "mapping_validation_outputs"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)

    # Test images
    cases = [
        ("square_256x256", make_test_image(256, 256)),
        ("wide_256x1024",  make_test_image(256, 1024)),
        ("tall_1024x256",  make_test_image(1024, 256)),
    ]

    # Example bbox in ORIGINAL IMAGE PIXELS: (y0,x0,y1,x1)
    # Choose something that spans across the vertical midline (so it hits TL/TR and BL/BR clearly).
    # You can change this per case if you want, but this works "reasonably" across all sizes:
    example_bbox_by_case = {
        "square_256x256": (60, 60, 190, 200),
        "wide_256x1024":  (40, 350, 210, 700),
        "tall_1024x256":  (300, 40, 800, 210),
    }

    for name, img in cases:
        q = example_bbox_by_case[name]
        run_one_case(model, processor, img, name, out_dir, query_bbox=q)

    print("\nDone. Check outputs in:", out_dir)


if __name__ == "__main__":
    main()