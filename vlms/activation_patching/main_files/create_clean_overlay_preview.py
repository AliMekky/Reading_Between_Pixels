#!/usr/bin/env python3
"""Create and validate a pixel-exact cleaned-overlay preview.

The cleaned image uses the decoded no-text image as its canvas and copies only
the requested region from an existing GUIC overlay. By default this is the
dataset's exact annotated text bounding box. The original diffusion crop is
also available as a comparison mode. Images are saved as PNG to avoid
introducing new JPEG differences.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image, ImageDraw, ImageFont


VALID_VARIANTS = (
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
)


def log(section: str, message: str) -> None:
    print("[{}] {}".format(section, message), flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_qid(value: Any) -> str:
    text = str(value).strip()
    return text.lstrip("0") or "0"


def find_sample(dataset: Dataset, question_id: str) -> Dict[str, Any]:
    wanted = canonical_qid(question_id)
    for index in range(len(dataset)):
        sample = dataset[index]
        if canonical_qid(sample.get("question_id", "")) == wanted:
            return sample
    raise ValueError("question_id={} not found".format(question_id))


def scaled_square_region(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    scale_factor: float,
) -> Tuple[int, int, int, int]:
    """Match the dataset generator's get_scaled_square_region geometry."""
    min_x, min_y, max_x, max_y = bbox_xyxy
    width = max_x - min_x
    height = max_y - min_y
    side_length = max(width, height) * scale_factor
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    half_side = side_length // 2

    left = max(center_x - half_side, 0)
    right = min(center_x + half_side, image_width)
    top = max(center_y - half_side, 0)
    bottom = min(center_y + half_side, image_height)

    if right - left > bottom - top:
        difference = right - left - (bottom - top)
        if top - difference / 2 < 0:
            top = 0
            bottom = right - left
        elif bottom + difference / 2 > image_height:
            bottom = image_height
            top = bottom - right + left
        else:
            top -= difference / 2
            bottom += difference / 2
    elif right - left < bottom - top:
        difference = bottom - top - (right - left)
        if left - difference / 2 < 0:
            left = 0
            right = bottom - top
        elif right + difference / 2 > image_width:
            right = image_width
            left = right - bottom + top
        else:
            left -= difference / 2
            right += difference / 2

    result = tuple(int(value) for value in (left, top, right, bottom))
    require(result[2] > result[0] and result[3] > result[1], "Invalid diffusion crop")
    require(result[2] - result[0] == result[3] - result[1], "Diffusion crop is not square")
    return result


def difference_metrics(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    require(first.shape == second.shape, "Cannot compare different image shapes")
    values = np.abs(first.astype(np.int16) - second.astype(np.int16)).mean(axis=2)[mask]
    return {
        "mean_absolute_difference": float(values.mean()) if values.size else 0.0,
        "maximum_absolute_difference": float(values.max()) if values.size else 0.0,
        "fraction_nonzero": float((values > 0).mean()) if values.size else 0.0,
    }


def load_font(size: int) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def labeled_panel(image: Image.Image, label: str, width: int) -> Image.Image:
    resized = image.copy()
    if resized.width != width:
        height = round(resized.height * width / resized.width)
        resized = resized.resize((width, height), Image.Resampling.LANCZOS)
    header = 44
    panel = Image.new("RGB", (resized.width, resized.height + header), "white")
    panel.paste(resized, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), label, fill="black", font=load_font(20))
    return panel


def expanded_box(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - margin),
        max(0, y0 - margin),
        min(image_width, x1 + margin),
        min(image_height, y1 + margin),
    )


def make_contact_sheet(
    no_text: Image.Image,
    original: Image.Image,
    cleaned: Image.Image,
    display_crop: Tuple[int, int, int, int],
) -> Image.Image:
    a = np.asarray(original, dtype=np.int16)
    b = np.asarray(cleaned, dtype=np.int16)
    removed = np.clip(np.abs(a - b) * 10, 0, 255).astype(np.uint8)
    removed_image = Image.fromarray(removed, mode="RGB")

    full_panels = [
        labeled_panel(no_text, "No text", 500),
        labeled_panel(original, "Original overlay", 500),
        labeled_panel(cleaned, "Cleaned overlay", 500),
        labeled_panel(removed_image, "Removed difference (10x)", 500),
    ]
    top_height = max(panel.height for panel in full_panels)
    top = Image.new("RGB", (sum(panel.width for panel in full_panels), top_height), "white")
    x = 0
    for panel in full_panels:
        top.paste(panel, (x, 0))
        x += panel.width

    zoom_box = display_crop
    zooms = [
        labeled_panel(no_text.crop(zoom_box), "No-text crop", 500),
        labeled_panel(original.crop(zoom_box), "Original crop", 500),
        labeled_panel(cleaned.crop(zoom_box), "Cleaned crop", 500),
    ]
    bottom_height = max(panel.height for panel in zooms)
    bottom = Image.new("RGB", (top.width, bottom_height), "white")
    x = (top.width - sum(panel.width for panel in zooms)) // 2
    for panel in zooms:
        bottom.paste(panel, (x, 0))
        x += panel.width

    sheet = Image.new("RGB", (top.width, top.height + bottom.height), "white")
    sheet.paste(top, (0, 0))
    sheet.paste(bottom, (0, top.height))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="../../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC",
    )
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="misleading_groundable")
    parser.add_argument(
        "--copy_region",
        choices=("text_bbox", "diffusion_crop"),
        default="text_bbox",
    )
    parser.add_argument("--scale_factor", type=float, default=3.0)
    parser.add_argument("--out_root", default="../clean_overlay_previews")
    args = parser.parse_args()

    log("CONFIG", "question_id={} variant={} copy_region={} scale_factor={}".format(
        args.question_id, args.variant, args.copy_region, args.scale_factor
    ))
    dataset = load_from_disk(args.dataset_dir)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["test"] if "test" in dataset else dataset[next(iter(dataset))]
    sample = find_sample(dataset, args.question_id)

    no_text = sample["notext"]["image"].convert("RGB")
    original = sample[args.variant]["image"].convert("RGB")
    require(no_text.size == original.size, "Paired image dimensions differ")
    bbox = tuple(float(value) for value in sample[args.variant]["bbox"])
    text_box = tuple(int(value) for value in bbox)
    diffusion_crop = scaled_square_region(bbox, no_text.width, no_text.height, args.scale_factor)
    copy_box = text_box if args.copy_region == "text_bbox" else diffusion_crop
    display_crop = expanded_box(copy_box, no_text.width, no_text.height, margin=30)
    log("GEOMETRY", "image_size={} text_bbox_xyxy={} diffusion_crop_xyxy={} copied_box_xyxy={}".format(
        no_text.size, bbox, diffusion_crop, copy_box
    ))

    cleaned = no_text.copy()
    cleaned.paste(original.crop(copy_box), (copy_box[0], copy_box[1]))

    base_array = np.asarray(no_text)
    original_array = np.asarray(original)
    cleaned_array = np.asarray(cleaned)
    outside_mask = np.ones((no_text.height, no_text.width), dtype=bool)
    outside_mask[copy_box[1] : copy_box[3], copy_box[0] : copy_box[2]] = False
    inside_mask = ~outside_mask

    outside_clean = difference_metrics(base_array, cleaned_array, outside_mask)
    inside_preservation = difference_metrics(original_array, cleaned_array, inside_mask)
    removed_outside = difference_metrics(original_array, cleaned_array, outside_mask)
    require(outside_clean["maximum_absolute_difference"] == 0.0, "Cleaned image changed outside copied box")
    require(inside_preservation["maximum_absolute_difference"] == 0.0, "Original copied box was not preserved")
    log("VALIDATION", "clean_vs_no_text outside_copied_box={}".format(outside_clean))
    log("VALIDATION", "clean_vs_original inside_copied_box={}".format(inside_preservation))
    log("DIAGNOSTIC", "removed_original_changes outside_copied_box={}".format(removed_outside))

    out_dir = (
        Path(args.out_root)
        / canonical_qid(sample["question_id"])
        / args.variant
        / args.copy_region
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "no_text": out_dir / "no_text.png",
        "original_overlay": out_dir / "original_overlay.png",
        "cleaned_overlay": out_dir / "cleaned_overlay.png",
        "comparison": out_dir / "comparison.png",
        "metadata": out_dir / "metadata.json",
    }
    no_text.save(paths["no_text"], format="PNG")
    original.save(paths["original_overlay"], format="PNG")
    cleaned.save(paths["cleaned_overlay"], format="PNG")
    make_contact_sheet(no_text, original, cleaned, display_crop).save(paths["comparison"], format="PNG")

    metadata = {
        "question_id": canonical_qid(sample["question_id"]),
        "variant": args.variant,
        "text": sample[args.variant]["text"],
        "text_bbox_xyxy": list(bbox),
        "diffusion_crop_xyxy": list(diffusion_crop),
        "copy_region": args.copy_region,
        "copied_box_xyxy": list(copy_box),
        "display_crop_xyxy": list(display_crop),
        "scale_factor": args.scale_factor,
        "outside_clean_validation": outside_clean,
        "inside_crop_preservation": inside_preservation,
        "removed_original_changes_outside_crop": removed_outside,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, path in paths.items():
        log("SAVE", "{}={}".format(name, path.resolve()))
    log("COMPLETE", "Cleaned overlay is pixel-identical to no-text outside the copied box")


if __name__ == "__main__":
    main()
