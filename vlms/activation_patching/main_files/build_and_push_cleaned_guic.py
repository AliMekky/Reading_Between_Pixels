#!/usr/bin/env python3
"""Add tight-box cleaned overlays to GUIC and optionally push the test split.

Every existing field is preserved. Each of the four overlay structs receives a
new ``cleaned_image`` field. The cleaned image starts from ``notext.image`` and
copies only the pixels inside that overlay's dataset-provided bounding box.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image as HFImage, load_from_disk
from PIL import Image


OVERLAY_VARIANTS = (
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


def clipped_bbox(
    raw_bbox: Iterable[Any],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    values = list(raw_bbox)
    require(len(values) == 4, "Expected an XYXY bounding box, got {}".format(values))
    x0, y0, x1, y1 = [float(value) for value in values]
    box = (
        max(0, int(math.floor(x0))),
        max(0, int(math.floor(y0))),
        min(width, int(math.ceil(x1))),
        min(height, int(math.ceil(y1))),
    )
    require(box[2] > box[0] and box[3] > box[1], "Invalid clipped bounding box {}".format(box))
    return box


def create_cleaned_image(
    no_text_image: Image.Image,
    overlay_image: Image.Image,
    raw_bbox: Iterable[Any],
) -> Image.Image:
    base = no_text_image.convert("RGB")
    overlay = overlay_image.convert("RGB")
    require(base.size == overlay.size, "No-text and overlay image dimensions differ")
    box = clipped_bbox(raw_bbox, base.width, base.height)
    cleaned = base.copy()
    cleaned.paste(overlay.crop(box), (box[0], box[1]))
    # PIL images without an inherited file format are encoded losslessly by the
    # datasets Image feature rather than retaining the source JPEG bytes.
    cleaned.format = "PNG"
    return cleaned


def augmented_features(dataset: Dataset) -> Features:
    fields: Dict[str, Any] = dict(dataset.features)
    for variant in OVERLAY_VARIANTS:
        require(variant in fields, "Dataset is missing overlay field {}".format(variant))
        variant_fields = dict(fields[variant])
        require("cleaned_image" not in variant_fields, "{} already has cleaned_image".format(variant))
        variant_fields["cleaned_image"] = HFImage()
        fields[variant] = variant_fields
    return Features(fields)


def augment_example(example: Dict[str, Any]) -> Dict[str, Any]:
    no_text = example["notext"]["image"]
    updates: Dict[str, Any] = {}
    for variant in OVERLAY_VARIANTS:
        item = dict(example[variant])
        item["cleaned_image"] = create_cleaned_image(no_text, item["image"], item["bbox"])
        updates[variant] = item
    return updates


def pixel_validation(dataset: Dataset) -> Dict[str, Any]:
    counts = {
        "rows": len(dataset),
        "pairs_expected": len(dataset) * len(OVERLAY_VARIANTS),
        "pairs_validated": 0,
        "outside_changed_pixels": 0,
        "inside_original_mismatch_pixels": 0,
        "dimension_failures": 0,
        "bbox_failures": 0,
    }
    variant_counts = {variant: 0 for variant in OVERLAY_VARIANTS}

    for index in range(len(dataset)):
        if index > 0 and index % 100 == 0:
            log("VALIDATE", "rows_checked={}/{}".format(index, len(dataset)))
        example = dataset[index]
        base = np.asarray(example["notext"]["image"].convert("RGB"), dtype=np.int16)
        height, width = base.shape[:2]
        for variant in OVERLAY_VARIANTS:
            item = example[variant]
            original = np.asarray(item["image"].convert("RGB"), dtype=np.int16)
            cleaned = np.asarray(item["cleaned_image"].convert("RGB"), dtype=np.int16)
            if original.shape != base.shape or cleaned.shape != base.shape:
                counts["dimension_failures"] += 1
                continue
            try:
                x0, y0, x1, y1 = clipped_bbox(item["bbox"], width, height)
            except Exception:
                counts["bbox_failures"] += 1
                continue

            box_mask = np.zeros((height, width), dtype=bool)
            box_mask[y0:y1, x0:x1] = True
            outside_changed = np.any(cleaned != base, axis=2) & ~box_mask
            inside_mismatch = np.any(cleaned != original, axis=2) & box_mask
            counts["outside_changed_pixels"] += int(outside_changed.sum())
            counts["inside_original_mismatch_pixels"] += int(inside_mismatch.sum())
            counts["pairs_validated"] += 1
            variant_counts[variant] += 1

    counts["variant_pair_counts"] = variant_counts
    counts["passed"] = bool(
        counts["pairs_validated"] == counts["pairs_expected"]
        and counts["outside_changed_pixels"] == 0
        and counts["inside_original_mismatch_pixels"] == 0
        and counts["dimension_failures"] == 0
        and counts["bbox_failures"] == 0
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        default="../../integrated_gradients/hf_dataset_GUIC/AHAAM__GUIC",
    )
    parser.add_argument(
        "--output_dir",
        default="../hf_dataset_GUIC_cleaned/AHAAM__GUIC",
    )
    parser.add_argument("--repo_id", default="AHAAM/GUIC")
    parser.add_argument("--split", default="test")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--push_existing",
        action="store_true",
        help="Validate and push an already-built output_dir without rebuilding it.",
    )
    parser.add_argument("--create_pr", action="store_true")
    parser.add_argument("--max_shard_size", default="500MB")
    args = parser.parse_args()

    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    temporary = output.with_name(output.name + ".building")
    log("CONFIG", "source={}".format(source))
    log("CONFIG", "output={}".format(output))
    log("CONFIG", "repo_id={} split={} push_to_hub={}".format(
        args.repo_id, args.split, args.push_to_hub
    ))
    if args.push_existing:
        require(args.push_to_hub, "--push_existing requires --push_to_hub")
        require(output.exists(), "Built output does not exist: {}".format(output))
        log("LOAD", "Using already-built augmented dataset: {}".format(output))
    else:
        require(source.exists(), "Source dataset does not exist: {}".format(source))
        require(not output.exists(), "Output already exists; refusing to overwrite: {}".format(output))
        require(not temporary.exists(), "Temporary output already exists: {}".format(temporary))

        dataset = load_from_disk(str(source))
        if isinstance(dataset, DatasetDict):
            require(args.split in dataset, "DatasetDict has no split {}".format(args.split))
            dataset = dataset[args.split]
        require(len(dataset) > 0, "Source dataset is empty")
        log("BUILD", "rows={} cleaned_pairs={}".format(
            len(dataset), len(dataset) * len(OVERLAY_VARIANTS)
        ))

        features = augmented_features(dataset)
        augmented = dataset.map(
            augment_example,
            features=features,
            load_from_cache_file=False,
            writer_batch_size=10,
            desc="Creating tight-box cleaned overlays",
        )
        require(len(augmented) == len(dataset), "Row count changed during augmentation")

        temporary.parent.mkdir(parents=True, exist_ok=True)
        augmented.save_to_disk(str(temporary), max_shard_size=args.max_shard_size)
        temporary.replace(output)
        log("SAVE", "Saved augmented dataset: {}".format(output))

    reloaded = load_from_disk(str(output))
    validation = pixel_validation(reloaded)
    log("VALIDATION", json.dumps(validation, sort_keys=True))
    require(validation["passed"], "Pixel validation failed")
    validation_path = output.parent / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    log("PASS", "All cleaned overlays preserve the original box and exactly match no-text outside it")

    if not args.push_to_hub:
        log("NEXT", "Authenticate with `hf auth login`, then rerun with --push_to_hub")
        log("COMPLETE", "Local dataset built and validated; remote dataset was not modified")
        return

    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    log("AUTH", "environment_token_present={}".format(token_present))
    commit = reloaded.push_to_hub(
        repo_id=args.repo_id,
        split=args.split,
        commit_message="Add pixel-exact tight-box cleaned overlay images",
        commit_description=(
            "Preserves all original fields and adds cleaned_image to each overlay struct. "
            "Cleaned images equal notext.image outside the dataset-provided text bbox."
        ),
        create_pr=args.create_pr,
        max_shard_size=args.max_shard_size,
    )
    log("PUSH", "commit_url={}".format(getattr(commit, "commit_url", commit)))
    log("COMPLETE", "Validated dataset pushed to {}/{}".format(args.repo_id, args.split))


if __name__ == "__main__":
    main()
