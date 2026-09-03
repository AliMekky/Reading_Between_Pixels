#!/usr/bin/env python3
"""Lock the exact GUIC subset shared by prior attention and IG analyses."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from transformers import LlavaNextConfig

from activation_patch_control_pilot import VARIANTS, build_regions_full_coverage
from activation_patch_llava_next_debug import (
    DATASET_DEFAULT,
    DATASET_REVISION_DEFAULT,
    MODEL_DEFAULT,
    build_packed_token_mapping,
    get_or_download_hf_dataset,
    image_size_to_num_views,
    log,
    save_json,
    validate_dataset_provenance,
)

SELECTION_VARIANTS = VARIANTS + ("irrelevant_word",)


def reason_category(exc: Exception) -> str:
    message = str(exc)
    if "overlaps text tokens" in message:
        return "text_object_token_overlap"
    if "matched-random regions" in message:
        return "matched_random_unavailable"
    if "maps to zero visual patch tokens" in message:
        return "region_maps_to_zero_tokens"
    if "bbox" in message.lower():
        return "invalid_bbox"
    return "other_geometry_failure"


def question_ids_from_output_directory(path: Path) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError("Question-ID source directory does not exist: {}".format(path))
    qids = {child.name for child in path.iterdir() if child.is_dir()}
    if not qids:
        raise RuntimeError("Question-ID source directory is empty: {}".format(path))
    return qids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=MODEL_DEFAULT)
    parser.add_argument("--hf_dataset", default=DATASET_DEFAULT)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION_DEFAULT)
    parser.add_argument("--hf_cache_dir", default="../hf_dataset_GUIC_cleaned")
    parser.add_argument("--dataset_validation_file", default="../hf_dataset_GUIC_cleaned/remote_validation.json")
    parser.add_argument(
        "--attention_qid_dir",
        default="../../attention_weights/llava-next_attentions/misleading_groundable/misleading_groundable",
    )
    parser.add_argument(
        "--ig_qid_dir",
        default="../../integrated_gradients/one_question_three_regions_mask_based_strict_sign",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--variants", default=",".join(SELECTION_VARIANTS))
    parser.add_argument("--out", default="activation_patch_confirmation_selection.json")
    args = parser.parse_args()
    selection_variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    valid_variants = ("correct_answer",) + SELECTION_VARIANTS
    if not selection_variants or len(selection_variants) != len(set(selection_variants)):
        raise ValueError("Variant selection is empty or contains duplicates")
    if not set(selection_variants).issubset(valid_variants):
        raise ValueError("Unknown selection variant")

    provenance = validate_dataset_provenance(args.dataset_validation_file, args.dataset_revision)
    ds = get_or_download_hf_dataset(
        args.hf_dataset, args.hf_cache_dir, args.split, args.dataset_revision
    )
    attention_ids = question_ids_from_output_directory(Path(args.attention_qid_dir))
    ig_ids = question_ids_from_output_directory(Path(args.ig_qid_dir))
    if attention_ids != ig_ids:
        raise RuntimeError(
            "Prior attention and IG question-ID sets differ: attention_only={}, ig_only={}".format(
                sorted(attention_ids - ig_ids)[:20], sorted(ig_ids - attention_ids)[:20]
            )
        )
    shared_ids = attention_ids

    config = LlavaNextConfig.from_pretrained(args.model_id, local_files_only=True)
    model_like = SimpleNamespace(config=config)
    tile_size = int(config.vision_config.image_size)
    selected = []
    counts: Counter[str] = Counter()
    dataset_ids = {str(value) for value in ds["question_id"]}
    missing_from_dataset = sorted(shared_ids - dataset_ids)
    if missing_from_dataset:
        raise RuntimeError("Shared prior IDs missing from pinned dataset: {}".format(missing_from_dataset))

    for dataset_index in range(len(ds)):
        qid = str(ds["question_id"][dataset_index])
        if qid not in shared_ids:
            continue
        sample = ds[dataset_index]
        no_text_image = sample["notext"]["image"].convert("RGB")
        image_hw = (no_text_image.height, no_text_image.width)
        num_views = image_size_to_num_views(
            image_hw, config.image_grid_pinpoints, tile_size
        )
        mapping = build_packed_token_mapping(model_like, image_hw, num_views)
        condition_counts: Dict[str, Dict[str, int]] = {}
        condition_regions: Dict[str, Dict[str, Any]] = {}
        for variant_index, variant in enumerate(selection_variants):
            regions = build_regions_full_coverage(
                sample=sample,
                variant=variant,
                mapping_tokens=mapping["tokens"],
                streams={"base", "mosaic"},
                min_overlap_fraction=args.min_overlap_fraction,
                image_hw=image_hw,
                seed=args.seed * 100000 + dataset_index * 10 + variant_index,
            )
            condition_counts[variant] = {
                name: int(region["token_count"])
                for name, region in regions.items()
            }
            condition_regions[variant] = {
                name: {
                    "token_count": int(region["token_count"]),
                    "clean_object_control": region.get("clean_object_control"),
                    "text_overlap_token_count": region.get("text_overlap_token_count"),
                    "random_disjoint_from_objects": region.get("disjoint_from_objects"),
                }
                for name, region in regions.items()
            }
            for object_name in ("correct_object_region", "grounded_object_region"):
                if object_name not in regions:
                    counts["{}_unavailable".format(object_name)] += 1
                elif not regions[object_name]["clean_object_control"]:
                    counts["{}_overlaps_text".format(object_name)] += 1
        selected.append({
            "question_id": qid,
            "dataset_index": dataset_index,
            "image_size_hw": list(image_hw),
            "region_token_counts": condition_counts,
            "region_diagnostics": condition_regions,
        })
        counts["selected"] += 1
        if (dataset_index + 1) % 25 == 0 or dataset_index + 1 == len(ds):
            log("PROGRESS", "rows={}/{} shared_selected={}/{}".format(
                dataset_index + 1, len(ds), counts["selected"], len(shared_ids),
            ))

    if len(selected) != len(shared_ids):
        raise RuntimeError("Selected {}, expected {} shared IDs".format(len(selected), len(shared_ids)))

    report: Dict[str, Any] = {
        "name": "activation_patch_shared_subset_window_analysis",
        "status": "locked_before_inference",
        "dataset": args.hf_dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_provenance": provenance,
        "split": args.split,
        "model_config": args.model_id,
        "seed": args.seed,
        "streams": ["base", "mosaic"],
        "variants": list(selection_variants),
        "min_overlap_fraction": args.min_overlap_fraction,
        "selection_used_model_outputs": False,
        "selection_method": "Exact question-ID set present in both prior LLaVA-NeXT attention and integrated-gradients outputs; no geometry-based sample exclusion.",
        "eligibility": [
            "Question ID is in the shared prior attention/integrated-gradients subset.",
            "Text region maps to at least one visual patch token.",
            "Object-control availability or overlap never removes a primary sample.",
            "Three random token sets match the text token count and packed-token composition.",
        ],
        "qid_sources": {
            "attention_output_directory": str(Path(args.attention_qid_dir).resolve()),
            "integrated_gradients_output_directory": str(Path(args.ig_qid_dir).resolve()),
            "sets_identical": True,
        },
        "coverage": {
            "dataset_rows": len(ds),
            "shared_prior_subset": len(shared_ids),
            "selected": len(selected),
            "primary_coverage_fraction": len(selected) / len(shared_ids),
            "geometry_rejected": 0,
            "counts_by_reason": dict(sorted(counts.items())),
        },
        "selected_samples": selected,
        "rejected_samples": [],
    }
    save_json(Path(args.out), report)
    log("SUMMARY", "dataset={} shared_prior_subset={} selected={} primary_coverage=100%".format(
        len(ds), len(shared_ids), len(selected),
    ))
    log("SAVE", args.out)
    log("COMPLETE", "Shared-subset selection locked without model inference")


if __name__ == "__main__":
    main()
