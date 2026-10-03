#!/usr/bin/env python3
"""Qwen spatial mapping used only by the generation-based causal study."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vlms"))
sys.path.insert(0, str(ROOT / "vlms/activation_patching/main_files"))

from qwen3_causal_common import build_qwen_mapping  # noqa: E402
from activation_patch_control_pilot import build_regions_full_coverage  # noqa: E402


POSITIVE_OVERLAP_EPSILON = 1e-12


def positive_overlap_regions(sample, variant, mapping, seed):
    """Select every merged token with a strictly positive bbox intersection."""
    image = sample["notext"]["image"]
    regions = build_regions_full_coverage(
        sample, variant, mapping["tokens"], {"base"}, POSITIVE_OVERLAP_EPSILON,
        (image.height, image.width), seed,
        allow_text_positive_overlap_fallback=False,
    )
    text = regions["text_region"]
    text["selection_method"] = "any_positive_bbox_intersection"
    text["requested_minimum_token_area_overlap"] = 0.0
    text["strictly_positive_overlap_required"] = True
    regions["all_image_tokens"]["includes_packing_newline_tokens"] = False
    regions["all_image_tokens"]["interpretation"] = "All Qwen merged visual tokens."
    return regions


def positive_overlap_region_positions(model, inputs, image, processor, sample, variant, seed):
    positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero(as_tuple=False).flatten().tolist()
    mapping = build_qwen_mapping(image, inputs["image_grid_thw"][0].tolist(), processor)
    if len(positions) != len(mapping["tokens"]):
        raise AssertionError(f"image placeholders ({len(positions)}) != mapped tokens ({len(mapping['tokens'])})")
    regions = positive_overlap_regions(sample, variant, mapping, seed)
    sequence = {
        name: [positions[index] for index in region["token_indices"]]
        for name, region in regions.items()
    }
    return positions, mapping, regions, sequence
