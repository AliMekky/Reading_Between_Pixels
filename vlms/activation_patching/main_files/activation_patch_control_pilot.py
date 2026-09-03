#!/usr/bin/env python3
"""Ten-sample activation-patching control pilot on cleaned GUIC images.

The pilot uses five predeclared decoder layers, both misleading conditions,
both intervention directions, annotated text/object regions, three independent
matched-random controls, and an all-image-token upper-bound intervention.
Selection is deterministic and depends only on annotation/mapping validity,
never on model predictions or patching effects.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from activation_patch_layer_sweep import boxes_overlap, stream_counts
from activation_patch_llava_next_debug import (
    ANSWER_LETTERS,
    DATASET_DEFAULT,
    DATASET_REVISION_DEFAULT,
    MODEL_DEFAULT,
    answer_token_id,
    build_options,
    build_packed_token_mapping,
    classify_change,
    format_mcq_prompt,
    forward_and_capture_resid_pre,
    forward_next_token_logits,
    forward_with_patch,
    get_decoder_layers,
    get_or_download_hf_dataset,
    image_difference_stats,
    image_placeholder_positions,
    image_size_to_num_views,
    log,
    maximum_absolute_difference,
    parse_streams,
    prepare_inputs,
    region_token_indices,
    require,
    save_json,
    summarize_logits,
    text_bbox_yxyx,
    validate_cleaned_overlay,
    validate_dataset_provenance,
)


VARIANTS = ("misleading_groundable", "misleading_ungroundable")
DIRECTIONS = ("restoration", "insertion")
REGIONS = (
    "text_region",
    "correct_object_region",
    "grounded_object_region",
    "matched_random_region_1",
    "matched_random_region_2",
    "matched_random_region_3",
    "all_image_tokens",
)
DEFAULT_LAYERS = (0, 8, 15, 23, 31)
DEFAULT_QUESTION_IDS = (
    "12771240", "19992587", "10106050", "03433670", "14616200",
    "07617263", "15159823", "10421697", "06958697", "1398602",
)


def object_bbox_yxyx(sample: Dict[str, Any], key: str) -> Tuple[float, float, float, float]:
    annotation = sample[key]
    x = float(annotation["x"])
    y = float(annotation["y"])
    width = float(annotation["w"])
    height = float(annotation["h"])
    require(x >= 0 and y >= 0 and width > 0 and height > 0, "Invalid {} object bbox".format(key))
    return y, x, y + height, x + width


def parse_layers(raw: str) -> List[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    require(values, "At least one layer is required")
    require(len(values) == len(set(values)), "Layer list contains duplicates")
    return values


def kind_counts(indices: Sequence[int], mapping_tokens: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    by_index = {int(token["token_idx"]): token for token in mapping_tokens}
    return dict(sorted(Counter(str(by_index[int(index)]["kind"]) for index in indices).items()))


def find_three_matched_random_regions(
    mapping_tokens: Sequence[Dict[str, Any]],
    text_bbox: Tuple[float, float, float, float],
    text_indices: Sequence[int],
    excluded_bboxes: Sequence[Tuple[float, float, float, float]],
    excluded_indices: Set[int],
    streams: Set[str],
    min_overlap_fraction: float,
    image_hw: Tuple[int, int],
    seed: int,
    attempts: int = 100000,
) -> List[Dict[str, Any]]:
    """Select three mutually disjoint controls matching text token composition."""
    target_counts = stream_counts(text_indices, mapping_tokens)
    image_h, image_w = image_hw
    y0, x0, y1, x1 = text_bbox
    box_h, box_w = y1 - y0, x1 - x0
    require(box_h > 0 and box_w > 0, "Text box has non-positive dimensions")
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    forbidden_bboxes = list(excluded_bboxes)
    forbidden_indices = set(int(index) for index in excluded_indices)

    for _ in range(attempts):
        candidate_y0 = rng.uniform(0.0, max(0.0, image_h - box_h))
        candidate_x0 = rng.uniform(0.0, max(0.0, image_w - box_w))
        candidate = (
            candidate_y0,
            candidate_x0,
            candidate_y0 + box_h,
            candidate_x0 + box_w,
        )
        if any(boxes_overlap(candidate, bbox) for bbox in forbidden_bboxes):
            continue
        indices = region_token_indices(mapping_tokens, candidate, streams, min_overlap_fraction)
        if not indices or forbidden_indices.intersection(indices):
            continue
        if stream_counts(indices, mapping_tokens) != target_counts:
            continue
        selected.append({
            "bbox_yxyx": list(candidate),
            "token_indices": indices,
            "token_count": len(indices),
            "stream_counts": target_counts,
        })
        forbidden_bboxes.append(candidate)
        forbidden_indices.update(indices)
        if len(selected) == 3:
            return selected
    raise RuntimeError(
        "Found {}/3 matched-random regions after {} attempts; target={}".format(
            len(selected), attempts, target_counts
        )
    )


def build_regions(
    sample: Dict[str, Any],
    variant: str,
    mapping_tokens: Sequence[Dict[str, Any]],
    streams: Set[str],
    min_overlap_fraction: float,
    image_hw: Tuple[int, int],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    text_bbox = text_bbox_yxyx(sample, variant)
    correct_bbox = object_bbox_yxyx(sample, "correct_answer")
    grounded_bbox = object_bbox_yxyx(sample, "misleading_groundable")
    boxes = {
        "text_region": text_bbox,
        "correct_object_region": correct_bbox,
        "grounded_object_region": grounded_bbox,
    }
    regions: Dict[str, Dict[str, Any]] = {}
    for name, bbox in boxes.items():
        indices = region_token_indices(mapping_tokens, bbox, streams, min_overlap_fraction)
        require(indices, "{} maps to zero visual patch tokens".format(name))
        regions[name] = {
            "bbox_yxyx": list(bbox),
            "token_indices": indices,
            "token_count": len(indices),
            "stream_counts": stream_counts(indices, mapping_tokens),
        }

    text_set = set(regions["text_region"]["token_indices"])
    for object_name in ("correct_object_region", "grounded_object_region"):
        overlap = sorted(text_set.intersection(regions[object_name]["token_indices"]))
        require(not overlap, "{} overlaps text tokens: {}".format(object_name, overlap))

    excluded_indices = set()
    for region in regions.values():
        excluded_indices.update(region["token_indices"])
    random_regions = find_three_matched_random_regions(
        mapping_tokens=mapping_tokens,
        text_bbox=text_bbox,
        text_indices=regions["text_region"]["token_indices"],
        excluded_bboxes=[text_bbox, correct_bbox, grounded_bbox],
        excluded_indices=excluded_indices,
        streams=streams,
        min_overlap_fraction=min_overlap_fraction,
        image_hw=image_hw,
        seed=seed,
    )
    for index, region in enumerate(random_regions, start=1):
        region["selection_seed"] = seed
        region["disjoint_from_text_objects_and_other_randoms"] = True
        regions["matched_random_region_{}".format(index)] = region

    all_indices = [int(token["token_idx"]) for token in mapping_tokens]
    regions["all_image_tokens"] = {
        "bbox_yxyx": None,
        "token_indices": all_indices,
        "token_count": len(all_indices),
        "token_kind_counts": kind_counts(all_indices, mapping_tokens),
        "includes_packing_newline_tokens": True,
        "interpretation": "Approximate upper bound from replacing the complete packed visual sequence.",
    }
    return regions


def find_full_coverage_matched_random_regions(
    mapping_tokens: Sequence[Dict[str, Any]],
    text_indices: Sequence[int],
    object_indices: Sequence[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """Draw three token-count/composition controls without rejecting a sample.

    Each draw exactly matches the text region's packed-token kind counts and
    excludes the text tokens. Object tokens are also excluded when enough
    candidates remain; otherwise the draw falls back to excluding text only
    and records that fact. Draws are independent, so they may overlap each
    other. This avoids selecting the evaluation population according to the
    availability of translated contiguous boxes.
    """
    by_kind: Dict[str, List[int]] = {}
    by_index = {int(token["token_idx"]): token for token in mapping_tokens}
    text_set = {int(index) for index in text_indices}
    object_set = {int(index) for index in object_indices}
    targets = kind_counts(text_indices, mapping_tokens)
    for token in mapping_tokens:
        index = int(token["token_idx"])
        if index not in text_set:
            by_kind.setdefault(str(token["kind"]), []).append(index)

    controls: List[Dict[str, Any]] = []
    for draw_index in range(3):
        rng = random.Random(seed + draw_index)
        chosen: List[int] = []
        avoided_objects = True
        for kind, count in sorted(targets.items()):
            candidates = by_kind.get(kind, [])
            preferred = [index for index in candidates if index not in object_set]
            pool = preferred
            if len(pool) < count:
                pool = candidates
                avoided_objects = False
            require(
                len(pool) >= count,
                "Not enough non-text {} tokens for a matched random control: need {}, found {}".format(
                    kind, count, len(pool)
                ),
            )
            chosen.extend(rng.sample(pool, count))
        chosen.sort()
        require(len(chosen) == len(text_indices), "Matched random token count differs from text")
        require(not text_set.intersection(chosen), "Matched random control overlaps text tokens")
        controls.append({
            "bbox_yxyx": None,
            "token_indices": chosen,
            "token_count": len(chosen),
            "stream_counts": targets,
            "selection_seed": seed + draw_index,
            "selection_method": "independent token sample matched by packed-token kind",
            "disjoint_from_text": True,
            "disjoint_from_objects": avoided_objects,
            "may_overlap_other_random_draws": True,
        })
    return controls


def build_regions_full_coverage(
    sample: Dict[str, Any],
    variant: str,
    mapping_tokens: Sequence[Dict[str, Any]],
    streams: Set[str],
    min_overlap_fraction: float,
    image_hw: Tuple[int, int],
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Build primary regions for all samples and optional object controls.

    Text, three matched-random controls, and all-image tokens are required.
    Object regions are included only when they map to at least one token. Their
    overlap with text is recorded instead of being used as an exclusion rule.
    """
    del image_hw  # Kept in the signature for parity with build_regions.
    text_bbox = text_bbox_yxyx(sample, variant)
    text_indices = region_token_indices(
        mapping_tokens, text_bbox, streams, min_overlap_fraction
    )
    require(text_indices, "text_region maps to zero visual patch tokens")
    regions: Dict[str, Dict[str, Any]] = {
        "text_region": {
            "bbox_yxyx": list(text_bbox),
            "token_indices": text_indices,
            "token_count": len(text_indices),
            "stream_counts": stream_counts(text_indices, mapping_tokens),
            "available": True,
        }
    }

    object_indices: List[int] = []
    text_set = set(text_indices)
    for name, key in (
        ("correct_object_region", "correct_answer"),
        ("grounded_object_region", "misleading_groundable"),
    ):
        bbox = object_bbox_yxyx(sample, key)
        indices = region_token_indices(mapping_tokens, bbox, streams, min_overlap_fraction)
        if not indices:
            continue
        overlap = sorted(text_set.intersection(indices))
        regions[name] = {
            "bbox_yxyx": list(bbox),
            "token_indices": indices,
            "token_count": len(indices),
            "stream_counts": stream_counts(indices, mapping_tokens),
            "available": True,
            "text_overlap_token_count": len(overlap),
            "text_overlap_token_fraction": len(overlap) / len(indices),
            "clean_object_control": len(overlap) == 0,
        }
        object_indices.extend(indices)

    random_regions = find_full_coverage_matched_random_regions(
        mapping_tokens=mapping_tokens,
        text_indices=text_indices,
        object_indices=object_indices,
        seed=seed,
    )
    for index, region in enumerate(random_regions, start=1):
        regions["matched_random_region_{}".format(index)] = region

    all_indices = [int(token["token_idx"]) for token in mapping_tokens]
    regions["all_image_tokens"] = {
        "bbox_yxyx": None,
        "token_indices": all_indices,
        "token_count": len(all_indices),
        "token_kind_counts": kind_counts(all_indices, mapping_tokens),
        "includes_packing_newline_tokens": True,
        "available": True,
        "interpretation": "Approximate upper bound from replacing the complete packed visual sequence.",
    }
    return regions


def selection_key(sample: Dict[str, Any]) -> str:
    return str(sample["question_id"])


def select_samples(
    ds: Any,
    model: LlavaNextForConditionalGeneration,
    sample_count: int,
    seed: int,
    streams: Set[str],
    min_overlap_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, str]]]:
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    selected: List[Dict[str, Any]] = []
    selected_regions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rejected: List[Dict[str, str]] = []
    cfg = model.config
    tile_size = int(cfg.vision_config.image_size)

    for dataset_index in order:
        sample = ds[dataset_index]
        qid = selection_key(sample)
        try:
            no_text_image = sample["notext"]["image"].convert("RGB")
            image_hw = (no_text_image.height, no_text_image.width)
            num_views = image_size_to_num_views(image_hw, cfg.image_grid_pinpoints, tile_size)
            mapping = build_packed_token_mapping(model, image_hw, num_views)
            per_variant = {}
            for variant_index, variant in enumerate(VARIANTS):
                per_variant[variant] = build_regions(
                    sample, variant, mapping["tokens"], streams,
                    min_overlap_fraction, image_hw,
                    seed * 100000 + dataset_index * 10 + variant_index,
                )
            selected.append(sample)
            selected_regions[qid] = per_variant
            log("SELECT", "accepted={}/{} qid={} geometry_and_controls=valid".format(
                len(selected), sample_count, qid
            ))
            if len(selected) == sample_count:
                break
        except Exception as exc:
            rejected.append({"question_id": qid, "reason": "{}: {}".format(type(exc).__name__, exc)})
            log("SELECT-SKIP", "qid={} reason={}".format(qid, rejected[-1]["reason"]))

    require(len(selected) == sample_count, "Selected only {}/{} valid samples".format(len(selected), sample_count))
    return selected, selected_regions, rejected


def record_key(record: Dict[str, Any]) -> Tuple[str, str, int, str, str]:
    return (
        str(record["question_id"]), str(record["variant"]), int(record["layer"]),
        str(record["direction"]), str(record["region"]),
    )


def save_checkpoint(
    path: Path,
    configuration: Dict[str, Any],
    selection: Dict[str, Any],
    samples: Dict[str, Any],
    records: List[Dict[str, Any]],
    expected_records: int,
    status: str,
) -> None:
    outcome_counts = dict(sorted(Counter(record["outcome"] for record in records).items()))
    report = {
        "status": status,
        "milestone": configuration["run_name"],
        "configuration": configuration,
        "selection": selection,
        "samples": samples,
        "records": records,
        "completion": {
            "expected_interventions": expected_records,
            "saved_interventions": len(records),
            "remaining_interventions": expected_records - len(records),
            "outcome_counts": outcome_counts,
        },
    }
    save_json(path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=MODEL_DEFAULT)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--hf_dataset", default=DATASET_DEFAULT)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION_DEFAULT)
    parser.add_argument("--hf_cache_dir", default="../hf_dataset_GUIC_cleaned")
    parser.add_argument("--dataset_validation_file", default="../hf_dataset_GUIC_cleaned/remote_validation.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_count", type=int, default=10)
    parser.add_argument("--question_ids", default=",".join(DEFAULT_QUESTION_IDS))
    parser.add_argument("--selection_file", default=None)
    parser.add_argument("--run_name", default="ten_sample_five_layer_region_control_pilot")
    parser.add_argument("--output_stem", default="control_pilot_10_samples_5_layers")
    parser.add_argument("--checkpoint_every_blocks", type=int, default=1)
    parser.add_argument("--layers", default=",".join(str(value) for value in DEFAULT_LAYERS))
    parser.add_argument("--streams", default="base,mosaic")
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--noop_tolerance", type=float, default=None)
    parser.add_argument("--out_dir", default="../control_pilot_outputs")
    args = parser.parse_args()

    require(args.sample_count > 0, "sample_count must be positive")
    require(args.checkpoint_every_blocks > 0, "checkpoint_every_blocks must be positive")
    require(0.0 <= args.min_overlap_fraction <= 1.0, "min_overlap_fraction must be in [0,1]")
    requested_layers = parse_layers(args.layers)
    streams = parse_streams(args.streams)
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    if args.device == "cpu":
        require(args.dtype == "float32", "CPU runs require float32")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    noop_tolerance = args.noop_tolerance
    if noop_tolerance is None:
        noop_tolerance = 1e-3 if dtype == torch.float16 else 1e-5

    provenance = validate_dataset_provenance(args.dataset_validation_file, args.dataset_revision)
    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, args.split, args.dataset_revision)
    log("CONFIG", "model={} dataset={} revision={} samples={} seed={}".format(
        args.model_id, args.hf_dataset, args.dataset_revision, args.sample_count, args.seed
    ))
    log("CONFIG", "layers={} variants={} directions={} regions={}".format(
        requested_layers, list(VARIANTS), list(DIRECTIONS), list(REGIONS)
    ))
    log("CONFIG", "device={} dtype={} hook=resid_pre streams={} overlap_threshold={}".format(
        device, dtype, sorted(streams), args.min_overlap_fraction
    ))
    log("PASS", "Dataset provenance validates all {}/{} cleaned pairs".format(
        provenance["pairs_validated"], provenance["pairs_expected"]
    ))

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = int(model.config.vision_config.patch_size)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    layers, layer_path = get_decoder_layers(model)
    for layer_index in requested_layers:
        require(0 <= layer_index < len(layers), "Layer {} outside [0, {})".format(layer_index, len(layers)))
    model_revision = getattr(model.config, "_commit_hash", None)
    log("MODEL", "revision={} decoder_path={} layers={}".format(model_revision, layer_path, len(layers)))

    tokenizer = processor.tokenizer
    answer_ids: Dict[str, int] = {}
    token_validation: Dict[str, Any] = {}
    for letter in ANSWER_LETTERS:
        token_id, encoded, decoded = answer_token_id(tokenizer, letter)
        answer_ids[letter] = token_id
        token_validation[letter] = {"token_id": token_id, "encoded": encoded, "decoded": decoded}
    require(len(set(answer_ids.values())) == 4, "Answer token IDs are not distinct")
    log("PASS", "Answer letters map to four distinct validated next tokens")

    selection_manifest = None
    if args.selection_file:
        with Path(args.selection_file).open("r", encoding="utf-8") as handle:
            selection_manifest = json.load(handle)
        require(selection_manifest["dataset_revision"] == args.dataset_revision, "Selection dataset revision mismatch")
        require(int(selection_manifest["seed"]) == args.seed, "Selection seed mismatch")
        selected_qids = [str(entry["question_id"]) for entry in selection_manifest["selected_samples"]]
    else:
        selected_qids = [value.strip() for value in args.question_ids.split(",") if value.strip()]
    require(len(selected_qids) == args.sample_count, "question_ids count must equal sample_count")
    require(len(selected_qids) == len(set(selected_qids)), "question_ids contains duplicates")
    dataset_qids = [str(value) for value in ds["question_id"]]
    qid_to_index = {qid: index for index, qid in enumerate(dataset_qids)}
    missing_qids = [qid for qid in selected_qids if qid not in qid_to_index]
    require(not missing_qids, "Predeclared question IDs missing from dataset: {}".format(missing_qids))
    selected = [ds[qid_to_index[qid]] for qid in selected_qids]
    selected_regions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    tile_size = int(model.config.vision_config.image_size)
    for selected_index, sample in enumerate(selected):
        qid = selection_key(sample)
        dataset_index = qid_to_index[qid]
        if selection_manifest is not None:
            manifest_entry = selection_manifest["selected_samples"][selected_index]
            require(int(manifest_entry["dataset_index"]) == dataset_index, "Dataset index changed for qid={}".format(qid))
        no_text_image = sample["notext"]["image"].convert("RGB")
        image_hw = (no_text_image.height, no_text_image.width)
        num_views = image_size_to_num_views(
            image_hw, model.config.image_grid_pinpoints, tile_size
        )
        mapping = build_packed_token_mapping(model, image_hw, num_views)
        selected_regions[qid] = {}
        for variant_index, variant in enumerate(VARIANTS):
            selected_regions[qid][variant] = build_regions(
                sample, variant, mapping["tokens"], streams,
                args.min_overlap_fraction, image_hw,
                args.seed * 100000 + dataset_index * 10 + variant_index,
            )
        log("SELECT", "validated={}/{} qid={} geometry_and_controls=valid".format(
            selected_index + 1, args.sample_count, qid
        ))
    selection = {
        "method": "predeclared question IDs chosen by seeded dataset-index shuffle and geometry-only eligibility; locked before inference",
        "seed": args.seed,
        "selected_question_ids": selected_qids,
        "selection_used_model_predictions_or_patching_effects": False,
        "selection_manifest": selection_manifest,
    }
    log("SELECTION", "fixed_before_inference qids={}".format(selected_qids))

    configuration = {
        "output_schema_version": 2,
        "run_name": args.run_name,
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "model_revision": model_revision,
        "dataset": args.hf_dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "overlay_image_field": "cleaned_image",
        "sample_count": args.sample_count,
        "selected_question_ids": selected_qids,
        "variants": list(VARIANTS),
        "directions": list(DIRECTIONS),
        "regions": list(REGIONS),
        "layers": requested_layers,
        "number_of_model_layers": len(layers),
        "hook_location": "resid_pre",
        "streams": sorted(streams),
        "min_overlap_fraction": args.min_overlap_fraction,
        "seed": args.seed,
        "dtype": str(dtype),
        "noop_tolerance": noop_tolerance,
        "checkpoint_every_blocks": args.checkpoint_every_blocks,
        "all_image_tokens_include_newlines": True,
        "decoding": "single next-token A-D constrained comparison; no generation",
    }
    output_path = Path(args.out_dir) / "{}.json".format(args.output_stem)
    records: List[Dict[str, Any]] = []
    samples_output: Dict[str, Any] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        prior_config = prior.get("configuration", {})
        for key in ("output_schema_version", "run_name", "model_id", "dataset_revision", "selected_question_ids", "variants", "directions", "regions", "layers", "streams", "seed", "dtype"):
            require(prior_config.get(key) == configuration[key], "Checkpoint mismatch for {}".format(key))
        records = prior.get("records", [])
        samples_output = prior.get("samples", {})
        log("RESUME", "loaded={} path={}".format(len(records), output_path))
    completed = {record_key(record) for record in records}
    blocks_since_checkpoint = 0
    require(len(completed) == len(records), "Checkpoint contains duplicate intervention records")
    expected_records = args.sample_count * len(requested_layers) * len(VARIANTS) * len(DIRECTIONS) * len(REGIONS)

    for sample_number, sample in enumerate(selected, start=1):
        qid = selection_key(sample)
        options, correct_letter, option_meta = build_options(sample, args.shuffle_options, args.seed)
        prompt = format_mcq_prompt(str(sample["question"]), options)
        sample_report = samples_output.setdefault(qid, {
            "question": str(sample["question"]),
            "options": options,
            "option_meta": option_meta,
            "correct_letter": correct_letter,
            "answer_token_validation": token_validation,
            "conditions": {},
        })
        no_text_image = sample["notext"]["image"].convert("RGB")

        for variant_index, variant in enumerate(VARIANTS):
            misleading_letter = option_meta["key_to_label"][variant]
            require(misleading_letter != correct_letter, "Correct and misleading letters coincide")
            original_overlay = sample[variant]["image"].convert("RGB")
            cleaned_overlay = sample[variant]["cleaned_image"].convert("RGB")
            text_bbox = text_bbox_yxyx(sample, variant)
            pixel_stats = image_difference_stats(no_text_image, cleaned_overlay, text_bbox)
            cleaned_validation = validate_cleaned_overlay(no_text_image, original_overlay, cleaned_overlay, text_bbox)
            require(pixel_stats["outside_pixels_changed"] == 0, "Outside-box pixels changed")
            require(pixel_stats["inside_pixels_changed"] > 0, "No inside-box pixels changed")
            require(cleaned_validation["cleaned_outside_no_text_mismatch_pixels"] == 0, "Outside mismatch")
            require(cleaned_validation["cleaned_inside_original_mismatch_pixels"] == 0, "Inside mismatch")

            no_text_inputs, no_text_prompt = prepare_inputs(processor, no_text_image, prompt, device, dtype)
            overlay_inputs, overlay_prompt = prepare_inputs(processor, cleaned_overlay, prompt, device, dtype)
            require(no_text_prompt == overlay_prompt, "Formatted prompts differ")
            require(torch.equal(no_text_inputs["input_ids"], overlay_inputs["input_ids"]), "Input IDs differ")
            no_text_positions = image_placeholder_positions(model, no_text_inputs["input_ids"])
            overlay_positions = image_placeholder_positions(model, overlay_inputs["input_ids"])
            require(no_text_positions == overlay_positions, "Image placeholder positions differ")
            image_hw = tuple(int(value) for value in no_text_inputs["image_sizes"][0].tolist())
            overlay_hw = tuple(int(value) for value in overlay_inputs["image_sizes"][0].tolist())
            require(image_hw == overlay_hw, "Paired image sizes differ")
            num_views = int(no_text_inputs["pixel_values"].shape[1]) if no_text_inputs["pixel_values"].ndim == 5 else 1
            mapping = build_packed_token_mapping(model, image_hw, num_views)
            require(len(no_text_positions) == len(mapping["tokens"]), "Packed mapping/placeholder count mismatch")
            require(mapping["summary"]["view_count_matches"], "View-count mapping check failed")

            regions = selected_regions[qid][variant]
            for name in REGIONS:
                require(name in regions and regions[name]["token_indices"], "Missing region {}".format(name))
                require(max(regions[name]["token_indices"]) < len(no_text_positions), "Region index out of range")
            region_positions = {
                name: [no_text_positions[index] for index in regions[name]["token_indices"]]
                for name in REGIONS
            }
            random_counts = [
                regions["matched_random_region_{}".format(i)]["stream_counts"]
                for i in range(1, 4)
            ]
            require(all(value == regions["text_region"]["stream_counts"] for value in random_counts), "Random composition mismatch")
            log("REGION", "sample={}/{} qid={} variant={} text={} correct_object={} grounded_object={} random_each={} all={}".format(
                sample_number, args.sample_count, qid, variant,
                regions["text_region"]["token_count"], regions["correct_object_region"]["token_count"],
                regions["grounded_object_region"]["token_count"],
                [regions["matched_random_region_{}".format(i)]["token_count"] for i in range(1, 4)],
                regions["all_image_tokens"]["token_count"],
            ))

            no_text_logits = forward_next_token_logits(model, no_text_inputs)
            overlay_logits = forward_next_token_logits(model, overlay_inputs)
            no_text_base = summarize_logits(no_text_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
            overlay_base = summarize_logits(overlay_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
            condition = sample_report["conditions"].setdefault(variant, {})
            condition.update({
                "misleading_letter": misleading_letter,
                "misleading_text": options[misleading_letter],
                "pixel_difference": pixel_stats,
                "cleaned_overlay_validation": cleaned_validation,
                "alignment": {
                    "sequence_length": int(no_text_inputs["input_ids"].shape[1]),
                    "image_placeholder_count": len(no_text_positions),
                    "image_size_hw": list(image_hw),
                    "num_views": num_views,
                    "mapping_summary": mapping["summary"],
                },
                "regions": regions,
                "baselines": {"no_text": no_text_base, "overlay": overlay_base},
            })
            log("BASELINE", "qid={} variant={} no_text_pred={} overlay_pred={} margin={:.6f}->{:.6f}".format(
                qid, variant, no_text_base["choice_constrained_prediction"],
                overlay_base["choice_constrained_prediction"],
                no_text_base["margin_correct_minus_misleading"],
                overlay_base["margin_correct_minus_misleading"],
            ))

            for layer_index in requested_layers:
                expected_keys = {
                    (qid, variant, layer_index, direction, region)
                    for direction in DIRECTIONS for region in REGIONS
                }
                if expected_keys.issubset(completed):
                    log("RESUME", "qid={} variant={} layer={} complete".format(qid, variant, layer_index))
                    continue
                require(not expected_keys.intersection(completed), "Partially saved layer block; refusing ambiguous resume")
                no_text_noop_logits, no_text_hidden = forward_and_capture_resid_pre(model, layers[layer_index], no_text_inputs)
                overlay_noop_logits, overlay_hidden = forward_and_capture_resid_pre(model, layers[layer_index], overlay_inputs)
                no_text_noop_error = maximum_absolute_difference(no_text_logits, no_text_noop_logits)
                overlay_noop_error = maximum_absolute_difference(overlay_logits, overlay_noop_logits)
                require(no_text_noop_error <= noop_tolerance, "No-text no-op failed")
                require(overlay_noop_error <= noop_tolerance, "Overlay no-op failed")
                log("NOOP", "qid={} variant={} layer={} no_text={:.8g} overlay={:.8g}".format(
                    qid, variant, layer_index, no_text_noop_error, overlay_noop_error
                ))

                block_records = []
                for region in REGIONS:
                    positions = region_positions[region]
                    restoration_logits, restoration_integrity = forward_with_patch(
                        model, layers[layer_index], overlay_inputs, no_text_hidden, positions
                    )
                    insertion_logits, insertion_integrity = forward_with_patch(
                        model, layers[layer_index], no_text_inputs, overlay_hidden, positions
                    )
                    for integrity in (restoration_integrity, insertion_integrity):
                        require(integrity["patched_donor_max_abs_difference_after"] == 0.0, "Patched values differ from donor")
                        require(integrity["unpatched_positions_max_direct_change"] == 0.0, "Hook changed unpatched values")
                        require(integrity["donor_recipient_max_abs_difference_before"] > 0.0, "Donor and recipient region values are identical")
                    patched_summaries = {
                        "restoration": summarize_logits(restoration_logits, tokenizer, answer_ids, correct_letter, misleading_letter),
                        "insertion": summarize_logits(insertion_logits, tokenizer, answer_ids, correct_letter, misleading_letter),
                    }
                    for direction in DIRECTIONS:
                        recipient = overlay_base if direction == "restoration" else no_text_base
                        patched = patched_summaries[direction]
                        effect = (
                            patched["margin_correct_minus_misleading"] - recipient["margin_correct_minus_misleading"]
                            if direction == "restoration"
                            else recipient["margin_correct_minus_misleading"] - patched["margin_correct_minus_misleading"]
                        )
                        require(math.isfinite(effect), "Non-finite patching effect")
                        integrity = restoration_integrity if direction == "restoration" else insertion_integrity
                        record = {
                            "question_id": qid,
                            "variant": variant,
                            "layer": layer_index,
                            "direction": direction,
                            "region": region,
                            "token_count": len(positions),
                            "effect": effect,
                            "recipient_margin": recipient["margin_correct_minus_misleading"],
                            "patched_margin": patched["margin_correct_minus_misleading"],
                            "recipient_prediction": recipient["choice_constrained_prediction"],
                            "patched_prediction": patched["choice_constrained_prediction"],
                            "recipient": recipient,
                            "patched": patched,
                            "outcome": classify_change(
                                recipient["choice_constrained_prediction"], patched["choice_constrained_prediction"],
                                correct_letter, misleading_letter,
                            ),
                            "noop": {
                                "no_text_max_logit_difference": no_text_noop_error,
                                "overlay_max_logit_difference": overlay_noop_error,
                            },
                            "integrity": integrity,
                        }
                        block_records.append(record)
                        log("PATCH", "qid={} variant={} layer={} direction={} region={} n={} effect={:+.6f} pred={}->{} outcome={}".format(
                            qid, variant, layer_index, direction, region, len(positions), effect,
                            record["recipient_prediction"], record["patched_prediction"], record["outcome"],
                        ))
                require(len(block_records) == len(DIRECTIONS) * len(REGIONS), "Layer block count mismatch")
                records.extend(block_records)
                completed.update(record_key(record) for record in block_records)
                blocks_since_checkpoint += 1
                if blocks_since_checkpoint >= args.checkpoint_every_blocks or len(records) == expected_records:
                    save_checkpoint(output_path, configuration, selection, samples_output, records, expected_records, "in_progress")
                    log("CHECKPOINT", "saved={}/{} sample={}/{} path={}".format(
                        len(records), expected_records, sample_number, args.sample_count, output_path
                    ))
                    blocks_since_checkpoint = 0
                else:
                    log("PROGRESS", "completed_in_memory={}/{} blocks_until_checkpoint={}".format(
                        len(records), expected_records, args.checkpoint_every_blocks - blocks_since_checkpoint
                    ))
                del no_text_hidden, overlay_hidden

    require(len(records) == expected_records, "Saved {}, expected {}".format(len(records), expected_records))
    require(len({record_key(record) for record in records}) == expected_records, "Final records are not unique")
    save_checkpoint(output_path, configuration, selection, samples_output, records, expected_records, "success")
    log("SUMMARY", "attempted_samples={} successful_samples={} geometry_skips={} interventions={}/{} failures=0".format(
        args.sample_count, args.sample_count, 0, len(records), expected_records
    ))
    log("SAVE", str(output_path))
    log("COMPLETE", "All {} control-pilot interventions passed validation".format(expected_records))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAIL", "{}: {}".format(type(exc).__name__, exc))
        raise
