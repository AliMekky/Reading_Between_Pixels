#!/usr/bin/env python3
"""One-sample, all-layer activation-patching sweep on cleaned GUIC pairs.

This second implementation milestone evaluates two misleading conditions, two
patch directions, and two regions (text and a token-count-matched random box)
at every LLaVA-NeXT decoder layer. Results are checkpointed after each layer.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from activation_patch_llava_next_debug import (
    ANSWER_LETTERS,
    DATASET_DEFAULT,
    DATASET_REVISION_DEFAULT,
    MODEL_DEFAULT,
    answer_token_id,
    build_options,
    build_packed_token_mapping,
    classify_change,
    find_sample_by_qid,
    format_mcq_prompt,
    forward_and_capture_resid_pre,
    forward_next_token_logits,
    forward_with_patch,
    get_decoder_layers,
    get_or_download_hf_dataset,
    image_difference_stats,
    image_placeholder_positions,
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
REGIONS = ("text_region", "matched_random_region")
DIRECTIONS = ("restoration", "insertion")


def boxes_overlap(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> bool:
    ay0, ax0, ay1, ax1 = first
    by0, bx0, by1, bx1 = second
    return max(ay0, by0) < min(ay1, by1) and max(ax0, bx0) < min(ax1, bx1)


def stream_counts(
    indices: Sequence[int], mapping_tokens: Sequence[Dict[str, Any]]
) -> Dict[str, int]:
    by_index = {int(token["token_idx"]): token for token in mapping_tokens}
    counts = {"base": 0, "mosaic": 0}
    for index in indices:
        kind = by_index[int(index)].get("kind")
        if kind == "base_patch":
            counts["base"] += 1
        elif kind == "mosaic_patch":
            counts["mosaic"] += 1
        else:
            raise RuntimeError("Selected non-patch token {} kind={}".format(index, kind))
    return counts


def matched_random_region(
    mapping_tokens: Sequence[Dict[str, Any]],
    text_bbox: Tuple[float, float, float, float],
    text_indices: Sequence[int],
    streams: Set[str],
    min_overlap_fraction: float,
    image_hw: Tuple[int, int],
    seed: int,
    attempts: int = 50000,
) -> Tuple[Tuple[float, float, float, float], List[int], Dict[str, int]]:
    """Find a non-overlapping random pixel box with identical stream counts."""
    target_counts = stream_counts(text_indices, mapping_tokens)
    target_set = set(int(index) for index in text_indices)
    image_h, image_w = image_hw
    y0, x0, y1, x1 = text_bbox
    box_h, box_w = y1 - y0, x1 - x0
    require(box_h > 0 and box_w > 0, "Text box has non-positive dimensions")
    require(box_h <= image_h and box_w <= image_w, "Text box is larger than the image")
    rng = random.Random(seed)

    for _ in range(attempts):
        candidate_y0 = rng.uniform(0.0, max(0.0, image_h - box_h))
        candidate_x0 = rng.uniform(0.0, max(0.0, image_w - box_w))
        candidate = (
            candidate_y0,
            candidate_x0,
            candidate_y0 + box_h,
            candidate_x0 + box_w,
        )
        if boxes_overlap(candidate, text_bbox):
            continue
        indices = region_token_indices(
            mapping_tokens, candidate, streams, min_overlap_fraction
        )
        if not indices or target_set.intersection(indices):
            continue
        if stream_counts(indices, mapping_tokens) == target_counts:
            return candidate, indices, target_counts
    raise RuntimeError(
        "Could not find a non-overlapping matched random region after {} attempts; "
        "target stream counts={}".format(attempts, target_counts)
    )


def record_key(record: Dict[str, Any]) -> Tuple[str, int, str, str]:
    return (
        str(record["variant"]),
        int(record["layer"]),
        str(record["direction"]),
        str(record["region"]),
    )


def load_checkpoint(path: Path, expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    configuration = saved.get("configuration", {})
    for key, value in expected.items():
        require(
            configuration.get(key) == value,
            "Checkpoint configuration mismatch for {}: {} != {}".format(
                key, configuration.get(key), value
            ),
        )
    records = saved.get("records", [])
    require(len({record_key(record) for record in records}) == len(records), "Checkpoint has duplicate records")
    log("RESUME", "Loaded {} completed interventions from {}".format(len(records), path))
    return records


def save_checkpoint(
    path: Path,
    configuration: Dict[str, Any],
    sample_info: Dict[str, Any],
    condition_metadata: Dict[str, Any],
    baselines: Dict[str, Any],
    records: List[Dict[str, Any]],
    expected_records: int,
    status: str,
) -> None:
    completed_by_variant = {
        variant: sum(record["variant"] == variant for record in records)
        for variant in VARIANTS
    }
    report = {
        "status": status,
        "milestone": "one_sample_all_layers_text_and_matched_random",
        "configuration": configuration,
        "sample": sample_info,
        "condition_metadata": condition_metadata,
        "baselines": baselines,
        "records": records,
        "completion": {
            "expected_interventions": expected_records,
            "saved_interventions": len(records),
            "completed_by_variant": completed_by_variant,
            "remaining_interventions": expected_records - len(records),
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
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--streams", default="base,mosaic")
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--noop_tolerance", type=float, default=None)
    parser.add_argument("--out_dir", default="../layer_sweep_outputs")
    args = parser.parse_args()

    require(0.0 <= args.min_overlap_fraction <= 1.0, "min_overlap_fraction must be in [0,1]")
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
    sample = find_sample_by_qid(ds, args.question_id)
    qid = str(sample["question_id"])

    log("CONFIG", "model={} dataset={} revision={} qid={}".format(
        args.model_id, args.hf_dataset, args.dataset_revision, qid
    ))
    log("CONFIG", "variants={} directions={} regions={} streams={}".format(
        list(VARIANTS), list(DIRECTIONS), list(REGIONS), sorted(streams)
    ))
    log("CONFIG", "device={} dtype={} seed={} overlap_threshold={}".format(
        device, dtype, args.seed, args.min_overlap_fraction
    ))
    log("PASS", "Dataset provenance validates all {}/{} cleaned pairs".format(
        provenance["pairs_validated"], provenance["pairs_expected"]
    ))

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    # Match the processor-loading path that passed the one-layer GPU debug run.
    # Passing use_fast=False also forces the slow Llama tokenizer, which adds an
    # unnecessary SentencePiece dependency in this environment.
    processor = LlavaNextProcessor.from_pretrained(
        args.model_id, revision=args.model_revision
    )
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = int(model.config.vision_config.patch_size)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    layers, layer_path = get_decoder_layers(model)
    model_revision = getattr(model.config, "_commit_hash", None)
    log("MODEL", "revision={} decoder_path={} layers={} processor_loading=validated_default".format(
        model_revision, layer_path, len(layers)
    ))

    options, correct_letter, option_meta = build_options(sample, args.shuffle_options, args.seed)
    prompt = format_mcq_prompt(str(sample["question"]), options)
    tokenizer = processor.tokenizer
    answer_ids = {}
    token_validation = {}
    for letter in ANSWER_LETTERS:
        token_id, encoded, decoded = answer_token_id(tokenizer, letter)
        answer_ids[letter] = token_id
        token_validation[letter] = {"token_id": token_id, "encoded": encoded, "decoded": decoded}
    require(len(set(answer_ids.values())) == 4, "Answer token IDs are not distinct")

    configuration = {
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "model_revision": model_revision,
        "dataset": args.hf_dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "question_id": qid,
        "overlay_image_field": "cleaned_image",
        "variants": list(VARIANTS),
        "directions": list(DIRECTIONS),
        "regions": list(REGIONS),
        "layers": list(range(len(layers))),
        "number_of_layers": len(layers),
        "hook_location": "resid_pre",
        "streams": sorted(streams),
        "min_overlap_fraction": args.min_overlap_fraction,
        "seed": args.seed,
        "dtype": str(dtype),
        "processor_loading": "validated_default",
        "noop_tolerance": noop_tolerance,
    }
    checkpoint_match = {
        key: configuration[key]
        for key in (
            "model_id", "dataset_revision", "question_id", "overlay_image_field",
            "variants", "directions", "regions", "layers", "streams",
            "min_overlap_fraction", "seed", "dtype",
        )
    }
    output_path = Path(args.out_dir) / qid / "text_random_all_layers.json"
    records = load_checkpoint(output_path, checkpoint_match)
    completed = {record_key(record) for record in records}
    expected_records = len(layers) * len(VARIANTS) * len(DIRECTIONS) * len(REGIONS)

    sample_info = {
        "question": str(sample["question"]),
        "options": options,
        "option_meta": option_meta,
        "correct_letter": correct_letter,
        "answer_token_validation": token_validation,
    }
    condition_metadata: Dict[str, Any] = {}
    baselines: Dict[str, Any] = {}

    no_text_image = sample["notext"]["image"].convert("RGB")
    for variant_index, variant in enumerate(VARIANTS):
        misleading_letter = option_meta["key_to_label"][variant]
        require(misleading_letter != correct_letter, "Correct and misleading letters coincide")
        original_overlay = sample[variant]["image"].convert("RGB")
        cleaned_overlay = sample[variant]["cleaned_image"].convert("RGB")
        text_bbox = text_bbox_yxyx(sample, variant)
        pixel_stats = image_difference_stats(no_text_image, cleaned_overlay, text_bbox)
        cleaned_validation = validate_cleaned_overlay(
            no_text_image, original_overlay, cleaned_overlay, text_bbox
        )
        require(pixel_stats["outside_pixels_changed"] == 0, "Outside-box pixels changed")
        require(pixel_stats["inside_pixels_changed"] > 0, "No inside-box pixels changed")
        require(cleaned_validation["cleaned_outside_no_text_mismatch_pixels"] == 0, "Outside-box mismatch")
        require(cleaned_validation["cleaned_inside_original_mismatch_pixels"] == 0, "Inside-box mismatch")

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
        overlay_views = int(overlay_inputs["pixel_values"].shape[1]) if overlay_inputs["pixel_values"].ndim == 5 else 1
        require(num_views == overlay_views, "Paired view counts differ")
        mapping = build_packed_token_mapping(model, image_hw, num_views)
        require(len(no_text_positions) == mapping["summary"]["total_packed_image_tokens"], "Mapping count mismatch")
        require(mapping["summary"]["view_count_matches"], "View-count mapping check failed")

        text_indices = region_token_indices(
            mapping["tokens"], text_bbox, streams, args.min_overlap_fraction
        )
        require(text_indices, "Text region maps to no tokens")
        random_seed = args.seed * 1000 + variant_index
        random_bbox, random_indices, target_counts = matched_random_region(
            mapping["tokens"], text_bbox, text_indices, streams,
            args.min_overlap_fraction, image_hw, random_seed,
        )
        require(len(random_indices) == len(text_indices), "Random/text token counts differ")
        require(not set(random_indices).intersection(text_indices), "Random/text token sets overlap")
        region_indices = {
            "text_region": text_indices,
            "matched_random_region": random_indices,
        }
        region_positions = {
            name: [no_text_positions[index] for index in indices]
            for name, indices in region_indices.items()
        }
        condition_metadata[variant] = {
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
            "regions": {
                "text_region": {
                    "bbox_yxyx": list(text_bbox),
                    "token_indices": text_indices,
                    "token_count": len(text_indices),
                    "stream_counts": target_counts,
                },
                "matched_random_region": {
                    "bbox_yxyx": list(random_bbox),
                    "token_indices": random_indices,
                    "token_count": len(random_indices),
                    "stream_counts": stream_counts(random_indices, mapping["tokens"]),
                    "selection_seed": random_seed,
                    "overlaps_text_tokens": False,
                    "overlaps_text_bbox": False,
                },
            },
        }
        log("PIXELS", "variant={} inside_changed={} outside_changed={}".format(
            variant, pixel_stats["inside_pixels_changed"], pixel_stats["outside_pixels_changed"]
        ))
        log("REGION", "variant={} text_tokens={} random_tokens={} stream_counts={} random_bbox={}".format(
            variant, len(text_indices), len(random_indices), target_counts,
            tuple(round(value, 2) for value in random_bbox),
        ))
        log("PASS", "variant={} cleaned alignment and matched-random validation passed".format(variant))

        no_text_logits = forward_next_token_logits(model, no_text_inputs)
        overlay_logits = forward_next_token_logits(model, overlay_inputs)
        no_text_base = summarize_logits(
            no_text_logits, tokenizer, answer_ids, correct_letter, misleading_letter
        )
        overlay_base = summarize_logits(
            overlay_logits, tokenizer, answer_ids, correct_letter, misleading_letter
        )
        baselines[variant] = {"no_text": no_text_base, "overlay": overlay_base}
        log("BASELINE", "variant={} no_text_margin={:.6f} overlay_margin={:.6f}".format(
            variant, no_text_base["margin_correct_minus_misleading"],
            overlay_base["margin_correct_minus_misleading"],
        ))

        for layer_index, layer in enumerate(layers):
            layer_keys = {
                (variant, layer_index, direction, region)
                for direction in DIRECTIONS for region in REGIONS
            }
            if layer_keys.issubset(completed):
                log("RESUME", "variant={} layer={} already complete".format(variant, layer_index))
                continue

            no_text_noop_logits, no_text_hidden = forward_and_capture_resid_pre(
                model, layer, no_text_inputs
            )
            overlay_noop_logits, overlay_hidden = forward_and_capture_resid_pre(
                model, layer, overlay_inputs
            )
            no_text_noop_error = maximum_absolute_difference(no_text_logits, no_text_noop_logits)
            overlay_noop_error = maximum_absolute_difference(overlay_logits, overlay_noop_logits)
            require(no_text_noop_error <= noop_tolerance, "No-text no-op failed at layer {}".format(layer_index))
            require(overlay_noop_error <= noop_tolerance, "Overlay no-op failed at layer {}".format(layer_index))
            log("NOOP", "variant={} layer={} no_text_error={:.8g} overlay_error={:.8g}".format(
                variant, layer_index, no_text_noop_error, overlay_noop_error
            ))

            new_records = []
            for region in REGIONS:
                positions = region_positions[region]
                restoration_logits, restoration_integrity = forward_with_patch(
                    model, layer, overlay_inputs, no_text_hidden, positions
                )
                insertion_logits, insertion_integrity = forward_with_patch(
                    model, layer, no_text_inputs, overlay_hidden, positions
                )
                for direction, integrity in (
                    ("restoration", restoration_integrity),
                    ("insertion", insertion_integrity),
                ):
                    require(integrity["patched_donor_max_abs_difference_after"] == 0.0, "Patched values differ from donor")
                    require(integrity["unpatched_positions_max_direct_change"] == 0.0, "Hook changed unpatched positions")
                    require(integrity["donor_recipient_max_abs_difference_before"] > 0.0, "Donor/recipient patch values are identical")

                restoration = summarize_logits(
                    restoration_logits, tokenizer, answer_ids, correct_letter, misleading_letter
                )
                insertion = summarize_logits(
                    insertion_logits, tokenizer, answer_ids, correct_letter, misleading_letter
                )
                values = {
                    "restoration": (
                        restoration,
                        restoration["margin_correct_minus_misleading"] - overlay_base["margin_correct_minus_misleading"],
                        overlay_base,
                        restoration_integrity,
                    ),
                    "insertion": (
                        insertion,
                        no_text_base["margin_correct_minus_misleading"] - insertion["margin_correct_minus_misleading"],
                        no_text_base,
                        insertion_integrity,
                    ),
                }
                for direction in DIRECTIONS:
                    patched, effect, recipient, integrity = values[direction]
                    require(math.isfinite(effect), "Non-finite patching effect")
                    result = {
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
                        "outcome": classify_change(
                            recipient["choice_constrained_prediction"],
                            patched["choice_constrained_prediction"],
                            correct_letter,
                            misleading_letter,
                        ),
                        "noop": {
                            "no_text_max_logit_difference": no_text_noop_error,
                            "overlay_max_logit_difference": overlay_noop_error,
                        },
                        "integrity": integrity,
                    }
                    new_records.append(result)
                    log("PATCH", "variant={} layer={} direction={} region={} tokens={} effect={:+.6f} outcome={} before_mean={:.6g}".format(
                        variant, layer_index, direction, region, len(positions), effect,
                        result["outcome"], integrity["donor_recipient_mean_abs_difference_before"],
                    ))

            require(len(new_records) == len(DIRECTIONS) * len(REGIONS), "Layer result count mismatch")
            records.extend(new_records)
            completed.update(record_key(record) for record in new_records)
            save_checkpoint(
                output_path, configuration, sample_info, condition_metadata,
                baselines, records, expected_records, "in_progress",
            )
            log("CHECKPOINT", "saved={}/{} path={}".format(len(records), expected_records, output_path))
            del no_text_hidden, overlay_hidden

    require(len(records) == expected_records, "Saved {} records, expected {}".format(len(records), expected_records))
    require(len({record_key(record) for record in records}) == expected_records, "Final records are not unique")
    save_checkpoint(
        output_path, configuration, sample_info, condition_metadata,
        baselines, records, expected_records, "success",
    )
    log("SUMMARY", "expected={} completed={} skipped=0 failures=0".format(expected_records, len(records)))
    log("SAVE", str(output_path))
    log("COMPLETE", "All 256 layer-sweep interventions passed validation")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAIL", "{}: {}".format(type(exc).__name__, exc))
        raise
