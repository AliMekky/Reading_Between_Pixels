#!/usr/bin/env python3
"""Shared-subset simultaneous layer-window activation-patching analysis."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from activation_patch_control_pilot import (
    DIRECTIONS,
    REGIONS,
    VARIANTS,
    build_regions_full_coverage,
)
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
    forward_next_token_logits,
    get_decoder_layers,
    get_or_download_hf_dataset,
    image_difference_stats,
    image_placeholder_positions,
    log,
    maximum_absolute_difference,
    parse_streams,
    prepare_inputs,
    require,
    save_json,
    summarize_logits,
    text_bbox_yxyx,
    validate_cleaned_overlay,
    validate_dataset_provenance,
)


WINDOWS = {
    "early_0_5": tuple(range(0, 6)),
    "middle_10_15": tuple(range(10, 16)),
    "late_26_31": tuple(range(26, 32)),
}
SUPPORTED_VARIANTS = ("correct_answer",) + VARIANTS + ("irrelevant_word",)


def strongest_incorrect_letter(logits, answer_ids, correct_letter):
    """Freeze the strongest no-text incorrect option as the correct-overlay comparator."""
    candidates = [letter for letter in ANSWER_LETTERS if letter != correct_letter]
    return max(candidates, key=lambda letter: float(logits[answer_ids[letter]].item()))


def correct_overlay_outcome(before_letter, after_letter, correct_letter):
    if before_letter != correct_letter and after_letter == correct_letter:
        return "correct_gain"
    if before_letter == correct_letter and after_letter != correct_letter:
        return "correct_loss"
    return "other_flip" if before_letter != after_letter else "no_prediction_change"


def oriented_margin_effect(direction, recipient_margin, patched_margin, correct_overlay):
    """Return a positive value for the intervention's hypothesized direction."""
    if correct_overlay:
        return recipient_margin - patched_margin if direction == "restoration" else patched_margin - recipient_margin
    return patched_margin - recipient_margin if direction == "restoration" else recipient_margin - patched_margin


@torch.no_grad()
def forward_and_capture_many_resid_pre(
    model: LlavaNextForConditionalGeneration,
    layers: Sequence[torch.nn.Module],
    layer_indices: Sequence[int],
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_index in layer_indices:
        def capture_hook(_module: torch.nn.Module, hook_args: Tuple[Any, ...], index: int = layer_index) -> None:
            require(index not in captured, "Capture hook called more than once for layer {}".format(index))
            require(hook_args and torch.is_tensor(hook_args[0]), "Capture hook received no hidden state")
            captured[index] = hook_args[0].detach().clone()
            return None

        handles.append(layers[layer_index].register_forward_pre_hook(capture_hook))
    try:
        logits = forward_next_token_logits(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    require(set(captured) == set(layer_indices), "Not every requested layer was captured")
    return logits, captured


@torch.no_grad()
def forward_with_window_patch(
    model: LlavaNextForConditionalGeneration,
    layers: Sequence[torch.nn.Module],
    layer_indices: Sequence[int],
    recipient_inputs: Dict[str, torch.Tensor],
    donor_hidden: Dict[int, torch.Tensor],
    sequence_positions: Sequence[int],
    require_nonzero_difference: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    integrity_by_layer: Dict[int, Dict[str, float]] = {}
    handles = []

    for layer_index in layer_indices:
        def patch_hook(
            _module: torch.nn.Module,
            hook_args: Tuple[Any, ...],
            index: int = layer_index,
        ) -> Tuple[Any, ...]:
            require(index not in integrity_by_layer, "Patch hook called more than once at layer {}".format(index))
            require(hook_args and torch.is_tensor(hook_args[0]), "Patch hook received no hidden state")
            hidden = hook_args[0]
            donor = donor_hidden[index].to(device=hidden.device, dtype=hidden.dtype)
            require(tuple(hidden.shape) == tuple(donor.shape), "Donor/recipient shapes differ at layer {}".format(index))
            positions = torch.as_tensor(sequence_positions, dtype=torch.long, device=hidden.device)
            require(positions.numel() > 0, "Window patch has no positions")
            require(int(positions.min()) >= 0 and int(positions.max()) < hidden.shape[1], "Patch position out of range")
            before = (hidden[:, positions, :] - donor[:, positions, :]).abs()
            patched = hidden.clone()
            patched[:, positions, :] = donor[:, positions, :]
            after = (patched[:, positions, :] - donor[:, positions, :]).abs()
            direct_delta = (patched - hidden).abs()
            direct_delta[:, positions, :] = 0
            integrity_by_layer[index] = {
                "donor_recipient_mean_abs_difference_before": float(before.mean().item()),
                "donor_recipient_max_abs_difference_before": float(before.max().item()),
                "patched_donor_max_abs_difference_after": float(after.max().item()),
                "unpatched_positions_max_direct_change": float(direct_delta.max().item()),
                "number_of_patched_positions": int(positions.numel()),
            }
            return (patched,) + tuple(hook_args[1:])

        handles.append(layers[layer_index].register_forward_pre_hook(patch_hook))
    try:
        logits = forward_next_token_logits(model, recipient_inputs)
    finally:
        for handle in handles:
            handle.remove()
    require(set(integrity_by_layer) == set(layer_indices), "Not every window patch hook ran")
    ordered = {str(index): integrity_by_layer[index] for index in layer_indices}
    nonzero_layers = sum(
        values["donor_recipient_max_abs_difference_before"] > 0.0
        for values in ordered.values()
    )
    for index, values in ordered.items():
        require(values["patched_donor_max_abs_difference_after"] == 0.0, "Patched values differ from donor at layer {}".format(index))
        require(values["unpatched_positions_max_direct_change"] == 0.0, "Unpatched values changed at layer {}".format(index))
    if require_nonzero_difference:
        require(nonzero_layers > 0, "Donor and recipient values are identical throughout the window")
    return logits, {
        "by_layer": ordered,
        "layers_with_nonzero_donor_recipient_difference": nonzero_layers,
        "number_of_window_layers": len(layer_indices),
        "zero_difference_control": nonzero_layers == 0,
    }


def record_key(record: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(record["question_id"]), str(record["variant"]),
        str(record["window"]), str(record["direction"]), str(record["region"]),
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
    completed_qids = sorted({str(record["question_id"]) for record in records})
    report = {
        "status": status,
        "milestone": "shared_subset_layer_window_analysis",
        "configuration": configuration,
        "selection": selection,
        "samples": samples,
        "records": records,
        "completion": {
            "expected_interventions": expected_records,
            "saved_interventions": len(records),
            "remaining_interventions": expected_records - len(records),
            "completed_question_ids": completed_qids,
            "outcome_counts": dict(sorted(Counter(record["outcome"] for record in records).items())),
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
    parser.add_argument("--selection_file", default="activation_patch_confirmation_selection.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--question_id", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--streams", default="base,mosaic")
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--noop_tolerance", type=float, default=None)
    parser.add_argument("--checkpoint_every_samples", type=int, default=5)
    parser.add_argument(
        "--regions",
        default=",".join(REGIONS),
        help="Comma-separated intervention regions selected from the manifest.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated overlay conditions.",
    )
    parser.add_argument(
        "--single_layer_sweep",
        action="store_true",
        help="Patch each decoder layer separately instead of the three fixed windows.",
    )
    parser.add_argument("--out", default="../confirmation_outputs/activation_patch_window_confirmation.json")
    args = parser.parse_args()

    require(args.checkpoint_every_samples > 0, "checkpoint_every_samples must be positive")
    requested_regions = tuple(value.strip() for value in args.regions.split(",") if value.strip())
    require(requested_regions, "At least one intervention region is required")
    require(len(requested_regions) == len(set(requested_regions)), "Region list contains duplicates")
    require(set(requested_regions).issubset(REGIONS), "Unknown intervention region requested")
    requested_variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    require(requested_variants, "At least one overlay condition is required")
    require(len(requested_variants) == len(set(requested_variants)), "Variant list contains duplicates")
    require(set(requested_variants).issubset(SUPPORTED_VARIANTS), "Unknown overlay condition requested")
    patch_units = (
        {"layer_{:02d}".format(index): (index,) for index in range(32)}
        if args.single_layer_sweep else WINDOWS
    )
    streams = parse_streams(args.streams)
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    if args.device == "cpu":
        require(args.dtype == "float32", "CPU requires float32")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    noop_tolerance = args.noop_tolerance
    if noop_tolerance is None:
        noop_tolerance = 1e-3 if dtype == torch.float16 else 1e-5

    provenance = validate_dataset_provenance(args.dataset_validation_file, args.dataset_revision)
    with Path(args.selection_file).open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    require(selection["status"] == "locked_before_inference", "Selection is not locked")
    require(selection["dataset_revision"] == args.dataset_revision, "Selection revision mismatch")
    require(selection["selection_used_model_outputs"] is False, "Selection used model outputs")
    if args.question_id is not None:
        matching = [
            entry for entry in selection["selected_samples"]
            if str(entry["question_id"]) == str(args.question_id)
        ]
        require(len(matching) == 1, "Requested question_id is not uniquely present in selection")
        selection = dict(selection)
        selection["selected_samples"] = matching
        selection["debug_question_id"] = str(args.question_id)
    if args.max_samples is not None:
        require(args.max_samples > 0, "max_samples must be positive")
        selection = dict(selection)
        selection["selected_samples"] = selection["selected_samples"][:args.max_samples]
        selection["debug_subset_of_locked_selection"] = True
    selected_entries = selection["selected_samples"]
    selected_qids = [str(entry["question_id"]) for entry in selected_entries]
    require(len(selected_qids) == len(set(selected_qids)), "Selection contains duplicate IDs")
    sample_count = len(selected_qids)
    expected_records = sum(
        sum(region in entry["region_token_counts"][variant] for region in requested_regions)
        * len(patch_units) * len(DIRECTIONS)
        for entry in selected_entries for variant in requested_variants
    )

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, args.split, args.dataset_revision)
    dataset_qids = [str(value) for value in ds["question_id"]]
    qid_to_index = {qid: index for index, qid in enumerate(dataset_qids)}
    for entry in selected_entries:
        qid = str(entry["question_id"])
        require(qid in qid_to_index, "Selected qid missing: {}".format(qid))
        require(qid_to_index[qid] == int(entry["dataset_index"]), "Dataset row changed for {}".format(qid))

    union_layers = sorted({index for values in patch_units.values() for index in values})
    log("CONFIG", "samples={} expected_interventions={} patch_units={}".format(sample_count, expected_records, patch_units))
    log("CONFIG", "model={} dataset={} revision={} hook=resid_pre".format(args.model_id, args.hf_dataset, args.dataset_revision))
    log("CONFIG", "device={} dtype={} streams={} overlap={} checkpoint_every_samples={}".format(
        device, dtype, sorted(streams), args.min_overlap_fraction, args.checkpoint_every_samples
    ))
    log("COVERAGE", json.dumps(selection["coverage"], sort_keys=True))
    log("PASS", "Dataset provenance validates {}/{} cleaned pairs".format(provenance["pairs_validated"], provenance["pairs_expected"]))

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
    require(max(union_layers) < len(layers), "Window layer outside model")
    model_revision = getattr(model.config, "_commit_hash", None)
    log("MODEL", "revision={} decoder_path={} layers={}".format(model_revision, layer_path, len(layers)))

    tokenizer = processor.tokenizer
    answer_ids = {}
    token_validation = {}
    for letter in ANSWER_LETTERS:
        token_id, encoded, decoded = answer_token_id(tokenizer, letter)
        answer_ids[letter] = token_id
        token_validation[letter] = {"token_id": token_id, "encoded": encoded, "decoded": decoded}
    require(len(set(answer_ids.values())) == 4, "Answer token IDs are not distinct")

    configuration = {
        "output_schema_version": 3 if "correct_answer" in requested_variants else 2,
        "model_id": args.model_id,
        "requested_model_revision": args.model_revision,
        "model_revision": model_revision,
        "dataset": args.hf_dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "overlay_image_field": "cleaned_image",
        "selected_question_ids": selected_qids,
        "sample_count": sample_count,
        "max_samples": args.max_samples,
        "windows": {name: list(values) for name, values in patch_units.items()},
        "sweep_mode": "single_layer" if args.single_layer_sweep else "fixed_windows",
        "variants": list(requested_variants),
        "directions": list(DIRECTIONS),
        "regions": list(requested_regions),
        "primary_regions": list(requested_regions),
        "object_control_policy": "Run when mapped; record text overlap; never exclude a primary sample.",
        "random_control_policy": "Three independent non-text token sets matched by packed-token kind; avoid object tokens when feasible.",
        "hook_location": "resid_pre",
        "streams": sorted(streams),
        "min_overlap_fraction": args.min_overlap_fraction,
        "seed": int(selection["seed"]),
        "shuffle_options": bool(args.shuffle_options),
        "dtype": str(dtype),
        "noop_tolerance": noop_tolerance,
        "checkpoint_every_samples": args.checkpoint_every_samples,
        "all_image_tokens_include_newlines": True,
        "decoding": "single next-token A-D constrained comparison; no generation",
        "correct_overlay_metric": "correct minus strongest incorrect no-text option, fixed per sample",
    }
    output_path = Path(args.out)
    records: List[Dict[str, Any]] = []
    samples_output: Dict[str, Any] = {}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        for key in ("output_schema_version", "model_id", "dataset_revision", "selected_question_ids", "windows", "variants", "directions", "regions", "streams", "seed", "shuffle_options", "dtype"):
            require(prior["configuration"].get(key) == configuration[key], "Checkpoint mismatch for {}".format(key))
        records = prior.get("records", [])
        samples_output = prior.get("samples", {})
        log("RESUME", "loaded={} path={}".format(len(records), output_path))
    completed = {record_key(record) for record in records}
    require(len(completed) == len(records), "Checkpoint contains duplicate records")
    persisted_counts = Counter(str(record["question_id"]) for record in records)
    expected_by_qid = {
        str(entry["question_id"]): sum(
            sum(region in entry["region_token_counts"][variant] for region in requested_regions)
            * len(patch_units) * len(DIRECTIONS)
            for variant in requested_variants
        )
        for entry in selected_entries
    }
    require(
        all(count == expected_by_qid[qid] for qid, count in persisted_counts.items()),
        "Checkpoint contains a partial sample",
    )
    samples_since_checkpoint = 0

    for sample_number, entry in enumerate(selected_entries, start=1):
        qid = str(entry["question_id"])
        expected_sample_keys = {
            (qid, variant, window, direction, region)
            for variant in requested_variants for window in patch_units
            for direction in DIRECTIONS
            for region in requested_regions
            if region in entry["region_token_counts"][variant]
        }
        if expected_sample_keys.issubset(completed):
            log("RESUME", "sample={}/{} qid={} complete".format(sample_number, sample_count, qid))
            continue
        require(not expected_sample_keys.intersection(completed), "Partial sample checkpoint for qid={}".format(qid))
        sample = ds[int(entry["dataset_index"])]
        options, correct_letter, option_meta = build_options(
            sample, args.shuffle_options, int(selection["seed"])
        )
        prompt = format_mcq_prompt(str(sample["question"]), options)
        sample_report = {
            "question": str(sample["question"]),
            "options": options,
            "option_meta": option_meta,
            "correct_letter": correct_letter,
            "answer_token_validation": token_validation,
            "conditions": {},
        }
        no_text_image = sample["notext"]["image"].convert("RGB")
        sample_records: List[Dict[str, Any]] = []

        for variant_index, variant in enumerate(requested_variants):
            target_letter = option_meta["key_to_label"][variant]
            correct_overlay = variant == "correct_answer"
            original_overlay = sample[variant]["image"].convert("RGB")
            cleaned_overlay = sample[variant]["cleaned_image"].convert("RGB")
            text_bbox = text_bbox_yxyx(sample, variant)
            pixel_stats = image_difference_stats(no_text_image, cleaned_overlay, text_bbox)
            clean_validation = validate_cleaned_overlay(no_text_image, original_overlay, cleaned_overlay, text_bbox)
            require(pixel_stats["outside_pixels_changed"] == 0 and pixel_stats["inside_pixels_changed"] > 0, "Cleaned pixel validation failed")
            require(clean_validation["cleaned_outside_no_text_mismatch_pixels"] == 0, "Outside mismatch")
            require(clean_validation["cleaned_inside_original_mismatch_pixels"] == 0, "Inside mismatch")

            no_text_inputs, no_text_prompt = prepare_inputs(processor, no_text_image, prompt, device, dtype)
            overlay_inputs, overlay_prompt = prepare_inputs(processor, cleaned_overlay, prompt, device, dtype)
            require(no_text_prompt == overlay_prompt, "Formatted prompts differ")
            require(torch.equal(no_text_inputs["input_ids"], overlay_inputs["input_ids"]), "Input IDs differ")
            no_text_positions = image_placeholder_positions(model, no_text_inputs["input_ids"])
            overlay_positions = image_placeholder_positions(model, overlay_inputs["input_ids"])
            require(no_text_positions == overlay_positions, "Image placeholder positions differ")
            image_hw = tuple(int(value) for value in no_text_inputs["image_sizes"][0].tolist())
            num_views = int(no_text_inputs["pixel_values"].shape[1]) if no_text_inputs["pixel_values"].ndim == 5 else 1
            mapping = build_packed_token_mapping(model, image_hw, num_views)
            require(len(no_text_positions) == len(mapping["tokens"]), "Mapping/placeholder count mismatch")
            require(mapping["summary"]["view_count_matches"], "View count mismatch")
            regions = build_regions_full_coverage(
                sample, variant, mapping["tokens"], streams,
                args.min_overlap_fraction, image_hw,
                int(selection["seed"]) * 100000 + int(entry["dataset_index"]) * 10 + variant_index,
            )
            expected_counts = entry["region_token_counts"][variant]
            require(set(regions) == set(expected_counts), "Region availability changed after selection")
            require(all(int(regions[name]["token_count"]) == int(expected_counts[name]) for name in regions), "Region mapping changed after selection")
            region_positions = {
                name: [no_text_positions[index] for index in regions[name]["token_indices"]]
                for name in regions
            }

            no_text_logits = forward_next_token_logits(model, no_text_inputs)
            overlay_logits = forward_next_token_logits(model, overlay_inputs)
            comparator_letter = (
                strongest_incorrect_letter(no_text_logits, answer_ids, correct_letter)
                if correct_overlay else target_letter
            )
            require(comparator_letter != correct_letter, "Metric comparator equals the correct answer")
            no_text_base = summarize_logits(no_text_logits, tokenizer, answer_ids, correct_letter, comparator_letter)
            overlay_base = summarize_logits(overlay_logits, tokenizer, answer_ids, correct_letter, comparator_letter)
            log("METRIC", "qid={} variant={} correct={} target={} comparator={} no_text_margin={:+.6f} overlay_margin={:+.6f}".format(
                qid, variant, correct_letter, target_letter, comparator_letter,
                no_text_base["margin_correct_minus_misleading"], overlay_base["margin_correct_minus_misleading"],
            ))
            no_text_noop_logits, no_text_hidden = forward_and_capture_many_resid_pre(model, layers, union_layers, no_text_inputs)
            overlay_noop_logits, overlay_hidden = forward_and_capture_many_resid_pre(model, layers, union_layers, overlay_inputs)
            no_text_noop_error = maximum_absolute_difference(no_text_logits, no_text_noop_logits)
            overlay_noop_error = maximum_absolute_difference(overlay_logits, overlay_noop_logits)
            require(no_text_noop_error <= noop_tolerance and overlay_noop_error <= noop_tolerance, "Multi-hook no-op validation failed")
            log("NOOP", "qid={} variant={} captured_layers={} no_text={:.8g} overlay={:.8g}".format(
                qid, variant, len(union_layers), no_text_noop_error, overlay_noop_error
            ))
            sample_report["conditions"][variant] = {
                "target_option_letter": target_letter,
                "target_option_text": options[target_letter],
                "comparator_option_letter": comparator_letter,
                "comparator_option_text": options[comparator_letter],
                "metric": "correct_minus_fixed_no_text_competitor" if correct_overlay else "correct_minus_condition_option",
                "pixel_difference": pixel_stats,
                "cleaned_overlay_validation": clean_validation,
                "alignment": {
                    "sequence_length": int(no_text_inputs["input_ids"].shape[1]),
                    "image_placeholder_count": len(no_text_positions),
                    "image_size_hw": list(image_hw),
                    "num_views": num_views,
                    "mapping_summary": mapping["summary"],
                },
                "regions": regions,
                "baselines": {"no_text": no_text_base, "overlay": overlay_base},
                "noop": {
                    "captured_layers": union_layers,
                    "no_text_max_logit_difference": no_text_noop_error,
                    "overlay_max_logit_difference": overlay_noop_error,
                },
            }

            for window_name, window_layers in patch_units.items():
                for region in requested_regions:
                    if region not in regions:
                        continue
                    positions = region_positions[region]
                    restoration_logits, restoration_integrity = forward_with_window_patch(
                        model, layers, window_layers, overlay_inputs, no_text_hidden, positions,
                        require_nonzero_difference=region in ("text_region", "all_image_tokens"),
                    )
                    insertion_logits, insertion_integrity = forward_with_window_patch(
                        model, layers, window_layers, no_text_inputs, overlay_hidden, positions,
                        require_nonzero_difference=region in ("text_region", "all_image_tokens"),
                    )
                    patched_by_direction = {
                        "restoration": summarize_logits(restoration_logits, tokenizer, answer_ids, correct_letter, comparator_letter),
                        "insertion": summarize_logits(insertion_logits, tokenizer, answer_ids, correct_letter, comparator_letter),
                    }
                    integrity_by_direction = {
                        "restoration": restoration_integrity,
                        "insertion": insertion_integrity,
                    }
                    for direction in DIRECTIONS:
                        recipient = overlay_base if direction == "restoration" else no_text_base
                        patched = patched_by_direction[direction]
                        effect = oriented_margin_effect(
                            direction, recipient["margin_correct_minus_misleading"],
                            patched["margin_correct_minus_misleading"], correct_overlay,
                        )
                        require(math.isfinite(effect), "Non-finite effect")
                        record = {
                            "question_id": qid,
                            "variant": variant,
                            "window": window_name,
                            "window_layers": list(window_layers),
                            "layer": int(window_layers[0]) if args.single_layer_sweep else None,
                            "direction": direction,
                            "target_option_key": variant,
                            "target_option_letter": target_letter,
                            "comparator_option_letter": comparator_letter,
                            "metric": "correct_minus_fixed_no_text_competitor" if correct_overlay else "correct_minus_condition_option",
                            "effect_orientation": (
                                "overlay_margin_minus_patched_margin" if direction == "restoration" else "patched_margin_minus_no_text_margin"
                            ) if correct_overlay else (
                                "patched_margin_minus_overlay_margin" if direction == "restoration" else "no_text_margin_minus_patched_margin"
                            ),
                            "region": region,
                            "token_count": len(positions),
                            "is_primary_region": region not in ("correct_object_region", "grounded_object_region"),
                            "clean_object_control": regions[region].get("clean_object_control"),
                            "text_overlap_token_count": regions[region].get("text_overlap_token_count"),
                            "effect": effect,
                            "recipient_margin": recipient["margin_correct_minus_misleading"],
                            "patched_margin": patched["margin_correct_minus_misleading"],
                            "recipient_prediction": recipient["choice_constrained_prediction"],
                            "patched_prediction": patched["choice_constrained_prediction"],
                            "recipient": recipient,
                            "patched": patched,
                            "outcome": correct_overlay_outcome(
                                recipient["choice_constrained_prediction"], patched["choice_constrained_prediction"], correct_letter,
                            ) if correct_overlay else classify_change(
                                recipient["choice_constrained_prediction"], patched["choice_constrained_prediction"],
                                correct_letter, target_letter,
                            ),
                            "target_flip": (
                                recipient["choice_constrained_prediction"] != target_letter
                                and patched["choice_constrained_prediction"] == target_letter
                            ),
                            "recovery_to_correct": (
                                recipient["choice_constrained_prediction"] != correct_letter
                                and patched["choice_constrained_prediction"] == correct_letter
                            ),
                            "loss_of_correct": (
                                recipient["choice_constrained_prediction"] == correct_letter
                                and patched["choice_constrained_prediction"] != correct_letter
                            ),
                            "integrity": integrity_by_direction[direction],
                        }
                        sample_records.append(record)
                        log("PATCH", "sample={}/{} qid={} variant={} window={} direction={} region={} n={} effect={:+.6f} pred={}->{} outcome={}".format(
                            sample_number, sample_count, qid, variant, window_name,
                            direction, region, len(positions), effect,
                            record["recipient_prediction"], record["patched_prediction"], record["outcome"],
                        ))
            del no_text_hidden, overlay_hidden

        require(len(sample_records) == expected_by_qid[qid], "Sample record count mismatch")
        records.extend(sample_records)
        completed.update(record_key(record) for record in sample_records)
        samples_output[qid] = sample_report
        samples_since_checkpoint += 1
        if samples_since_checkpoint >= args.checkpoint_every_samples or len(records) == expected_records:
            save_checkpoint(output_path, configuration, selection, samples_output, records, expected_records, "in_progress")
            samples_since_checkpoint = 0
            log("CHECKPOINT", "samples={}/{} records={}/{} path={}".format(
                sample_number, sample_count, len(records), expected_records, output_path
            ))
        else:
            log("PROGRESS", "samples={}/{} records_in_memory={}/{} next_checkpoint_in={}".format(
                sample_number, sample_count, len(records), expected_records,
                args.checkpoint_every_samples - samples_since_checkpoint,
            ))

    require(len(records) == expected_records, "Saved {}, expected {}".format(len(records), expected_records))
    require(len({record_key(record) for record in records}) == expected_records, "Final records are not unique")
    save_checkpoint(output_path, configuration, selection, samples_output, records, expected_records, "success")
    log("SUMMARY", "selected_samples={} completed_samples={} records={}/{} failures=0".format(
        sample_count, sample_count, len(records), expected_records
    ))
    log("SAVE", str(output_path))
    log("COMPLETE", "All {} shared-subset window interventions passed validation".format(expected_records))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAIL", "{}: {}".format(type(exc).__name__, exc))
        raise
