#!/usr/bin/env python3
"""Milestone 4: validate attention routes across fixed windows on one sample."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from attention_edge_mask import add_directed_edge_block
from inspect_attention_path_setup import (
    align_raw_to_expanded,
    char_span,
    decoded_group,
    positions_for_char_span,
)
from test_one_sample_attention_path import (
    ACTIVATION_ROOT,
    DATASET_REVISION,
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_VALIDATION,
    MODEL,
    MODEL_REVISION,
    log,
    selected_qids,
    write_json,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
EXPERIMENT_ROOT = HERE.parent
ACTIVATION_MAIN = ACTIVATION_ROOT / "main_files"
sys.path.insert(0, str(ACTIVATION_MAIN))

from activation_patch_control_pilot import build_regions_full_coverage, kind_counts  # noqa: E402
from activation_patch_llava_next_debug import (  # noqa: E402
    ANSWER_LETTERS,
    answer_token_id,
    build_options,
    build_packed_token_mapping,
    find_sample_by_qid,
    format_mcq_prompt,
    forward_next_token_logits,
    get_decoder_layers,
    get_or_download_hf_dataset,
    image_placeholder_positions,
    prepare_inputs,
    require,
    summarize_logits,
    text_bbox_yxyx,
    validate_cleaned_overlay,
    validate_dataset_provenance,
)


WINDOWS = {"early": list(range(0, 6)), "transfer": list(range(8, 14)), "late": list(range(17, 23))}
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "debug_outputs" / "milestone_4_one_sample_windows.json"


def token_destinations(processor: Any, formatted: str, expanded_ids: Sequence[int], image_token_id: int) -> Dict[str, List[int]]:
    raw = processor.tokenizer(formatted, add_special_tokens=True, return_offsets_mapping=True)
    raw_ids = [int(value) for value in raw["input_ids"]]
    offsets = [tuple(map(int, value)) for value in raw["offset_mapping"]]
    raw_to_expanded, aligned_images = align_raw_to_expanded(raw_ids, expanded_ids, image_token_id)
    question_span = char_span(formatted, "Question: ", "\n\nOptions:")
    option_span = char_span(formatted, "Options:\n", "\n\nAnswer with only the letter")
    groups = {
        "Q": positions_for_char_span(offsets, raw_to_expanded, question_span),
        "P": positions_for_char_span(offsets, raw_to_expanded, option_span),
        "D": [len(expanded_ids) - 1],
    }
    require(groups["Q"] and groups["P"], "Q or P token group is empty")
    require(set(groups["Q"]).isdisjoint(groups["P"]), "Q and P token groups overlap")
    require(max(aligned_images) < min(groups["Q"]) < min(groups["P"]) < groups["D"][0], "Token groups violate causal order")
    return groups


@torch.no_grad()
def forward_with_window_block(
    model: LlavaNextForConditionalGeneration,
    layers: Sequence[torch.nn.Module],
    inputs: Dict[str, torch.Tensor],
    source_positions: Optional[Sequence[int]],
    destination_positions: Sequence[int],
    forward_fn=forward_next_token_logits,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    sources = [] if source_positions is None else sorted(set(int(value) for value in source_positions))
    destinations = sorted(set(int(value) for value in destination_positions))
    require(destinations, "Destination group is empty")
    states: Dict[int, Dict[str, Any]] = {}
    handles = []

    for layer in layers:
        attention = layer.self_attn
        layer_index = int(attention.layer_idx)
        state: Dict[str, Any] = {"layer": layer_index, "pre_calls": 0, "post_calls": 0}
        states[layer_index] = state

        def pre_hook(_module, args, kwargs, state=state):
            mask = kwargs.get("attention_mask")
            require(torch.is_tensor(mask) and mask.ndim == 4, "Layer {} lacks a 4D mask".format(state["layer"]))
            require(max(destinations) < mask.shape[-2], "Destination outside query axis")
            combined = mask.clone()
            if sources:
                require(min(sources) >= 0 and max(sources) < mask.shape[-1], "Source outside key axis")
                for destination in destinations:
                    require(bool(torch.all(mask[..., destination, sources] == 0)), "An edge was already masked")
                combined = add_directed_edge_block(mask, sources, destinations)
            require(bool(torch.all(combined[mask != 0] == mask[mask != 0])), "Existing mask changed")
            for destination in destinations:
                require(bool(torch.any(combined[..., destination, :] == 0)), "A destination row became fully masked")
            kwargs["attention_mask"] = combined
            state.update({
                "pre_calls": state["pre_calls"] + 1,
                "mask_shape": list(mask.shape),
                "existing_mask_preserved": True,
                "destination_rows_have_valid_keys": True,
            })
            return args, kwargs

        def post_hook(_module, _args, output, state=state):
            weights = output[1]
            require(torch.is_tensor(weights) and weights.ndim == 4, "Eager attention returned no probabilities")
            destination_index = torch.as_tensor(destinations, dtype=torch.long, device=weights.device)
            rows = weights[0, :, destination_index, :]
            state["post_calls"] += 1
            state["attention_shape"] = list(weights.shape)
            state["destination_row_sum_max_error"] = float((rows.sum(-1) - 1.0).abs().max().item())
            if sources:
                source_index = torch.as_tensor(sources, dtype=torch.long, device=weights.device)
                state["blocked_probability_max"] = float(rows[..., source_index].abs().max().item())

        handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(attention.register_forward_hook(post_hook))

    try:
        logits = forward_fn(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    integrity = [states[index] for index in sorted(states)]
    for state in integrity:
        require(state["pre_calls"] == 1 and state["post_calls"] == 1, "Layer hook count mismatch")
        require(state["destination_row_sum_max_error"] <= 2e-3, "Attention row failed to normalize")
        if sources:
            require(state["blocked_probability_max"] == 0.0, "Blocked probability is nonzero")
    return logits, integrity


def transition(before: str, after: str, correct: str, displayed: str) -> str:
    if before == after:
        return "no_prediction_change"
    if before == displayed and after == correct:
        return "displayed_to_correct"
    if before == correct and after == displayed:
        return "correct_to_displayed"
    return "other_option_flip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--variant", default="misleading_groundable", choices=("misleading_groundable", "misleading_ungroundable", "irrelevant_word"))
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--model_id", default=MODEL)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--hf_cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset_validation_file", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--selection_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require(torch.cuda.is_available(), "CUDA is required")
    require(args.model_revision == MODEL_REVISION and args.dataset_revision == DATASET_REVISION, "Pinned revision mismatch")
    require(args.question_id in selected_qids(args.selection_manifest), "qid is outside the shared 305")
    provenance = validate_dataset_provenance(str(args.dataset_validation_file), args.dataset_revision)
    log("CONFIG", "continuation=activation_patching qid={} variant={} windows={}".format(args.question_id, args.variant, WINDOWS))
    log("CONFIG", "conditions=overlay,no_text paths=T->Q,T->P,T->D,C->D,G->D,R1->D,T+C->D")
    log("EXPECTED", "42 interventions; all blocked probabilities=0; DID and T+C interaction recompute from raw margins")

    dataset = get_or_download_hf_dataset("anonymous/GUIC", str(args.hf_cache_dir), "test", args.dataset_revision)
    sample = find_sample_by_qid(dataset, args.question_id)
    no_text_image = sample["notext"]["image"].convert("RGB")
    overlay_image = sample[args.variant]["cleaned_image"].convert("RGB")
    cleaned = validate_cleaned_overlay(
        no_text_image, sample[args.variant]["image"].convert("RGB"), overlay_image,
        text_bbox_yxyx(sample, args.variant),
    )
    require(cleaned["cleaned_outside_no_text_mismatch_pixels"] == 0 and cleaned["cleaned_inside_original_mismatch_pixels"] == 0, "Cleaned overlay validation failed")

    log("MODEL", "loading pinned FP16 model with eager attention")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to("cuda").eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    processor.patch_size = int(model.config.vision_config.patch_size)
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    layers, layer_path = get_decoder_layers(model)
    require(len(layers) == 32 and model.config.text_config._attn_implementation == "eager", "Model architecture/backend mismatch")

    options, correct_letter, option_meta = build_options(sample, shuffle=True, seed=args.seed)
    displayed_letter = option_meta["key_to_label"][args.variant]
    answer_ids = {letter: answer_token_id(processor.tokenizer, letter)[0] for letter in ANSWER_LETTERS}
    prompt = format_mcq_prompt(str(sample["question"]), options)
    condition_inputs = {}
    formatted_reference = None
    for condition, image in (("overlay", overlay_image), ("no_text", no_text_image)):
        inputs, formatted = prepare_inputs(processor, image, prompt, torch.device("cuda"), torch.float16)
        condition_inputs[condition] = inputs
        if formatted_reference is None:
            formatted_reference = formatted
        require(formatted == formatted_reference, "Formatted prompts differ")
    require(torch.equal(condition_inputs["overlay"]["input_ids"], condition_inputs["no_text"]["input_ids"]), "Paired input IDs differ")

    inputs = condition_inputs["overlay"]
    expanded_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
    image_positions = image_placeholder_positions(model, inputs["input_ids"])
    destinations = token_destinations(processor, formatted_reference, expanded_ids, int(model.config.image_token_index))
    image_size = tuple(int(value) for value in inputs["image_sizes"][0].tolist())
    views = int(inputs["pixel_values"].shape[1])
    mapping = build_packed_token_mapping(model, image_size, views)
    require(len(image_positions) == mapping["summary"]["total_packed_image_tokens"], "Packed mapping mismatch")
    regions = build_regions_full_coverage(sample, args.variant, mapping["tokens"], {"base", "mosaic"}, 0.25, image_size, args.seed)
    require(regions.get("correct_object_region", {}).get("clean_object_control") is True, "C is not a clean object control")
    require(regions.get("grounded_object_region", {}).get("clean_object_control") is True, "G is not a clean object control")

    def sequence_positions(region_name: str) -> List[int]:
        packed = regions[region_name]["token_indices"]
        return [int(image_positions[index]) for index in packed]

    sources = {
        "T": sequence_positions("text_region"),
        "C": sequence_positions("correct_object_region"),
        "G": sequence_positions("grounded_object_region"),
        "R1": sequence_positions("matched_random_region_1"),
    }
    text_kinds = kind_counts(regions["text_region"]["token_indices"], mapping["tokens"])
    random_kinds = kind_counts(regions["matched_random_region_1"]["token_indices"], mapping["tokens"])
    require(len(sources["T"]) == len(sources["R1"]) and text_kinds == random_kinds, "T/R1 controls are unmatched")
    require(set(sources["T"]).isdisjoint(sources["C"]) and set(sources["T"]).isdisjoint(sources["G"]), "Object control overlaps T")
    source_groups = {**sources, "T+C": sorted(set(sources["T"]) | set(sources["C"]))}
    for source in source_groups.values():
        require(max(source) < min(destinations["Q"]), "Visual source is not before Q/P/D")
    log("TOKENS", "sequence={} image={} Q={} P={} D={} T={} C={} G={} R1={}".format(
        len(expanded_ids), len(image_positions), len(destinations["Q"]), len(destinations["P"]),
        destinations["D"][0], *(len(sources[name]) for name in ("T", "C", "G", "R1"))
    ))
    log("TOKENS", "Q={!r} P={!r} D={!r}".format(
        decoded_group(processor.tokenizer, expanded_ids, destinations["Q"]),
        decoded_group(processor.tokenizer, expanded_ids, destinations["P"]),
        decoded_group(processor.tokenizer, expanded_ids, destinations["D"]),
    ))

    paths = [
        ("T_to_Q", "T", "Q"), ("T_to_P", "T", "P"), ("T_to_D", "T", "D"),
        ("C_to_D", "C", "D"), ("G_to_D", "G", "D"), ("R1_to_D", "R1", "D"),
        ("T_plus_C_to_D", "T+C", "D"),
    ]
    all_window_layers = sorted({index for window in WINDOWS.values() for index in window})
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "continuation_of": "activation_patching",
        "configuration": {
            "model_id": args.model_id, "model_revision": args.model_revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "dataset": "anonymous/GUIC", "dataset_revision": args.dataset_revision,
            "question_id": str(sample["question_id"]), "variant": args.variant,
            "backend": "eager", "dtype": "float16", "seed": args.seed,
            "windows": WINDOWS, "decoder_path": layer_path,
        },
        "tokens": {
            "correct_letter": correct_letter, "displayed_letter": displayed_letter,
            "answer_token_ids": answer_ids, "destinations": destinations,
            "destination_decoded": {name: decoded_group(processor.tokenizer, expanded_ids, positions) for name, positions in destinations.items()},
            "sources": source_groups, "T_kind_counts": text_kinds, "R1_kind_counts": random_kinds,
        },
        "validation": {"dataset_provenance": provenance, "cleaned_overlay": cleaned},
        "baselines": {}, "records": [], "derived": {},
    }

    for condition, condition_batch in condition_inputs.items():
        baseline_logits = forward_next_token_logits(model, condition_batch)
        noop_logits, noop_integrity = forward_with_window_block(
            model, [layers[index] for index in all_window_layers], condition_batch, None, destinations["D"]
        )
        noop_difference = float((baseline_logits - noop_logits).abs().max().item())
        require(noop_difference <= 1e-3, "{} no-op exceeded tolerance".format(condition))
        baseline = summarize_logits(baseline_logits, processor.tokenizer, answer_ids, correct_letter, displayed_letter)
        noop = summarize_logits(noop_logits, processor.tokenizer, answer_ids, correct_letter, displayed_letter)
        require(baseline["choice_constrained_prediction"] == noop["choice_constrained_prediction"], "No-op changed prediction")
        report["baselines"][condition] = {
            "baseline": baseline, "empty_mask_noop": noop,
            "noop_max_full_vocab_logit_difference": noop_difference,
            "noop_integrity": noop_integrity,
        }
        log("NOOP", "condition={} max_logit_difference={:.6g} baseline_margin={:.6f}".format(
            condition, noop_difference, baseline["margin_correct_minus_misleading"]
        ))

        for window_name, layer_indices in WINDOWS.items():
            for path_name, source_name, destination_name in paths:
                logits, integrity = forward_with_window_block(
                    model, [layers[index] for index in layer_indices], condition_batch,
                    source_groups[source_name], destinations[destination_name],
                )
                summary = summarize_logits(logits, processor.tokenizer, answer_ids, correct_letter, displayed_letter)
                effect = summary["margin_correct_minus_misleading"] - baseline["margin_correct_minus_misleading"]
                require(math.isfinite(effect), "Non-finite intervention effect")
                record = {
                    "condition": condition, "window": window_name, "layers": layer_indices,
                    "path": path_name, "source_group": source_name, "destination_group": destination_name,
                    "source_count": len(source_groups[source_name]), "destination_count": len(destinations[destination_name]),
                    "directed_edges_per_head": len(source_groups[source_name]) * len(destinations[destination_name]) * len(layer_indices),
                    "summary": summary, "margin_change": effect,
                    "transition": transition(baseline["choice_constrained_prediction"], summary["choice_constrained_prediction"], correct_letter, displayed_letter),
                    "integrity": integrity,
                }
                report["records"].append(record)
                write_json(args.output, report)
                log("PATH", "condition={} window={} path={} effect={:+.6f} pred={}->{}".format(
                    condition, window_name, path_name, effect,
                    baseline["choice_constrained_prediction"], summary["choice_constrained_prediction"],
                ))

    lookup = {(row["condition"], row["window"], row["path"]): row for row in report["records"]}
    require(len(lookup) == 42, "Expected 42 unique intervention records")
    did = {}
    interactions = {}
    for window_name in WINDOWS:
        for path_name in ("T_to_Q", "T_to_P", "T_to_D"):
            did["{}_{}".format(window_name, path_name)] = (
                lookup[("overlay", window_name, path_name)]["margin_change"]
                - lookup[("no_text", window_name, path_name)]["margin_change"]
            )
        for condition in ("overlay", "no_text"):
            baseline_margin = report["baselines"][condition]["baseline"]["margin_correct_minus_misleading"]
            joint = lookup[(condition, window_name, "T_plus_C_to_D")]["summary"]["margin_correct_minus_misleading"]
            text = lookup[(condition, window_name, "T_to_D")]["summary"]["margin_correct_minus_misleading"]
            obj = lookup[(condition, window_name, "C_to_D")]["summary"]["margin_correct_minus_misleading"]
            interactions["{}_{}".format(condition, window_name)] = joint - text - obj + baseline_margin
    report["derived"] = {"text_path_difference_in_differences": did, "T_C_factorial_interactions": interactions}
    report["status"] = "complete"
    report["all_validation_passed"] = True
    write_json(args.output, report)
    log("DERIVED", "DID={}".format({key: round(value, 6) for key, value in did.items()}))
    log("DERIVED", "T+C interactions={}".format({key: round(value, 6) for key, value in interactions.items()}))
    log("OUTPUT", str(args.output))
    log("COMPLETE", "Milestone 4 one-sample window and route validation passed; records=42")


if __name__ == "__main__":
    main()
