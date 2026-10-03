#!/usr/bin/env python3
"""Milestone 6: full shared-305 attention-path intervention shard."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from attention_edge_mask import add_directed_edge_block
from test_one_sample_attention_path import (
    DATASET_REVISION, DEFAULT_CACHE, DEFAULT_MANIFEST, DEFAULT_VALIDATION,
    MODEL, MODEL_REVISION, log, write_json,
)
from test_one_sample_attention_windows import forward_with_window_block, token_destinations, transition


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ACTIVATION_MAIN = PROJECT_ROOT / "vlms" / "activation_patching" / "main_files"
sys.path.insert(0, str(ACTIVATION_MAIN))

from activation_patch_control_pilot import build_regions_full_coverage, kind_counts  # noqa: E402
from activation_patch_llava_next_debug import (  # noqa: E402
    ANSWER_LETTERS, answer_token_id, build_options, build_packed_token_mapping,
    find_sample_by_qid, format_mcq_prompt, forward_next_token_logits,
    get_decoder_layers, get_or_download_hf_dataset, image_placeholder_positions,
    prepare_inputs, require, summarize_logits, text_bbox_yxyx,
    validate_cleaned_overlay, validate_dataset_provenance,
)


VALID_VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")
WINDOWS = {
    "early": list(range(0, 6)),
    "transfer": list(range(8, 14)),
    "late": list(range(17, 23)),
}
BASE_PATH_INSTANCES = 14
OBJECT_PATH_INSTANCES = 8


def strongest_incorrect_letter(logits: torch.Tensor, answer_ids: Dict[str, int], correct_letter: str) -> str:
    """Freeze the strongest no-text incorrect option for the correct-overlay margin."""
    candidates = [letter for letter in ANSWER_LETTERS if letter != correct_letter]
    return max(candidates, key=lambda letter: float(logits[answer_ids[letter]].item()))


def correct_overlay_transition(before: str, after: str, correct: str) -> str:
    if before != correct and after == correct:
        return "correct_gain"
    if before == correct and after != correct:
        return "correct_loss"
    return "other_option_flip" if before != after else "no_prediction_change"


def load_manifest(path: Path, dataset_revision: str) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    qids = [str(row["question_id"]) for row in manifest["selected_samples"]]
    require(manifest["dataset_revision"] == dataset_revision, "Manifest revision mismatch")
    require(len(qids) == 305 and len(set(qids)) == 305, "Manifest must contain 305 unique qids")
    return manifest


def clean_control(entry: Dict[str, Any], variant: str, region: str) -> bool:
    return entry["region_diagnostics"][variant].get(region, {}).get("clean_object_control") is True


def shard_entries(manifest: Dict[str, Any], shard_id: int, num_shards: int) -> List[Dict[str, Any]]:
    require(num_shards > 0 and 0 <= shard_id < num_shards, "Invalid shard")
    return manifest["selected_samples"][shard_id::num_shards]


def expected_slots(entries: Sequence[Dict[str, Any]], variant: str) -> int:
    total = 0
    for entry in entries:
        paths = BASE_PATH_INSTANCES
        paths += OBJECT_PATH_INSTANCES * clean_control(entry, variant, "correct_object_region")
        paths += OBJECT_PATH_INSTANCES * clean_control(entry, variant, "grounded_object_region")
        total += 2 * len(WINDOWS) * paths
    return total


def backend_forward(model: Any, inputs: Dict[str, torch.Tensor], backend: str) -> torch.Tensor:
    model.set_attn_implementation(backend)
    require(model.config.text_config._attn_implementation == backend, "Failed to select " + backend)
    return forward_next_token_logits(model, inputs)


def accessible_edge_count(sources: Sequence[int], destinations: Sequence[int]) -> int:
    return sum(source <= destination for destination in destinations for source in sources)


@torch.no_grad()
def forward_with_causal_window_block(
    model: LlavaNextForConditionalGeneration,
    layers: Sequence[torch.nn.Module],
    inputs: Dict[str, torch.Tensor],
    source_positions: Sequence[int],
    destination_positions: Sequence[int],
    forward_fn=forward_next_token_logits,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Block all causally available source-to-destination edges in a window."""
    sources = sorted(set(int(value) for value in source_positions))
    destinations = sorted(set(int(value) for value in destination_positions))
    require(sources and destinations, "Source and destination groups must be nonempty")
    require(accessible_edge_count(sources, destinations) > 0, "Path has no causally available edge")
    states: Dict[int, Dict[str, Any]] = {}
    handles = []

    for layer in layers:
        attention = layer.self_attn
        layer_index = int(attention.layer_idx)
        state: Dict[str, Any] = {"layer": layer_index, "pre_calls": 0, "post_calls": 0}
        states[layer_index] = state

        def pre_hook(_module, args, kwargs, state=state):
            mask = kwargs.get("attention_mask")
            require(torch.is_tensor(mask) and mask.ndim == 4, "Attention layer lacks a 4D mask")
            require(max(sources) < mask.shape[-1] and max(destinations) < mask.shape[-2], "Path index outside mask")
            selected = mask[0, 0][torch.as_tensor(destinations, device=mask.device)[:, None],
                                   torch.as_tensor(sources, device=mask.device)[None, :]]
            available = int((selected == 0).sum().item())
            require(available > 0, "Runtime mask exposes no selected edge")
            combined = add_directed_edge_block(mask, sources, destinations)
            require(bool(torch.all(combined[mask != 0] == mask[mask != 0])), "Existing masks changed")
            for destination in destinations:
                require(bool(torch.any(combined[..., destination, :] == 0)), "A query row became fully masked")
            kwargs["attention_mask"] = combined
            state.update({
                "pre_calls": state["pre_calls"] + 1,
                "mask_shape": list(mask.shape),
                "requested_edges_per_head": len(sources) * len(destinations),
                "blocked_available_edges_per_head": available,
                "existing_masks_preserved": True,
            })
            return args, kwargs

        def post_hook(_module, _args, output, state=state):
            weights = output[1]
            require(torch.is_tensor(weights) and weights.ndim == 4, "Eager attention returned no probabilities")
            query_ids = torch.as_tensor(destinations, dtype=torch.long, device=weights.device)
            source_ids = torch.as_tensor(sources, dtype=torch.long, device=weights.device)
            rows = weights[0, :, query_ids, :]
            state["post_calls"] += 1
            state["destination_row_sum_max_error"] = float((rows.sum(-1) - 1.0).abs().max().item())
            state["blocked_probability_max"] = float(rows[..., source_ids].abs().max().item())

        handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(attention.register_forward_hook(post_hook))

    try:
        logits = forward_fn(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    integrity = [states[index] for index in sorted(states)]
    for state in integrity:
        require(state["pre_calls"] == 1 and state["post_calls"] == 1, "Hook count mismatch")
        require(state["blocked_probability_max"] == 0.0, "Blocked probability is nonzero")
        require(state["destination_row_sum_max_error"] <= 2e-3, "Attention row failed to normalize")
    return logits, integrity


def path_specs(groups: Dict[str, List[int]], c_clean: bool, g_clean: bool) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    def add(name: str, source: str, destination: str, conceptual: Optional[str] = None) -> None:
        specs.append({
            "path_instance": name,
            "conceptual_path": conceptual or name,
            "source_group": source,
            "destination_group": destination,
            "sources": groups[source],
            "destinations": groups[destination],
        })

    for destination in ("Q", "P", "D"):
        add("T_to_" + destination, "T", destination)
        for index in (1, 2, 3):
            add("R{}_to_{}".format(index, destination), "R" + str(index), destination, "R_to_" + destination)
    add("Q_to_D", "Q", "D")
    add("P_to_D", "P", "D")

    for object_name, is_clean in (("C", c_clean), ("G", g_clean)):
        if not is_clean:
            continue
        for destination in ("Q", "P", "D"):
            add("{}_to_{}".format(object_name, destination), object_name, destination)
        add("T_to_" + object_name, "T", object_name)
        add(object_name + "_to_T", object_name, "T")
        joint = "T+" + object_name
        groups[joint] = sorted(set(groups["T"]) | set(groups[object_name]))
        for destination in ("Q", "P", "D"):
            add("{}_to_{}".format(joint, destination), joint, destination)
    return specs


def configuration(args: argparse.Namespace, entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": 6,
        "continuation_of": "activation_patching",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset": "anonymous/GUIC",
        "dataset_revision": args.dataset_revision,
        "variant": args.variant,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "question_ids": [str(row["question_id"]) for row in entries],
        "sample_count": len(entries),
        "windows": WINDOWS,
        "image_states": ["overlay", "no_text"],
        "conceptual_path_count_maximum": 24,
        "path_instance_count_maximum": 30,
        "random_controls_per_destination": 3,
        "object_paths_require_clean_nonoverlapping_control": True,
        "expected_record_slots": expected_slots(entries, args.variant),
        "seed": args.seed,
        "dtype": "float16",
        "intervention_backend": "eager",
        "comparison_backend": "sdpa",
        "margin_metric": (
            "correct_minus_fixed_strongest_no_text_incorrect"
            if args.variant == "correct_answer" else "correct_minus_condition_option"
        ),
    }


def validate_or_write_config(path: Path, config: Dict[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        require(saved == config, "Existing shard configuration differs")
    else:
        write_json(path, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    parser.add_argument("--shard_id", required=True, type=int)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--selection", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--model_id", default=MODEL)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--hf_cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset_validation_file", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug-only prefix length; zero uses the full shard")
    args = parser.parse_args()

    require(args.model_revision == MODEL_REVISION and args.dataset_revision == DATASET_REVISION, "Pinned revision mismatch")
    manifest = load_manifest(args.selection, args.dataset_revision)
    entries = shard_entries(manifest, args.shard_id, args.num_shards)
    require(args.max_samples >= 0, "max_samples must be nonnegative")
    if args.max_samples:
        entries = entries[:args.max_samples]
    config = configuration(args, entries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    validate_or_write_config(args.output_dir / "configuration.json", config)
    c_count = sum(clean_control(row, args.variant, "correct_object_region") for row in entries)
    g_count = sum(clean_control(row, args.variant, "grounded_object_region") for row in entries)
    log("CONFIG", "variant={} shard={}/{} samples={} expected_slots={}".format(
        args.variant, args.shard_id, args.num_shards, len(entries), config["expected_record_slots"]))
    log("CONFIG", "windows={} clean_C={} clean_G={} maximum_paths=24 conceptual/30 instances".format(WINDOWS, c_count, g_count))
    log("EXPECTED", "each complete sample is an atomic checkpoint; restart skips validated complete sample files")
    if args.preflight_only:
        log("COMPLETE", "Preflight passed without loading model")
        return

    require(torch.cuda.is_available(), "CUDA is required")
    provenance = validate_dataset_provenance(str(args.dataset_validation_file), args.dataset_revision)
    dataset = get_or_download_hf_dataset("anonymous/GUIC", str(args.hf_cache_dir), "test", args.dataset_revision)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager",
    ).to("cuda").eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    processor.patch_size = int(model.config.vision_config.patch_size)
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    layers, layer_path = get_decoder_layers(model)
    require(len(layers) == 32, "Expected 32 decoder layers")

    saved_slots = 0
    unavailable_slots = 0
    completed_samples = 0
    resumed_samples = 0
    max_blocked_probability = 0.0
    max_row_sum_error = 0.0
    max_noop_difference = 0.0
    max_backend_difference = 0.0
    backend_prediction_disagreements = 0

    for sample_number, entry in enumerate(entries, start=1):
        qid = str(entry["question_id"])
        sample_path = samples_dir / (qid + ".json")
        if sample_path.exists():
            saved = json.loads(sample_path.read_text(encoding="utf-8"))
            require(saved.get("status") == "complete" and saved.get("question_id") == qid, "Invalid sample checkpoint")
            saved_slots += len(saved["records"])
            unavailable_slots += sum(row["status"] == "structurally_unavailable" for row in saved["records"])
            for baseline in saved["baselines"].values():
                max_noop_difference = max(max_noop_difference, baseline["noop_max_full_vocab_logit_difference"])
                max_backend_difference = max(max_backend_difference, baseline["sdpa_eager_max_full_vocab_logit_difference"])
                backend_prediction_disagreements += int(not baseline["sdpa_eager_choice_prediction_equal"])
            for row in saved["records"]:
                if row["status"] != "complete":
                    continue
                max_blocked_probability = max(
                    max_blocked_probability, max(item["blocked_probability_max"] for item in row["integrity"])
                )
                max_row_sum_error = max(
                    max_row_sum_error, max(item["destination_row_sum_max_error"] for item in row["integrity"])
                )
            completed_samples += 1
            resumed_samples += 1
            log("RESUME", "{}/{} qid={} slots={}".format(sample_number, len(entries), qid, len(saved["records"])))
            continue

        sample = find_sample_by_qid(dataset, qid)
        no_text_image = sample["notext"]["image"].convert("RGB")
        overlay_image = sample[args.variant]["cleaned_image"].convert("RGB")
        cleaned = validate_cleaned_overlay(
            no_text_image, sample[args.variant]["image"].convert("RGB"), overlay_image,
            text_bbox_yxyx(sample, args.variant),
        )
        require(cleaned["cleaned_outside_no_text_mismatch_pixels"] == 0, "Outside pixels differ")
        require(cleaned["cleaned_inside_original_mismatch_pixels"] == 0, "Inside pixels differ")
        options, correct_letter, option_meta = build_options(sample, shuffle=True, seed=args.seed)
        target_letter = option_meta["key_to_label"][args.variant]
        answer_ids = {letter: answer_token_id(processor.tokenizer, letter)[0] for letter in ANSWER_LETTERS}
        prompt = format_mcq_prompt(str(sample["question"]), options)
        batches: Dict[str, Dict[str, torch.Tensor]] = {}
        formatted_reference = None
        for image_state, image in (("overlay", overlay_image), ("no_text", no_text_image)):
            batch, formatted = prepare_inputs(processor, image, prompt, torch.device("cuda"), torch.float16)
            batches[image_state] = batch
            if formatted_reference is None:
                formatted_reference = formatted
            require(formatted == formatted_reference, "Paired prompts differ")
        require(torch.equal(batches["overlay"]["input_ids"], batches["no_text"]["input_ids"]), "Paired token IDs differ")

        batch = batches["overlay"]
        expanded_ids = [int(value) for value in batch["input_ids"][0].tolist()]
        image_positions = image_placeholder_positions(model, batch["input_ids"])
        destinations = token_destinations(processor, formatted_reference, expanded_ids, int(model.config.image_token_index))
        image_size = tuple(int(value) for value in batch["image_sizes"][0].tolist())
        mapping = build_packed_token_mapping(model, image_size, int(batch["pixel_values"].shape[1]))
        require(len(image_positions) == mapping["summary"]["total_packed_image_tokens"], "Packed mapping mismatch")
        regions = build_regions_full_coverage(sample, args.variant, mapping["tokens"], {"base", "mosaic"}, 0.25, image_size, args.seed)

        def positions(region: str) -> List[int]:
            return [int(image_positions[index]) for index in regions[region]["token_indices"]]

        groups = {
            "T": positions("text_region"),
            "R1": positions("matched_random_region_1"),
            "R2": positions("matched_random_region_2"),
            "R3": positions("matched_random_region_3"),
            "Q": destinations["Q"], "P": destinations["P"], "D": destinations["D"],
        }
        runtime_c_clean = regions.get("correct_object_region", {}).get("clean_object_control") is True
        runtime_g_clean = regions.get("grounded_object_region", {}).get("clean_object_control") is True
        require(runtime_c_clean == clean_control(entry, args.variant, "correct_object_region"), "C availability changed from manifest")
        require(runtime_g_clean == clean_control(entry, args.variant, "grounded_object_region"), "G availability changed from manifest")
        if runtime_c_clean:
            groups["C"] = positions("correct_object_region")
        if runtime_g_clean:
            groups["G"] = positions("grounded_object_region")
        text_kinds = kind_counts(regions["text_region"]["token_indices"], mapping["tokens"])
        for index in (1, 2, 3):
            random_region = "matched_random_region_" + str(index)
            require(len(groups["R" + str(index)]) == len(groups["T"]), "Random count mismatch")
            require(kind_counts(regions[random_region]["token_indices"], mapping["tokens"]) == text_kinds, "Random composition mismatch")
        specs = path_specs(groups, runtime_c_clean, runtime_g_clean)
        expected_sample_slots = 2 * len(WINDOWS) * len(specs)
        require(expected_sample_slots == 2 * len(WINDOWS) * (
            BASE_PATH_INSTANCES + OBJECT_PATH_INSTANCES * runtime_c_clean + OBJECT_PATH_INSTANCES * runtime_g_clean
        ), "Sample slot count mismatch")
        log("SAMPLE", "{}/{} qid={} seq={} T={} C_clean={} G_clean={} paths={} slots={}".format(
            sample_number, len(entries), qid, len(expanded_ids), len(groups["T"]), runtime_c_clean,
            runtime_g_clean, len(specs), expected_sample_slots))

        correct_overlay = args.variant == "correct_answer"
        comparator_letter = (
            strongest_incorrect_letter(
                backend_forward(model, batches["no_text"], "eager"), answer_ids, correct_letter
            ) if correct_overlay else target_letter
        )
        require(comparator_letter != correct_letter, "Margin comparator equals correct answer")
        log("METRIC", "qid={} target={} correct={} comparator={} metric={}".format(
            qid, target_letter, correct_letter, comparator_letter,
            "correct_minus_fixed_no_text_competitor" if correct_overlay else "correct_minus_condition_option",
        ))

        baselines: Dict[str, Any] = {}
        for image_state, inputs in batches.items():
            sdpa_logits = backend_forward(model, inputs, "sdpa")
            eager_logits = backend_forward(model, inputs, "eager")
            noop_logits, noop_integrity = forward_with_window_block(
                model, [layers[0]], inputs, None, destinations["D"]
            )
            noop_difference = float((eager_logits - noop_logits).abs().max().item())
            require(noop_difference <= 1e-3, "No-op mismatch")
            eager = summarize_logits(eager_logits, processor.tokenizer, answer_ids, correct_letter, comparator_letter)
            sdpa = summarize_logits(sdpa_logits, processor.tokenizer, answer_ids, correct_letter, comparator_letter)
            noop = summarize_logits(noop_logits, processor.tokenizer, answer_ids, correct_letter, comparator_letter)
            require(eager["choice_constrained_prediction"] == noop["choice_constrained_prediction"], "No-op prediction changed")
            backend_difference = float((sdpa_logits - eager_logits).abs().max().item())
            baselines[image_state] = {
                "eager": eager, "sdpa": sdpa, "empty_mask_noop": noop,
                "noop_integrity": noop_integrity,
                "noop_max_full_vocab_logit_difference": noop_difference,
                "sdpa_eager_max_full_vocab_logit_difference": backend_difference,
                "sdpa_eager_choice_prediction_equal": sdpa["choice_constrained_prediction"] == eager["choice_constrained_prediction"],
            }
            max_noop_difference = max(max_noop_difference, noop_difference)
            max_backend_difference = max(max_backend_difference, backend_difference)
            backend_prediction_disagreements += int(sdpa["choice_constrained_prediction"] != eager["choice_constrained_prediction"])

        records: List[Dict[str, Any]] = []
        for image_state, inputs in batches.items():
            baseline = baselines[image_state]["eager"]
            for window_name, layer_indices in WINDOWS.items():
                selected_layers = [layers[index] for index in layer_indices]
                for spec in specs:
                    available_edges = accessible_edge_count(spec["sources"], spec["destinations"])
                    common = {
                        "question_id": qid, "variant": args.variant, "image_state": image_state,
                        "window": window_name, "layers": layer_indices,
                        "path_instance": spec["path_instance"], "conceptual_path": spec["conceptual_path"],
                        "source_group": spec["source_group"], "destination_group": spec["destination_group"],
                        "source_count": len(spec["sources"]), "destination_count": len(spec["destinations"]),
                        "requested_edges_per_head": len(spec["sources"]) * len(spec["destinations"]),
                        "causally_available_edges_per_head": available_edges,
                    }
                    if available_edges == 0:
                        records.append({**common, "status": "structurally_unavailable", "margin_change": None})
                        continue
                    logits, integrity = forward_with_causal_window_block(
                        model, selected_layers, inputs, spec["sources"], spec["destinations"]
                    )
                    summary = summarize_logits(logits, processor.tokenizer, answer_ids, correct_letter, comparator_letter)
                    effect = summary["margin_correct_minus_misleading"] - baseline["margin_correct_minus_misleading"]
                    require(math.isfinite(effect), "Non-finite effect")
                    records.append({
                        **common, "status": "complete", "summary": summary, "margin_change": effect,
                        "transition": (
                            correct_overlay_transition(
                                baseline["choice_constrained_prediction"], summary["choice_constrained_prediction"], correct_letter
                            ) if correct_overlay else transition(
                                baseline["choice_constrained_prediction"], summary["choice_constrained_prediction"],
                                correct_letter, target_letter,
                            )
                        ),
                        "integrity": integrity,
                    })
                    max_blocked_probability = max(max_blocked_probability, max(row["blocked_probability_max"] for row in integrity))
                    max_row_sum_error = max(max_row_sum_error, max(row["destination_row_sum_max_error"] for row in integrity))

        require(len(records) == expected_sample_slots, "Saved sample slot count mismatch")
        sample_report = {
            "status": "complete", "question_id": qid, "dataset_index": int(entry["dataset_index"]),
            "variant": args.variant, "shard_id": args.shard_id,
            "correct_letter": correct_letter, "displayed_letter": target_letter,
            "comparator_letter": comparator_letter,
            "metric": (
                "correct_minus_fixed_no_text_competitor"
                if correct_overlay else "correct_minus_condition_option"
            ),
            "options": options, "answer_token_ids": answer_ids,
            "clean_object_controls": {"C": runtime_c_clean, "G": runtime_g_clean},
            "token_counts": {name: len(values) for name, values in groups.items()},
            "cleaned_overlay_validation": cleaned, "baselines": baselines, "records": records,
        }
        write_json(sample_path, sample_report)
        saved_slots += len(records)
        unavailable_slots += sum(row["status"] == "structurally_unavailable" for row in records)
        completed_samples += 1
        log("PROGRESS", "samples={}/{} slots={}/{} unavailable={} checkpoint={}".format(
            completed_samples, len(entries), saved_slots, config["expected_record_slots"], unavailable_slots, sample_path))

    require(completed_samples == len(entries), "Completed sample count mismatch")
    require(saved_slots == config["expected_record_slots"], "Saved slot count mismatch")
    completion = {
        "status": "complete", "variant": args.variant, "shard_id": args.shard_id,
        "samples": completed_samples, "resumed_samples": resumed_samples,
        "saved_record_slots": saved_slots, "expected_record_slots": config["expected_record_slots"],
        "structurally_unavailable_slots": unavailable_slots,
        "completed_interventions": saved_slots - unavailable_slots,
        "max_blocked_probability": max_blocked_probability,
        "max_attention_row_sum_error": max_row_sum_error,
        "max_noop_logit_difference": max_noop_difference,
        "max_sdpa_eager_logit_difference": max_backend_difference,
        "sdpa_eager_prediction_disagreements": backend_prediction_disagreements,
        "dataset_provenance": provenance, "decoder_path": layer_path,
    }
    write_json(args.output_dir / "completion.json", completion)
    log("COMPLETE", "Milestone 6 variant={} shard={}/{} samples={} slots={} unavailable={} validation_failures=0".format(
        args.variant, args.shard_id, args.num_shards, completed_samples, saved_slots, unavailable_slots))


if __name__ == "__main__":
    main()
