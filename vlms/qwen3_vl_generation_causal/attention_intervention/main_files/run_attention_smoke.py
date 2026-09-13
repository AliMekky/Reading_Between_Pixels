#!/usr/bin/env python3
"""Validate generation-specific attention paths on one Qwen3-VL-8B example."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "vlms"), str(ROOT / "vlms/activation_patching/main_files"),
                str(ROOT / "vlms/qwen3_vl_generation_causal/main_files"),
                str(ROOT / "vlms/attention_path_intervention/main_files")]

from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from inspect_attention_path_setup import align_raw_to_expanded, positions_for_char_span  # noqa: E402
from run_attention_path_full import accessible_edge_count, forward_with_causal_window_block  # noqa: E402
from test_one_sample_attention_windows import forward_with_window_block  # noqa: E402
from qwen3_causal_common import cache_multimodal_inputs  # noqa: E402
from run_step3_multilayer_patch import cached_logits, candidate_batch, score_from_logits  # noqa: E402
from sequence_scoring import answer_margin, strongest_incorrect  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (CACHE, PROMPT_INSTRUCTION, REVISIONS, SELECTION,
                                         VARIANTS, log, prepare, save_json)  # noqa: E402


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
WINDOWS = {f"layers_{start:02d}_{start + 5:02d}": list(range(start, start + 6))
           for start in range(0, 36, 6)}


def question_positions(processor, formatted, answer, expanded_ids, image_token_id, question):
    raw = processor.tokenizer(formatted + answer, add_special_tokens=True, return_offsets_mapping=True)
    mapping, _ = align_raw_to_expanded(
        [int(x) for x in raw["input_ids"]], [int(x) for x in expanded_ids], image_token_id,
    )
    start = formatted.index(question)
    positions = positions_for_char_span(raw["offset_mapping"], mapping, (start, start + len(question)))
    if not positions:
        raise AssertionError("question token group is empty")
    return positions


def build_paths(groups, c_clean, g_clean):
    paths = []

    def add(name, source, destination):
        paths.append((name, source, destination))

    for destination in ("Q", "A"):
        add(f"T_to_{destination}", "T", destination)
        for index in (1, 2, 3):
            add(f"R{index}_to_{destination}", f"R{index}", destination)
        add(f"V_to_{destination}", "V", destination)
    add("Q_to_A", "Q", "A")
    for object_name, clean in (("C", c_clean), ("G", g_clean)):
        if not clean:
            continue
        for destination in ("Q", "A"):
            add(f"{object_name}_to_{destination}", object_name, destination)
            joint = f"T+{object_name}"
            groups[joint] = sorted(set(groups["T"]) | set(groups[object_name]))
            add(f"{joint}_to_{destination}", joint, destination)
        add(f"T_to_{object_name}", "T", object_name)
        add(f"{object_name}_to_T", object_name, "T")
    return paths


def compact_integrity(values):
    return {
        "layers": [int(x["layer"]) for x in values],
        "blocked_probability_max": max(float(x.get("blocked_probability_max", 0)) for x in values),
        "row_sum_max_error": max(float(x["destination_row_sum_max_error"]) for x in values),
        "blocked_available_edges_per_head": [int(x.get("blocked_available_edges_per_head", 0)) for x in values],
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="misleading_groundable", choices=VARIANTS)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    selected = {str(x["question_id"]): int(x["dataset_index"])
                for x in json.loads(SELECTION.read_text())["selected_samples"]}
    if len(selected) != 305 or args.question_id not in selected:
        raise AssertionError("question must belong to the shared 305-question manifest")
    sample = load_from_disk(str(CACHE))[selected[args.question_id]]
    question = str(sample["question"])
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {question}"
    images = {"no_text": sample["notext"]["image"].convert("RGB"),
              "overlay": sample[args.variant]["cleaned_image"].convert("RGB")}
    clean = validate_cleaned_overlay(
        images["no_text"], sample[args.variant]["image"].convert("RGB"), images["overlay"],
        text_bbox_yxyx(sample, args.variant),
    )
    if clean["cleaned_outside_no_text_mismatch_pixels"] or clean["cleaned_inside_original_mismatch_pixels"]:
        raise AssertionError("cleaned-image pixel validation failed")

    log("CONFIG", f"stage=attention_smoke model={MODEL_ID} variant={args.variant} qid={args.question_id}")
    log("CONFIG", f"windows={WINDOWS} prompt_has_options={('Options:' in prompt)} backend=eager")
    log("EXPECTED", "full multi-token A; overlay/no-text DID; all pathways; blocked_probability=0; row_error<=2e-3")
    revision = REVISIONS[MODEL_ID]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    layers = model.model.language_model.layers
    if len(layers) != 36 or model.config.text_config._attn_implementation != "eager":
        raise AssertionError("unexpected layer count or attention backend")

    prompt_batches, spatial = {}, {}
    for state, image in images.items():
        prompt_batches[state], _ = prepare(processor, image, prompt, "", "cuda")
        spatial[state] = positive_overlap_region_positions(
            model, prompt_batches[state], image, processor, sample, args.variant,
            args.seed * 100000 + selected[args.question_id],
        )
    if not torch.equal(prompt_batches["no_text"]["input_ids"], prompt_batches["overlay"]["input_ids"]):
        raise AssertionError("paired prompt IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0] or spatial["no_text"][1]["summary"] != spatial["overlay"][1]["summary"]:
        raise AssertionError("paired image-token mappings differ")
    _, mapping, regions, sequence_regions = spatial["no_text"]
    required = ["text_region", "matched_random_region_1", "matched_random_region_2",
                "matched_random_region_3", "all_image_tokens"]
    if any(name not in regions for name in required):
        raise AssertionError("required spatial region is missing")
    text_count = len(sequence_regions["text_region"])
    if any(len(sequence_regions[f"matched_random_region_{i}"]) != text_count for i in (1, 2, 3)):
        raise AssertionError("random controls do not match the text-token count")

    groups = {"T": sequence_regions["text_region"],
              "R1": sequence_regions["matched_random_region_1"],
              "R2": sequence_regions["matched_random_region_2"],
              "R3": sequence_regions["matched_random_region_3"]}
    groups["V"] = sorted(set(sequence_regions["all_image_tokens"]) - set(groups["T"]))
    c_clean = regions.get("correct_object_region", {}).get("clean_object_control") is True
    g_clean = regions.get("grounded_object_region", {}).get("clean_object_control") is True
    if c_clean:
        groups["C"] = sequence_regions["correct_object_region"]
    if g_clean:
        groups["G"] = sequence_regions["grounded_object_region"]
    paths = build_paths(groups, c_clean, g_clean)
    log("MAPPING", f"grid={mapping['summary']['merged_grid_hw']} image_tokens={len(spatial['no_text'][0])} "
                   f"T={len(groups['T'])} R={[len(groups[f'R{i}']) for i in (1,2,3)]} "
                   f"V={len(groups['V'])} C={len(groups.get('C', []))} G={len(groups.get('G', []))}")
    log("PATHS", f"instances={len(paths)} names={[x[0] for x in paths]}")

    references = {key: str(sample[key]["text"]) for key in VARIANTS}
    comparison_key = args.variant
    if args.variant == "correct_answer":
        initial = {}
        for key, answer in references.items():
            batch, start = candidate_batch(processor, images["no_text"], prompt, answer, "cuda",
                                           prompt_batches["no_text"]["input_ids"])
            initial[key] = score_from_logits(processor, answer, model(**batch, use_cache=False).logits,
                                             batch["input_ids"], start)
        comparison_key = strongest_incorrect(initial)
    answers = {"correct": references["correct_answer"], "comparison": references[comparison_key]}

    runs = {state: {} for state in images}
    noop_max = cache_max = 0.0
    for state, image in images.items():
        for role, answer in answers.items():
            batch, start = candidate_batch(processor, image, prompt, answer, "cuda",
                                           prompt_batches[state]["input_ids"])
            formatted = prepare(processor, image, prompt, "", "cuda")[1]
            expanded = [int(x) for x in batch["input_ids"][0]]
            q_positions = question_positions(processor, formatted, answer, expanded,
                                             int(model.config.image_token_id), question)
            a_positions = list(range(start - 1, len(expanded) - 1))
            if len(a_positions) != len(expanded) - start or set(q_positions) & set(a_positions):
                raise AssertionError("invalid Q/A token groups")
            cached = cache_multimodal_inputs(model, batch)
            full = model(**batch, use_cache=False).logits.detach()
            baseline_logits = cached_logits(model, cached).detach()
            noop_logits, noop_check = forward_with_window_block(
                model, list(layers), cached, None, a_positions, forward_fn=cached_logits,
            )
            cache_error = float((full - baseline_logits).abs().max())
            noop_error = float((baseline_logits - noop_logits).abs().max())
            if cache_error > 1e-3 or noop_error > 1e-3:
                raise AssertionError(f"baseline reproduction failed: cache={cache_error} noop={noop_error}")
            cache_max, noop_max = max(cache_max, cache_error), max(noop_max, noop_error)
            runs[state][role] = {"batch": batch, "cached": cached, "start": start,
                                 "Q": q_positions, "A": a_positions,
                                 "score": score_from_logits(processor, answer, baseline_logits,
                                                            batch["input_ids"], start),
                                 "noop_integrity": compact_integrity(noop_check)}
            log("TOKENS", f"state={state} role={role} answer={answer!r} answer_tokens={len(a_positions)} "
                          f"Q={len(q_positions)} A={a_positions} cache_error={cache_error:.3g} noop={noop_error:.3g}")

    baseline_margins = {state: answer_margin(runs[state]["correct"]["score"],
                                             runs[state]["comparison"]["score"])
                        for state in images}
    records = []
    max_blocked = max_row_error = 0.0
    for state in ("no_text", "overlay"):
        for window, layer_ids in WINDOWS.items():
            complete = unavailable = 0
            for path, source_name, destination_name in paths:
                scores, checks = {}, {}
                for role, answer in answers.items():
                    run = runs[state][role]
                    role_groups = {**groups, "Q": run["Q"], "A": run["A"]}
                    source, destination = role_groups[source_name], role_groups[destination_name]
                    if accessible_edge_count(source, destination) == 0:
                        checks[role] = None
                        continue
                    logits, integrity = forward_with_causal_window_block(
                        model, [layers[i] for i in layer_ids], run["cached"], source, destination,
                        forward_fn=cached_logits,
                    )
                    check = compact_integrity(integrity)
                    max_blocked = max(max_blocked, check["blocked_probability_max"])
                    max_row_error = max(max_row_error, check["row_sum_max_error"])
                    scores[role] = score_from_logits(processor, answer, logits,
                                                    run["batch"]["input_ids"], run["start"])
                    checks[role] = check
                common = {"image_state": state, "window": window, "layers": layer_ids,
                          "path": path, "source_group": source_name,
                          "destination_group": destination_name}
                if len(scores) != 2:
                    records.append({**common, "status": "structurally_unavailable"})
                    unavailable += 1
                    continue
                margin = answer_margin(scores["correct"], scores["comparison"])
                change = margin - baseline_margins[state]
                if not math.isfinite(change):
                    raise AssertionError("non-finite margin change")
                records.append({**common, "status": "complete", "baseline_margin": baseline_margins[state],
                                "blocked_margin": margin, "margin_change": change,
                                "scores": scores, "integrity": checks})
                complete += 1
            log("WINDOW", f"state={state} window={window} complete={complete} unavailable={unavailable}")

    expected = 2 * len(WINDOWS) * len(paths)
    if len(records) != expected:
        raise AssertionError(f"record accounting failed: {len(records)} != {expected}")
    lookup = {(x["image_state"], x["window"], x["path"]): x for x in records}
    derived = []
    for window in WINDOWS:
        for path, _, destination in paths:
            overlay = lookup[("overlay", window, path)]
            no_text = lookup[("no_text", window, path)]
            if overlay["status"] != "complete" or no_text["status"] != "complete":
                continue
            did = overlay["margin_change"] - no_text["margin_change"]
            derived.append({"window": window, "path": path, "destination_group": destination,
                            "difference_in_differences": did,
                            "oriented_effect": -did if args.variant == "correct_answer" else did})

    interactions = []
    for window in WINDOWS:
        for object_name, clean_flag in (("C", c_clean), ("G", g_clean)):
            if not clean_flag:
                continue
            for destination in ("Q", "A"):
                values = {}
                for state in ("no_text", "overlay"):
                    joint = lookup[(state, window, f"T+{object_name}_to_{destination}")]["margin_change"]
                    text = lookup[(state, window, f"T_to_{destination}")]["margin_change"]
                    obj = lookup[(state, window, f"{object_name}_to_{destination}")]["margin_change"]
                    values[state] = joint - text - obj
                interactions.append({"window": window, "object": object_name,
                                     "destination_group": destination,
                                     "no_text_interaction": values["no_text"],
                                     "overlay_interaction": values["overlay"],
                                     "overlay_specific_interaction": values["overlay"] - values["no_text"]})

    report = {"status": "complete", "model_id": MODEL_ID, "model_revision": revision,
              "question_id": args.question_id, "variant": args.variant, "comparison_key": comparison_key,
              "prompt": prompt, "windows": WINDOWS, "paths": [x[0] for x in paths],
              "answers": answers, "mapping": mapping["summary"],
              "region_counts": {name: len(values) for name, values in groups.items()},
              "clean_object_controls": {"C": c_clean, "G": g_clean},
              "baseline_margins": baseline_margins, "records": records,
              "derived_path_effects": derived, "factorial_interactions": interactions,
              "validation": {"cleaned_overlay": clean, "expected_records": expected,
                             "saved_records": len(records), "max_cached_logit_difference": cache_max,
                             "max_noop_logit_difference": noop_max,
                             "max_blocked_probability": max_blocked,
                             "max_attention_row_sum_error": max_row_error}}
    save_json(args.output, report)
    log("VALIDATION", f"records={len(records)}/{expected} cache={cache_max:.3g} noop={noop_max:.3g} "
                      f"blocked={max_blocked:.3g} row_error={max_row_error:.3g}")
    log("OUTPUT", f"saved={args.output}")
    log("PASS", "generation-specific attention-path smoke test complete")


if __name__ == "__main__":
    main()
