#!/usr/bin/env python3
"""Run generation-specific Qwen3-VL-8B attention intervention on one shard."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "vlms"), str(ROOT / "vlms/activation_patching/main_files"),
                str(ROOT / "vlms/qwen3_vl_generation_causal/main_files"),
                str(ROOT / "vlms/attention_path_intervention/main_files")]

from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from run_attention_path_full import accessible_edge_count, forward_with_causal_window_block  # noqa: E402
from test_one_sample_attention_windows import forward_with_window_block  # noqa: E402
from qwen3_causal_common import cache_multimodal_inputs  # noqa: E402
from run_step3_multilayer_patch import cached_logits, candidate_batch, score_from_logits  # noqa: E402
from sequence_scoring import answer_margin, strongest_incorrect  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (CACHE, PROMPT_INSTRUCTION, REVISIONS, VARIANTS,
                                         log, prepare, save_json)  # noqa: E402
from run_attention_smoke import (MODEL_ID, WINDOWS, build_paths, compact_integrity,
                                 question_positions)  # noqa: E402


def candidate_run(model, processor, layers, image, prompt, prompt_ids, question, answer):
    batch, start = candidate_batch(processor, image, prompt, answer, "cuda", prompt_ids)
    _, formatted = prepare(processor, image, prompt, "", "cuda")
    expanded = [int(x) for x in batch["input_ids"][0]]
    q_positions = question_positions(
        processor, formatted, answer, expanded, int(model.config.image_token_id), question,
    )
    a_positions = list(range(start - 1, len(expanded) - 1))
    if not a_positions or len(a_positions) != len(expanded) - start or set(q_positions) & set(a_positions):
        raise AssertionError("invalid Q/A groups")
    cached = cache_multimodal_inputs(model, batch)
    full = model(**batch, use_cache=False).logits.detach()
    baseline_logits = cached_logits(model, cached).detach()
    noop_logits, noop = forward_with_window_block(
        model, list(layers), cached, None, a_positions, forward_fn=cached_logits,
    )
    cache_error = float((full - baseline_logits).abs().max())
    noop_error = float((baseline_logits - noop_logits).abs().max())
    if cache_error > 1e-3 or noop_error > 1e-3:
        raise AssertionError(f"baseline reproduction failed cache={cache_error} noop={noop_error}")
    return {"batch": batch, "cached": cached, "start": start, "Q": q_positions, "A": a_positions,
            "score": score_from_logits(processor, answer, baseline_logits, batch["input_ids"], start),
            "cache_error": cache_error, "noop_error": noop_error,
            "noop_integrity": compact_integrity(noop)}


@torch.inference_mode()
def run_sample(model, processor, layers, sample, entry, variant, seed):
    qid = str(entry["question_id"]); question = str(sample["question"])
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {question}"
    images = {"no_text": sample["notext"]["image"].convert("RGB"),
              "overlay": sample[variant]["cleaned_image"].convert("RGB")}
    clean = validate_cleaned_overlay(images["no_text"], sample[variant]["image"].convert("RGB"),
                                     images["overlay"], text_bbox_yxyx(sample, variant))
    if clean["cleaned_outside_no_text_mismatch_pixels"] or clean["cleaned_inside_original_mismatch_pixels"]:
        raise AssertionError("cleaned-image pixel validation failed")

    prompt_batches, spatial = {}, {}
    for state, image in images.items():
        prompt_batches[state], _ = prepare(processor, image, prompt, "", "cuda")
        spatial[state] = positive_overlap_region_positions(
            model, prompt_batches[state], image, processor, sample, variant,
            seed * 100000 + int(entry["dataset_index"]),
        )
    if not torch.equal(prompt_batches["no_text"]["input_ids"], prompt_batches["overlay"]["input_ids"]):
        raise AssertionError("paired prompt IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0] or spatial["no_text"][1]["summary"] != spatial["overlay"][1]["summary"]:
        raise AssertionError("paired visual mapping differs")
    _, mapping, regions, sequence_regions = spatial["no_text"]
    required = ("text_region", "matched_random_region_1", "matched_random_region_2",
                "matched_random_region_3", "all_image_tokens")
    if any(name not in regions for name in required):
        raise AssertionError("required spatial group missing")
    text_count = len(sequence_regions["text_region"])
    if any(len(sequence_regions[f"matched_random_region_{i}"]) != text_count for i in (1, 2, 3)):
        raise AssertionError("random controls are not count matched")
    groups = {"T": sequence_regions["text_region"],
              **{f"R{i}": sequence_regions[f"matched_random_region_{i}"] for i in (1, 2, 3)}}
    groups["V"] = sorted(set(sequence_regions["all_image_tokens"]) - set(groups["T"]))
    c_clean = regions.get("correct_object_region", {}).get("clean_object_control") is True
    g_clean = regions.get("grounded_object_region", {}).get("clean_object_control") is True
    if c_clean: groups["C"] = sequence_regions["correct_object_region"]
    if g_clean: groups["G"] = sequence_regions["grounded_object_region"]
    paths = build_paths(groups, c_clean, g_clean)

    references = {key: str(sample[key]["text"]) for key in VARIANTS}
    comparison_key = variant
    if variant == "correct_answer":
        candidates = {}
        for key, answer in references.items():
            batch, start = candidate_batch(processor, images["no_text"], prompt, answer, "cuda",
                                           prompt_batches["no_text"]["input_ids"])
            candidates[key] = score_from_logits(processor, answer, model(**batch, use_cache=False).logits,
                                                batch["input_ids"], start)
        comparison_key = strongest_incorrect(candidates)
    answers = {"correct": references["correct_answer"], "comparison": references[comparison_key]}
    runs = {state: {role: candidate_run(model, processor, layers, image, prompt,
                                        prompt_batches[state]["input_ids"], question, answer)
                    for role, answer in answers.items()} for state, image in images.items()}
    baseline_margins = {state: answer_margin(values["correct"]["score"], values["comparison"]["score"])
                        for state, values in runs.items()}

    records = []; max_blocked = max_row = 0.0
    for state in ("no_text", "overlay"):
        for window, layer_ids in WINDOWS.items():
            for path, source_name, destination_name in paths:
                scores, checks, available = {}, {}, True
                for role, answer in answers.items():
                    run = runs[state][role]; role_groups = {**groups, "Q": run["Q"], "A": run["A"]}
                    source, destination = role_groups[source_name], role_groups[destination_name]
                    if accessible_edge_count(source, destination) == 0:
                        available = False; break
                    logits, integrity = forward_with_causal_window_block(
                        model, [layers[i] for i in layer_ids], run["cached"], source, destination,
                        forward_fn=cached_logits,
                    )
                    check = compact_integrity(integrity)
                    max_blocked = max(max_blocked, check["blocked_probability_max"])
                    max_row = max(max_row, check["row_sum_max_error"])
                    scores[role] = score_from_logits(processor, answer, logits,
                                                    run["batch"]["input_ids"], run["start"])
                    checks[role] = check
                common = {"image_state": state, "window": window, "layers": layer_ids, "path": path,
                          "source_group": source_name, "destination_group": destination_name}
                if not available:
                    records.append({**common, "status": "structurally_unavailable"}); continue
                margin = answer_margin(scores["correct"], scores["comparison"])
                change = margin - baseline_margins[state]
                if not math.isfinite(change): raise AssertionError("non-finite margin change")
                records.append({**common, "status": "complete", "baseline_margin": baseline_margins[state],
                                "blocked_margin": margin, "margin_change": change,
                                "scores": scores, "integrity": checks})

    expected = 2 * len(WINDOWS) * len(paths)
    if len(records) != expected: raise AssertionError("record accounting failed")
    lookup = {(x["image_state"], x["window"], x["path"]): x for x in records}
    derived = []
    for window in WINDOWS:
        for path, _, destination in paths:
            overlay, no_text = lookup[("overlay", window, path)], lookup[("no_text", window, path)]
            if overlay["status"] != "complete" or no_text["status"] != "complete": continue
            did = overlay["margin_change"] - no_text["margin_change"]
            derived.append({"window": window, "path": path, "destination_group": destination,
                            "difference_in_differences": did,
                            "oriented_effect": -did if variant == "correct_answer" else did})
    interactions = []
    for window in WINDOWS:
        for object_name, clean_flag in (("C", c_clean), ("G", g_clean)):
            if not clean_flag: continue
            for destination in ("Q", "A"):
                values = {}
                for state in ("no_text", "overlay"):
                    values[state] = (lookup[(state, window, f"T+{object_name}_to_{destination}")]["margin_change"]
                                     - lookup[(state, window, f"T_to_{destination}")]["margin_change"]
                                     - lookup[(state, window, f"{object_name}_to_{destination}")]["margin_change"])
                interactions.append({"window": window, "object": object_name,
                                     "destination_group": destination, "no_text_interaction": values["no_text"],
                                     "overlay_interaction": values["overlay"],
                                     "overlay_specific_interaction": values["overlay"] - values["no_text"]})
    validation = {"expected_records": expected, "saved_records": len(records),
                  "max_cached_logit_difference": max(r["cache_error"] for s in runs.values() for r in s.values()),
                  "max_noop_logit_difference": max(r["noop_error"] for s in runs.values() for r in s.values()),
                  "max_blocked_probability": max_blocked, "max_attention_row_sum_error": max_row,
                  "structurally_unavailable_records": sum(x["status"] != "complete" for x in records)}
    return {"status": "complete", "model_id": MODEL_ID, "model_revision": REVISIONS[MODEL_ID],
            "question_id": qid, "dataset_index": int(entry["dataset_index"]), "variant": variant,
            "comparison_key": comparison_key, "prompt": prompt, "answers": answers, "windows": WINDOWS,
            "paths": [x[0] for x in paths], "mapping": mapping["summary"], "groups": groups,
            "clean_object_controls": {"C": c_clean, "G": g_clean}, "baseline_margins": baseline_margins,
            "records": records, "derived_path_effects": derived,
            "factorial_interactions": interactions, "validation": validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    manifest = json.loads(args.selection.read_text()); entries = manifest["selected_samples"]
    qids = [str(x["question_id"]) for x in entries]
    if len(qids) != 305 or len(set(qids)) != 305 or not 0 <= args.shard_id < args.num_shards:
        raise AssertionError("invalid shared-305 manifest or shard")
    entries = entries[args.shard_id::args.num_shards]
    if args.max_samples: entries = entries[:args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "configuration.json", {
        "status": "configured", "model_id": MODEL_ID, "model_revision": REVISIONS[MODEL_ID],
        "variant": args.variant, "question_ids": [str(x["question_id"]) for x in entries],
        "windows": WINDOWS, "backend": "eager", "prompt_format": "open_ended_no_options",
        "answer_metric": "mean_logprob_full_reference_sequence", "shard_id": args.shard_id,
        "num_shards": args.num_shards, "seed": args.seed,
    })
    log("CONFIG", f"model={MODEL_ID} variant={args.variant} shard={args.shard_id}/{args.num_shards} "
                  f"questions={len(entries)} windows={WINDOWS}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISIONS[MODEL_ID], dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISIONS[MODEL_ID])
    layers = model.model.language_model.layers
    if len(layers) != 36 or model.config.text_config._attn_implementation != "eager":
        raise AssertionError("architecture/backend mismatch")
    dataset = load_from_disk(str(CACHE)); samples_dir = args.output_dir / "samples"; samples_dir.mkdir(exist_ok=True)
    completed = resumed = records = unavailable = 0; maxima = {"cache": 0., "noop": 0., "blocked": 0., "row": 0.}
    for number, entry in enumerate(entries, 1):
        qid = str(entry["question_id"]); path = samples_dir / f"{qid}.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("status") != "complete": raise AssertionError(f"invalid checkpoint {path}")
            resumed += 1
        else:
            started = time.perf_counter(); sample = dataset[int(entry["dataset_index"])]
            if str(sample["question_id"]) != qid: raise AssertionError("dataset index mismatch")
            report = run_sample(model, processor, layers, sample, entry, args.variant, args.seed)
            save_json(path, report)
            log("TIMING", f"question={number}/{len(entries)} qid={qid} seconds={time.perf_counter()-started:.2f}")
        validation = report["validation"]; completed += 1; records += validation["saved_records"]
        unavailable += validation["structurally_unavailable_records"]
        for short, key in (("cache", "max_cached_logit_difference"), ("noop", "max_noop_logit_difference"),
                           ("blocked", "max_blocked_probability"), ("row", "max_attention_row_sum_error")):
            maxima[short] = max(maxima[short], float(validation[key]))
        log("CHECKPOINT", f"completed={completed}/{len(entries)} qid={qid} records={records} "
                          f"unavailable={unavailable} resumed={resumed}")
    completion = {"status": "complete", "samples": completed, "expected_samples": len(entries),
                  "resumed_samples": resumed, "saved_records": records,
                  "structurally_unavailable_records": unavailable, "validation_failures": 0,
                  "maximum_validation_values": maxima}
    save_json(args.output_dir / "completion.json", completion)
    log("PASS", f"variant={args.variant} shard={args.shard_id}/{args.num_shards} completion={completion}")


if __name__ == "__main__":
    main()
