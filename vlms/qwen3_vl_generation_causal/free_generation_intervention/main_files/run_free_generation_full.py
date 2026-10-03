#!/usr/bin/env python3
"""Run late text-to-answer blocking during free generation on one data shard."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "vlms"),
                str(ROOT / "vlms/activation_patching/main_files"),
                str(ROOT / "vlms/qwen3_vl_generation_causal/main_files")]

from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from run_free_generation_gate import (LAYERS, MODEL_ID, REGIONS, generate_with_block,
                                      image_hash, ordinary_generation, prepare,
                                      previous_record)  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (CACHE, PROMPT_INSTRUCTION, REVISIONS, SELECTION,
                                         VARIANTS, log, save_json)  # noqa: E402


def run_sample(model, processor, layers, sample, entry, variant, max_new_tokens, seed):
    qid = str(entry["question_id"])
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
    if "Options:" in prompt or "A, B, C, or D" in prompt:
        raise AssertionError("MCQ content leaked into the open-ended prompt")

    images = {
        "no_text": sample["notext"]["image"].convert("RGB"),
        "overlay": sample[variant]["cleaned_image"].convert("RGB"),
    }
    clean = validate_cleaned_overlay(
        images["no_text"], sample[variant]["image"].convert("RGB"), images["overlay"],
        text_bbox_yxyx(sample, variant),
    )
    if clean["cleaned_outside_no_text_mismatch_pixels"] or clean["cleaned_inside_original_mismatch_pixels"]:
        raise AssertionError("cleaned overlay failed pixel validation")

    batches, spatial = {}, {}
    for state, image in images.items():
        batches[state], _ = prepare(processor, image, prompt, "", "cuda")
        spatial[state] = positive_overlap_region_positions(
            model, batches[state], image, processor, sample, variant,
            seed * 100000 + int(entry["dataset_index"]),
        )
    if not torch.equal(batches["no_text"]["input_ids"], batches["overlay"]["input_ids"]):
        raise AssertionError("paired prompt token IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0]:
        raise AssertionError("paired visual sequence positions differ")

    sequence_regions = spatial["no_text"][3]
    text_count = len(sequence_regions["text_region"])
    if not text_count or any(len(sequence_regions[name]) != text_count for name in REGIONS[1:]):
        raise AssertionError("text/random region counts are invalid")

    report = {
        "status": "running", "model_id": MODEL_ID, "model_revision": REVISIONS[MODEL_ID],
        "question_id": qid, "dataset_index": int(entry["dataset_index"]), "variant": variant,
        "question": str(sample["question"]), "prompt": prompt,
        "references": {key: str(sample[key]["text"]) for key in VARIANTS},
        "layers": LAYERS, "max_new_tokens": max_new_tokens,
        "region_counts": {name: len(sequence_regions[name]) for name in REGIONS},
        "cleaned_overlay_validation": clean, "states": {},
    }
    selected_layers = [layers[index] for index in LAYERS]
    validation_maxima = {"blocked_probability": 0.0, "attention_row_sum_error": 0.0}

    for state in ("no_text", "overlay"):
        baseline = generate_with_block(
            model, processor, batches[state], selected_layers, None, max_new_tokens,
        )
        prior_variant = "notext" if state == "no_text" else variant
        prior = previous_record(MODEL_ID, prior_variant, qid)
        prior_check = {
            "available": prior is not None,
            "revision_match": prior is not None and prior.get("model_revision") == REVISIONS[MODEL_ID],
            "image_hash_match": prior is not None and prior.get("image_sha256") == image_hash(images[state]),
            "tokens_match": prior is not None and prior.get("output_token_ids") == baseline["output_token_ids"],
        }
        if not (prior_check["available"] and prior_check["revision_match"] and prior_check["image_hash_match"]):
            raise AssertionError(f"{state}: prior baseline is not comparable: {prior_check}")
        eager_recheck = None
        if not prior_check["tokens_match"]:
            eager_recheck = ordinary_generation(
                model, processor, batches[state], max_new_tokens,
            )
            if eager_recheck["output_token_ids"] != baseline["output_token_ids"]:
                raise AssertionError(f"{state}: empty hook changes eager generation")
            log("BACKEND", f"qid={qid} state={state} prior_sdpa={prior['raw_response']!r} "
                           f"current_eager={baseline['raw_response']!r} hook_noop=True")

        interventions = {}
        for region in REGIONS:
            result = generate_with_block(
                model, processor, batches[state], selected_layers,
                sequence_regions[region], max_new_tokens,
            )
            result["changed_from_baseline"] = result["output_token_ids"] != baseline["output_token_ids"]
            interventions[region] = result
            validation_maxima["blocked_probability"] = max(
                validation_maxima["blocked_probability"], result["integrity"]["max_blocked_probability"])
            validation_maxima["attention_row_sum_error"] = max(
                validation_maxima["attention_row_sum_error"], result["integrity"]["max_attention_row_sum_error"])
        report["states"][state] = {
            "image_sha256": image_hash(images[state]), "prior_check": prior_check,
            "empty_hook_baseline": baseline, "ordinary_eager_recheck": eager_recheck,
            "interventions": interventions,
        }

    core_generations = 2 * (1 + len(REGIONS))
    eager_rechecks = sum(value["ordinary_eager_recheck"] is not None for value in report["states"].values())
    report["validation"] = {
        "paired_prompt_ids_equal": True, "paired_visual_positions_equal": True,
        "prior_baseline_comparability_checked": True, "text_random_counts_matched": True,
        "core_generations": core_generations, "eager_rechecks": eager_rechecks,
        "saved_generations": core_generations + eager_rechecks,
        "prior_token_mismatches": sum(
            not value["prior_check"]["tokens_match"] for value in report["states"].values()
        ),
        "maximum_integrity_values": validation_maxima,
    }
    report["status"] = "complete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    entries = json.loads(args.selection.read_text())["selected_samples"]
    qids = [str(entry["question_id"]) for entry in entries]
    if len(qids) != 305 or len(set(qids)) != 305 or not 0 <= args.shard_id < args.num_shards:
        raise AssertionError("invalid shared-305 manifest or shard")
    entries = entries[args.shard_id::args.num_shards]
    if args.max_samples:
        entries = entries[:args.max_samples]
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "configuration.json", {
        "status": "configured", "model_id": MODEL_ID, "model_revision": REVISIONS[MODEL_ID],
        "variant": args.variant, "layers": LAYERS, "path": "T_to_current_answer_query",
        "regions": REGIONS, "question_ids": [str(entry["question_id"]) for entry in entries],
        "prompt_format": "open_ended_no_options", "decoding": "greedy_with_kv_cache",
        "max_new_tokens": args.max_new_tokens, "shard_id": args.shard_id,
        "num_shards": args.num_shards, "seed": args.seed,
    })
    log("CONFIG", f"model={MODEL_ID} variant={args.variant} shard={args.shard_id}/{args.num_shards} "
                  f"questions={len(entries)} layers={LAYERS} regions={REGIONS}")
    log("EXPECTED", "305 unique questions across shards; 10 generations/question; exact prior baseline reproduction; zero integrity failures")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISIONS[MODEL_ID], dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISIONS[MODEL_ID])
    layers = model.model.language_model.layers
    if len(layers) != 36 or model.config.text_config._attn_implementation != "eager":
        raise AssertionError("architecture/backend mismatch")

    dataset = load_from_disk(str(CACHE))
    completed = resumed = generations = changed = backend_mismatches = 0
    maxima = {"blocked_probability": 0.0, "attention_row_sum_error": 0.0}
    for number, entry in enumerate(entries, 1):
        qid = str(entry["question_id"])
        path = samples_dir / f"{qid}.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("status") != "complete" or report.get("variant") != args.variant:
                raise AssertionError(f"invalid checkpoint {path}")
            resumed += 1
        else:
            started = time.perf_counter()
            sample = dataset[int(entry["dataset_index"])]
            if str(sample["question_id"]) != qid:
                raise AssertionError("dataset index mismatch")
            report = run_sample(
                model, processor, layers, sample, entry, args.variant,
                args.max_new_tokens, args.seed,
            )
            save_json(path, report)
            log("TIMING", f"question={number}/{len(entries)} qid={qid} seconds={time.perf_counter()-started:.2f}")
        completed += 1
        generations += int(report["validation"]["saved_generations"])
        backend_mismatches += int(report["validation"].get("prior_token_mismatches", 0))
        changed += sum(
            int(run["changed_from_baseline"])
            for state in report["states"].values() for run in state["interventions"].values()
        )
        for key, value in report["validation"]["maximum_integrity_values"].items():
            maxima[key] = max(maxima[key], float(value))
        log("CHECKPOINT", f"completed={completed}/{len(entries)} qid={qid} generations={generations} "
                          f"changed={changed} backend_mismatches={backend_mismatches} resumed={resumed}")

    completion = {
        "status": "complete", "variant": args.variant, "samples": completed,
        "expected_samples": len(entries), "resumed_samples": resumed,
        "saved_generations": generations, "changed_interventions": changed,
        "prior_backend_token_mismatches": backend_mismatches,
        "validation_failures": 0, "maximum_integrity_values": maxima,
    }
    if completed != len(entries) or generations < completed * 2 * (1 + len(REGIONS)):
        raise AssertionError("completion accounting failed")
    save_json(args.output_dir / "completion.json", completion)
    log("PASS", f"variant={args.variant} shard={args.shard_id}/{args.num_shards} completion={completion}")


if __name__ == "__main__":
    main()
