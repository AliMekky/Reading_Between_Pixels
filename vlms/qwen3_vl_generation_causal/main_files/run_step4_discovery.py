#!/usr/bin/env python3
"""All-layer Qwen3-VL activation-patching discovery screen."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "vlms"))
sys.path.insert(0, str(ROOT / "vlms/activation_patching/main_files"))

from qwen3_causal_common import cache_multimodal_inputs  # noqa: E402
from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from run_step3_multilayer_patch import (  # noqa: E402
    REGIONS, cached_logits, candidate_batch, capture_resid_pre, oriented,
    patch_one_layer, patch_visual_source_bundle, score_from_logits, validate_integrity,
)
from sequence_scoring import answer_margin, strongest_incorrect  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (  # noqa: E402
    CACHE, PROMPT_INSTRUCTION, REVISIONS, VARIANTS, log, prepare, save_json,
)


MODEL_LAYERS = {
    "Qwen/Qwen3-VL-2B-Instruct": 28,
    "Qwen/Qwen3-VL-8B-Instruct": 36,
}


def preference(margin):
    return "correct" if margin > 0 else "comparison"


def load_selection(path, shard_id, num_shards, expected_questions):
    manifest = json.loads(path.read_text())
    entries = manifest["selected_samples"]
    qids = [str(row["question_id"]) for row in entries]
    if len(qids) != expected_questions or len(set(qids)) != expected_questions:
        raise AssertionError(f"selection must contain {expected_questions} unique questions")
    return manifest, entries[shard_id::num_shards]


def score_candidates(model, processor, image, prompt, prompt_ids, answers, device):
    scores = {}
    for key, answer in answers.items():
        batch, start = candidate_batch(processor, image, prompt, answer, device, prompt_ids)
        logits = model(**batch, use_cache=False).logits.detach()
        scores[key] = score_from_logits(processor, answer, logits, batch["input_ids"], start)
    return scores


@torch.inference_mode()
def run_sample(args, model, processor, layers, sample, entry):
    qid = str(entry["question_id"])
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
    images = {
        "no_text": sample["notext"]["image"].convert("RGB"),
        "overlay": sample[args.variant]["cleaned_image"].convert("RGB"),
    }
    clean = validate_cleaned_overlay(
        images["no_text"], sample[args.variant]["image"].convert("RGB"), images["overlay"],
        text_bbox_yxyx(sample, args.variant),
    )
    if clean["cleaned_outside_no_text_mismatch_pixels"] or clean["cleaned_inside_original_mismatch_pixels"]:
        raise AssertionError("cleaned overlay failed pixel validation")

    prompts, spatial = {}, {}
    seed = args.region_seed * 100000 + int(entry["dataset_index"])
    for state, image in images.items():
        prompts[state], _ = prepare(processor, image, prompt, "", args.device)
        spatial[state] = positive_overlap_region_positions(
            model, prompts[state], image, processor, sample, args.variant, seed,
        )
    if not torch.equal(prompts["no_text"]["input_ids"], prompts["overlay"]["input_ids"]):
        raise AssertionError("paired prompt token IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0] or spatial["no_text"][1]["summary"] != spatial["overlay"][1]["summary"]:
        raise AssertionError("paired visual positions or grids differ")
    regions, sequence_regions = spatial["no_text"][2], spatial["no_text"][3]
    if any(name not in regions for name in REGIONS):
        raise AssertionError("required spatial control missing")
    text_count = len(regions["text_region"]["token_indices"])
    if any(len(regions[f"matched_random_region_{i}"]["token_indices"]) != text_count for i in (1, 2, 3)):
        raise AssertionError("random controls are not count matched")

    references = {key: str(sample[key]["text"]) for key in VARIANTS}
    comparison_key = args.variant
    comparator_selection_scores = None
    if args.variant == "correct_answer":
        comparator_selection_scores = score_candidates(
            model, processor, images["no_text"], prompt, prompts["no_text"]["input_ids"],
            references, args.device,
        )
        comparison_key = strongest_incorrect(comparator_selection_scores)
    answers = {"correct": references["correct_answer"], "comparison": references[comparison_key]}

    runs = {state: {} for state in images}
    for state, image in images.items():
        for role, answer in answers.items():
            batch, start = candidate_batch(
                processor, image, prompt, answer, args.device, prompts[state]["input_ids"],
            )
            full_logits = model(**batch, use_cache=False).logits.detach()
            cached = cache_multimodal_inputs(model, batch)
            noop_logits, hidden = capture_resid_pre(model, cached, range(len(layers)))
            error = float((full_logits - noop_logits).abs().max())
            if error > 1e-3:
                raise AssertionError(f"cached no-op error={error}")
            runs[state][role] = {
                "batch": batch, "cached": cached, "start": start, "hidden": hidden,
                "score": score_from_logits(processor, answer, noop_logits, batch["input_ids"], start),
                "noop_max_logit_difference": error,
            }

    baseline_scores = {state: {role: run["score"] for role, run in roles.items()} for state, roles in runs.items()}
    baseline_margins = {
        state: answer_margin(roles["correct"]["score"], roles["comparison"]["score"])
        for state, roles in runs.items()
    }
    records = []
    for layer in range(len(layers)):
        for region in REGIONS:
            for direction, recipient_state, donor_state in (
                ("restoration", "overlay", "no_text"), ("insertion", "no_text", "overlay"),
            ):
                patched_scores, checks = {}, {}
                for role, answer in answers.items():
                    recipient = runs[recipient_state][role]
                    logits, check = patch_one_layer(
                        model, recipient["cached"], layer, sequence_regions[region],
                        runs[donor_state][role]["hidden"][layer],
                    )
                    validate_integrity(check)
                    patched_scores[role] = score_from_logits(
                        processor, answer, logits, recipient["batch"]["input_ids"], recipient["start"],
                    )
                    checks[role] = check
                patched_margin = answer_margin(patched_scores["correct"], patched_scores["comparison"])
                raw = patched_margin - baseline_margins[recipient_state]
                if not math.isfinite(raw):
                    raise AssertionError("non-finite single-layer effect")
                records.append({
                    "intervention_type": "single_layer_resid_pre", "layer": layer,
                    "region": region, "direction": direction, "recipient_state": recipient_state,
                    "token_count": len(sequence_regions[region]),
                    "recipient_margin": baseline_margins[recipient_state], "patched_margin": patched_margin,
                    "raw_margin_change": raw,
                    "oriented_effect": oriented(raw, direction, args.variant == "correct_answer"),
                    "recipient_preference": preference(baseline_margins[recipient_state]),
                    "patched_preference": preference(patched_margin),
                    "patched_scores": patched_scores, "integrity": checks,
                })
        log("LAYER", f"qid={qid} layer={layer}/{len(layers)-1} records={(layer+1)*len(REGIONS)*2}")

    for region in REGIONS:
        visual_indices = regions[region]["token_indices"]
        for direction, recipient_state, donor_state in (
            ("restoration", "overlay", "no_text"), ("insertion", "no_text", "overlay"),
        ):
            patched_scores, checks = {}, {}
            for role, answer in answers.items():
                recipient, donor = runs[recipient_state][role], runs[donor_state][role]
                logits, check = patch_visual_source_bundle(
                    model, recipient["cached"], donor["cached"], sequence_regions[region], visual_indices,
                )
                validate_integrity(check["initial_visual_embedding"])
                for stream in check["deepstack_additions"]:
                    validate_integrity(stream)
                patched_scores[role] = score_from_logits(
                    processor, answer, logits, recipient["batch"]["input_ids"], recipient["start"],
                )
                checks[role] = check
            patched_margin = answer_margin(patched_scores["correct"], patched_scores["comparison"])
            raw = patched_margin - baseline_margins[recipient_state]
            if not math.isfinite(raw):
                raise AssertionError("non-finite source-bundle effect")
            records.append({
                "intervention_type": "visual_source_bundle", "layer": None,
                "components": ["initial_visual_embedding", "deepstack_0", "deepstack_1", "deepstack_2"],
                "region": region, "direction": direction, "recipient_state": recipient_state,
                "token_count": len(visual_indices), "recipient_margin": baseline_margins[recipient_state],
                "patched_margin": patched_margin, "raw_margin_change": raw,
                "oriented_effect": oriented(raw, direction, args.variant == "correct_answer"),
                "recipient_preference": preference(baseline_margins[recipient_state]),
                "patched_preference": preference(patched_margin),
                "patched_scores": patched_scores, "integrity": checks,
            })

    expected = (len(layers) + 1) * len(REGIONS) * 2
    if len(records) != expected:
        raise AssertionError(f"records={len(records)} expected={expected}")
    return {
        "status": "complete", "step": args.step, "model_id": args.model_id,
        "model_revision": REVISIONS[args.model_id], "question_id": qid,
        "dataset_index": int(entry["dataset_index"]), "variant": args.variant,
        "comparison_key": comparison_key, "answers": answers,
        "comparator_selection_scores": comparator_selection_scores,
        "layers": list(range(len(layers))), "regions": {name: regions[name] for name in REGIONS},
        "mapping": spatial["no_text"][1]["summary"], "baseline_scores": baseline_scores,
        "baseline_margins": baseline_margins, "cleaned_overlay_validation": clean,
        "noop": {state: {role: run["noop_max_logit_difference"] for role, run in roles.items()}
                 for state, roles in runs.items()},
        "records": records, "expected_records": expected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct", choices=tuple(MODEL_LAYERS))
    parser.add_argument("--step", default="4")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--expected_questions", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--region_seed", type=int, default=271828)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    expected_layers = MODEL_LAYERS[args.model_id]
    if not 0 <= args.shard_id < args.num_shards:
        raise AssertionError("invalid shard")
    manifest, entries = load_selection(
        args.selection, args.shard_id, args.num_shards, args.expected_questions,
    )
    if args.max_samples:
        entries = entries[:args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "status": "configured", "step": args.step, "model_id": args.model_id,
        "model_revision": REVISIONS[args.model_id], "variant": args.variant,
        "selection": str(args.selection),
        "selection_hash": manifest.get("question_id_sha256", hashlib.sha256("\n".join(
            str(row["question_id"]) for row in manifest["selected_samples"]
        ).encode()).hexdigest()),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "question_ids": [str(row["question_id"]) for row in entries],
        "layers": list(range(expected_layers)), "regions": list(REGIONS),
        "directions": ["restoration", "insertion"], "hook": "resid_pre",
        "mapping": "any_positive_bbox_intersection", "region_seed": args.region_seed,
    }
    save_json(args.output_dir / "configuration.json", config)
    log("CONFIG", f"model={args.model_id} revision={REVISIONS[args.model_id]} variant={args.variant} shard={args.shard_id}/{args.num_shards}")
    log("CONFIG", f"questions={len(entries)} layers=0..{expected_layers-1} regions={REGIONS} directions=restoration,insertion")
    log("RUNTIME", f"torch={torch.__version__} transformers={transformers.__version__} device={args.device} dtype=float16")
    expected_per_question = (expected_layers + 1) * len(REGIONS) * 2
    log("EXPECTED", f"{expected_per_question} records/question; no-op<=1e-3; exact patched values; random counts equal text; finite effects")
    if args.preflight_only:
        log("PASS", f"Step {args.step} preflight complete without loading model weights")
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, revision=REVISIONS[args.model_id], dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, revision=REVISIONS[args.model_id])
    layers = model.model.language_model.layers
    if len(layers) != expected_layers:
        raise AssertionError(f"expected {expected_layers} decoder layers, found {len(layers)}")
    dataset = load_from_disk(str(CACHE))
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    completed = resumed = total_records = 0
    for number, entry in enumerate(entries, 1):
        qid = str(entry["question_id"])
        path = samples_dir / f"{qid}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("status") != "complete" or saved.get("expected_records") != len(saved.get("records", [])):
                raise AssertionError(f"invalid checkpoint: {path}")
            completed += 1
            resumed += 1
            total_records += len(saved["records"])
            log("RESUME", f"question={number}/{len(entries)} qid={qid} records={len(saved['records'])}")
            continue
        started = time.perf_counter()
        sample = dataset[int(entry["dataset_index"])]
        if str(sample["question_id"]) != qid:
            raise AssertionError("dataset index does not match frozen question ID")
        log("SAMPLE", f"start={number}/{len(entries)} qid={qid}")
        report = run_sample(args, model, processor, layers, sample, entry)
        save_json(path, report)
        completed += 1
        total_records += len(report["records"])
        log("CHECKPOINT", f"completed={completed}/{len(entries)} qid={qid} records={total_records} path={path}")
        log("TIMING", f"qid={qid} seconds={time.perf_counter()-started:.2f}")
    completion = {
        "status": "complete", "attempted_questions": len(entries),
        "successful_questions": completed, "skipped_questions": 0,
        "resumed_questions": resumed, "saved_records": total_records,
        "expected_records": len(entries) * expected_per_question,
    }
    if completion["saved_records"] != completion["expected_records"]:
        raise AssertionError(f"completion accounting failed: {completion}")
    save_json(args.output_dir / "completion.json", completion)
    log("PASS", f"Step 4 shard complete questions={completed} records={total_records}")


if __name__ == "__main__":
    main()
