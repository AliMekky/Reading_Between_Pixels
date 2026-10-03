#!/usr/bin/env python3
"""One-sample multi-layer and DeepStack-aware activation-patching test."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "vlms"))
sys.path.insert(0, str(ROOT / "vlms/activation_patching/main_files"))

from qwen3_causal_common import cache_multimodal_inputs  # noqa: E402
from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from sequence_scoring import answer_margin, scored_token_logprobs, strongest_incorrect, summarize_answer_score  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (  # noqa: E402
    CACHE, PROMPT_INSTRUCTION, REVISIONS, SELECTION, VARIANTS, log, prepare, save_json,
)


REGIONS = (
    "text_region", "matched_random_region_1", "matched_random_region_2",
    "matched_random_region_3", "all_image_tokens",
)


def layer_ids(layer_count: int):
    values = [0, 1, 2, 3, layer_count // 4, layer_count // 2, 3 * layer_count // 4, layer_count - 1]
    return sorted(set(values))


def score_from_logits(processor, answer, logits, input_ids, answer_start):
    ids = input_ids[0, answer_start:].tolist()
    return summarize_answer_score(
        answer, ids,
        [processor.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in ids],
        scored_token_logprobs(logits, input_ids, answer_start),
    )


def cached_logits(model, cached):
    hidden = model.language_model(**cached, use_cache=False).last_hidden_state
    return model.lm_head(hidden)


@torch.inference_mode()
def capture_resid_pre(model, cached: Dict, layers: Iterable[int]):
    captured, handles = {}, []
    for layer in layers:
        def hook(_module, args, layer=layer):
            captured[layer] = args[0].detach().clone()
        handles.append(model.model.language_model.layers[layer].register_forward_pre_hook(hook))
    try:
        logits = cached_logits(model, cached).detach()
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


@torch.inference_mode()
def patch_one_layer(model, cached, layer, positions, donor):
    integrity = {}

    def hook(_module, args):
        original = args[0]
        patched = original.clone()
        donor_values = donor[:, positions].to(patched.device, patched.dtype)
        recipient_values = original[:, positions]
        patched[:, positions] = donor_values
        direct = (patched - original).abs()
        direct[:, positions] = 0
        integrity.update({
            "donor_recipient_mean_difference": float((donor_values - recipient_values).abs().mean()),
            "donor_recipient_max_difference": float((donor_values - recipient_values).abs().max()),
            "patched_to_donor_max_difference": float((patched[:, positions] - donor_values).abs().max()),
            "unpatched_direct_max_change": float(direct.max()),
        })
        return (patched,) + args[1:]

    handle = model.model.language_model.layers[layer].register_forward_pre_hook(hook)
    try:
        logits = cached_logits(model, cached).detach()
    finally:
        handle.remove()
    return logits, integrity


def replace_positions(recipient, donor, positions):
    patched = recipient.clone()
    donor_values = donor[positions].to(patched.device, patched.dtype)
    before = recipient[positions]
    patched[positions] = donor_values
    direct = (patched - recipient).abs()
    direct[positions] = 0
    return patched, {
        "position_count": len(positions),
        "donor_recipient_mean_difference": float((donor_values - before).abs().mean()),
        "donor_recipient_max_difference": float((donor_values - before).abs().max()),
        "patched_to_donor_max_difference": float((patched[positions] - donor_values).abs().max()),
        "unpatched_direct_max_change": float(direct.max()),
    }


@torch.inference_mode()
def patch_visual_source_bundle(model, recipient, donor, sequence_positions, visual_indices):
    patched = dict(recipient)
    initial, initial_check = replace_positions(
        recipient["inputs_embeds"][0], donor["inputs_embeds"][0], sequence_positions
    )
    patched["inputs_embeds"] = initial.unsqueeze(0)
    patched_deepstack, checks = [], []
    for recipient_stream, donor_stream in zip(
        recipient["deepstack_visual_embeds"], donor["deepstack_visual_embeds"]
    ):
        stream, check = replace_positions(recipient_stream, donor_stream, visual_indices)
        patched_deepstack.append(stream)
        checks.append(check)
    patched["deepstack_visual_embeds"] = patched_deepstack
    logits = cached_logits(model, patched).detach()
    return logits, {"initial_visual_embedding": initial_check, "deepstack_additions": checks}


def candidate_batch(processor, image, prompt, answer, device, prompt_ids):
    batch, _ = prepare(processor, image, prompt, answer, device)
    start = int(prompt_ids.shape[1])
    if not torch.equal(batch["input_ids"][:, :start], prompt_ids):
        raise AssertionError("answer suffix changed prompt tokens")
    return batch, start


def oriented(raw_change, direction, correct_overlay):
    if correct_overlay:
        return raw_change if direction == "insertion" else -raw_change
    return -raw_change if direction == "insertion" else raw_change


def validate_integrity(check):
    if check["patched_to_donor_max_difference"] > 1e-6 or check["unpatched_direct_max_change"] > 1e-6:
        raise AssertionError(f"patch integrity failed: {check}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, choices=tuple(REVISIONS))
    parser.add_argument("--variant", default="misleading_groundable", choices=VARIANTS)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    manifest = json.loads(SELECTION.read_text())
    selected = {str(row["question_id"]): int(row["dataset_index"]) for row in manifest["selected_samples"]}
    if len(selected) != 305 or args.question_id not in selected:
        raise AssertionError("invalid shared-305 selection")
    sample = load_from_disk(str(CACHE))[selected[args.question_id]]
    revision = REVISIONS[args.model_id]
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, revision=revision, dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, revision=revision)
    layers = layer_ids(len(model.model.language_model.layers))
    log("CONFIG", f"step=3 model={args.model_id} revision={revision} variant={args.variant} qid={args.question_id}")
    log("CONFIG", f"layers={layers} directions=restoration,insertion regions={REGIONS} plus=visual_source_bundle")
    log("EXPECTED", "all scores finite; no-op <=1e-3; patch-to-donor=0; unpatched direct change=0; no directional effect is required")

    images = {
        "no_text": sample["notext"]["image"].convert("RGB"),
        "overlay": sample[args.variant]["cleaned_image"].convert("RGB"),
    }
    clean = validate_cleaned_overlay(
        images["no_text"], sample[args.variant]["image"].convert("RGB"),
        images["overlay"], text_bbox_yxyx(sample, args.variant),
    )
    if clean["cleaned_outside_no_text_mismatch_pixels"] or clean["cleaned_inside_original_mismatch_pixels"]:
        raise AssertionError("cleaned image validation failed")

    prompt_batches, spatial = {}, {}
    for state, image in images.items():
        prompt_batches[state], _ = prepare(processor, image, prompt, "", args.device)
        spatial[state] = positive_overlap_region_positions(
            model, prompt_batches[state], image, processor, sample, args.variant, args.seed
        )
    if not torch.equal(prompt_batches["no_text"]["input_ids"], prompt_batches["overlay"]["input_ids"]):
        raise AssertionError("paired prompt IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0]:
        raise AssertionError("paired image positions differ")
    regions = spatial["no_text"][2]
    sequence_regions = spatial["no_text"][3]
    if any(name not in regions for name in REGIONS):
        raise AssertionError("required region missing")
    log("MAPPING", f"grid={spatial['no_text'][1]['summary']['merged_grid_hw']} image_tokens={len(spatial['no_text'][0])} regions={[(name,len(regions[name]['token_indices'])) for name in REGIONS]}")

    references = {key: str(sample[key]["text"]) for key in VARIANTS}
    comparison_key = args.variant
    if args.variant == "correct_answer":
        scores = {}
        for key, answer in references.items():
            batch, start = candidate_batch(processor, images["no_text"], prompt, answer, args.device, prompt_batches["no_text"]["input_ids"])
            logits = model(**batch, use_cache=False).logits
            scores[key] = score_from_logits(processor, answer, logits, batch["input_ids"], start)
        comparison_key = strongest_incorrect(scores)
    answers = {"correct": references["correct_answer"], "comparison": references[comparison_key]}

    runs = {state: {} for state in images}
    for state, image in images.items():
        for role, answer in answers.items():
            batch, start = candidate_batch(processor, image, prompt, answer, args.device, prompt_batches[state]["input_ids"])
            full_logits = model(**batch, use_cache=False).logits.detach()
            cached = cache_multimodal_inputs(model, batch)
            noop_logits, hidden = capture_resid_pre(model, cached, layers)
            error = float((full_logits - noop_logits).abs().max())
            if error > 1e-3:
                raise AssertionError(f"cached no-op failed state={state} role={role}: {error}")
            runs[state][role] = {
                "batch": batch, "cached": cached, "start": start, "hidden": hidden,
                "score": score_from_logits(processor, answer, noop_logits, batch["input_ids"], start),
                "noop_max_logit_difference": error,
            }
            log("NOOP", f"state={state} role={role} max_logit_difference={error:.3g}")

    baseline_margins = {
        state: answer_margin(runs[state]["correct"]["score"], runs[state]["comparison"]["score"])
        for state in images
    }
    records = []
    for layer in layers:
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
                        processor, answer, logits, recipient["batch"]["input_ids"], recipient["start"]
                    )
                    checks[role] = check
                margin = answer_margin(patched_scores["correct"], patched_scores["comparison"])
                raw = margin - baseline_margins[recipient_state]
                if not math.isfinite(raw):
                    raise AssertionError("non-finite layer effect")
                records.append({
                    "intervention_type": "single_layer_resid_pre", "layer": layer,
                    "region": region, "direction": direction, "recipient_state": recipient_state,
                    "token_count": len(sequence_regions[region]), "recipient_margin": baseline_margins[recipient_state],
                    "patched_margin": margin, "raw_margin_change": raw,
                    "oriented_effect": oriented(raw, direction, args.variant == "correct_answer"),
                    "recipient_preference": "correct" if baseline_margins[recipient_state] > 0 else "comparison",
                    "patched_preference": "correct" if margin > 0 else "comparison", "integrity": checks,
                })
                log("PATCH", f"type=resid_pre layer={layer} region={region} direction={direction} raw={raw:.6f} oriented={records[-1]['oriented_effect']:.6f}")

    for region in REGIONS:
        visual_indices = regions[region]["token_indices"]
        for direction, recipient_state, donor_state in (
            ("restoration", "overlay", "no_text"), ("insertion", "no_text", "overlay"),
        ):
            patched_scores, checks = {}, {}
            for role, answer in answers.items():
                recipient, donor = runs[recipient_state][role], runs[donor_state][role]
                logits, check = patch_visual_source_bundle(
                    model, recipient["cached"], donor["cached"], sequence_regions[region], visual_indices
                )
                validate_integrity(check["initial_visual_embedding"])
                for stream_check in check["deepstack_additions"]:
                    validate_integrity(stream_check)
                patched_scores[role] = score_from_logits(
                    processor, answer, logits, recipient["batch"]["input_ids"], recipient["start"]
                )
                checks[role] = check
            margin = answer_margin(patched_scores["correct"], patched_scores["comparison"])
            raw = margin - baseline_margins[recipient_state]
            if not math.isfinite(raw):
                raise AssertionError("non-finite source-bundle effect")
            records.append({
                "intervention_type": "visual_source_bundle", "layer": None,
                "components": ["initial_visual_embedding", "deepstack_0", "deepstack_1", "deepstack_2"],
                "region": region, "direction": direction, "recipient_state": recipient_state,
                "token_count": len(visual_indices), "recipient_margin": baseline_margins[recipient_state],
                "patched_margin": margin, "raw_margin_change": raw,
                "oriented_effect": oriented(raw, direction, args.variant == "correct_answer"),
                "recipient_preference": "correct" if baseline_margins[recipient_state] > 0 else "comparison",
                "patched_preference": "correct" if margin > 0 else "comparison", "integrity": checks,
            })
            log("PATCH", f"type=source_bundle region={region} direction={direction} raw={raw:.6f} oriented={records[-1]['oriented_effect']:.6f}")

    expected_records = (len(layers) + 1) * len(REGIONS) * 2
    if len(records) != expected_records:
        raise AssertionError(f"record count {len(records)} != {expected_records}")
    report = {
        "status": "complete", "step": 3, "model_id": args.model_id, "model_revision": revision,
        "question_id": args.question_id, "variant": args.variant, "comparison_key": comparison_key,
        "answers": answers, "layers": layers, "regions": {name: regions[name] for name in REGIONS},
        "baseline_margins": baseline_margins, "cleaned_overlay_validation": clean,
        "noop": {state: {role: value["noop_max_logit_difference"] for role, value in roles.items()} for state, roles in runs.items()},
        "records": records, "expected_records": expected_records,
    }
    save_json(args.output, report)
    log("OUTPUT", f"saved={args.output} records={len(records)}")
    log("PASS", "multi-layer, bidirectional, control, and DeepStack-aware patching complete")


if __name__ == "__main__":
    main()
