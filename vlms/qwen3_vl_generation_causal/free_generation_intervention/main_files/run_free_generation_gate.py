#!/usr/bin/env python3
"""Validate cached free generation with late Qwen3-VL text-to-answer blocking."""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [
    str(ROOT / "vlms"),
    str(ROOT / "vlms/activation_patching/main_files"),
    str(ROOT / "vlms/qwen3_vl_generation_causal/main_files"),
    str(ROOT / "vlms/attention_path_intervention/main_files"),
]

from activation_patch_llava_next_debug import text_bbox_yxyx, validate_cleaned_overlay  # noqa: E402
from attention_edge_mask import add_directed_edge_block  # noqa: E402
from spatial_mapping import positive_overlap_region_positions  # noqa: E402
from validate_multitoken_scoring import (  # noqa: E402
    CACHE, PROMPT_INSTRUCTION, REVISIONS, SELECTION, VARIANTS,
    image_hash, log, prepare, previous_record, save_json,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LAYERS = list(range(30, 36))
REGIONS = ("text_region", "matched_random_region_1", "matched_random_region_2",
           "matched_random_region_3")


def clone_batch(batch):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def decode_generation(model, processor, generated, prompt_length, max_new_tokens):
    token_ids = [int(value) for value in generated.sequences[0, prompt_length:].tolist()]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    return {
        "raw_response": processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
        "output_token_ids": token_ids,
        "output_token_count": len(token_ids),
        "termination_reason": (
            "eos" if token_ids and token_ids[-1] in eos_ids
            else "max_tokens" if len(token_ids) >= max_new_tokens else "model_stop"
        ),
    }


@torch.inference_mode()
def generate_with_block(model, processor, inputs, layers, sources, max_new_tokens):
    """Block fixed source keys at answer queries; optionally supply sources by layer."""
    states, handles = {}, []
    for layer in layers:
        attention = layer.self_attn
        layer_id = int(attention.layer_idx)
        layer_sources = sources.get(layer_id, []) if isinstance(sources, dict) else sources
        source_ids = [] if layer_sources is None else sorted(set(int(value) for value in layer_sources))
        state = {"layer": layer_id, "calls": []}
        states[layer_id] = state

        def pre_hook(_module, args, kwargs, state=state, source_ids=source_ids):
            mask = kwargs.get("attention_mask")
            if not torch.is_tensor(mask) or mask.ndim != 4:
                raise AssertionError(f"layer {state['layer']} lacks a 4D attention mask")
            query_count, key_count = map(int, mask.shape[-2:])
            destination = query_count - 1
            if source_ids:
                if min(source_ids) < 0 or max(source_ids) >= key_count:
                    raise AssertionError("source lies outside the cached key axis")
                if not bool(torch.all(mask[..., destination, source_ids] == 0)):
                    raise AssertionError("a requested text/random edge was already masked")
                kwargs["attention_mask"] = add_directed_edge_block(mask, source_ids, [destination])
            state["current"] = {
                "query_count": query_count,
                "key_count": key_count,
                "destination": destination,
            }
            return args, kwargs

        def post_hook(_module, _args, output, state=state, source_ids=source_ids):
            weights = output[1]
            if not torch.is_tensor(weights) or weights.ndim != 4:
                raise AssertionError("eager attention returned no attention probabilities")
            current = state.pop("current")
            row = weights[0, :, current["destination"], :]
            current["row_sum_max_error"] = float((row.sum(-1) - 1).abs().max())
            current["blocked_probability_max"] = (
                float(row[:, source_ids].abs().max()) if source_ids else 0.0
            )
            state["calls"].append(current)

        handles.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(attention.register_forward_hook(post_hook))

    prompt_length = int(inputs["input_ids"].shape[1])
    try:
        generated = model.generate(
            **clone_batch(inputs), max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, return_dict_in_generate=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    generation = decode_generation(model, processor, generated, prompt_length, max_new_tokens)
    call_counts = {layer: len(state["calls"]) for layer, state in states.items()}
    if len(set(call_counts.values())) != 1 or next(iter(call_counts.values())) != generation["output_token_count"]:
        raise AssertionError(f"hook/generation call mismatch: {call_counts}")
    calls = [call for state in states.values() for call in state["calls"]]
    blocked_max = max(call["blocked_probability_max"] for call in calls)
    row_error = max(call["row_sum_max_error"] for call in calls)
    if blocked_max != 0 or row_error > 2e-3:
        raise AssertionError(f"attention integrity failed: blocked={blocked_max} row={row_error}")
    if generation["output_token_count"] > 1 and not any(
        call["query_count"] == 1 and call["key_count"] > prompt_length for call in calls
    ):
        raise AssertionError("cached single-query decoding was not observed")
    generation["integrity"] = {
        "layers": sorted(states), "calls_per_layer": call_counts,
        "max_blocked_probability": blocked_max,
        "max_attention_row_sum_error": row_error,
        "first_mask_shape_qk": [calls[0]["query_count"], calls[0]["key_count"]],
        "last_mask_shape_qk": [calls[-1]["query_count"], calls[-1]["key_count"]],
        "cached_single_query_observed": any(call["query_count"] == 1 for call in calls),
    }
    return generation


@torch.inference_mode()
def ordinary_generation(model, processor, inputs, max_new_tokens):
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **clone_batch(inputs), max_new_tokens=max_new_tokens, do_sample=False,
        use_cache=True, return_dict_in_generate=True,
    )
    return decode_generation(model, processor, generated, prompt_length, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="misleading_groundable", choices=VARIANTS)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = {str(row["question_id"]): int(row["dataset_index"])
                for row in json.loads(SELECTION.read_text())["selected_samples"]}
    if len(selected) != 305 or args.question_id not in selected:
        raise AssertionError("question must belong to the shared 305-question sample")
    sample = load_from_disk(str(CACHE))[selected[args.question_id]]
    prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
    if "Options:" in prompt or "A, B, C, or D" in prompt:
        raise AssertionError("MCQ content leaked into the generation prompt")

    revision = REVISIONS[MODEL_ID]
    log("CONFIG", f"model={MODEL_ID} revision={revision} variant={args.variant} qid={args.question_id}")
    log("CONFIG", f"path=T->current_answer_query layers={LAYERS} regions={REGIONS} cache=True")
    log("EXPECTED", "SDPA=eager=noop tokens; blocked probability=0; row error<=2e-3; cached q=1 observed")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    layers = model.model.language_model.layers
    if len(layers) != 36:
        raise AssertionError(f"expected 36 layers, found {len(layers)}")

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

    batches, spatial = {}, {}
    for state, image in images.items():
        batches[state], _ = prepare(processor, image, prompt, "", "cuda")
        spatial[state] = positive_overlap_region_positions(
            model, batches[state], image, processor, sample, args.variant,
            args.seed * 100000 + selected[args.question_id],
        )
    if not torch.equal(batches["no_text"]["input_ids"], batches["overlay"]["input_ids"]):
        raise AssertionError("paired prompt token IDs differ")
    if spatial["no_text"][0] != spatial["overlay"][0]:
        raise AssertionError("paired visual sequence positions differ")
    sequence_regions = spatial["no_text"][3]
    text_count = len(sequence_regions["text_region"])
    if not text_count or any(
        len(sequence_regions[f"matched_random_region_{index}"]) != text_count for index in (1, 2, 3)
    ):
        raise AssertionError("text/random region counts are invalid")
    log("MAPPING", f"text_tokens={text_count} random_tokens={[len(sequence_regions[x]) for x in REGIONS[1:]]}")

    report = {
        "status": "running", "model_id": MODEL_ID, "model_revision": revision,
        "question_id": args.question_id, "variant": args.variant, "question": str(sample["question"]),
        "prompt": prompt, "layers": LAYERS, "max_new_tokens": args.max_new_tokens,
        "region_counts": {name: len(sequence_regions[name]) for name in REGIONS},
        "cleaned_overlay_validation": clean, "states": {},
    }
    for state in ("no_text", "overlay"):
        prior_variant = "notext" if state == "no_text" else args.variant
        model.set_attn_implementation("sdpa")
        sdpa = ordinary_generation(model, processor, batches[state], args.max_new_tokens)
        prior = previous_record(MODEL_ID, prior_variant, args.question_id)
        prior_check = {
            "available": prior is not None,
            "revision_match": prior is not None and prior.get("model_revision") == revision,
            "image_hash_match": prior is not None and prior.get("image_sha256") == image_hash(images[state]),
            "tokens_match": prior is not None and prior.get("output_token_ids") == sdpa["output_token_ids"],
        }
        if prior_check["revision_match"] and prior_check["image_hash_match"] and not prior_check["tokens_match"]:
            raise AssertionError(f"{state}: SDPA generation does not reproduce the prior run")

        model.set_attn_implementation("eager")
        eager = ordinary_generation(model, processor, batches[state], args.max_new_tokens)
        noop = generate_with_block(
            model, processor, batches[state], [layers[index] for index in LAYERS], None,
            args.max_new_tokens,
        )
        if sdpa["output_token_ids"] != eager["output_token_ids"]:
            raise AssertionError(f"{state}: eager generation differs from SDPA")
        if eager["output_token_ids"] != noop["output_token_ids"]:
            raise AssertionError(f"{state}: empty attention hook changes generation")

        interventions = {}
        for region in REGIONS:
            result = generate_with_block(
                model, processor, batches[state], [layers[index] for index in LAYERS],
                sequence_regions[region], args.max_new_tokens,
            )
            interventions[region] = result
            log("GENERATION", f"state={state} region={region} response={result['raw_response']!r} "
                              f"tokens={result['output_token_count']} changed={result['output_token_ids'] != eager['output_token_ids']}")
        report["states"][state] = {
            "image_sha256": image_hash(images[state]), "prior_check": prior_check,
            "sdpa_baseline": sdpa, "eager_baseline": eager, "empty_hook_noop": noop,
            "interventions": interventions,
        }
        save_json(args.output, report)

    report["status"] = "complete"
    report["validation"] = {
        "paired_prompt_ids_equal": True, "paired_visual_positions_equal": True,
        "sdpa_eager_tokens_equal": True, "eager_noop_tokens_equal": True,
        "all_blocked_probabilities_zero": True, "all_row_errors_within_2e3": True,
    }
    save_json(args.output, report)
    log("OUTPUT", f"saved={args.output}")
    log("PASS", "free-generation attention-intervention gate complete")


if __name__ == "__main__":
    main()
