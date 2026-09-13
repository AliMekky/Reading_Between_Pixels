#!/usr/bin/env python3
"""Milestone 3: one real sample, layer, and attention-path intervention."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from attention_edge_mask import add_directed_edge_block


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
ACTIVATION_MAIN = PROJECT_ROOT / "vlms" / "activation_patching" / "main_files"
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


MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"
DATASET_REVISION = "27b45899d1154ef1f08ce5c40d45d2468e4ea3e2"
ACTIVATION_ROOT = PROJECT_ROOT / "vlms" / "activation_patching"
DEFAULT_CACHE = ACTIVATION_ROOT / "hf_dataset_GUIC_cleaned"
DEFAULT_VALIDATION = DEFAULT_CACHE / "remote_validation.json"
DEFAULT_MANIFEST = ACTIVATION_ROOT / "main_files" / "activation_patch_final_selection_shared_305_three_conditions.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "debug_outputs" / "milestone_3_one_sample_t_to_d.json"


def log(section: str, message: str) -> None:
    print("[{}] {}".format(section, message), flush=True)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_qids(path: Path) -> Sequence[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    qids = [str(row["question_id"]) for row in report["selected_samples"]]
    require(len(qids) == 305 and len(set(qids)) == 305, "Shared manifest must contain 305 unique qids")
    require(report["dataset_revision"] == DATASET_REVISION, "Manifest dataset revision mismatch")
    return qids


@torch.no_grad()
def forward_with_edge_hook(
    model: LlavaNextForConditionalGeneration,
    attention_module: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    destination: int,
    blocked_sources: Optional[Sequence[int]],
    audit_sources: Dict[str, Sequence[int]],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run one forward pass while cloning or modifying one layer's additive mask."""
    state: Dict[str, Any] = {"pre_calls": 0, "post_calls": 0}

    def pre_hook(_module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        mask = kwargs.get("attention_mask")
        require(torch.is_tensor(mask) and mask.ndim == 4, "Target attention layer did not receive a 4D mask")
        require(0 <= destination < mask.shape[-2], "Destination query is outside the attention mask")
        original = mask
        combined = original.clone()
        sources = [] if blocked_sources is None else [int(position) for position in blocked_sources]
        if sources:
            require(max(sources) < mask.shape[-1] and min(sources) >= 0, "Source key is outside the attention mask")
            require(bool(torch.all(original[..., destination, sources] == 0)), "Requested edges were already masked")
            combined = add_directed_edge_block(original, sources, [destination])
        require(bool(torch.all(combined[original != 0] == original[original != 0])), "Existing mask entries changed")
        require(bool(torch.any(combined[..., destination, :] == 0)), "Intervention fully masked the destination row")
        kwargs["attention_mask"] = combined
        state.update({
            "pre_calls": state["pre_calls"] + 1,
            "mask_shape": list(mask.shape),
            "blocked_source_count": len(sources),
            "blocked_directed_edges_per_head": len(sources),
            "existing_mask_preserved": True,
            "destination_has_valid_keys": True,
        })
        return args, kwargs

    def post_hook(_module: torch.nn.Module, _args: Tuple[Any, ...], output: Tuple[torch.Tensor, torch.Tensor]):
        weights = output[1]
        require(torch.is_tensor(weights) and weights.ndim == 4, "Eager attention did not return probabilities")
        require(destination < weights.shape[-2], "Destination is outside returned attention")
        destination_weights = weights[0, :, destination, :]
        state["post_calls"] += 1
        state["attention_shape"] = list(weights.shape)
        state["destination_row_sum_max_error"] = float((destination_weights.sum(-1) - 1.0).abs().max().item())
        state["audit"] = {}
        for name, positions in audit_sources.items():
            indices = torch.as_tensor(list(positions), dtype=torch.long, device=weights.device)
            values = destination_weights[:, indices]
            state["audit"][name] = {
                "source_count": int(indices.numel()),
                "attention_mass_mean_over_heads": float(values.sum(-1).mean().item()),
                "probability_max": float(values.abs().max().item()),
            }

    pre_handle = attention_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = attention_module.register_forward_hook(post_hook)
    try:
        logits = forward_next_token_logits(model, inputs)
    finally:
        pre_handle.remove()
        post_handle.remove()
    require(state["pre_calls"] == 1 and state["post_calls"] == 1, "Attention hooks must each run exactly once")
    require(state["destination_row_sum_max_error"] <= 2e-3, "Attention row failed to renormalize")
    if blocked_sources:
        for name, positions in audit_sources.items():
            if list(positions) == list(blocked_sources):
                require(state["audit"][name]["probability_max"] == 0.0, "Blocked attention probability is nonzero")
    return logits, state


def log_choice_logits(name: str, summary: Dict[str, Any]) -> None:
    values = summary["choice_logits"]
    log("RESULT", "{} prediction={} margin={:.6f} logits={}".format(
        name,
        summary["choice_constrained_prediction"],
        summary["margin_correct_minus_misleading"],
        ", ".join("{}:{:.4f}".format(letter, values[letter]) for letter in ANSWER_LETTERS),
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--variant", default="misleading_groundable", choices=("misleading_groundable", "misleading_ungroundable", "irrelevant_word"))
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--model_id", default=MODEL)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--hf_cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset_validation_file", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--selection_manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require(torch.cuda.is_available(), "CUDA is required for the real-model test")
    require(args.dataset_revision == DATASET_REVISION, "Dataset revision is not the approved revision")
    require(args.model_revision == MODEL_REVISION, "Model revision is not the approved revision")
    require(args.question_id in selected_qids(args.selection_manifest), "qid is outside the shared 305-sample subset")
    provenance = validate_dataset_provenance(str(args.dataset_validation_file), args.dataset_revision)
    log("CONFIG", "continuation=activation_patching model={} revision={}".format(args.model_id, args.model_revision))
    log("CONFIG", "dataset_revision={} qid={} variant={} layer={} path=T->D control=R1->D".format(
        args.dataset_revision, args.question_id, args.variant, args.layer
    ))
    log("CONFIG", "backend=eager device=cuda dtype=float16 seed={} image_field=cleaned_image".format(args.seed))
    log("EXPECTED", "baseline=no-op; T and R1 have matched token composition; blocked probabilities=0")

    dataset = get_or_download_hf_dataset("AHAAM/GUIC", str(args.hf_cache_dir), "test", args.dataset_revision)
    sample = find_sample_by_qid(dataset, args.question_id)
    image = sample[args.variant]["cleaned_image"].convert("RGB")
    original_overlay = sample[args.variant]["image"].convert("RGB")
    no_text = sample["notext"]["image"].convert("RGB")
    cleaned = validate_cleaned_overlay(no_text, original_overlay, image, text_bbox_yxyx(sample, args.variant))
    require(cleaned["cleaned_outside_no_text_mismatch_pixels"] == 0, "Cleaned overlay differs outside text bbox")
    require(cleaned["cleaned_inside_original_mismatch_pixels"] == 0, "Cleaned overlay differs from original inside text bbox")
    log("PASS", "cleaned image and remote provenance validated; pairs={}".format(provenance["pairs_validated"]))

    log("MODEL", "loading pinned model with attn_implementation=eager")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    processor.patch_size = int(model.config.vision_config.patch_size)
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    layers, layer_path = get_decoder_layers(model)
    require(0 <= args.layer < len(layers), "Layer is outside decoder")
    require(model.config.text_config._attn_implementation == "eager", "Text model did not resolve eager attention")
    attention_module = layers[args.layer].self_attn
    log("PASS", "decoder={} layers={} target_attention={} backend=eager".format(layer_path, len(layers), type(attention_module).__name__))

    options, correct_letter, option_meta = build_options(sample, shuffle=True, seed=args.seed)
    misleading_letter = option_meta["key_to_label"][args.variant]
    prompt = format_mcq_prompt(str(sample["question"]), options)
    answer_ids = {letter: answer_token_id(processor.tokenizer, letter)[0] for letter in ANSWER_LETTERS}
    require(len(set(answer_ids.values())) == 4, "A-D token IDs are not distinct")
    inputs, formatted = prepare_inputs(processor, image, prompt, torch.device("cuda"), torch.float16)
    image_positions = image_placeholder_positions(model, inputs["input_ids"])
    image_size = tuple(int(value) for value in inputs["image_sizes"][0].tolist())
    views = int(inputs["pixel_values"].shape[1])
    mapping = build_packed_token_mapping(model, image_size, views)
    require(len(image_positions) == mapping["summary"]["total_packed_image_tokens"], "Packed mapping count mismatch")
    regions = build_regions_full_coverage(
        sample, args.variant, mapping["tokens"], {"base", "mosaic"}, 0.25, image_size, args.seed
    )
    text_packed = regions["text_region"]["token_indices"]
    random_packed = regions["matched_random_region_1"]["token_indices"]
    require(len(text_packed) == len(random_packed), "Text and random token counts differ")
    require(kind_counts(text_packed, mapping["tokens"]) == kind_counts(random_packed, mapping["tokens"]), "Text/random compositions differ")
    require(set(text_packed).isdisjoint(random_packed), "Text and random controls overlap")
    text_sequence = [image_positions[index] for index in text_packed]
    random_sequence = [image_positions[index] for index in random_packed]
    destination = int(inputs["input_ids"].shape[1] - 1)
    require(max(text_sequence + random_sequence) < destination, "Source tokens are not causally before D")
    log("TOKENS", "sequence={} image={} D={} T={} R1={} composition={}".format(
        inputs["input_ids"].shape[1], len(image_positions), destination,
        len(text_sequence), len(random_sequence), kind_counts(text_packed, mapping["tokens"])
    ))
    log("TOKENS", "D_decoded={!r} formatted_prompt_chars={}".format(
        processor.tokenizer.decode([int(inputs["input_ids"][0, destination])], skip_special_tokens=False), len(formatted)
    ))

    baseline_logits = forward_next_token_logits(model, inputs)
    noop_logits, noop_integrity = forward_with_edge_hook(
        model, attention_module, inputs, destination, None,
        {"text_region": text_sequence, "matched_random_region_1": random_sequence},
    )
    noop_difference = float((baseline_logits - noop_logits).abs().max().item())
    require(noop_difference <= 1e-3, "Empty-mask no-op changed logits beyond tolerance")
    require(noop_integrity["audit"]["text_region"]["attention_mass_mean_over_heads"] > 0.0, "T->D had zero baseline attention")
    require(noop_integrity["audit"]["matched_random_region_1"]["attention_mass_mean_over_heads"] > 0.0, "R1->D had zero baseline attention")
    log("NOOP", "max_full_vocab_logit_difference={:.6g} prediction_expected_unchanged".format(noop_difference))

    text_logits, text_integrity = forward_with_edge_hook(
        model, attention_module, inputs, destination, text_sequence, {"blocked_text_region": text_sequence}
    )
    random_logits, random_integrity = forward_with_edge_hook(
        model, attention_module, inputs, destination, random_sequence, {"blocked_random_region_1": random_sequence}
    )
    summaries = {
        "baseline": summarize_logits(baseline_logits, processor.tokenizer, answer_ids, correct_letter, misleading_letter),
        "empty_mask_noop": summarize_logits(noop_logits, processor.tokenizer, answer_ids, correct_letter, misleading_letter),
        "block_T_to_D": summarize_logits(text_logits, processor.tokenizer, answer_ids, correct_letter, misleading_letter),
        "block_R1_to_D": summarize_logits(random_logits, processor.tokenizer, answer_ids, correct_letter, misleading_letter),
    }
    require(summaries["baseline"]["choice_constrained_prediction"] == summaries["empty_mask_noop"]["choice_constrained_prediction"], "No-op changed prediction")
    for name, summary in summaries.items():
        require(all(math.isfinite(value) for value in summary["choice_logits"].values()), "Non-finite logits in {}".format(name))
        log_choice_logits(name, summary)

    baseline_margin = summaries["baseline"]["margin_correct_minus_misleading"]
    effects = {
        "block_T_to_D_margin_change": summaries["block_T_to_D"]["margin_correct_minus_misleading"] - baseline_margin,
        "block_R1_to_D_margin_change": summaries["block_R1_to_D"]["margin_correct_minus_misleading"] - baseline_margin,
    }
    effects["text_minus_random_margin_change"] = effects["block_T_to_D_margin_change"] - effects["block_R1_to_D_margin_change"]
    log("EFFECT", "T->D={:.6f} R1->D={:.6f} text-minus-random={:.6f}".format(
        effects["block_T_to_D_margin_change"], effects["block_R1_to_D_margin_change"], effects["text_minus_random_margin_change"]
    ))

    report = {
        "schema_version": 1,
        "continuation_of": "activation_patching",
        "configuration": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "dataset": "AHAAM/GUIC",
            "dataset_revision": args.dataset_revision,
            "question_id": str(sample["question_id"]),
            "variant": args.variant,
            "layer": args.layer,
            "backend": "eager",
            "dtype": "float16",
            "seed": args.seed,
            "image_field": "cleaned_image",
        },
        "tokens": {
            "correct_letter": correct_letter,
            "misleading_letter": misleading_letter,
            "answer_token_ids": answer_ids,
            "sequence_length": int(inputs["input_ids"].shape[1]),
            "image_placeholder_count": len(image_positions),
            "decision_position": destination,
            "decision_token_decoded": processor.tokenizer.decode([int(inputs["input_ids"][0, destination])], skip_special_tokens=False),
            "text_region": {"packed": text_packed, "sequence": text_sequence, "kind_counts": kind_counts(text_packed, mapping["tokens"])},
            "matched_random_region_1": {"packed": random_packed, "sequence": random_sequence, "kind_counts": kind_counts(random_packed, mapping["tokens"])},
        },
        "validation": {
            "dataset_provenance": provenance,
            "cleaned_overlay": cleaned,
            "noop_max_full_vocab_logit_difference": noop_difference,
            "noop_tolerance": 1e-3,
            "empty_mask_hook": noop_integrity,
            "block_T_to_D": text_integrity,
            "block_R1_to_D": random_integrity,
        },
        "runs": summaries,
        "effects": effects,
        "all_validation_passed": True,
    }
    write_json(args.output, report)
    log("OUTPUT", str(args.output))
    log("COMPLETE", "Milestone 3 one-sample attention-path validation passed")


if __name__ == "__main__":
    main()
