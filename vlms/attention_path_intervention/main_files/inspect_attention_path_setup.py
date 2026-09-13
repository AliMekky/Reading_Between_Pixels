#!/usr/bin/env python3
"""Milestone 1: inspect attention modules and token groups without inference."""

import os

os.environ.setdefault("USE_TF", "0")

import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch
import transformers
from accelerate import init_empty_weights
from datasets import load_from_disk
from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.models.mistral.modeling_mistral import MistralAttention, eager_attention_forward


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
ACTIVATION_MAIN = PROJECT_ROOT / "vlms" / "activation_patching" / "main_files"
sys.path.insert(0, str(ACTIVATION_MAIN))

from activation_patch_control_pilot import build_regions_full_coverage, kind_counts  # noqa: E402
from activation_patch_llava_next_debug import (  # noqa: E402
    build_options,
    build_packed_token_mapping,
    find_sample_by_qid,
    format_mcq_prompt,
    image_size_to_num_views,
    require,
)


MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
MODEL_REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"
DATASET_REVISION = "27b45899d1154ef1f08ce5c40d45d2468e4ea3e2"
DEFAULT_DATASET = PROJECT_ROOT / "vlms" / "activation_patching" / "hf_dataset_GUIC_cleaned" / "AHAAM__GUIC"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "debug_outputs" / "milestone_1_static_inspection.json"


def log(section: str, message: str) -> None:
    print("[{}] {}".format(section, message), flush=True)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def char_span(text: str, start_marker: str, end_marker: str) -> Tuple[int, int]:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    require(end > start, "Empty span between {!r} and {!r}".format(start_marker, end_marker))
    return start, end


def align_raw_to_expanded(
    raw_ids: Sequence[int], expanded_ids: Sequence[int], image_token_id: int
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Align tokenizer IDs to processor IDs while allowing image expansion."""
    mapping: Dict[int, List[int]] = {}
    expanded_image_positions: List[int] = []
    cursor = 0
    for raw_index, token_id in enumerate(raw_ids):
        if token_id == image_token_id:
            start = cursor
            while cursor < len(expanded_ids) and expanded_ids[cursor] == image_token_id:
                expanded_image_positions.append(cursor)
                cursor += 1
            require(cursor > start, "Raw image token did not expand to image placeholders")
            mapping[raw_index] = list(range(start, cursor))
        else:
            require(cursor < len(expanded_ids), "Expanded sequence ended during alignment")
            require(
                expanded_ids[cursor] == token_id,
                "Token alignment failed at raw={} expanded={}: {} != {}".format(
                    raw_index, cursor, token_id, expanded_ids[cursor]
                ),
            )
            mapping[raw_index] = [cursor]
            cursor += 1
    require(cursor == len(expanded_ids), "Unaligned expanded tokens remain: {}".format(len(expanded_ids) - cursor))
    return mapping, expanded_image_positions


def positions_for_char_span(
    offsets: Sequence[Sequence[int]], raw_to_expanded: Dict[int, List[int]], span: Tuple[int, int]
) -> List[int]:
    start, end = span
    positions: List[int] = []
    for raw_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            positions.extend(raw_to_expanded[raw_index])
    return positions


def decoded_group(tokenizer: Any, input_ids: Sequence[int], positions: Sequence[int]) -> str:
    return tokenizer.decode([input_ids[index] for index in positions], skip_special_tokens=False)


def sequence_region(
    region: Dict[str, Any], image_positions: Sequence[int], mapping_tokens: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    packed = [int(value) for value in region["token_indices"]]
    require(packed, "Visual region contains no packed tokens")
    require(min(packed) >= 0 and max(packed) < len(image_positions), "Packed visual index out of range")
    sequence = [int(image_positions[index]) for index in packed]
    return {
        "available": True,
        "bbox_yxyx": region.get("bbox_yxyx"),
        "packed_token_count": len(packed),
        "packed_token_kind_counts": kind_counts(packed, mapping_tokens),
        "packed_token_indices": packed,
        "sequence_positions": sequence,
        "clean_object_control": region.get("clean_object_control"),
        "text_overlap_token_count": region.get("text_overlap_token_count"),
    }


def edge_summary(name: str, source: Sequence[int], destination: Sequence[int]) -> Dict[str, Any]:
    require(source and destination, "{} has an empty source or destination".format(name))
    causal = all(source_position <= destination_position for destination_position in destination for source_position in source)
    return {
        "path": name,
        "source_positions": list(source),
        "destination_positions": list(destination),
        "query_count": len(destination),
        "key_value_count": len(source),
        "directed_edge_count_per_head": len(source) * len(destination),
        "causally_reachable": causal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=MODEL)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--question_id", default="14412508")
    parser.add_argument("--variant", default="misleading_groundable")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    log("CONFIG", "model={} revision={}".format(args.model_id, args.model_revision))
    log("CONFIG", "dataset={} revision={} qid={} variant={}".format(
        args.dataset_dir, args.dataset_revision, args.question_id, args.variant
    ))
    log("CONFIG", "No model weights, GPU inference, or attention intervention will be used")

    config = LlavaNextConfig.from_pretrained(
        args.model_id, revision=args.model_revision, local_files_only=True
    )
    processor = LlavaNextProcessor.from_pretrained(
        args.model_id, revision=args.model_revision, local_files_only=True
    )
    with init_empty_weights():
        empty_model = LlavaNextForConditionalGeneration(config)
    layers = empty_model.language_model.layers
    attention_classes = sorted({
        "{}.{}".format(type(layer.self_attn).__module__, type(layer.self_attn).__name__)
        for layer in layers
    })
    backends = sorted({str(layer.self_attn.config._attn_implementation) for layer in layers})
    attention_signature = str(inspect.signature(MistralAttention.forward))
    eager_signature = str(inspect.signature(eager_attention_forward))
    eager_source = inspect.getsource(eager_attention_forward)
    additive_mask_before_softmax = (
        "attn_weights = attn_weights + causal_mask" in eager_source
        and eager_source.index("attn_weights = attn_weights + causal_mask")
        < eager_source.index("softmax(attn_weights")
    )
    require(len(layers) == int(config.text_config.num_hidden_layers), "Decoder layer count mismatch")
    require(attention_classes == ["transformers.models.mistral.modeling_mistral.MistralAttention"], "Unexpected attention class")
    require(additive_mask_before_softmax, "Eager attention does not apply an additive mask before softmax")
    log("ATTENTION", "layers={} heads={} kv_heads={} resolved_backend={}".format(
        len(layers), config.text_config.num_attention_heads,
        config.text_config.num_key_value_heads, ",".join(backends)
    ))
    log("ATTENTION", "class={} additive_4d_mask_before_softmax={}".format(
        attention_classes[0], additive_mask_before_softmax
    ))
    log("EXPECTED", "Full interventions must force eager attention for both baselines and blocked runs")

    dataset = load_from_disk(str(args.dataset_dir))
    sample = find_sample_by_qid(dataset, args.question_id)
    qid = str(sample["question_id"])
    options, correct_letter, option_meta = build_options(sample, shuffle=True, seed=args.seed)
    question = str(sample["question"])
    prompt = format_mcq_prompt(question, options)
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image"}],
    }]
    formatted = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    image = sample["notext"]["image"].convert("RGB")
    batch = processor(images=image, text=formatted, return_tensors="pt")
    expanded_ids = [int(value) for value in batch["input_ids"][0].tolist()]
    raw = processor.tokenizer(
        formatted, add_special_tokens=True, return_offsets_mapping=True
    )
    raw_ids = [int(value) for value in raw["input_ids"]]
    offsets = [tuple(map(int, value)) for value in raw["offset_mapping"]]
    image_token_id = int(config.image_token_index)
    raw_to_expanded, image_positions = align_raw_to_expanded(raw_ids, expanded_ids, image_token_id)

    q_span = char_span(formatted, "Question: ", "\n\nOptions:")
    p_span = char_span(formatted, "Options:\n", "\n\nAnswer with only the letter")
    q_positions = positions_for_char_span(offsets, raw_to_expanded, q_span)
    p_positions = positions_for_char_span(offsets, raw_to_expanded, p_span)
    d_positions = [len(expanded_ids) - 1]
    require(q_positions and p_positions, "Question or option token group is empty")
    require(set(q_positions).isdisjoint(p_positions), "Question and option token groups overlap")
    require(max(image_positions) < min(q_positions) < min(p_positions) < d_positions[0], "Token groups violate expected causal order")
    require(decoded_group(processor.tokenizer, expanded_ids, q_positions).strip() == question.strip(), "Q tokens do not decode to the question")
    log("TOKENS", "sequence={} raw={} image_placeholders={} Q={} P={} D={}".format(
        len(expanded_ids), len(raw_ids), len(image_positions),
        len(q_positions), len(p_positions), d_positions[0]
    ))
    log("TOKENS", "Q decoded={!r}".format(decoded_group(processor.tokenizer, expanded_ids, q_positions)))
    log("TOKENS", "P decoded={!r}".format(decoded_group(processor.tokenizer, expanded_ids, p_positions)))
    log("TOKENS", "D token={!r}".format(decoded_group(processor.tokenizer, expanded_ids, d_positions)))

    image_hw = (image.height, image.width)
    num_views = int(batch["pixel_values"].shape[1])
    expected_views = image_size_to_num_views(
        image_hw, config.image_grid_pinpoints, int(config.vision_config.image_size)
    )
    require(num_views == expected_views, "Image-view count mismatch")
    proxy = SimpleNamespace(config=config)
    packed_mapping = build_packed_token_mapping(proxy, image_hw, num_views)
    require(
        len(image_positions) == packed_mapping["summary"]["total_packed_image_tokens"],
        "Packed mapping does not match expanded image-placeholder count",
    )
    regions = build_regions_full_coverage(
        sample=sample,
        variant=args.variant,
        mapping_tokens=packed_mapping["tokens"],
        streams={"base", "mosaic"},
        min_overlap_fraction=args.min_overlap_fraction,
        image_hw=image_hw,
        seed=args.seed,
    )
    visual_groups: Dict[str, Dict[str, Any]] = {}
    names = {
        "T": "text_region",
        "C": "correct_object_region",
        "G": "grounded_object_region",
        "R1": "matched_random_region_1",
        "R2": "matched_random_region_2",
        "R3": "matched_random_region_3",
    }
    for short, region_name in names.items():
        if region_name not in regions:
            visual_groups[short] = {"available": False, "region": region_name}
            log("REGION", "{} {} unavailable".format(short, region_name))
            continue
        result = sequence_region(regions[region_name], image_positions, packed_mapping["tokens"])
        result["region"] = region_name
        visual_groups[short] = result
        log("REGION", "{} {} tokens={} kinds={} seq_preview={}".format(
            short, region_name, result["packed_token_count"],
            result["packed_token_kind_counts"], result["sequence_positions"][:10]
        ))
    require(visual_groups["T"]["available"], "T is unavailable")
    for random_name in ("R1", "R2", "R3"):
        require(
            visual_groups[random_name]["packed_token_kind_counts"]
            == visual_groups["T"]["packed_token_kind_counts"],
            "{} does not match T token composition".format(random_name),
        )

    linguistic_groups = {
        "Q": {"sequence_positions": q_positions, "decoded": decoded_group(processor.tokenizer, expanded_ids, q_positions)},
        "P": {"sequence_positions": p_positions, "decoded": decoded_group(processor.tokenizer, expanded_ids, p_positions)},
        "D": {"sequence_positions": d_positions, "decoded": decoded_group(processor.tokenizer, expanded_ids, d_positions)},
    }
    edges: List[Dict[str, Any]] = []
    for source_name in ("T", "C", "G", "R1", "R2", "R3"):
        source = visual_groups[source_name]
        if not source["available"]:
            continue
        for destination_name in ("Q", "P", "D"):
            edges.append(edge_summary(
                "{}->{}".format(source_name, destination_name),
                source["sequence_positions"], linguistic_groups[destination_name]["sequence_positions"],
            ))
    edges.append(edge_summary("Q->D", q_positions, d_positions))
    edges.append(edge_summary("P->D", p_positions, d_positions))
    require(all(edge["causally_reachable"] for edge in edges), "At least one path violates causal ordering")
    for edge in edges:
        log("PATH", "{} queries={} keys={} edges/head={} causal={}".format(
            edge["path"], edge["query_count"], edge["key_value_count"],
            edge["directed_edge_count_per_head"], edge["causally_reachable"]
        ))

    report = {
        "status": "success",
        "milestone": "static_model_prompt_and_token_group_inspection",
        "configuration": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_dir": str(args.dataset_dir.resolve()),
            "dataset_revision": args.dataset_revision,
            "question_id": qid,
            "variant": args.variant,
            "seed": args.seed,
            "min_overlap_fraction": args.min_overlap_fraction,
            "device": "none; no weights or inference",
        },
        "software": {"transformers": transformers.__version__, "torch": torch.__version__},
        "attention": {
            "decoder_layer_path": "language_model.layers",
            "decoder_layers": len(layers),
            "attention_heads": int(config.text_config.num_attention_heads),
            "key_value_heads": int(config.text_config.num_key_value_heads),
            "attention_classes": attention_classes,
            "resolved_default_backends_in_empty_model": backends,
            "required_intervention_backend": "eager",
            "mistral_attention_forward_signature": attention_signature,
            "eager_attention_signature": eager_signature,
            "additive_mask_applied_before_softmax": additive_mask_before_softmax,
            "edge_specific_masking_technical_verdict": "supported through per-layer 4D additive masks under eager attention",
        },
        "prompt": {
            "unformatted": prompt,
            "formatted": formatted,
            "options": options,
            "correct_letter": correct_letter,
            "option_order": option_meta,
            "raw_token_count": len(raw_ids),
            "expanded_sequence_length": len(expanded_ids),
            "image_token_id": image_token_id,
            "image_placeholder_count": len(image_positions),
            "image_placeholder_first_last": [image_positions[0], image_positions[-1]],
        },
        "linguistic_groups": linguistic_groups,
        "visual_mapping_summary": packed_mapping["summary"],
        "visual_groups": visual_groups,
        "paths": edges,
        "validation": {
            "decoder_layer_count_matches_config": True,
            "all_attention_layers_same_class": True,
            "additive_mask_precedes_softmax": True,
            "raw_and_expanded_tokens_align": True,
            "packed_mapping_matches_image_placeholders": True,
            "question_decodes_exactly": True,
            "question_option_groups_disjoint": True,
            "causal_order_is_image_then_question_then_options_then_decision": True,
            "all_paths_causally_reachable": True,
            "three_random_controls_match_text_token_composition": True,
        },
        "expected_next_step": "Milestone 2 synthetic attention-mask test; do not load model weights yet.",
    }
    write_json(args.output, report)
    log("SAVE", str(args.output.resolve()))
    log("COMPLETE", "Milestone 1 static inspection passed; no model inference was run")


if __name__ == "__main__":
    main()
