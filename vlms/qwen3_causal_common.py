#!/usr/bin/env python3
"""Shared validated Qwen3-VL utilities for causal intervention experiments."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_MAIN = PROJECT_ROOT / "vlms" / "activation_patching" / "main_files"
sys.path.insert(0, str(ACTIVATION_MAIN))

from activation_patch_control_pilot import build_regions_full_coverage  # noqa: E402
from activation_patch_llava_next_debug import (  # noqa: E402
    ANSWER_LETTERS, answer_token_id, build_options, format_mcq_prompt,
    get_decoder_layers, get_or_download_hf_dataset, image_placeholder_positions,
    require, summarize_logits, text_bbox_yxyx, validate_cleaned_overlay,
    validate_dataset_provenance,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATASET_REVISION = "27b45899d1154ef1f08ce5c40d45d2468e4ea3e2"
VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")


def log(section: str, message: str) -> None:
    print(f"[{section}] {message}", flush=True)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_qwen3(device: str = "cuda", attention_backend: str = "sdpa"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation=attention_backend,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    layers, path = get_decoder_layers(model)
    require(len(layers) == 36, f"Expected 36 text layers, found {len(layers)}")
    return model, processor, layers, path


def prepare_inputs(processor: Any, image: Any, prompt: str, device: str = "cuda"):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = processor(text=[formatted], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if key == "pixel_values":
                value = value.to(torch.float16)
            inputs[key] = value
    return inputs, formatted


@torch.no_grad()
def cache_multimodal_inputs(model: Any, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Encode one Qwen image once while retaining its DeepStack decoder inputs."""
    input_ids = inputs["input_ids"]
    embeddings = model.get_input_embeddings()(input_ids)
    image_splits, deepstack = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    image_features = torch.cat(image_splits, dim=0).to(embeddings.device, embeddings.dtype)
    image_mask, _ = model.model.get_placeholder_mask(
        input_ids, inputs_embeds=embeddings, image_features=image_features,
    )
    embeddings = embeddings.masked_scatter(image_mask, image_features)
    position_ids, _ = model.model.get_rope_index(
        input_ids, inputs["image_grid_thw"], None, attention_mask=inputs.get("attention_mask"),
    )
    return {
        "inputs_embeds": embeddings,
        "attention_mask": inputs.get("attention_mask"),
        "position_ids": position_ids,
        "visual_pos_masks": image_mask[..., 0],
        "deepstack_visual_embeds": deepstack,
    }


@torch.no_grad()
def forward_cached_next_token_logits(model: Any, inputs: Dict[str, Any]) -> torch.Tensor:
    outputs = model.language_model(**inputs, use_cache=False)
    return model.lm_head(outputs.last_hidden_state[:, -1:, :])[0, -1].detach().float().cpu()


def build_qwen_mapping(image: Any, image_grid_thw: Sequence[int], processor: Any) -> Dict[str, Any]:
    temporal, grid_h, grid_w = map(int, image_grid_thw)
    patch = int(processor.image_processor.patch_size)
    merge = int(processor.image_processor.merge_size)
    require(temporal == 1, "Only still images are supported")
    require(grid_h % merge == 0 and grid_w % merge == 0, "Visual grid is not merge-aligned")
    merged_h, merged_w = grid_h // merge, grid_w // merge
    image_w, image_h = image.size
    resized_h, resized_w = grid_h * patch, grid_w * patch
    tokens = []
    for row in range(merged_h):
        for col in range(merged_w):
            tokens.append({
                "token_idx": row * merged_w + col,
                # One Qwen merged-patch stream; this compatibility label lets the
                # validated region-control utility treat it as a single stream.
                "kind": "base_patch", "qwen_kind": "merged_patch",
                "row": row, "col": col,
                "bbox": (
                    row * merge * patch * image_h / resized_h,
                    col * merge * patch * image_w / resized_w,
                    (row + 1) * merge * patch * image_h / resized_h,
                    (col + 1) * merge * patch * image_w / resized_w,
                ),
            })
    return {"tokens": tokens, "summary": {
        "original_size_hw": [image_h, image_w], "image_grid_thw": [temporal, grid_h, grid_w],
        "merged_grid_hw": [merged_h, merged_w], "patch_size": patch, "merge_size": merge,
        "total_packed_image_tokens": len(tokens), "stream": "qwen_merged_patch",
    }}


def mapped_regions(sample: Dict[str, Any], variant: str, mapping: Dict[str, Any], seed: int):
    image = sample["notext"]["image"]
    regions = build_regions_full_coverage(
        sample, variant, mapping["tokens"], {"base"}, .25,
        (image.height, image.width), seed,
        allow_text_positive_overlap_fallback=True,
    )
    regions["all_image_tokens"]["includes_packing_newline_tokens"] = False
    regions["all_image_tokens"]["interpretation"] = "All Qwen merged visual tokens."
    return regions


def load_selection(path: Path) -> Dict[str, Any]:
    selection = json.loads(path.read_text())
    qids = [str(row["question_id"]) for row in selection["selected_samples"]]
    require(selection["dataset_revision"] == DATASET_REVISION, "Dataset revision mismatch")
    require(len(qids) == len(set(qids)) == 305, "Selection must contain 305 unique questions")
    return selection


def answer_ids(processor: Any) -> Dict[str, int]:
    values = {letter: answer_token_id(processor.tokenizer, letter)[0] for letter in ANSWER_LETTERS}
    require(len(set(values.values())) == 4, "A-D answer token IDs are not distinct")
    return values


def sample_inputs(sample: Dict[str, Any], variant: str, processor: Any, seed: int):
    options, correct, option_meta = build_options(sample, True, seed)
    prompt = format_mcq_prompt(str(sample["question"]), options)
    images = {
        "no_text": sample["notext"]["image"].convert("RGB"),
        "overlay": sample[variant]["cleaned_image"].convert("RGB"),
    }
    original = sample[variant]["image"].convert("RGB")
    clean = validate_cleaned_overlay(images["no_text"], original, images["overlay"], text_bbox_yxyx(sample, variant))
    require(clean["cleaned_outside_no_text_mismatch_pixels"] == 0, "Outside pixels differ")
    require(clean["cleaned_inside_original_mismatch_pixels"] == 0, "Inside pixels differ")
    batches, formatted = {}, None
    for state, image in images.items():
        batches[state], current = prepare_inputs(processor, image, prompt)
        formatted = current if formatted is None else formatted
        require(current == formatted, "Paired formatted prompts differ")
    require(torch.equal(batches["no_text"]["input_ids"], batches["overlay"]["input_ids"]), "Paired token IDs differ")
    require(torch.equal(batches["no_text"]["image_grid_thw"], batches["overlay"]["image_grid_thw"]), "Paired grids differ")
    return options, correct, option_meta, images, batches, formatted, clean


def region_positions(model: Any, inputs: Dict[str, torch.Tensor], image: Any, processor: Any,
                     sample: Dict[str, Any], variant: str, seed: int):
    positions = image_placeholder_positions(model, inputs["input_ids"])
    grid = inputs["image_grid_thw"][0].tolist()
    mapping = build_qwen_mapping(image, grid, processor)
    require(len(positions) == len(mapping["tokens"]),
            f"Image placeholders ({len(positions)}) != mapped tokens ({len(mapping['tokens'])})")
    regions = mapped_regions(sample, variant, mapping, seed)
    sequence = {name: [positions[index] for index in region["token_indices"]]
                for name, region in regions.items()}
    return positions, mapping, regions, sequence
