#!/usr/bin/env python3
"""One-sample, one-layer LLaVA-NeXT residual-stream activation patching.

This is the first implementation milestone from
``activation_patching_experiment_profile.md``. It intentionally supports only:

* one GUIC question;
* one misleading overlay condition;
* one decoder layer;
* the annotated text region; and
* both restoration and insertion directions.

The script performs strict validation before accepting an intervention result:
configuration, answer tokens, paired prompt/image-token alignment, spatial
region mapping, no-op hook reproduction, patch integrity, and finite output
metrics. Results and validation details are saved as JSON.
"""

import os

# Avoid importing a broken optional TensorFlow installation through Transformers.
os.environ.setdefault("USE_TF", "0")

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution


MODEL_DEFAULT = "llava-hf/llava-v1.6-mistral-7b-hf"
DATASET_DEFAULT = "AHAAM/GUIC"
DATASET_REVISION_DEFAULT = "27b45899d1154ef1f08ce5c40d45d2468e4ea3e2"
VALID_VARIANTS = ("misleading_groundable", "misleading_ungroundable")
VALID_STREAMS = ("base", "mosaic")
ANSWER_LETTERS = ("A", "B", "C", "D")


def log(section: str, message: str) -> None:
    print("[{}] {}".format(section, message), flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(
    dataset_id: str,
    local_cache_root: str,
    split: str,
    revision: Optional[str] = None,
) -> Dataset:
    cache_root = Path(local_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / sanitize_repo_id(dataset_id)
    if cache_dir.exists():
        log("DATA", "Loading cached dataset: {}".format(cache_dir))
        ds = load_from_disk(str(cache_dir))
    else:
        log("DATA", "Downloading dataset: {} split={} revision={}".format(dataset_id, split, revision))
        ds = load_dataset(dataset_id, split=split, revision=revision)
        try:
            ds.save_to_disk(str(cache_dir))
            log("DATA", "Saved dataset cache: {}".format(cache_dir))
        except Exception as exc:
            log("WARN", "Could not save dataset cache: {}".format(exc))
    if isinstance(ds, DatasetDict):
        require(split in ds, "DatasetDict has no split {!r}".format(split))
        ds = ds[split]
    return ds


def find_sample_by_qid(ds: Dataset, question_id: str) -> Dict[str, Any]:
    requested = str(question_id)
    canonical = requested.lstrip("0") or "0"
    for index in range(len(ds)):
        sample = ds[index]
        current = str(sample.get("question_id", ""))
        if current == requested or (current.lstrip("0") or "0") == canonical:
            return sample
    raise ValueError("question_id={} not found in dataset".format(question_id))


def build_options(
    sample: Dict[str, Any],
    shuffle: bool,
    seed: int,
) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
    candidates = [
        {"key": "correct_answer", "text": sample["correct_answer"]["text"]},
        {"key": "misleading_groundable", "text": sample["misleading_groundable"]["text"]},
        {"key": "misleading_ungroundable", "text": sample["misleading_ungroundable"]["text"]},
        {"key": "irrelevant_word", "text": sample["irrelevant_word"]["text"]},
    ]
    qid = str(sample.get("question_id", ""))
    if shuffle:
        rng = random.Random("{}_{}".format(seed, qid))
        rng.shuffle(candidates)

    options = {ANSWER_LETTERS[i]: item["text"] for i, item in enumerate(candidates)}
    label_to_key = {ANSWER_LETTERS[i]: item["key"] for i, item in enumerate(candidates)}
    key_to_label = {key: label for label, key in label_to_key.items()}
    correct_letter = key_to_label["correct_answer"]
    meta = {
        "order": [item["key"] for item in candidates],
        "label_to_key": label_to_key,
        "key_to_label": key_to_label,
        "shuffle": bool(shuffle),
        "seed": int(seed),
    }
    return options, correct_letter, meta


def format_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    lines = [
        "Answer the following multiple-choice question by selecting the correct option.",
        "",
        "Question: {}".format(question),
        "",
        "Options:",
    ]
    for letter in ANSWER_LETTERS:
        lines.append("{}) {}".format(letter, options[letter]))
    lines.extend(["", "Answer with only the letter (A, B, C, or D):"])
    return "\n".join(lines)


def prepare_inputs(
    processor: LlavaNextProcessor,
    image: Image.Image,
    prompt_text: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[Dict[str, torch.Tensor], str]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]
    formatted = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(images=image, text=formatted, return_tensors="pt")
    inputs: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if key == "pixel_values":
                value = value.to(model_dtype)
            inputs[key] = value
    return inputs, formatted


def answer_token_id(tokenizer: Any, letter: str) -> Tuple[int, str, str]:
    for candidate in (" {}".format(letter), letter):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) != 1:
            continue
        token_id = int(ids[0])
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded.strip().upper() == letter:
            return token_id, candidate, decoded
    raise ValueError("Answer letter {!r} does not map to one validated token".format(letter))


def parse_answer_letter(text: str) -> Optional[str]:
    stripped = str(text).strip().upper()
    if stripped in ANSWER_LETTERS:
        return stripped
    return None


def get_decoder_layers(model: LlavaNextForConditionalGeneration) -> Tuple[Sequence[torch.nn.Module], str]:
    candidates = [
        ("language_model.model.layers", lambda: model.language_model.model.layers),
        ("language_model.layers", lambda: model.language_model.layers),
        ("model.layers", lambda: model.model.layers),
    ]
    for path, getter in candidates:
        try:
            layers = getter()
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers, path
    raise AttributeError("Could not locate decoder layer list on LLaVA-NeXT model")


def image_placeholder_positions(
    model: LlavaNextForConditionalGeneration,
    input_ids: torch.Tensor,
) -> List[int]:
    token_id = getattr(model.config, "image_token_index", None)
    if token_id is None:
        token_id = getattr(model.config, "image_token_id", None)
    require(token_id is not None, "Model config has no image token ID")
    return (input_ids[0] == int(token_id)).nonzero(as_tuple=True)[0].tolist()


def get_anyres_grid_shape(
    image_size_hw: Tuple[int, int],
    grid_pinpoints: Sequence[Sequence[int]],
    tile_size: int,
) -> Tuple[int, int]:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return int(best_h // tile_size), int(best_w // tile_size)


def image_size_to_num_views(
    image_size_hw: Tuple[int, int],
    grid_pinpoints: Sequence[Sequence[int]],
    tile_size: int,
) -> int:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return int((best_h // tile_size) * (best_w // tile_size) + 1)


def unpadded_grid_shape(
    current_h: int,
    current_w: int,
    original_h: int,
    original_w: int,
) -> Tuple[int, int]:
    original_aspect = float(original_w) / float(original_h)
    current_aspect = float(current_w) / float(current_h)
    if original_aspect > current_aspect:
        new_h = (original_h * current_w) // original_w
        padding = (current_h - new_h) // 2
        return int(current_h - 2 * padding), int(current_w)
    new_w = (original_w * current_h) // original_h
    padding = (current_w - new_w) // 2
    return int(current_h), int(current_w - 2 * padding)


def grid_bbox_to_original(
    row: int,
    col: int,
    grid_h: int,
    grid_w: int,
    original_h: int,
    original_w: int,
) -> Tuple[float, float, float, float]:
    return (
        float(row) * original_h / grid_h,
        float(col) * original_w / grid_w,
        float(row + 1) * original_h / grid_h,
        float(col + 1) * original_w / grid_w,
    )


def build_packed_token_mapping(
    model: LlavaNextForConditionalGeneration,
    original_size_hw: Tuple[int, int],
    num_views_provided: int,
) -> Dict[str, Any]:
    """Map packed LLaVA-NeXT image tokens to original-image coordinates."""
    cfg = model.config
    vision_cfg = cfg.vision_config
    tile_size = int(vision_cfg.image_size)
    patch_size = int(vision_cfg.patch_size)
    patches_per_side = tile_size // patch_size
    cls_removed = cfg.vision_feature_select_strategy == "default"
    base_tokens = patches_per_side * patches_per_side + (0 if cls_removed else 1)

    tile_h, tile_w = get_anyres_grid_shape(
        original_size_hw,
        cfg.image_grid_pinpoints,
        tile_size,
    )
    mosaic_h = tile_h * patches_per_side
    mosaic_w = tile_w * patches_per_side
    original_h, original_w = original_size_hw
    unpadded_h, unpadded_w = unpadded_grid_shape(
        mosaic_h,
        mosaic_w,
        original_h,
        original_w,
    )
    row_width = unpadded_w + 1
    mosaic_tokens = unpadded_h * row_width
    tokens: List[Dict[str, Any]] = []

    for token_idx in range(base_tokens):
        if not cls_removed and token_idx == 0:
            tokens.append({"token_idx": token_idx, "kind": "base_cls", "bbox": None})
            continue
        patch_idx = token_idx if cls_removed else token_idx - 1
        row = patch_idx // patches_per_side
        col = patch_idx % patches_per_side
        tokens.append(
            {
                "token_idx": token_idx,
                "kind": "base_patch",
                "row": int(row),
                "col": int(col),
                "bbox": grid_bbox_to_original(
                    row,
                    col,
                    patches_per_side,
                    patches_per_side,
                    original_h,
                    original_w,
                ),
            }
        )

    for local_idx in range(mosaic_tokens):
        token_idx = base_tokens + local_idx
        row = local_idx // row_width
        col = local_idx % row_width
        if col == unpadded_w:
            tokens.append(
                {
                    "token_idx": token_idx,
                    "kind": "newline",
                    "row": int(row),
                    "col": None,
                    "bbox": None,
                }
            )
            continue
        tokens.append(
            {
                "token_idx": token_idx,
                "kind": "mosaic_patch",
                "row": int(row),
                "col": int(col),
                "bbox": grid_bbox_to_original(
                    row,
                    col,
                    unpadded_h,
                    unpadded_w,
                    original_h,
                    original_w,
                ),
            }
        )

    expected_views = image_size_to_num_views(
        original_size_hw,
        cfg.image_grid_pinpoints,
        tile_size,
    )
    return {
        "summary": {
            "original_size_hw": list(original_size_hw),
            "tile_size": tile_size,
            "patch_size": patch_size,
            "patches_per_side": patches_per_side,
            "tile_layout_hw": [tile_h, tile_w],
            "mosaic_pre_unpad_hw": [mosaic_h, mosaic_w],
            "mosaic_unpadded_hw": [unpadded_h, unpadded_w],
            "base_tokens": base_tokens,
            "mosaic_tokens_including_newlines": mosaic_tokens,
            "total_packed_image_tokens": base_tokens + mosaic_tokens,
            "num_views_provided": int(num_views_provided),
            "expected_num_views": expected_views,
            "view_count_matches": int(num_views_provided) == expected_views,
        },
        "tokens": tokens,
    }


def overlap_fraction_of_token(
    token_bbox: Tuple[float, float, float, float],
    region_bbox: Tuple[float, float, float, float],
) -> float:
    ty0, tx0, ty1, tx1 = token_bbox
    ry0, rx0, ry1, rx1 = region_bbox
    iy0, ix0 = max(ty0, ry0), max(tx0, rx0)
    iy1, ix1 = min(ty1, ry1), min(tx1, rx1)
    intersection = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    token_area = max(0.0, ty1 - ty0) * max(0.0, tx1 - tx0)
    return 0.0 if token_area <= 0 else float(intersection / token_area)


def region_token_indices(
    mapping_tokens: Sequence[Dict[str, Any]],
    region_bbox_yxyx: Tuple[float, float, float, float],
    streams: Set[str],
    min_overlap_fraction: float,
) -> List[int]:
    allowed_kinds = set()
    if "base" in streams:
        allowed_kinds.add("base_patch")
    if "mosaic" in streams:
        allowed_kinds.add("mosaic_patch")
    selected = []
    for token in mapping_tokens:
        if token.get("kind") not in allowed_kinds or token.get("bbox") is None:
            continue
        if overlap_fraction_of_token(tuple(token["bbox"]), region_bbox_yxyx) >= min_overlap_fraction:
            selected.append(int(token["token_idx"]))
    return selected


def text_bbox_yxyx(sample: Dict[str, Any], variant: str) -> Tuple[float, float, float, float]:
    bbox = sample[variant].get("bbox")
    require(bbox is not None and len(bbox) == 4, "{} has no valid text bbox".format(variant))
    x0, y0, x1, y1 = [float(value) for value in bbox]
    require(x1 > x0 and y1 > y0, "Invalid text bbox: {}".format(bbox))
    return y0, x0, y1, x1


def image_difference_stats(
    no_text_image: Image.Image,
    overlay_image: Image.Image,
    bbox_yxyx: Tuple[float, float, float, float],
) -> Dict[str, float]:
    a = np.asarray(no_text_image.convert("RGB"), dtype=np.float32)
    b = np.asarray(overlay_image.convert("RGB"), dtype=np.float32)
    require(a.shape == b.shape, "Paired image pixel shapes do not match")
    diff = np.abs(a - b).mean(axis=2)
    height, width = diff.shape
    y0, x0, y1, x1 = bbox_yxyx
    y0i, y1i = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
    x0i, x1i = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
    mask = np.zeros_like(diff, dtype=bool)
    mask[y0i:y1i, x0i:x1i] = True
    inside = diff[mask]
    outside = diff[~mask]
    changed = np.any(a != b, axis=2)
    return {
        "mean_abs_difference_all": float(diff.mean()),
        "mean_abs_difference_inside_text_bbox": float(inside.mean()) if inside.size else float("nan"),
        "mean_abs_difference_outside_text_bbox": float(outside.mean()) if outside.size else float("nan"),
        "fraction_outside_pixels_changed": float((outside > 0).mean()) if outside.size else float("nan"),
        "inside_pixels_changed": int((changed & mask).sum()),
        "outside_pixels_changed": int((changed & ~mask).sum()),
        "total_pixels": int(height * width),
    }


def validate_cleaned_overlay(
    no_text_image: Image.Image,
    original_overlay_image: Image.Image,
    cleaned_overlay_image: Image.Image,
    bbox_yxyx: Tuple[float, float, float, float],
) -> Dict[str, int]:
    base = np.asarray(no_text_image.convert("RGB"))
    original = np.asarray(original_overlay_image.convert("RGB"))
    cleaned = np.asarray(cleaned_overlay_image.convert("RGB"))
    require(base.shape == original.shape == cleaned.shape, "Cleaned-pair image dimensions differ")
    height, width = base.shape[:2]
    y0, x0, y1, x1 = bbox_yxyx
    y0i, y1i = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
    x0i, x1i = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0i:y1i, x0i:x1i] = True
    outside_mismatch = np.any(cleaned != base, axis=2) & ~mask
    inside_mismatch = np.any(cleaned != original, axis=2) & mask
    return {
        "cleaned_outside_no_text_mismatch_pixels": int(outside_mismatch.sum()),
        "cleaned_inside_original_mismatch_pixels": int(inside_mismatch.sum()),
        "bbox_pixel_count": int(mask.sum()),
    }


def validate_dataset_provenance(path: str, expected_revision: str) -> Dict[str, Any]:
    validation_path = Path(path).resolve()
    require(validation_path.exists(), "Dataset validation record not found: {}".format(validation_path))
    with validation_path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    require(record.get("passed") is True, "Dataset validation record did not pass")
    recorded_revision = record.get("commit_sha")
    require(
        recorded_revision == expected_revision,
        "Dataset revision mismatch: expected {}, validation record has {}".format(
            expected_revision, recorded_revision
        ),
    )
    require(record.get("outside_changed_pixels") == 0, "Dataset validation reports outside-box changes")
    require(record.get("inside_original_mismatch_pixels") == 0, "Dataset validation reports inside-box mismatches")
    return record


def clone_inputs(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Model forward does not intentionally mutate these tensors, so cloning is unnecessary
    # and would substantially increase memory use for pixel_values.
    return {key: value for key, value in inputs.items()}


@torch.no_grad()
def forward_next_token_logits(
    model: LlavaNextForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    forward_parameters = inspect.signature(model.forward).parameters
    forward_kwargs: Dict[str, Any] = {
        "use_cache": False,
        "return_dict": True,
    }
    # Transformers renamed this argument in recent releases. Keeping the
    # compatibility branch explicit prevents a full-sequence logits allocation.
    if "logits_to_keep" in forward_parameters:
        forward_kwargs["logits_to_keep"] = 1
    elif "num_logits_to_keep" in forward_parameters:
        forward_kwargs["num_logits_to_keep"] = 1
    outputs = model(**clone_inputs(inputs), **forward_kwargs)
    return outputs.logits[0, -1].detach().float().cpu()


@torch.no_grad()
def forward_and_capture_resid_pre(
    model: LlavaNextForConditionalGeneration,
    layer: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(_module: torch.nn.Module, hook_args: Tuple[Any, ...]) -> None:
        require(len(hook_args) > 0 and torch.is_tensor(hook_args[0]), "Layer pre-hook did not receive hidden states")
        captured["hidden"] = hook_args[0].detach().clone()
        return None

    handle = layer.register_forward_pre_hook(capture_hook)
    try:
        logits = forward_next_token_logits(model, inputs)
    finally:
        handle.remove()
    require("hidden" in captured, "Residual-stream capture hook was not called")
    return logits, captured["hidden"]


@torch.no_grad()
def forward_with_patch(
    model: LlavaNextForConditionalGeneration,
    layer: torch.nn.Module,
    recipient_inputs: Dict[str, torch.Tensor],
    donor_hidden: torch.Tensor,
    sequence_positions: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    integrity: Dict[str, float] = {}
    hook_calls = {"count": 0}

    def patch_hook(_module: torch.nn.Module, hook_args: Tuple[Any, ...]) -> Tuple[Any, ...]:
        require(len(hook_args) > 0 and torch.is_tensor(hook_args[0]), "Patch hook did not receive hidden states")
        hidden = hook_args[0]
        require(tuple(hidden.shape) == tuple(donor_hidden.shape), "Donor and recipient hidden-state shapes differ")
        positions = torch.as_tensor(sequence_positions, dtype=torch.long, device=hidden.device)
        require(positions.numel() > 0, "No sequence positions supplied to patch hook")
        require(int(positions.min()) >= 0 and int(positions.max()) < hidden.shape[1], "Patch position out of range")

        donor = donor_hidden.to(device=hidden.device, dtype=hidden.dtype)
        before_difference = (hidden[:, positions, :] - donor[:, positions, :]).abs()
        patched = hidden.clone()
        patched[:, positions, :] = donor[:, positions, :]
        after_error = (patched[:, positions, :] - donor[:, positions, :]).abs()

        direct_delta = (patched - hidden).abs()
        direct_delta[:, positions, :] = 0
        integrity.update(
            {
                "donor_recipient_mean_abs_difference_before": float(before_difference.mean().item()),
                "donor_recipient_max_abs_difference_before": float(before_difference.max().item()),
                "patched_donor_max_abs_difference_after": float(after_error.max().item()),
                "unpatched_positions_max_direct_change": float(direct_delta.max().item()),
                "number_of_patched_positions": int(positions.numel()),
            }
        )
        hook_calls["count"] += 1
        return (patched,) + tuple(hook_args[1:])

    handle = layer.register_forward_pre_hook(patch_hook)
    try:
        logits = forward_next_token_logits(model, recipient_inputs)
    finally:
        handle.remove()
    require(hook_calls["count"] == 1, "Patch hook call count was {}, expected 1".format(hook_calls["count"]))
    return logits, integrity


def summarize_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    answer_ids: Dict[str, int],
    correct_letter: str,
    misleading_letter: str,
) -> Dict[str, Any]:
    correct_logit = float(logits[answer_ids[correct_letter]].item())
    misleading_logit = float(logits[answer_ids[misleading_letter]].item())
    global_id = int(torch.argmax(logits).item())
    global_text = tokenizer.decode([global_id], skip_special_tokens=False)
    choice_letter = max(ANSWER_LETTERS, key=lambda letter: float(logits[answer_ids[letter]].item()))
    values = [correct_logit, misleading_logit, correct_logit - misleading_logit]
    require(all(math.isfinite(value) for value in values), "Found non-finite target logit or margin")
    return {
        "correct_logit": correct_logit,
        "misleading_logit": misleading_logit,
        "margin_correct_minus_misleading": correct_logit - misleading_logit,
        "global_next_token_id": global_id,
        "global_next_token_text": global_text,
        "global_next_token_letter": parse_answer_letter(global_text),
        "choice_constrained_prediction": choice_letter,
        "choice_logits": {letter: float(logits[token_id].item()) for letter, token_id in answer_ids.items()},
    }


def classify_change(
    before_letter: str,
    after_letter: str,
    correct_letter: str,
    misleading_letter: str,
) -> str:
    if before_letter != correct_letter and after_letter == correct_letter:
        return "recovery"
    if before_letter == correct_letter and after_letter == misleading_letter:
        return "misleading_flip"
    if before_letter != after_letter:
        return "other_flip"
    return "no_prediction_change"


def maximum_absolute_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    require(tuple(a.shape) == tuple(b.shape), "Cannot compare tensors with different shapes")
    return float((a - b).abs().max().item())


def parse_streams(raw: str) -> Set[str]:
    streams = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = streams.difference(VALID_STREAMS)
    require(bool(streams), "At least one stream is required")
    require(not unknown, "Unknown streams: {}".format(sorted(unknown)))
    return streams


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=MODEL_DEFAULT)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--hf_dataset", default=DATASET_DEFAULT)
    parser.add_argument("--dataset_revision", default=DATASET_REVISION_DEFAULT)
    parser.add_argument("--hf_cache_dir", default="../hf_dataset_GUIC_cleaned")
    parser.add_argument(
        "--dataset_validation_file",
        default="../hf_dataset_GUIC_cleaned/remote_validation.json",
    )
    parser.add_argument("--overlay_image_field", choices=("cleaned_image", "image"), default="cleaned_image")
    parser.add_argument("--split", default="test")
    parser.add_argument("--question_id", required=True)
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="misleading_groundable")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--hook_location", choices=("resid_pre",), default="resid_pre")
    parser.add_argument("--streams", default="base,mosaic")
    parser.add_argument("--min_overlap_fraction", type=float, default=0.25)
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--noop_tolerance", type=float, default=None)
    parser.add_argument("--out_dir", default="../debug_outputs")
    args = parser.parse_args()

    require(0.0 <= args.min_overlap_fraction <= 1.0, "min_overlap_fraction must be in [0,1]")
    streams = parse_streams(args.streams)
    if args.device == "cuda":
        require(torch.cuda.is_available(), "--device cuda was requested but CUDA is unavailable")
    if args.device == "cpu":
        require(args.dtype == "float32", "CPU debug runs require --dtype float32")
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    noop_tolerance = args.noop_tolerance
    if noop_tolerance is None:
        noop_tolerance = 1e-3 if dtype == torch.float16 else 1e-5

    log("CONFIG", "model_id={}".format(args.model_id))
    log("CONFIG", "requested_model_revision={}".format(args.model_revision or "default"))
    log("CONFIG", "dataset={} revision={} split={} question_id={}".format(
        args.hf_dataset, args.dataset_revision, args.split, args.question_id
    ))
    log("CONFIG", "overlay_image_field={}".format(args.overlay_image_field))
    log("CONFIG", "variant={} directions=restoration,insertion".format(args.variant))
    log("CONFIG", "layer={} hook_location={}".format(args.layer, args.hook_location))
    log("CONFIG", "streams={} min_overlap_fraction={}".format(sorted(streams), args.min_overlap_fraction))
    log("CONFIG", "device={} dtype={} seed={}".format(device, dtype, args.seed))
    log("CONFIG", "decoding=single_next_token choice_evaluation=constrained_A-D")
    log("EXPECTED", "One sample, one layer, text-region patching in both directions")

    provenance = validate_dataset_provenance(args.dataset_validation_file, args.dataset_revision)
    log("DATA", "validated_remote_commit={} validated_pairs={}/{}".format(
        provenance["commit_sha"], provenance["pairs_validated"], provenance["pairs_expected"]
    ))
    log("PASS", "Pinned cleaned dataset provenance and dataset-wide pixel audit validated")
    ds = get_or_download_hf_dataset(
        args.hf_dataset, args.hf_cache_dir, args.split, args.dataset_revision
    )
    sample = find_sample_by_qid(ds, args.question_id)
    qid = str(sample.get("question_id"))

    log("MODEL", "Loading {}".format(args.model_id))
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(args.model_id, revision=args.model_revision)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = int(model.config.vision_config.patch_size)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    layers, layer_path = get_decoder_layers(model)
    require(0 <= args.layer < len(layers), "layer={} outside [0, {})".format(args.layer, len(layers)))
    layer_module = layers[args.layer]
    revision = getattr(model.config, "_commit_hash", None)
    log("MODEL", "revision={} decoder_path={} num_layers={}".format(revision, layer_path, len(layers)))
    log("PASS", "Requested layer exists and hook location is resid_pre")

    options, correct_letter, option_meta = build_options(sample, args.shuffle_options, args.seed)
    misleading_letter = option_meta["key_to_label"][args.variant]
    prompt_text = format_mcq_prompt(str(sample.get("question", "")), options)
    tokenizer = processor.tokenizer

    token_validation: Dict[str, Any] = {}
    answer_ids: Dict[str, int] = {}
    for letter in ANSWER_LETTERS:
        token_id, encoded_form, decoded = answer_token_id(tokenizer, letter)
        answer_ids[letter] = token_id
        token_validation[letter] = {
            "token_id": token_id,
            "encoded_form": encoded_form,
            "decoded": decoded,
        }
    require(len(set(answer_ids.values())) == len(ANSWER_LETTERS), "A-D answer token IDs are not distinct")
    require(correct_letter != misleading_letter, "Correct and misleading letters are equal")
    require(answer_ids[correct_letter] != answer_ids[misleading_letter], "Correct and misleading token IDs are equal")
    log("TOKENS", "qid={} correct={} {!r} token_id={} decoded={!r}".format(
        qid, correct_letter, options[correct_letter], answer_ids[correct_letter], token_validation[correct_letter]["decoded"]
    ))
    log("TOKENS", "qid={} misleading={} {!r} token_id={} decoded={!r}".format(
        qid, misleading_letter, options[misleading_letter], answer_ids[misleading_letter], token_validation[misleading_letter]["decoded"]
    ))
    log("PASS", "All A-D answer letters map to distinct validated single tokens")

    no_text_image = sample["notext"]["image"].convert("RGB")
    require(args.overlay_image_field in sample[args.variant], "Overlay field is missing: {}.{}".format(
        args.variant, args.overlay_image_field
    ))
    original_overlay_image = sample[args.variant]["image"].convert("RGB")
    overlay_image = sample[args.variant][args.overlay_image_field].convert("RGB")
    bbox = text_bbox_yxyx(sample, args.variant)
    pixel_difference = image_difference_stats(no_text_image, overlay_image, bbox)
    cleaned_overlay_validation = validate_cleaned_overlay(
        no_text_image, original_overlay_image, overlay_image, bbox
    )
    log("PIXELS", "mean_abs_diff inside_text={:.4f} outside_text={:.4f}".format(
        pixel_difference["mean_abs_difference_inside_text_bbox"],
        pixel_difference["mean_abs_difference_outside_text_bbox"],
    ))
    log("PIXELS", "outside_changed={} inside_changed={} total_pixels={}".format(
        pixel_difference["outside_pixels_changed"],
        pixel_difference["inside_pixels_changed"],
        pixel_difference["total_pixels"],
    ))
    log("PIXELS", "cleaned_outside_base_mismatch={} cleaned_inside_original_mismatch={}".format(
        cleaned_overlay_validation["cleaned_outside_no_text_mismatch_pixels"],
        cleaned_overlay_validation["cleaned_inside_original_mismatch_pixels"],
    ))
    if args.overlay_image_field == "cleaned_image":
        require(pixel_difference["outside_pixels_changed"] == 0, "Cleaned overlay changes pixels outside text bbox")
        require(
            cleaned_overlay_validation["cleaned_outside_no_text_mismatch_pixels"] == 0,
            "Cleaned overlay does not exactly match no-text outside bbox",
        )
        require(
            cleaned_overlay_validation["cleaned_inside_original_mismatch_pixels"] == 0,
            "Cleaned overlay does not exactly match original overlay inside bbox",
        )
        require(pixel_difference["inside_pixels_changed"] > 0, "Cleaned overlay introduces no changed text-box pixels")
        log("PASS", "Cleaned pair differs only inside the annotated text bbox")

    no_text_inputs, no_text_formatted = prepare_inputs(processor, no_text_image, prompt_text, device, dtype)
    overlay_inputs, overlay_formatted = prepare_inputs(processor, overlay_image, prompt_text, device, dtype)
    require(no_text_formatted == overlay_formatted, "Formatted prompts differ between paired runs")
    require(torch.equal(no_text_inputs["input_ids"], overlay_inputs["input_ids"]), "Paired input_ids differ")
    require(tuple(no_text_inputs["input_ids"].shape) == tuple(overlay_inputs["input_ids"].shape), "Sequence shapes differ")

    no_text_img_pos = image_placeholder_positions(model, no_text_inputs["input_ids"])
    overlay_img_pos = image_placeholder_positions(model, overlay_inputs["input_ids"])
    require(no_text_img_pos == overlay_img_pos, "Image-placeholder positions do not align")
    no_text_size = tuple(int(value) for value in no_text_inputs["image_sizes"][0].tolist())
    overlay_size = tuple(int(value) for value in overlay_inputs["image_sizes"][0].tolist())
    require(no_text_size == overlay_size, "Original image sizes differ")
    no_text_views = int(no_text_inputs["pixel_values"].shape[1]) if no_text_inputs["pixel_values"].ndim == 5 else 1
    overlay_views = int(overlay_inputs["pixel_values"].shape[1]) if overlay_inputs["pixel_values"].ndim == 5 else 1
    require(no_text_views == overlay_views, "Paired runs have different view counts")

    mapping = build_packed_token_mapping(model, no_text_size, no_text_views)
    expected_tokens = int(mapping["summary"]["total_packed_image_tokens"])
    require(len(no_text_img_pos) == expected_tokens, "Placeholder count {} != mapping count {}".format(len(no_text_img_pos), expected_tokens))
    require(mapping["summary"]["view_count_matches"], "Processor view count does not match mapping expectation")
    alignment = {
        "prompt_equal": True,
        "input_ids_equal": True,
        "sequence_length": int(no_text_inputs["input_ids"].shape[1]),
        "image_placeholder_count": len(no_text_img_pos),
        "packed_mapping_count": expected_tokens,
        "original_image_size_hw": list(no_text_size),
        "num_views": no_text_views,
        "mapping_summary": mapping["summary"],
    }
    log("ALIGN", "seq_len={} image_placeholders={} mapping_tokens={} views={} image_hw={}".format(
        alignment["sequence_length"], len(no_text_img_pos), expected_tokens, no_text_views, no_text_size
    ))
    log("PASS", "Prompt tokens and packed visual-token positions align one-to-one")

    image_token_indices = region_token_indices(
        mapping["tokens"],
        bbox,
        streams,
        args.min_overlap_fraction,
    )
    require(len(image_token_indices) > 0, "Text bbox mapped to zero visual tokens")
    require(min(image_token_indices) >= 0 and max(image_token_indices) < len(no_text_img_pos), "Mapped token index out of range")
    sequence_positions = [no_text_img_pos[index] for index in image_token_indices]
    require(len(set(sequence_positions)) == len(sequence_positions), "Duplicate sequence positions selected")
    region_validation = {
        "region": "text_region",
        "bbox_yxyx": list(bbox),
        "streams": sorted(streams),
        "min_overlap_fraction": float(args.min_overlap_fraction),
        "image_token_indices": image_token_indices,
        "sequence_positions": sequence_positions,
        "number_of_tokens": len(image_token_indices),
    }
    log("REGION", "text_bbox_yxyx={} mapped_tokens={} token_preview={} sequence_preview={}".format(
        tuple(round(value, 2) for value in bbox),
        len(image_token_indices),
        image_token_indices[:12],
        sequence_positions[:12],
    ))
    log("PASS", "Text region maps to valid packed visual-token positions")

    # Normal forwards establish the reference logits.
    no_text_logits = forward_next_token_logits(model, no_text_inputs)
    overlay_logits = forward_next_token_logits(model, overlay_inputs)

    # Capture hooks are also the required no-op-hook reproductions.
    no_text_noop_logits, no_text_hidden = forward_and_capture_resid_pre(model, layer_module, no_text_inputs)
    overlay_noop_logits, overlay_hidden = forward_and_capture_resid_pre(model, layer_module, overlay_inputs)
    no_text_noop_error = maximum_absolute_difference(no_text_logits, no_text_noop_logits)
    overlay_noop_error = maximum_absolute_difference(overlay_logits, overlay_noop_logits)

    no_text_base = summarize_logits(no_text_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    overlay_base = summarize_logits(overlay_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    no_text_noop = summarize_logits(no_text_noop_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    overlay_noop = summarize_logits(overlay_noop_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    no_text_noop_prediction_same = no_text_base["choice_constrained_prediction"] == no_text_noop["choice_constrained_prediction"]
    overlay_noop_prediction_same = overlay_base["choice_constrained_prediction"] == overlay_noop["choice_constrained_prediction"]
    require(no_text_noop_error <= noop_tolerance, "No-text no-op error {} > tolerance {}".format(no_text_noop_error, noop_tolerance))
    require(overlay_noop_error <= noop_tolerance, "Overlay no-op error {} > tolerance {}".format(overlay_noop_error, noop_tolerance))
    require(no_text_noop_prediction_same and overlay_noop_prediction_same, "No-op hook changed a constrained prediction")
    log("NOOP", "no_text max_logit_diff={:.8g} prediction_same={}".format(no_text_noop_error, no_text_noop_prediction_same))
    log("NOOP", "no_text original_correct={:.6f} original_misleading={:.6f} original_margin={:.6f}".format(
        no_text_base["correct_logit"],
        no_text_base["misleading_logit"],
        no_text_base["margin_correct_minus_misleading"],
    ))
    log("NOOP", "no_text hooked_correct={:.6f} hooked_misleading={:.6f} hooked_margin={:.6f}".format(
        no_text_noop["correct_logit"],
        no_text_noop["misleading_logit"],
        no_text_noop["margin_correct_minus_misleading"],
    ))
    log("NOOP", "overlay max_logit_diff={:.8g} prediction_same={}".format(overlay_noop_error, overlay_noop_prediction_same))
    log("NOOP", "overlay original_correct={:.6f} original_misleading={:.6f} original_margin={:.6f}".format(
        overlay_base["correct_logit"],
        overlay_base["misleading_logit"],
        overlay_base["margin_correct_minus_misleading"],
    ))
    log("NOOP", "overlay hooked_correct={:.6f} hooked_misleading={:.6f} hooked_margin={:.6f}".format(
        overlay_noop["correct_logit"],
        overlay_noop["misleading_logit"],
        overlay_noop["margin_correct_minus_misleading"],
    ))
    log("PASS", "No-op capture hooks reproduce both recipient baselines within tolerance={}".format(noop_tolerance))

    require(tuple(no_text_hidden.shape) == tuple(overlay_hidden.shape), "Paired residual-stream shapes differ")
    require(no_text_hidden.shape[1] == no_text_inputs["input_ids"].shape[1], "Residual sequence length differs from input sequence length")

    restoration_logits, restoration_integrity = forward_with_patch(
        model,
        layer_module,
        overlay_inputs,
        no_text_hidden,
        sequence_positions,
    )
    insertion_logits, insertion_integrity = forward_with_patch(
        model,
        layer_module,
        no_text_inputs,
        overlay_hidden,
        sequence_positions,
    )

    patch_tolerance = 0.0
    for name, integrity in (("restoration", restoration_integrity), ("insertion", insertion_integrity)):
        require(integrity["patched_donor_max_abs_difference_after"] <= patch_tolerance, "{} patched values do not equal donor".format(name))
        require(integrity["unpatched_positions_max_direct_change"] <= patch_tolerance, "{} hook directly changed unpatched positions".format(name))
        require(integrity["donor_recipient_max_abs_difference_before"] > 0.0, "{} donor and recipient selected activations are identical".format(name))
        log("PATCH", "direction={} region=text_region layer={} positions={} before_mean={:.6g} before_max={:.6g} after_error={:.6g} unpatched_change={:.6g}".format(
            name,
            args.layer,
            integrity["number_of_patched_positions"],
            integrity["donor_recipient_mean_abs_difference_before"],
            integrity["donor_recipient_max_abs_difference_before"],
            integrity["patched_donor_max_abs_difference_after"],
            integrity["unpatched_positions_max_direct_change"],
        ))
    log("PASS", "Patch integrity checks passed in both directions")

    restoration = summarize_logits(restoration_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    insertion = summarize_logits(insertion_logits, tokenizer, answer_ids, correct_letter, misleading_letter)
    restoration_effect = restoration["margin_correct_minus_misleading"] - overlay_base["margin_correct_minus_misleading"]
    insertion_effect = no_text_base["margin_correct_minus_misleading"] - insertion["margin_correct_minus_misleading"]
    require(math.isfinite(restoration_effect) and math.isfinite(insertion_effect), "Non-finite intervention effect")
    restoration_change = classify_change(
        overlay_base["choice_constrained_prediction"],
        restoration["choice_constrained_prediction"],
        correct_letter,
        misleading_letter,
    )
    insertion_change = classify_change(
        no_text_base["choice_constrained_prediction"],
        insertion["choice_constrained_prediction"],
        correct_letter,
        misleading_letter,
    )

    log("RESULT", "restoration before_margin={:.6f} after_margin={:.6f} effect={:.6f} before_pred={} after_pred={} outcome={}".format(
        overlay_base["margin_correct_minus_misleading"],
        restoration["margin_correct_minus_misleading"],
        restoration_effect,
        overlay_base["choice_constrained_prediction"],
        restoration["choice_constrained_prediction"],
        restoration_change,
    ))
    log("EXPECTED", "Positive restoration effect means movement toward the correct answer")
    log("RESULT", "insertion before_margin={:.6f} after_margin={:.6f} effect={:.6f} before_pred={} after_pred={} outcome={}".format(
        no_text_base["margin_correct_minus_misleading"],
        insertion["margin_correct_minus_misleading"],
        insertion_effect,
        no_text_base["choice_constrained_prediction"],
        insertion["choice_constrained_prediction"],
        insertion_change,
    ))
    log("EXPECTED", "Positive insertion effect means movement toward the misleading answer")

    report: Dict[str, Any] = {
        "status": "success",
        "milestone": "one_sample_one_layer_text_region",
        "configuration": {
            "model_id": args.model_id,
            "requested_model_revision": args.model_revision,
            "model_revision": revision,
            "transformers_version": __import__("transformers").__version__,
            "device": str(device),
            "dtype": str(dtype),
            "dataset": args.hf_dataset,
            "dataset_revision": args.dataset_revision,
            "dataset_validation_file": str(Path(args.dataset_validation_file).resolve()),
            "overlay_image_field": args.overlay_image_field,
            "split": args.split,
            "question_id": qid,
            "variant": args.variant,
            "layer": int(args.layer),
            "hook_location": args.hook_location,
            "decoder_layer_path": layer_path,
            "number_of_layers": len(layers),
            "shuffle_options": bool(args.shuffle_options),
            "seed": int(args.seed),
            "decoding": "single_next_token",
            "choice_evaluation": "constrained_A-D",
            "streams": sorted(streams),
            "min_overlap_fraction": float(args.min_overlap_fraction),
            "noop_tolerance": float(noop_tolerance),
        },
        "sample": {
            "question": str(sample.get("question", "")),
            "options": options,
            "option_meta": option_meta,
            "correct_letter": correct_letter,
            "misleading_letter": misleading_letter,
            "correct_text": options[correct_letter],
            "misleading_text": options[misleading_letter],
        },
        "answer_token_validation": token_validation,
        "paired_input_alignment": alignment,
        "pixel_difference": pixel_difference,
        "cleaned_overlay_validation": cleaned_overlay_validation,
        "dataset_provenance": provenance,
        "region_mapping": region_validation,
        "no_op_validation": {
            "no_text_max_absolute_logit_difference": no_text_noop_error,
            "overlay_max_absolute_logit_difference": overlay_noop_error,
            "no_text_prediction_same": no_text_noop_prediction_same,
            "overlay_prediction_same": overlay_noop_prediction_same,
            "passed": True,
        },
        "baselines": {
            "no_text": no_text_base,
            "misleading_overlay": overlay_base,
        },
        "interventions": {
            "restoration": {
                "recipient": "misleading_overlay",
                "donor": "no_text",
                "patched": restoration,
                "effect": restoration_effect,
                "outcome": restoration_change,
                "integrity": restoration_integrity,
            },
            "insertion": {
                "recipient": "no_text",
                "donor": "misleading_overlay",
                "patched": insertion,
                "effect": insertion_effect,
                "outcome": insertion_change,
                "integrity": insertion_integrity,
            },
        },
        "checklist": {
            "configuration_validation": "passed",
            "answer_token_validation": "passed",
            "paired_input_alignment": "passed",
            "region_mapping": "passed",
            "unpatched_baseline_reproduction": "passed",
            "patch_integrity": "passed",
            "intervention_result_validation": "passed",
            "dataset_scale_progress_summary": "not_applicable_to_debug_milestone",
        },
    }

    output_path = (
        Path(args.out_dir)
        / qid
        / args.variant
        / "layer_{:02d}_text_region_{}_debug.json".format(args.layer, args.overlay_image_field)
    )
    save_json(output_path, report)
    log("SAVE", "Saved structured report: {}".format(output_path))
    log("COMPLETE", "All milestone validation checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("FAIL", "{}: {}".format(type(exc).__name__, exc))
        raise
