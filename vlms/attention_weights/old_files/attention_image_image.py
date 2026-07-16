# cache_prefill_img_self_attn_with_mapping_npz_chattemplate.py
"""
Prefill-step (prompt self-attention) caching for Llava-NeXT / LLaVA-NeXT-style models:
- Uses processor.apply_chat_template(conversation, add_generation_prompt=True)
- Runs ONE prefill forward pass with output_attentions=True to get prompt self-attn
- Builds packed image-token mapping (token_idx -> kind/row/col/bbox) like your IG script
- Extracts image-token x image-token attention matrices per layer:
    * base-only (base_patch x base_patch)
    * mosaic-only (mosaic_patch x mosaic_patch)
    * optional full image-token (all packed image tokens, incl newline/base_cls if enabled)
- Saves into ONE compressed NPZ per sample (or optionally one NPZ per layer)

Outputs (per sample NPZ):
    - base_attn_LBB:  (L, B, B) float16  (B = #base_patch tokens)
    - mosaic_attn_LMM:(L, M, M) float16  (M = #mosaic_patch tokens)
    - (optional) img_attn_LNN: (L, N, N) float16  (N = #all packed image tokens)
    - base_token_idx: int32 indices into packed token space [0..N-1] for base_patch tokens
    - mosaic_token_idx:int32 indices into packed token space [0..N-1] for mosaic_patch tokens
    - meta: json string
    - packed_mapping_summary: json string
    - packed_mapping_tokens: json string (full mapping list)
    - image_placeholder_positions: int32 array (prompt indices for packed image tokens; len=N)
    - prompt_input_ids: int32 array
    - attention_mask: int32 array (if present)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List, Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.image_processing_utils import select_best_resolution


# -----------------------------
# Dataset utils
# -----------------------------
def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace(" ", "_")


def get_or_download_hf_dataset(dataset_id: str, local_cache_root: str, split: str = "test") -> Dataset:
    local_cache_root = Path(local_cache_root)
    local_cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = local_cache_root / sanitize_repo_id(dataset_id)

    if cache_dir.exists():
        print(f"Loading dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    print(f"Downloading dataset '{dataset_id}' from Hugging Face...")
    ds = load_dataset(dataset_id, split=split)
    try:
        ds.save_to_disk(str(cache_dir))
        print(f"Saved dataset to cache: {cache_dir}")
    except Exception as e:
        print(f"Warning: failed to save dataset to disk: {e}")
    return ds


def load_qid_whitelist(path: str) -> Set[str]:
    qids = set()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                qids.add(s)
    return qids


# -----------------------------
# Prompt formatting
# -----------------------------
def format_mcq_prompt(question: str, options: Dict[str, str]) -> str:
    instruction = "Answer the following multiple-choice question by selecting the correct option."
    prompt = f"{instruction}\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Options:\n"
    for k in ["A", "B", "C", "D"]:
        if k in options:
            prompt += f"{k}) {options[k]}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def build_options_from_sample(sample: Dict, shuffle: bool = False, seed: int = 0) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
    labels = ["A", "B", "C", "D"]
    candidates = [
        {"key": "correct_answer", "text": sample["correct_answer"]["text"]},
        {"key": "misleading_groundable", "text": sample["misleading_groundable"]["text"]},
        {"key": "misleading_ungroundable", "text": sample["misleading_ungroundable"]["text"]},
        {"key": "irrelevant_word", "text": sample["irrelevant_word"]["text"]},
    ]

    qid = str(sample.get("question_id", ""))
    if shuffle:
        import random
        rng = random.Random(f"{seed}_{qid}")
        rng.shuffle(candidates)

    options = {labels[i]: candidates[i]["text"] for i in range(4)}
    correct_index = next(i for i, c in enumerate(candidates) if c["key"] == "correct_answer")
    correct_letter = labels[correct_index]

    option_meta = {
        "order": [c["key"] for c in candidates],
        "label_to_key": {labels[i]: candidates[i]["key"] for i in range(4)},
        "shuffle": bool(shuffle),
        "seed": int(seed),
    }
    return options, correct_letter, option_meta


# -----------------------------
# Image placeholder positions
# -----------------------------
def get_image_placeholder_positions(model, input_ids: torch.Tensor) -> List[int]:
    image_token_id = model.config.image_token_id
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    return pos.tolist()


# -----------------------------
# Packed image-token mapping logic
# -----------------------------
def get_anyres_image_grid_shape(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> Tuple[int, int]:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    return best_h // tile_size, best_w // tile_size


def image_size_to_num_tiles_plus_base(image_size_hw: Tuple[int, int], grid_pinpoints, tile_size: int) -> int:
    best_h, best_w = select_best_resolution(list(image_size_hw), grid_pinpoints)
    num_tiles = 0
    for _ in range(0, best_h, tile_size):
        for _ in range(0, best_w, tile_size):
            num_tiles += 1
    num_tiles += 1  # base tile
    return num_tiles


def unpad_params(current_h: int, current_w: int, original_h: int, original_w: int) -> Dict[str, Any]:
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h
    if orig_ar > curr_ar:
        scale = current_w / original_w
        new_h = int(round(original_h * scale, 7))
        pad = (current_h - new_h) // 2
        return {"mode": "slice_h", "scale": scale, "pad": pad, "new_h": new_h, "new_w": current_w}
    else:
        scale = current_h / original_h
        new_w = int(round(original_w * scale, 7))
        pad = (current_w - new_w) // 2
        return {"mode": "slice_w", "scale": scale, "pad": pad, "new_h": current_h, "new_w": new_w}


def bbox_from_unpadded_grid_to_original(
    r0: float, c0: float, r1: float, c1: float,
    current_h: int, current_w: int,
    original_h: int, original_w: int,
) -> Tuple[float, float, float, float]:
    orig_ar = original_w / original_h
    curr_ar = current_w / current_h
    scale = (current_w / original_w) if (orig_ar > curr_ar) else (current_h / original_h)

    y0, y1 = r0 / scale, r1 / scale
    x0, x1 = c0 / scale, c1 / scale

    y0 = max(0.0, min(float(original_h), y0))
    y1 = max(0.0, min(float(original_h), y1))
    x0 = max(0.0, min(float(original_w), x0))
    x1 = max(0.0, min(float(original_w), x1))
    return (y0, x0, y1, x1)


def build_packed_image_token_map_for_one_image(
    *,
    model: LlavaNextForConditionalGeneration,
    original_size_hw: Tuple[int, int],      # (H, W)
    num_views_provided: int,                # pixel_values.shape[1] if anyres
    include_newline_tokens: bool = True,
) -> Dict[str, Any]:
    cfg = model.config
    vision_cfg = cfg.vision_config

    tile_size = int(vision_cfg.image_size)
    patch_size = int(vision_cfg.patch_size)
    patches_per_side = tile_size // patch_size
    per_tile_patch_tokens = patches_per_side * patches_per_side

    cls_removed = (cfg.vision_feature_select_strategy == "default")
    per_tile_tokens_after_select = per_tile_patch_tokens if cls_removed else (per_tile_patch_tokens + 1)

    expected_tiles_plus_base = image_size_to_num_tiles_plus_base(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    used_tiles_plus_base = min(num_views_provided, expected_tiles_plus_base)
    non_base_tiles = max(0, used_tiles_plus_base - 1)

    num_tile_h, num_tile_w = get_anyres_image_grid_shape(original_size_hw, cfg.image_grid_pinpoints, tile_size)
    expected_non_base_tiles_for_layout = num_tile_h * num_tile_w

    mosaic_h = num_tile_h * patches_per_side
    mosaic_w = num_tile_w * patches_per_side

    orig_h, orig_w = original_size_hw
    params = unpad_params(mosaic_h, mosaic_w, orig_h, orig_w)
    if params["mode"] == "slice_h":
        unpadded_h, unpadded_w = params["new_h"], mosaic_w
    else:
        unpadded_h, unpadded_w = mosaic_h, params["new_w"]

    row_width_with_newline = unpadded_w + (1 if include_newline_tokens else 0)

    base_tokens = per_tile_tokens_after_select
    mosaic_tokens = unpadded_h * row_width_with_newline
    total_packed_image_tokens = base_tokens + mosaic_tokens

    tokens: List[Dict[str, Any]] = []

    # Base tokens
    for t in range(base_tokens):
        if not cls_removed and t == 0:
            tokens.append({"token_idx": t, "kind": "base_cls", "row": None, "col": None, "bbox": None})
            continue
        patch_t = t if cls_removed else (t - 1)
        r = patch_t // patches_per_side
        c = patch_t % patches_per_side
        bbox = bbox_from_unpadded_grid_to_original(
            r, c, r + 1, c + 1,
            patches_per_side, patches_per_side,
            orig_h, orig_w
        )
        tokens.append({"token_idx": t, "kind": "base_patch", "row": int(r), "col": int(c), "bbox": bbox})

    # Mosaic + newline tokens
    base_offset = base_tokens
    for k in range(mosaic_tokens):
        token_idx = base_offset + k
        row = k // row_width_with_newline
        col = k % row_width_with_newline
        if include_newline_tokens and col == unpadded_w:
            tokens.append({"token_idx": token_idx, "kind": "newline", "row": int(row), "col": None, "bbox": None})
            continue
        bbox = bbox_from_unpadded_grid_to_original(
            row, col, row + 1, col + 1,
            unpadded_h, unpadded_w,
            orig_h, orig_w
        )
        tokens.append({"token_idx": token_idx, "kind": "mosaic_patch", "row": int(row), "col": int(col), "bbox": bbox})

    return {
        "summary": {
            "original_size_hw": original_size_hw,
            "tile_size": tile_size,
            "patch_size": patch_size,
            "patches_per_side": patches_per_side,
            "cls_removed": cls_removed,
            "num_views_provided": num_views_provided,
            "expected_tiles_plus_base_from_image_sizes": expected_tiles_plus_base,
            "used_tiles_plus_base_after_slicing": used_tiles_plus_base,
            "non_base_tiles_used": non_base_tiles,
            "tile_layout_hw": (num_tile_h, num_tile_w),
            "expected_non_base_tiles_for_layout": expected_non_base_tiles_for_layout,
            "mosaic_pre_unpad_hw_in_patches": (mosaic_h, mosaic_w),
            "mosaic_unpadded_hw_in_patches": (unpadded_h, unpadded_w),
            "newline_tokens_enabled": include_newline_tokens,
            "row_width_with_newline": row_width_with_newline,
            "base_tokens": base_tokens,
            "mosaic_tokens": mosaic_tokens,
            "total_packed_image_tokens": total_packed_image_tokens,
        },
        "tokens": tokens
    }


# -----------------------------
# Prefill (prompt) self-attention extraction
# -----------------------------
@torch.no_grad()
def prefill_head_avg_attn_prompt_self(
    model: LlavaNextForConditionalGeneration,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns head-averaged prompt self-attention per layer.
    Shape: (L, S, S) float16 on CPU
    """
    S = int(inputs["input_ids"].shape[1])

    out = model(
        **inputs,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )
    attns = out.attentions  # tuple(L) of (B,H,S,S)

    if attns is None or len(attns) == 0:
        raise RuntimeError("No attentions returned. Ensure attn_implementation='eager' at model load.")

    layer_mats = []
    for layer_attn in attns:
        if layer_attn is None or layer_attn.ndim != 4:
            raise RuntimeError(f"Unexpected attention tensor: {None if layer_attn is None else layer_attn.shape}")
        # (B,H,S,S) -> (S,S)
        A = layer_attn[0].mean(dim=0)
        layer_mats.append(A)

    A = torch.stack(layer_mats, dim=0)  # (L,S,S)
    attn_np = A.to(torch.float16).cpu().numpy()

    info = {"num_layers": int(attn_np.shape[0]), "prompt_seq_len": int(S)}
    return attn_np, info


def split_image_token_indices(mapping_tokens: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices in packed token space [0..N-1] for base_patch and mosaic_patch tokens.
    Excludes newline and base_cls by default.
    """
    base_idx: List[int] = []
    mosaic_idx: List[int] = []
    for t in mapping_tokens:
        k = t["kind"]
        i = int(t["token_idx"])
        if k == "base_patch":
            base_idx.append(i)
        elif k == "mosaic_patch":
            mosaic_idx.append(i)
    return np.asarray(base_idx, dtype=np.int64), np.asarray(mosaic_idx, dtype=np.int64)


def extract_image_self_attn_per_layer(
    attn_LSS: np.ndarray,         # (L,S,S)
    img_pos: List[int],           # length N (prompt indices for packed image tokens)
    base_idx: np.ndarray,         # indices into packed token space (0..N-1)
    mosaic_idx: np.ndarray,       # indices into packed token space (0..N-1)
    save_full_img: bool,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    img_pos = np.asarray(img_pos, dtype=np.int64)  # (N,)

    # (L,N,N) using advanced indexing:
    # first select rows by img_pos -> (L,N,S), then cols by img_pos -> (L,N,N)
    img_attn = attn_LSS[:, img_pos][:, :, img_pos]  # (L,N,N)

    base_attn = img_attn[:, base_idx][:, :, base_idx]       # (L,B,B)
    mosaic_attn = img_attn[:, mosaic_idx][:, :, mosaic_idx] # (L,M,M)

    if not save_full_img:
        img_attn = None

    return img_attn, base_attn, mosaic_attn


# -----------------------------
# Save NPZ
# -----------------------------
def save_npz_sample(
    out_dir: str,
    qid: str,
    variant: str,
    *,
    meta: Dict[str, Any],
    mapping: Dict[str, Any],
    image_placeholder_positions: List[int],
    prompt_input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
    base_attn_LBB: np.ndarray,
    mosaic_attn_LMM: np.ndarray,
    base_idx: np.ndarray,
    mosaic_idx: np.ndarray,
    img_attn_LNN: Optional[np.ndarray] = None,
    mode: str = "prefill_img_self",
):
    sample_dir = Path(out_dir) / variant / qid
    sample_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sample_dir / f"{mode}.npz"

    save_dict: Dict[str, Any] = {
        "meta": json.dumps(meta),
        "packed_mapping_summary": json.dumps(mapping["summary"]),
        "packed_mapping_tokens": json.dumps(mapping["tokens"]),
        "image_placeholder_positions": np.asarray(image_placeholder_positions, dtype=np.int32),
        "prompt_input_ids": prompt_input_ids.astype(np.int32),
        "base_token_idx": np.asarray(base_idx, dtype=np.int32),
        "mosaic_token_idx": np.asarray(mosaic_idx, dtype=np.int32),
        "base_attn_LBB": base_attn_LBB.astype(np.float16),
        "mosaic_attn_LMM": mosaic_attn_LMM.astype(np.float16),
    }
    if img_attn_LNN is not None:
        save_dict["img_attn_LNN"] = img_attn_LNN.astype(np.float16)

    if attention_mask is not None:
        save_dict["attention_mask"] = attention_mask.astype(np.int32)

    np.savez_compressed(str(npz_path), **save_dict)


def save_npz_per_layer(
    out_dir: str,
    qid: str,
    variant: str,
    *,
    meta: Dict[str, Any],
    mapping: Dict[str, Any],
    image_placeholder_positions: List[int],
    prompt_input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray],
    base_attn_LBB: np.ndarray,
    mosaic_attn_LMM: np.ndarray,
    base_idx: np.ndarray,
    mosaic_idx: np.ndarray,
    img_attn_LNN: Optional[np.ndarray],
    mode: str = "prefill_img_self",
    layers_to_save: Optional[List[int]] = None,
):
    """
    Writes one NPZ per layer under: out_dir/variant/qid/layers/{mode}_layerXX.npz
    """
    L = int(base_attn_LBB.shape[0])
    if layers_to_save is None:
        layers = list(range(L))
    else:
        layers = []
        for x in layers_to_save:
            if x < 0:
                x = L + x
            if 0 <= x < L:
                layers.append(x)
        layers = sorted(set(layers))

    layer_dir = Path(out_dir) / variant / qid / "layers"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for li in layers:
        npz_path = layer_dir / f"{mode}_layer{li:02d}.npz"
        meta_layer = dict(meta)
        meta_layer["layer_index"] = int(li)

        save_dict: Dict[str, Any] = {
            "meta": json.dumps(meta_layer),
            "packed_mapping_summary": json.dumps(mapping["summary"]),
            "packed_mapping_tokens": json.dumps(mapping["tokens"]),
            "image_placeholder_positions": np.asarray(image_placeholder_positions, dtype=np.int32),
            "prompt_input_ids": prompt_input_ids.astype(np.int32),
            "base_token_idx": np.asarray(base_idx, dtype=np.int32),
            "mosaic_token_idx": np.asarray(mosaic_idx, dtype=np.int32),
            "base_attn_BB": base_attn_LBB[li].astype(np.float16),
            "mosaic_attn_MM": mosaic_attn_LMM[li].astype(np.float16),
        }
        if img_attn_LNN is not None:
            save_dict["img_attn_NN"] = img_attn_LNN[li].astype(np.float16)
        if attention_mask is not None:
            save_dict["attention_mask"] = attention_mask.astype(np.int32)

        np.savez_compressed(str(npz_path), **save_dict)


def parse_layers_arg(s: str) -> Optional[List[int]]:
    """
    Examples:
      "all" -> None
      "0,1,2" -> [0,1,2]
      "-1" -> [-1]
      "0,5,-1" -> [0,5,-1]
    """
    s = s.strip().lower()
    if s == "all" or s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    parser.add_argument("--hf_cache_dir", type=str, default="../integrated_gradients/hf_dataset_GUIC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--variant", type=str, default="misleading_groundable")
    parser.add_argument("--qid_file", type=str, default="../inference/no_overlap_question_ids.txt")
    parser.add_argument("--out_dir", type=str, default="attn_cache_prefill")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # prompt consistency
    parser.add_argument("--shuffle_options", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # mapping
    parser.add_argument("--include_newline_tokens", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500, help="End index (exclusive). 0 = no limit.")

    # saving control
    parser.add_argument("--save_full_img_attn", action="store_true",
                        help="If set, also save img_attn_LNN for all packed image tokens (can be huge).")
    parser.add_argument("--per_layer_files", action="store_true",
                        help="If set, also save one NPZ per layer under variant/qid/layers/.")
    parser.add_argument("--layers", type=str, default="all",
                        help='Layers to save for per-layer files, e.g. "all" or "0,1,2,-1"')

    args = parser.parse_args()

    qids = load_qid_whitelist(args.qid_file)
    print(f"Loaded {len(qids)} question_ids from: {args.qid_file}")

    ds = get_or_download_hf_dataset(args.hf_dataset, args.hf_cache_dir, split=args.split)
    if isinstance(ds, DatasetDict):
        ds = ds[args.split]

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    device_t = torch.device(device)

    print(f"Loading model: {args.model_id} on {device}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for output_attentions reliably
    ).to(device_t)
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(args.model_id)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    existing_subdirs = [d.name for d in out_root.iterdir() if d.is_dir()]
    existing_subdirs = set(existing_subdirs)
    print(f"Found {len(existing_subdirs)} existing subdirectories in {out_root}")

    layers_to_save = parse_layers_arg(args.layers)

    kept = 0
    for i in tqdm(range(len(ds)), desc=f"Caching prefill img self-attn [{args.variant}]"):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i >= args.end:
            break

        sample = ds[i]
        qid = str(sample.get("question_id", f"unknown_{i}"))
        if qid not in qids:
            continue
        if qid in existing_subdirs:
            # Skip only if the primary file exists; keep same behavior as your script.
            maybe = (Path(args.out_dir) / args.variant / qid / "prefill_img_self.npz")
            if maybe.exists():
                continue

        # choose image variant
        if args.variant == "notext":
            img = sample["notext"]["image"]
        else:
            if args.variant not in sample:
                continue
            img = sample[args.variant]["image"]
        img = img.convert("RGB")

        # build prompt
        options, correct_letter, option_meta = build_options_from_sample(
            sample, shuffle=args.shuffle_options, seed=args.seed
        )
        prompt_text = format_mcq_prompt(sample.get("question", ""), options)

        # evaluator-style chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,  # NOTE: correct kw is tokenize
        )

        inputs = processor(images=img, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(device_t) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16 if device == "cuda" else torch.float32)

        # prompt ids + placeholder positions
        input_ids = inputs["input_ids"]
        img_pos = get_image_placeholder_positions(model, input_ids)

        # mapping prerequisites
        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is None:
            raise RuntimeError("Processor did not return image_sizes; packed mapping needs original H,W.")
        orig_h = int(image_sizes[0, 0].item())
        orig_w = int(image_sizes[0, 1].item())

        pixel_values = inputs.get("pixel_values", None)
        if pixel_values is None:
            raise RuntimeError("Processor did not return pixel_values; packed mapping needs num_views_provided.")
        num_views = int(pixel_values.shape[1]) if pixel_values.ndim == 5 else 1

        packed_mapping = build_packed_image_token_map_for_one_image(
            model=model,
            original_size_hw=(orig_h, orig_w),
            num_views_provided=num_views,
            include_newline_tokens=bool(args.include_newline_tokens),
        )

        expected_img_tokens = int(packed_mapping["summary"]["total_packed_image_tokens"])
        if len(img_pos) != expected_img_tokens:
            print(
                f"[{qid}] Packed mapping mismatch: len(image_placeholder_positions)={len(img_pos)} != expected={expected_img_tokens}. "
                "Skipping because alignment is unsafe."
            )
            continue

        # prefill prompt self-attn (L,S,S)
        attn_LSS, attn_info = prefill_head_avg_attn_prompt_self(model, inputs)

        # split token indices in packed space
        base_idx, mosaic_idx = split_image_token_indices(packed_mapping["tokens"])

        # extract matrices
        img_attn_LNN, base_attn_LBB, mosaic_attn_LMM = extract_image_self_attn_per_layer(
            attn_LSS=attn_LSS,
            img_pos=img_pos,
            base_idx=base_idx,
            mosaic_idx=mosaic_idx,
            save_full_img=bool(args.save_full_img_attn),
        )

        # meta
        meta = {
            "question_id": qid,
            "variant": args.variant,
            "prompt_seq_len": int(input_ids.shape[1]),
            "image_placeholder_count": len(img_pos),  # N (all packed image tokens)
            "correct_letter": correct_letter,
            "option_meta": option_meta,
            "original_image_size_hw": [orig_h, orig_w],
            "num_views_provided": num_views,
            "include_newline_tokens": bool(args.include_newline_tokens),
            "N_img_tokens_total": int(len(img_pos)),
            "N_base_patch_tokens": int(base_idx.size),
            "N_mosaic_patch_tokens": int(mosaic_idx.size),
            **attn_info,
        }

        prompt_input_ids = input_ids[0].detach().cpu().numpy()
        attention_mask = inputs["attention_mask"][0].detach().cpu().numpy() if "attention_mask" in inputs else None

        # save one NPZ per sample
        save_npz_sample(
            args.out_dir,
            qid,
            args.variant,
            meta=meta,
            mapping=packed_mapping,
            image_placeholder_positions=img_pos,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            base_attn_LBB=base_attn_LBB,
            mosaic_attn_LMM=mosaic_attn_LMM,
            base_idx=base_idx,
            mosaic_idx=mosaic_idx,
            img_attn_LNN=img_attn_LNN,
            mode="prefill_img_self",
        )

        # optionally save per-layer NPZs
        if args.per_layer_files:
            save_npz_per_layer(
                args.out_dir,
                qid,
                args.variant,
                meta=meta,
                mapping=packed_mapping,
                image_placeholder_positions=img_pos,
                prompt_input_ids=prompt_input_ids,
                attention_mask=attention_mask,
                base_attn_LBB=base_attn_LBB,
                mosaic_attn_LMM=mosaic_attn_LMM,
                base_idx=base_idx,
                mosaic_idx=mosaic_idx,
                img_attn_LNN=img_attn_LNN,  # may be None
                mode="prefill_img_self",
                layers_to_save=layers_to_save,
            )

        kept += 1
        if args.max_samples > 0 and kept >= args.max_samples:
            break

    print(f"Done. Cached {kept} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()