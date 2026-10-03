#!/usr/bin/env python3
"""Fixed 12-question comparison of text-only, gray, clean, and overlay inputs."""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(BASE / "free_generation_intervention/main_files"), str(BASE / "main_files")]
from run_free_generation_gate import (MODEL_ID, REVISIONS, CACHE, SELECTION, VARIANTS, prepare,
    ordinary_generation, torch, load_from_disk, AutoProcessor, Qwen3VLForConditionalGeneration,
    save_json, log, text_bbox_yxyx, validate_cleaned_overlay)
from validate_multitoken_scoring import to_device, image_hash
from sequence_scoring import summarize_answer_score, scored_token_logprobs

OUTPUT = HERE / "outputs/input_comparison_12"
INSTRUCTION = "Answer the question. Give only the short answer without explanation."
STATES = ("question_only", "gray_image", "clean_image", *VARIANTS)


def prepare_input(processor, image, prompt, suffix=""):
    if image is not None:
        return prepare(processor, image, prompt, suffix, "cuda")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[formatted + suffix], return_tensors="pt", padding=True)
    return to_device(dict(batch), "cuda"), formatted


@contextlib.contextmanager
def reuse_visual_features(model, batch):
    """Memoize vision only; decoder and full-sequence scoring are unchanged."""
    if "pixel_values" not in batch:
        yield {"vision_cached": False}
        return
    module = model.model
    original = module.get_image_features
    features = original(batch["pixel_values"], batch["image_grid_thw"])
    calls = {"vision_cached": True, "reused_calls": 0}

    def cached(pixel_values, image_grid_thw):
        assert torch.equal(pixel_values, batch["pixel_values"])
        assert torch.equal(image_grid_thw, batch["image_grid_thw"])
        calls["reused_calls"] += 1
        return features

    had_instance_attribute = "get_image_features" in module.__dict__
    module.get_image_features = cached
    try:
        yield calls
    finally:
        if had_instance_attribute:
            module.get_image_features = original
        else:
            delattr(module, "get_image_features")


def score(model, processor, batch, prompt_length, answer):
    logits = model(**batch, use_cache=False).logits
    values = scored_token_logprobs(logits, batch["input_ids"], prompt_length)
    ids = batch["input_ids"][0, prompt_length:].tolist()
    result = summarize_answer_score(answer, ids,
        [processor.tokenizer.decode([x], skip_special_tokens=False) for x in ids], values)
    assert all(math.isfinite(v) and v <= 1e-6 for v in values)
    assert abs(result["sum_logprob"] - result["token_count"] * result["mean_logprob"]) < 1e-8
    del logits
    return result


@torch.inference_mode()
def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    source = json.loads((HERE / "outputs/pilot_12_cached/configuration.json").read_text())
    entries = [{"question_id": str(e["question_id"]), "dataset_index": int(e["dataset_index"])}
               for e in source["selected_samples"]]
    assert len(entries) == 12 and len({e["question_id"] for e in entries}) == 12
    if args.full:
        manifest = json.loads(SELECTION.read_text())["selected_samples"]
        assert len(manifest) == 305
        entries = [{"question_id": str(e["question_id"]), "dataset_index": int(e["dataset_index"])}
                   for e in sorted(manifest, key=lambda x: str(x["question_id"]))][args.shard::args.shards]
        OUTPUT = HERE / "outputs/input_comparison_305" / f"shard_{args.shard:02d}"
    else:
        assert args.shards == 1
    expected = len(entries) * len(STATES)
    config = {"version": 1, "model": MODEL_ID, "revision": REVISIONS[MODEL_ID],
        "selected_samples": entries, "selection_seed": None if args.full else source["seed"], "selection": "shared 305 manifest" if args.full else "unchanged pilot sample",
        "instruction": INSTRUCTION, "states": list(STATES), "candidate_keys": list(VARIANTS),
        "gray_rgb": [128, 128, 128], "gray_size": "same as original image",
        "backend": "eager", "dtype": "float16", "max_new_tokens": 32, "do_sample": False,
        "scores": "full teacher-forced candidate; mean/sum/first-token log probability; excludes EOS",
        "optimization": "reuse identical vision features within each image; no decoder-prefix reuse",
        "validation": "all four cached scores and generation versus uncached for every visual state on first question"}
    if args.full:
        config.update(shard=args.shard, shards=args.shards)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT / "configuration.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "resume configuration mismatch"
    save_json(config_path, config)
    log("CONFIG", f"{len(entries)} fixed questions, {len(STATES)} states; {expected} generations, {expected*4} candidate scores; prompt={INSTRUCTION!r}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
        revision=REVISIONS[MODEL_ID], dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISIONS[MODEL_ID])
    dataset = load_from_disk(str(CACHE))
    start = time.monotonic()
    completed = 0
    for index, entry in enumerate(entries):
        qid = entry["question_id"]
        sample = dataset[entry["dataset_index"]]
        question = str(sample["question"])
        prompt = f"{INSTRUCTION}\n\nQuestion: {question}"
        assert "Options:" not in prompt
        refs = {v: str(sample[v]["text"]) for v in VARIANTS}
        original = sample["notext"]["image"].convert("RGB")
        images = {"question_only": None, "gray_image": Image.new("RGB", original.size, (128, 128, 128)),
            "clean_image": original, **{v: sample[v]["cleaned_image"].convert("RGB") for v in VARIANTS}}
        aligned_ids = aligned_grid = None
        for state in STATES:
            path = OUTPUT / "samples" / f"{qid}_{state}.json"
            image = images[state]
            batch, formatted = prepare_input(processor, image, prompt)
            prompt_ids = batch["input_ids"]
            prompt_length = int(prompt_ids.shape[1])
            grid = batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else None
            image_tokens = int((prompt_ids == model.config.image_token_id).sum())
            if image is None:
                assert grid is None and image_tokens == 0 and "pixel_values" not in batch
            else:
                assert image.size == original.size and image_tokens > 0
                if aligned_ids is None:
                    aligned_ids, aligned_grid = prompt_ids.clone(), grid
                assert torch.equal(prompt_ids, aligned_ids) and grid == aligned_grid
            if path.exists():
                previous = json.loads(path.read_text())
                assert previous["status"] == "complete" and previous["references"] == refs
                assert previous["prompt_token_ids"] == prompt_ids[0].tolist()
                completed += 1
                continue
            record = {"question_id": qid, "dataset_index": entry["dataset_index"], "state": state,
                "question": question, "prompt": prompt, "formatted_prompt": formatted, "references": refs,
                "prompt_token_ids": prompt_ids[0].tolist(), "image_token_count": image_tokens,
                "image_grid_thw": grid, "image_size_wh": list(image.size) if image else None,
                "image_sha256": image_hash(image) if image else None,
                "paired_visual_ids_and_grid_match": image is not None, "scores": {}}
            if state in VARIANTS:
                check = validate_cleaned_overlay(original, sample[state]["image"].convert("RGB"), image,
                                                 text_bbox_yxyx(sample, state))
                assert check["cleaned_outside_no_text_mismatch_pixels"] == 0
                assert check["cleaned_inside_original_mismatch_pixels"] == 0
                record["cleaned_overlay_validation"] = check
            candidates = {}
            for key, answer in refs.items():
                candidate, candidate_formatted = prepare_input(processor, image, prompt, answer)
                assert candidate_formatted == formatted
                assert torch.equal(candidate["input_ids"][:, :prompt_length], prompt_ids)
                assert candidate["input_ids"].shape[1] > prompt_length
                candidates[key] = candidate
            # Starting each context fresh also guards against stale multimodal RoPE state.
            model.model.rope_deltas = None
            with reuse_visual_features(model, batch) as cache_info:
                for key, answer in refs.items():
                    record["scores"][key] = score(model, processor, candidates[key], prompt_length, answer)
                record["generation"] = ordinary_generation(model, processor, batch, 32)
            record["vision_cache"] = cache_info
            if index == 0 and image is not None:
                errors = []
                for key, answer in refs.items():
                    native = score(model, processor, candidates[key], prompt_length, answer)
                    cached = record["scores"][key]
                    assert native["token_ids"] == cached["token_ids"]
                    errors.extend(abs(a-b) for a,b in zip(native["token_logprobs"],cached["token_logprobs"]))
                native_generation = ordinary_generation(model, processor, batch, 32)
                assert native_generation["output_token_ids"] == record["generation"]["output_token_ids"]
                assert max(errors) <= 1e-3, f"vision memoization changed scores: {max(errors)}"
                record["cache_validation"] = {"max_token_logprob_error": max(errors), "generation_tokens_identical": True}
            record["status"] = "complete"
            save_json(path, record)
            completed += 1
            log("RESULT", f"{completed}/{expected} {qid}/{state}: {record['generation']['raw_response']!r}; image_tokens={image_tokens}; elapsed={(time.monotonic()-start)/60:.1f}min")
            del candidates, batch
        log("QUESTION", f"{index+1}/{len(entries)} complete: {qid}; identical grids across six visual inputs PASS")
    assert completed == expected
    save_json(OUTPUT / "completion.json", {"status": "complete", "questions": len(entries),
        "input_records": completed, "candidate_scores": completed*4, "generations": completed,
        "elapsed_seconds": time.monotonic()-start})
    log("PASS", f"input comparison complete; {expected} generations and {expected*4} candidate scores")
    if args.full:
        # The final successful shard writes the report without a fifth Slurm job.
        import fcntl
        import subprocess
        parent = OUTPUT.parent
        with (parent / "analysis.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if all((parent / f"shard_{i:02d}" / "completion.json").exists() for i in range(args.shards)):
                if not (parent / "analysis_completion.json").exists():
                    subprocess.run([sys.executable, str(HERE / "analyze_full_input_comparison.py")], check=True)


if __name__ == "__main__":
    main()
