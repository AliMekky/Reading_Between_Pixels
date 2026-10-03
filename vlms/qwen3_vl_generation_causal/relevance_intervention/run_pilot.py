#!/usr/bin/env python3
"""Exploratory question/text blocking with paired no-text and random controls."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(BASE / "free_generation_intervention/main_files"),
                str(BASE / "attention_intervention/main_files")]
from run_free_generation_gate import (MODEL_ID, REVISIONS, CACHE, SELECTION, VARIANTS,
    PROMPT_INSTRUCTION, prepare, save_json, log, generate_with_block, ordinary_generation,
    positive_overlap_region_positions, text_bbox_yxyx, validate_cleaned_overlay,
    torch, load_from_disk, AutoProcessor, Qwen3VLForConditionalGeneration)
from run_attention_smoke import question_positions

Q_WINDOWS = {"middle": list(range(18, 24)), "late": list(range(30, 36))}
T_WINDOW = list(range(30, 36))


def plans(question, text, control):
    def combine(q_layers=(), visual=()):
        result = {}
        for layer in q_layers:
            result[layer] = list(question)
        if visual:
            for layer in T_WINDOW:
                result[layer] = sorted(set(result.get(layer, []) + list(visual)))
        return result
    result = {"baseline": {}, "T": combine(visual=text), "R": combine(visual=control)}
    for name, window in Q_WINDOWS.items():
        result[f"Q_{name}"] = combine(window)
        result[f"Q_{name}+T"] = combine(window, text)
        result[f"Q_{name}+R"] = combine(window, control)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/pilot_12")
    parser.add_argument("--reuse-prefix", action="store_true",
                        help="Reuse unaffected prompt KV states; validate against uncached generation")
    args = parser.parse_args()
    entries = json.loads(SELECTION.read_text())["selected_samples"]
    if len(entries) != 305 or not 1 <= args.samples <= len(entries):
        raise ValueError("expected shared 305-question manifest and valid sample count")
    entries = random.Random(args.seed).sample(sorted(entries, key=lambda x: str(x["question_id"])), args.samples)
    config = {"model": MODEL_ID, "revision": REVISIONS[MODEL_ID], "seed": args.seed,
              "selected_samples": entries, "q_windows": Q_WINDOWS, "t_window": T_WINDOW,
              "max_new_tokens": 32, "dtype": "float16", "decoding": "greedy", "backend": "eager",
              "variants": list(VARIANTS), "random_controls": 1, "version": 1}
    if args.reuse_prefix:
        config.update({"version": 2, "reuse_prompt_prefix": True,
                       "prefix_validation": "baseline_every_state_all_arms_first_question"})
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("output belongs to a different configuration")
    save_json(config_path, config)
    expected = args.samples * len(VARIANTS) * 2 * 9
    log("CONFIG", json.dumps(config))
    log("EXPECTED", f"{expected} saved generations; no-op identical to ordinary eager; blocked probability=0; row error<=0.002")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISIONS[MODEL_ID], dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISIONS[MODEL_ID])
    layers = model.model.language_model.layers
    assert len(layers) == 36 and model.config.text_config._attn_implementation == "eager"
    hook_layers = [layers[i] for i in sorted(set(T_WINDOW + Q_WINDOWS["middle"]))]
    dataset = load_from_disk(str(CACHE))
    started, completed = time.monotonic(), 0
    for entry in entries:
        qid, index = str(entry["question_id"]), int(entry["dataset_index"])
        sample = dataset[index]
        prompt = f"{PROMPT_INSTRUCTION}\n\nQuestion: {sample['question']}"
        for variant in VARIANTS:
            path = args.output / "samples" / f"{qid}_{variant}.json"
            record = json.loads(path.read_text()) if path.exists() else {
                "question_id": qid, "variant": variant, "question": sample["question"],
                "references": {v: str(sample[v]["text"]) for v in VARIANTS}, "states": {}}
            images = {"no_text": sample["notext"]["image"].convert("RGB"),
                      "overlay": sample[variant]["cleaned_image"].convert("RGB")}
            clean = validate_cleaned_overlay(images["no_text"], sample[variant]["image"].convert("RGB"),
                                            images["overlay"], text_bbox_yxyx(sample, variant))
            assert clean["cleaned_outside_no_text_mismatch_pixels"] == 0
            assert clean["cleaned_inside_original_mismatch_pixels"] == 0
            record["pixel_validation"] = clean
            paired_ids = paired_visual = paired_groups = None
            for state, image in images.items():
                batch, formatted = prepare(processor, image, prompt, "", "cuda")
                visual, _, _, groups = positive_overlap_region_positions(
                    model, batch, image, processor, sample, variant, args.seed * 100000 + index)
                ids = batch["input_ids"][0].tolist()
                if paired_ids is not None:
                    assert paired_ids == ids and paired_visual == visual and paired_groups == groups
                paired_ids, paired_visual, paired_groups = ids, visual, groups
                q = question_positions(processor, formatted, "", ids, model.config.image_token_id, sample["question"])
                t, r = groups["text_region"], groups["matched_random_region_1"]
                assert t and len(t) == len(r) and not set(q) & set(visual)
                assert not set(t) & set(r) and max(q) < len(ids) - 1
                record["positions"] = {"Q": q, "T": t, "R": r}
                record["alignment_passed"] = True
                log("MAPPING", f"qid={qid} {variant}/{state}: Q={len(q)} T={len(t)} R={len(r)}; equal-size/disjoint/alignment PASS")
                results = record["states"].setdefault(state, {})
                prefix = None
                if args.reuse_prefix and len(results) < 9:
                    # Only the final prompt query and generated queries are intervened on.
                    # Causal masking makes earlier prompt KV states independent of these
                    # interventions. Drop the final query's cached KV before reusing.
                    with torch.inference_mode():
                        prefetched = model(**batch, use_cache=True)
                        prefix = prefetched.past_key_values
                        prefix.crop(len(ids) - 1)
                        del prefetched
                    assert prefix.get_seq_length() == len(ids) - 1
                for arm, sources in plans(q, t, r).items():
                    if arm not in results:
                        current_batch = dict(batch)
                        if prefix is not None:
                            current_batch["past_key_values"] = copy.deepcopy(prefix)
                        result = generate_with_block(model, processor, current_batch, hook_layers, sources, 32)
                        if arm == "baseline":
                            plain = ordinary_generation(model, processor, batch, 32)
                            assert plain["output_token_ids"] == result["output_token_ids"], "no-op altered generation"
                            result["no_op_passed"] = True
                        if args.reuse_prefix and entry == entries[0]:
                            uncached = generate_with_block(model, processor, batch, hook_layers, sources, 32)
                            assert uncached["output_token_ids"] == result["output_token_ids"], "prefix reuse altered intervention generation"
                            result["prefix_equivalence_passed"] = True
                        del current_batch
                        result["sources_by_layer"] = sources
                        results[arm] = result
                        save_json(path, record)
                    completed += 1
                    log("RESULT", f"{completed}/{expected} {qid}/{variant}/{state}/{arm}: {results[arm]['raw_response']!r}")
            record["status"] = "complete"
            save_json(path, record)
            log("PROGRESS", f"saved={completed}/{expected}; elapsed={(time.monotonic()-started)/60:.1f}min checkpoint={path}")
    assert completed == expected
    save_json(args.output / "completion.json", {"status": "complete", "questions": args.samples,
        "saved_generations": completed, "elapsed_seconds": time.monotonic() - started})
    log("PASS", f"pilot complete: {completed} records; run summarize_pilot.py for conservative exact-match results")


if __name__ == "__main__":
    main()
