#!/usr/bin/env python3
"""Recover eager full-answer baselines for the fixed relevance pilot; no new interventions."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(BASE / "attention_intervention/main_files"),
                str(BASE / "free_generation_intervention/main_files")]
from run_attention_full import candidate_run
from run_free_generation_gate import (MODEL_ID, REVISIONS, CACHE, prepare, torch,
    load_from_disk, AutoProcessor, Qwen3VLForConditionalGeneration, save_json, log)


@torch.inference_mode()
def main():
    pilot = HERE / "outputs/pilot_12_cached"
    config = json.loads((pilot / "configuration.json").read_text())
    output = HERE / "outputs/relevance_score_followup/pilot_baselines"
    output.mkdir(parents=True, exist_ok=True)
    variants = ("misleading_groundable", "misleading_ungroundable", "irrelevant_word")
    assert config["model"] == MODEL_ID and config["revision"] == REVISIONS[MODEL_ID]
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
        revision=REVISIONS[MODEL_ID], dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation="eager").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISIONS[MODEL_ID])
    dataset = load_from_disk(str(CACHE))
    layers = model.model.language_model.layers
    done = 0
    for entry in config["selected_samples"]:
        qid = str(entry["question_id"])
        sample = dataset[int(entry["dataset_index"])]
        for variant in variants:
            target = output / f"{qid}_{variant}.json"
            if target.exists():
                saved = json.loads(target.read_text())
                assert saved["status"] == "complete" and saved["model_revision"] == REVISIONS[MODEL_ID]
                done += 1
                continue
            paths = list((BASE / "attention_intervention/outputs/full" / variant).glob(f"shard_*/samples/{qid}.json"))
            assert len(paths) == 1
            original = json.loads(paths[0].read_text())
            assert original["dataset_index"] == int(entry["dataset_index"])
            result = {"question_id": qid, "variant": variant, "model_id": MODEL_ID,
                "model_revision": REVISIONS[MODEL_ID], "backend": "eager",
                "source_attention_file": str(paths[0]), "states": {}}
            for state, image in (("no_text", sample["notext"]["image"]),
                                 ("overlay", sample[variant]["cleaned_image"])):
                image = image.convert("RGB")
                batch, _ = prepare(processor, image, original["prompt"], "", "cuda")
                scores = {}
                errors = []
                for role, answer in original["answers"].items():
                    run = candidate_run(model, processor, layers, image, original["prompt"],
                        batch["input_ids"], str(sample["question"]), answer)
                    scores[role] = run["score"]
                    errors.extend([run["cache_error"], run["noop_error"]])
                    del run
                margin = scores["correct"]["mean_logprob"] - scores["comparison"]["mean_logprob"]
                error = abs(margin - original["baseline_margins"][state])
                result["states"][state] = {"scores": scores, "margin": margin,
                    "original_margin": original["baseline_margins"][state],
                    "margin_reproduction_error": error, "max_cache_noop_error": max(errors)}
                # Never silently combine different model baselines.
                if error > 1e-3:
                    result["status"] = "baseline_mismatch"
                    save_json(target.with_suffix(".mismatch.json"), result)
                    raise AssertionError(f"{qid}/{variant}/{state}: original margin differs by {error}")
            result["status"] = "complete"
            save_json(target, result)
            done += 1
            log("BASELINE", f"{done}/36 {qid}/{variant}: exact eager margin reproduction PASS")
    assert done == 36
    save_json(output / "completion.json", {"status": "complete", "pairs": done,
        "questions": len(config["selected_samples"]), "candidate_scores": done * 4})


if __name__ == "__main__":
    main()
