#!/usr/bin/env python3
"""Classify free-generation intervention outputs and export only novel unresolved texts."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OPEN_EVAL = ROOT / "vlms/open_ended_evaluation"
sys.path.insert(0, str(OPEN_EVAL / "main_files"))

from evaluate_deterministic import (REFERENCE_IDS, classify, judge_reference_orders,  # noqa: E402
                                    normalize_answer)


VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")
REGIONS = ("text_region", "matched_random_region_1", "matched_random_region_2", "matched_random_region_3")
MODEL_SLUG = "Qwen__Qwen3-VL-8B-Instruct"


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def semantic_id(question_id, response):
    payload = f"{question_id}|{normalize_answer(response)}"
    return "oei-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--baseline_classification_dir", type=Path,
                        default=OPEN_EVAL / "outputs/final_classification_six_models" / MODEL_SLUG)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = {
        condition: {str(row["question_id"]): row for row in read_jsonl(
            args.baseline_classification_dir / f"{condition}.jsonl"
        )}
        for condition in ("notext", *VARIANTS)
    }
    records, unresolved, counts = [], {}, Counter()
    seen = set()

    for variant in VARIANTS:
        sample_paths = sorted(
            path for shard in (0, 1)
            for path in (args.input_dir / variant / f"shard_{shard}" / "samples").glob("*.json")
        )
        if len(sample_paths) != 305:
            raise AssertionError(f"{variant}: expected 305 samples, found {len(sample_paths)}")
        for path in sample_paths:
            sample = json.loads(path.read_text())
            qid = str(sample["question_id"])
            if (variant, qid) in seen or sample.get("status") != "complete":
                raise AssertionError(f"duplicate or incomplete sample: {variant}/{qid}")
            seen.add((variant, qid))
            references = sample["references"]

            for state in ("no_text", "overlay"):
                condition = "notext" if state == "no_text" else variant
                inherited = baseline[condition].get(qid)
                if inherited is None:
                    raise AssertionError(f"missing prior classification: {condition}/{qid}")
                runs = {"baseline": sample["states"][state]["empty_hook_baseline"],
                        **sample["states"][state]["interventions"]}
                response_cache = {}

                for intervention, generation in runs.items():
                    record_id = f"q3-free-intervention|{variant}|{qid}|{state}|{intervention}"
                    base = {
                        "record_id": record_id, "status": "ok", "model_id": sample["model_id"],
                        "question_id": qid, "variant": variant, "image_state": state,
                        "intervention": intervention, "layers": sample["layers"],
                        "question": sample["question"], "raw_response": generation["raw_response"],
                        "output_token_ids": generation["output_token_ids"],
                        "output_token_count": generation["output_token_count"],
                        "reference_answers": references,
                    }
                    result = classify(base, args.threshold)
                    normalized = result["normalized_response"]
                    if normalized in response_cache:
                        source = response_cache[normalized]
                        result.update({key: source.get(key) for key in (
                            "final_category", "final_stage", "judge_case_id", "classification_source"
                        )})
                    elif normalized == inherited["normalized_response"]:
                        result.update({
                            "final_category": inherited["final_category"],
                            "final_stage": inherited["final_stage"],
                            "judge_case_id": None, "classification_source": "inherited_prior_judgment",
                        })
                    elif not result["needs_llm_judge"]:
                        result.update({
                            "final_category": result["deterministic_category"],
                            "final_stage": result["deterministic_stage"],
                            "judge_case_id": None, "classification_source": "deterministic",
                        })
                    else:
                        case_id = semantic_id(qid, generation["raw_response"])
                        result.update({
                            "final_category": None, "final_stage": None,
                            "judge_case_id": case_id, "classification_source": "pending_judges",
                        })
                        if case_id not in unresolved:
                            semantic_record = {**result, "record_id": case_id}
                            order_1, order_2 = judge_reference_orders(semantic_record, args.seed)
                            unresolved[case_id] = {
                                "judge_case_id": case_id, "record_id": case_id,
                                "question_id": qid, "model_id": sample["model_id"],
                                "variant": variant, "question": sample["question"],
                                "raw_response": generation["raw_response"], "references": references,
                                "judge_1_reference_order": order_1,
                                "judge_2_reference_order": order_2,
                            }
                    response_cache[normalized] = result
                    counts[("stage", result["deterministic_stage"])] += 1
                    counts[("source", result["classification_source"])] += 1
                    records.append(result)

    expected = 4 * 305 * 2 * (1 + len(REGIONS))
    if len(records) != expected or len(seen) != 4 * 305:
        raise AssertionError(f"record accounting failed: records={len(records)} samples={len(seen)}")
    if any(set(row["normalized_references"]) != set(REFERENCE_IDS) for row in records):
        raise AssertionError("reference IDs are incomplete")

    with (args.output_dir / "records.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    judge_dir = args.output_dir / MODEL_SLUG
    judge_dir.mkdir(exist_ok=True)
    with (judge_dir / "unresolved_for_judges.jsonl").open("w") as handle:
        for row in unresolved.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "complete", "records": len(records), "sample_condition_pairs": len(seen),
        "unique_unresolved_for_judges": len(unresolved),
        "deterministic_stage_counts": {key: value for (kind, key), value in counts.items() if kind == "stage"},
        "classification_source_counts": {key: value for (kind, key), value in counts.items() if kind == "source"},
        "edit_similarity_threshold": args.threshold,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[PASS] {summary}")


if __name__ == "__main__":
    main()
