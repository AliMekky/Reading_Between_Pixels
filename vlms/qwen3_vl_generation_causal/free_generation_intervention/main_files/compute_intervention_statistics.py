#!/usr/bin/env python3
"""Compute question-paired causal generation metrics and text/random contrasts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")
STATES = ("no_text", "overlay")
REGIONS = ("text_region", "matched_random_region_1", "matched_random_region_2", "matched_random_region_3")
METRICS = (
    "answer_changed", "accuracy_change", "target_following_change", "target_removal",
    "recovery", "targeted_recovery", "harmful_flip", "targeted_harmful_flip",
    "transition_to_other_or_ambiguous",
)


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def measures(before, after, token_changed, target):
    before_correct, after_correct = before == "correct_answer", after == "correct_answer"
    before_target, after_target = before == target, after == target
    return {
        "answer_changed": float(token_changed),
        "accuracy_change": float(after_correct) - float(before_correct),
        "target_following_change": float(after_target) - float(before_target),
        "target_removal": float(before_target and not after_target),
        "recovery": float(not before_correct and after_correct),
        "targeted_recovery": float(before_target and after_correct and before != after),
        "harmful_flip": float(before_correct and not after_correct),
        "targeted_harmful_flip": float(before_correct and after_target and before != after),
        "transition_to_other_or_ambiguous": float(before != after and after in {"other", "ambiguous"}),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lookup = {}
    with args.records.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["variant"], str(row["question_id"]), row["image_state"], row["intervention"])
            if key in lookup:
                raise AssertionError(f"duplicate record {key}")
            lookup[key] = row
    if len(lookup) != 12200:
        raise AssertionError(f"expected 12200 records, found {len(lookup)}")

    metric_rows, contrast_rows, transitions = [], [], []
    rng = np.random.default_rng(args.seed)
    for variant in VARIANTS:
        qids = sorted({key[1] for key in lookup if key[0] == variant})
        if len(qids) != 305:
            raise AssertionError(f"{variant}: expected 305 questions")
        indices = rng.integers(0, len(qids), size=(args.resamples, len(qids)))
        arrays = {}
        for state in STATES:
            for region in REGIONS:
                values = {metric: [] for metric in METRICS}
                for qid in qids:
                    baseline = lookup[(variant, qid, state, "baseline")]
                    blocked = lookup[(variant, qid, state, region)]
                    current = measures(
                        baseline["final_category"], blocked["final_category"],
                        baseline["output_token_ids"] != blocked["output_token_ids"], variant,
                    )
                    for metric, value in current.items():
                        values[metric].append(value)
                    transitions.append({
                        "variant": variant, "question_id": qid, "image_state": state, "region": region,
                        "baseline_response": baseline["raw_response"], "blocked_response": blocked["raw_response"],
                        "baseline_category": baseline["final_category"], "blocked_category": blocked["final_category"],
                        **current,
                    })
                for metric, vector in values.items():
                    vector = np.asarray(vector, dtype=float)
                    arrays[(state, region, metric)] = vector
                    low, high = interval(vector, indices)
                    metric_rows.append({
                        "variant": variant, "image_state": state, "region": region, "metric": metric,
                        "n": len(qids), "estimate": float(vector.mean()), "ci_low": low, "ci_high": high,
                    })

        for state in STATES:
            for metric in METRICS:
                text = arrays[(state, "text_region", metric)]
                random = np.mean([arrays[(state, region, metric)] for region in REGIONS[1:]], axis=0)
                contrast = text - random
                low, high = interval(contrast, indices)
                contrast_rows.append({
                    "variant": variant, "contrast": "text_minus_mean_random", "image_state": state,
                    "metric": metric, "n": len(qids), "estimate": float(contrast.mean()),
                    "ci_low": low, "ci_high": high,
                })
        for metric in METRICS:
            overlay = arrays[("overlay", "text_region", metric)] - np.mean(
                [arrays[("overlay", region, metric)] for region in REGIONS[1:]], axis=0)
            no_text = arrays[("no_text", "text_region", metric)] - np.mean(
                [arrays[("no_text", region, metric)] for region in REGIONS[1:]], axis=0)
            contrast = overlay - no_text
            low, high = interval(contrast, indices)
            contrast_rows.append({
                "variant": variant, "contrast": "overlay_minus_no_text_text_minus_random",
                "image_state": "difference_in_differences", "metric": metric, "n": len(qids),
                "estimate": float(contrast.mean()), "ci_low": low, "ci_high": high,
            })

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "contrasts.csv", contrast_rows)
    write_csv(args.output_dir / "transitions.csv", transitions)
    summary = {
        "status": "complete", "records": len(lookup), "questions_per_variant": 305,
        "metric_rows": len(metric_rows), "contrast_rows": len(contrast_rows),
        "transition_rows": len(transitions), "bootstrap_resamples": args.resamples,
        "bootstrap_unit": "question_id", "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[PASS] {summary}")


if __name__ == "__main__":
    main()
