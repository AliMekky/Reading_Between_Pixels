#!/usr/bin/env python3
"""Compute paired six-model open-ended metrics and question-bootstrap intervals."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = (
    "notext", "correct_answer", "misleading_groundable",
    "misleading_ungroundable", "irrelevant_word",
)
TARGET = {condition: condition for condition in CONDITIONS if condition != "notext"}
TRANSITION_TYPES = (
    "helpful_targeted_flip", "harmful_targeted_flip", "irrelevant_adoption",
    "robust_correct", "persistently_wrong", "other_flip", "other_stable",
)


def read_by_question(path):
    with path.open() as handle:
        return {str(row["question_id"]): row for row in map(json.loads, filter(str.strip, handle))}


def interval(values, indices):
    values = np.asarray(values, dtype=float)
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def conditional_interval(indicator, eligible, indices):
    """Question bootstrap for a transition rate within its eligible subset."""
    indicator, eligible = np.asarray(indicator), np.asarray(eligible)
    denominator = eligible[indices].sum(axis=1)
    estimates = np.divide(indicator[indices].sum(axis=1), denominator,
                          out=np.full(len(indices), np.nan), where=denominator != 0)
    valid = estimates[np.isfinite(estimates)]
    return float(np.quantile(valid, .025)), float(np.quantile(valid, .975))


def wilson_interval(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def exact_mcnemar(before, after):
    harmed = int(np.sum(before & ~after))
    helped = int(np.sum(~before & after))
    discordant = harmed + helped
    if discordant == 0:
        return harmed, helped, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(harmed, helped) + 1)) / (2 ** discordant)
    return harmed, helped, min(1.0, 2 * tail)


def holm_adjust(p_values):
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def metric_row(model, condition, name, values, indices, interval_type="bootstrap"):
    if interval_type == "wilson":
        low, high = wilson_interval(int(np.sum(values)), len(values))
    else:
        low, high = interval(values, indices)
    return {
        "model": model, "condition": condition, "metric": name,
        "n": len(values), "estimate": float(np.mean(values)),
        "ci_low": low, "ci_high": high,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, transitions, transition_summary, tests = [], [], [], []
    metric_arrays = {}

    for model_dir in sorted(path for path in args.classified_dir.iterdir() if path.is_dir()):
        data = {condition: read_by_question(model_dir / f"{condition}.jsonl") for condition in CONDITIONS}
        qids = sorted(data["notext"])
        if any(set(data[condition]) != set(qids) for condition in CONDITIONS):
            raise AssertionError(f"{model_dir.name}: unaligned question IDs")
        rng = np.random.default_rng(args.seed)
        bootstrap_indices = rng.integers(0, len(qids), size=(args.resamples, len(qids)))
        baseline_correct = np.array([data["notext"][qid]["final_category"] == "correct_answer" for qid in qids])
        model_arrays = {("notext", "accuracy"): baseline_correct.astype(float)}
        rows.append(metric_row(model_dir.name, "notext", "accuracy", baseline_correct, bootstrap_indices, "wilson"))
        model_tests = []

        for condition, target in TARGET.items():
            current = np.array([data[condition][qid]["final_category"] for qid in qids])
            baseline = np.array([data["notext"][qid]["final_category"] for qid in qids])
            current_correct = current == "correct_answer"
            current_target = current == target
            baseline_target = baseline == target
            arrays = {
                "accuracy": current_correct.astype(float),
                "accuracy_change": current_correct.astype(int) - baseline_correct.astype(int),
                "target_following": current_target.astype(float),
                "adjusted_target_following": current_target.astype(int) - baseline_target.astype(int),
            }
            model_arrays.update({(condition, name): values for name, values in arrays.items()})
            rows.extend([
                metric_row(model_dir.name, condition, "accuracy", arrays["accuracy"], bootstrap_indices, "wilson"),
                metric_row(model_dir.name, condition, "accuracy_change", arrays["accuracy_change"], bootstrap_indices),
                metric_row(model_dir.name, condition, "target_following", arrays["target_following"], bootstrap_indices),
                metric_row(model_dir.name, condition, "adjusted_target_following", arrays["adjusted_target_following"], bootstrap_indices),
            ])
            harmed, helped, p_value = exact_mcnemar(baseline_correct, current_correct)
            model_tests.append({
                "model": model_dir.name, "condition": condition,
                "baseline_correct_overlay_wrong": harmed,
                "baseline_wrong_overlay_correct": helped,
                "p_value": p_value,
            })
            for qid, before, after in zip(qids, baseline, current):
                if condition == "correct_answer" and before != "correct_answer" and after == "correct_answer":
                    transition = "helpful_targeted_flip"
                elif condition in ("misleading_groundable", "misleading_ungroundable") and before == "correct_answer" and after == target:
                    transition = "harmful_targeted_flip"
                elif condition == "irrelevant_word" and before != target and after == target:
                    transition = "irrelevant_adoption"
                elif before == "correct_answer" and after == "correct_answer":
                    transition = "robust_correct"
                elif before == after and before != "correct_answer":
                    transition = "persistently_wrong"
                elif before != after:
                    transition = "other_flip"
                else:
                    transition = "other_stable"
                transitions.append({
                    "model": model_dir.name, "condition": condition,
                    "question_id": qid, "baseline_category": before,
                    "overlay_category": after, "transition": transition,
                })

            condition_rows = [row for row in transitions if row["model"] == model_dir.name and row["condition"] == condition]
            labels = np.array([row["transition"] for row in condition_rows])
            transition_counts = Counter(labels)
            eligibility = {
                "helpful_targeted_flip": ~baseline_correct,
                "harmful_targeted_flip": baseline_correct,
                "irrelevant_adoption": ~baseline_target,
                "robust_correct": baseline_correct,
                "persistently_wrong": ~baseline_correct,
                "other_flip": np.ones(len(qids), dtype=bool),
                "other_stable": np.ones(len(qids), dtype=bool),
            }
            for transition_name in TRANSITION_TYPES:
                count = transition_counts[transition_name]
                eligible_mask = eligibility[transition_name]
                eligible = int(eligible_mask.sum())
                indicator = labels == transition_name
                all_low, all_high = interval(indicator, bootstrap_indices)
                eligible_low, eligible_high = conditional_interval(indicator, eligible_mask, bootstrap_indices)
                transition_summary.append({
                    "model": model_dir.name, "condition": condition,
                    "transition": transition_name, "count": count,
                    "total_questions": len(qids), "rate_all": count / len(qids),
                    "rate_all_ci_low": all_low, "rate_all_ci_high": all_high,
                    "eligible_questions": eligible,
                    "rate_eligible": count / eligible if eligible else None,
                    "rate_eligible_ci_low": eligible_low, "rate_eligible_ci_high": eligible_high,
                })

        grounded_follow = model_arrays[("misleading_groundable", "adjusted_target_following")]
        ungrounded_follow = model_arrays[("misleading_ungroundable", "adjusted_target_following")]
        harm_contrast = model_arrays[("misleading_groundable", "accuracy")] - model_arrays[("misleading_ungroundable", "accuracy")]
        follow_contrast = ungrounded_follow - grounded_follow
        model_arrays[("ungrounded_minus_grounded", "following_contrast")] = follow_contrast
        model_arrays[("ungrounded_minus_grounded", "accuracy_harm_contrast")] = harm_contrast
        rows.extend([
            metric_row(model_dir.name, "ungrounded_minus_grounded", "following_contrast", follow_contrast, bootstrap_indices),
            metric_row(model_dir.name, "ungrounded_minus_grounded", "accuracy_harm_contrast", harm_contrast, bootstrap_indices),
        ])
        adjusted = holm_adjust([row["p_value"] for row in model_tests])
        for test, adjusted_p in zip(model_tests, adjusted):
            test["holm_adjusted_p"] = adjusted_p
            tests.append(test)
        metric_arrays[model_dir.name] = model_arrays

    rng = np.random.default_rng(args.seed)
    question_count = len(next(iter(next(iter(metric_arrays.values())).values())))
    bootstrap_indices = rng.integers(0, question_count, size=(args.resamples, question_count))
    metric_keys = sorted(next(iter(metric_arrays.values())))
    for condition, name in metric_keys:
        values = np.mean([arrays[(condition, name)] for arrays in metric_arrays.values()], axis=0)
        rows.append(metric_row("unweighted_model_average", condition, name, values, bootstrap_indices))

    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "transitions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=transitions[0].keys())
        writer.writeheader(); writer.writerows(transitions)
    with (args.output_dir / "transition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=transition_summary[0].keys())
        writer.writeheader(); writer.writerows(transition_summary)
    with (args.output_dir / "mcnemar_holm.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tests[0].keys())
        writer.writeheader(); writer.writerows(tests)
    summary = {
        "models": len(metric_arrays),
        "metric_rows": len(rows), "transition_rows": len(transitions),
        "transition_summary_rows": len(transition_summary), "mcnemar_tests": len(tests),
        "bootstrap_resamples": args.resamples, "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[PASS] {summary}")


if __name__ == "__main__":
    main()
