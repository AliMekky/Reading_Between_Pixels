#!/usr/bin/env python3
"""Build behavioral-subgroup tables from the three completed overlay reports."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from plot_style import VARIANTS


GROUPS = ("all", "fooled", "robust", "helped", "consistently_wrong")


def behavior_group(no_text, overlay, correct):
    if no_text == correct and overlay != correct:
        return "fooled"
    if no_text == correct and overlay == correct:
        return "robust"
    if no_text != correct and overlay == correct:
        return "helped"
    return "consistently_wrong"


def mean_ci(values, rng, draws):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return values.mean(), *np.quantile(means, (.025, .975))


def difference_ci(first, second, rng, draws):
    first, second = np.asarray(first), np.asarray(second)
    values = np.empty(draws)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        a = first[rng.integers(0, len(first), size=(stop - start, len(first)))].mean(axis=1)
        b = second[rng.integers(0, len(second), size=(stop - start, len(second)))].mean(axis=1)
        values[start:stop] = a - b
    return first.mean() - second.mean(), *np.quantile(values, (.025, .975))


def bh_adjust(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    effects, memberships, count_rows = {}, {}, []

    for variant, path in zip(VARIANTS, args.inputs):
        print("[LOAD] {}".format(path), flush=True)
        with Path(path).open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") != "success" or report["configuration"]["variants"] != [variant]:
            raise ValueError("Invalid report: {}".format(path))
        qids = [str(qid) for qid in report["configuration"]["selected_question_ids"]]
        correct = {str(qid): sample["correct_letter"] for qid, sample in report["samples"].items()}
        no_text, overlay, target = {}, {}, {}
        for record in report["records"]:
            if record["region"] != "text_region":
                continue
            qid, direction, layer = str(record["question_id"]), record["direction"], int(record["layer"])
            effects[(variant, qid, direction, layer)] = float(record["effect"])
            if layer == 0:
                if direction == "insertion":
                    no_text[qid] = record["recipient_prediction"]
                else:
                    overlay[qid] = record["recipient_prediction"]
                    target[qid] = record["target_option_letter"]
        for qid in qids:
            memberships[(variant, qid)] = behavior_group(no_text[qid], overlay[qid], correct[qid])
        for group in GROUPS[1:]:
            selected = [qid for qid in qids if memberships[(variant, qid)] == group]
            target_n = sum(overlay[qid] == target[qid] for qid in selected)
            count_rows.append({
                "variant": variant, "group": group, "n": len(selected),
                "overlay_target_prediction_n": target_n,
                "overlay_target_prediction_rate": target_n / len(selected) if selected else "",
            })
        count_rows.append({
            "variant": variant, "group": "prediction_changed", "n": sum(no_text[qid] != overlay[qid] for qid in qids),
            "overlay_target_prediction_n": "", "overlay_target_prediction_rate": "",
        })
        count_rows.append({
            "variant": variant, "group": "prediction_stable", "n": sum(no_text[qid] == overlay[qid] for qid in qids),
            "overlay_target_prediction_n": "", "overlay_target_prediction_rate": "",
        })
        del report
        gc.collect()
        print("[VALID] variant={} qids={} groups={}".format(
            variant, len(qids), {group: sum(memberships[(variant, qid)] == group for qid in qids) for group in GROUPS[1:]}), flush=True)

    summary = []
    grouped_values = {}
    for variant in VARIANTS:
        variant_qids = [qid for (current, qid) in memberships if current == variant]
        for direction in ("restoration", "insertion"):
            for layer in range(32):
                for group in GROUPS:
                    selected = variant_qids if group == "all" else [qid for qid in variant_qids if memberships[(variant, qid)] == group]
                    values = np.asarray([effects[(variant, qid, direction, layer)] for qid in selected])
                    grouped_values[(variant, direction, layer, group)] = values
                    mean, low, high = mean_ci(values, rng, args.bootstrap_draws)
                    summary.append({
                        "variant": variant, "direction": direction, "layer": layer,
                        "group": group, "n": len(values), "mean_effect": mean,
                        "ci_95_low": low, "ci_95_high": high,
                    })

    comparisons = []
    for variant in VARIANTS:
        for direction in ("restoration", "insertion"):
            direction_rows = []
            for layer in range(32):
                fooled = grouped_values[(variant, direction, layer, "fooled")]
                robust = grouped_values[(variant, direction, layer, "robust")]
                mean, low, high = difference_ci(fooled, robust, rng, args.bootstrap_draws)
                direction_rows.append({
                    "variant": variant, "direction": direction, "layer": layer,
                    "fooled_n": len(fooled), "robust_n": len(robust),
                    "fooled_minus_robust_mean": mean, "ci_95_low": low, "ci_95_high": high,
                    "mann_whitney_p_value": mannwhitneyu(fooled, robust, alternative="two-sided").pvalue,
                })
            for row, q_value in zip(direction_rows, bh_adjust([r["mann_whitney_p_value"] for r in direction_rows])):
                row["fdr_q_value"] = q_value
            comparisons.extend(direction_rows)

    output = Path(args.output_dir)
    write_csv(output / "behavior_group_counts.csv", count_rows)
    write_csv(output / "behavior_breakdown_summary.csv", summary)
    write_csv(output / "fooled_vs_robust_comparisons.csv", comparisons)
    metadata = {"status": "success", "sample_count_per_condition": 305,
                "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
                "definitions": {
                    "fooled": "no-text correct; overlay wrong",
                    "robust": "no-text correct; overlay correct",
                    "helped": "no-text wrong; overlay correct",
                    "consistently_wrong": "no-text wrong; overlay wrong",
                }}
    (output / "behavior_breakdown_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[COMPLETE] summary_rows={} comparison_rows={} count_rows={}".format(
        len(summary), len(comparisons), len(count_rows)), flush=True)


if __name__ == "__main__":
    main()
