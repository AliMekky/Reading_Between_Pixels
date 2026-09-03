#!/usr/bin/env python3
"""Summarize the 305-question single-layer activation-patching sweep."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


VARIANTS = {
    "misleading_groundable": "Grounded misleading",
    "misleading_ungroundable": "Ungrounded misleading",
    "irrelevant_word": "Irrelevant option text",
}
DIRECTIONS = ("restoration", "insertion")
RAW_REGIONS = (
    "text_region", "correct_object_region", "grounded_object_region",
    "matched_random_region_1", "matched_random_region_2", "matched_random_region_3",
    "all_image_tokens",
)
REGIONS = (
    "text_region", "correct_object_region", "grounded_object_region",
    "matched_random_mean", "all_image_tokens",
)
LABELS = {
    "text_region": "Text",
    "correct_object_region": "Correct object (clean subset)",
    "grounded_object_region": "Grounded object (clean subset)",
    "matched_random_mean": "Matched random (3-region mean)",
    "all_image_tokens": "All image tokens",
}
COLORS = {
    "text_region": "#d62728",
    "correct_object_region": "#1f77b4",
    "grounded_object_region": "#2ca02c",
    "matched_random_mean": "#7f7f7f",
    "all_image_tokens": "#9467bd",
}


def mean_ci(values, seed, draws):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_p_value(values):
    array = np.asarray(values, dtype=np.float64)
    if np.all(array == 0):
        return 1.0
    return float(wilcoxon(array, zero_method="zsplit", alternative="two-sided").pvalue)


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--comparisons_csv", required=True)
    parser.add_argument("--object_zoom_output", required=True)
    parser.add_argument("--condition_output", required=True)
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "success":
        raise ValueError("Layer-sweep report is incomplete")
    if report["configuration"].get("sweep_mode") != "single_layer":
        raise ValueError("Input is not a single-layer sweep")
    if tuple(report["configuration"]["regions"]) != RAW_REGIONS:
        raise ValueError("Unexpected region configuration")

    sample_count = int(report["configuration"]["sample_count"])
    per_sample = {}
    clean_object = {}
    for record in report["records"]:
        key = (
            str(record["question_id"]), str(record["variant"]),
            str(record["direction"]), int(record["layer"]), str(record["region"]),
        )
        if key in per_sample:
            raise ValueError("Duplicate record {}".format(key))
        per_sample[key] = float(record["effect"])
        clean_object[key] = record.get("clean_object_control")

    layers = list(range(32))
    grouped = defaultdict(list)
    qids = report["configuration"]["selected_question_ids"]
    for qid in qids:
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                for layer in layers:
                    prefix = (qid, variant, direction, layer)
                    random_mean = float(np.mean([
                        per_sample[prefix + ("matched_random_region_1",)],
                        per_sample[prefix + ("matched_random_region_2",)],
                        per_sample[prefix + ("matched_random_region_3",)],
                    ]))
                    grouped[(variant, direction, layer, "text_region")].append(
                        per_sample[prefix + ("text_region",)]
                    )
                    grouped[(variant, direction, layer, "matched_random_mean")].append(random_mean)
                    grouped[(variant, direction, layer, "all_image_tokens")].append(
                        per_sample[prefix + ("all_image_tokens",)]
                    )
                    for object_region in ("correct_object_region", "grounded_object_region"):
                        key = prefix + (object_region,)
                        if key in per_sample and clean_object[key] is True:
                            grouped[(variant, direction, layer, object_region)].append(per_sample[key])
    summary_rows = []
    lookup = {}
    for variant_index, variant in enumerate(VARIANTS):
        for direction_index, direction in enumerate(DIRECTIONS):
            for layer in layers:
                for region_index, region in enumerate(REGIONS):
                    values = grouped[(variant, direction, layer, region)]
                    if region not in ("correct_object_region", "grounded_object_region") and len(values) != sample_count:
                        raise ValueError("Incomplete group {}: {}".format((variant, direction, layer, region), len(values)))
                    if not values:
                        raise ValueError("No valid values for {}".format((variant, direction, layer, region)))
                    mean, low, high = mean_ci(
                        values,
                        args.seed + variant_index * 100000 + direction_index * 10000 + layer * 10 + region_index,
                        args.bootstrap_draws,
                    )
                    row = {
                        "variant": variant, "direction": direction, "layer": layer,
                        "region": region, "n": len(values), "mean_effect": mean,
                        "ci_95_low": low, "ci_95_high": high,
                    }
                    summary_rows.append(row)
                    lookup[(variant, direction, layer, region)] = row
    write_csv(args.summary_csv, summary_rows)

    comparison_rows = []
    for direction_index, direction in enumerate(DIRECTIONS):
        for layer in layers:
            grounded = np.asarray([
                per_sample[(qid, "misleading_groundable", direction, layer, "text_region")]
                for qid in qids
            ])
            ungrounded = np.asarray([
                per_sample[(qid, "misleading_ungroundable", direction, layer, "text_region")]
                for qid in qids
            ])
            irrelevant = np.asarray([
                per_sample[(qid, "irrelevant_word", direction, layer, "text_region")]
                for qid in qids
            ])
            random_values = {}
            for variant in VARIANTS:
                random_values[variant] = np.asarray([
                    np.mean([
                        per_sample[(qid, variant, direction, layer, "matched_random_region_1")],
                        per_sample[(qid, variant, direction, layer, "matched_random_region_2")],
                        per_sample[(qid, variant, direction, layer, "matched_random_region_3")],
                    ]) for qid in qids
                ])
            row = {"direction": direction, "layer": layer, "n": sample_count}
            arrays = {
                "ungrounded_minus_grounded": ungrounded - grounded,
                "grounded_minus_irrelevant": grounded - irrelevant,
                "ungrounded_minus_irrelevant": ungrounded - irrelevant,
            }
            for variant, text_values in (
                ("grounded", grounded), ("ungrounded", ungrounded),
                ("irrelevant", irrelevant),
            ):
                full_variant = {
                    "grounded": "misleading_groundable",
                    "ungrounded": "misleading_ungroundable",
                    "irrelevant": "irrelevant_word",
                }[variant]
                arrays["{}_text_minus_random".format(variant)] = text_values - random_values[full_variant]
            for index, (name, values) in enumerate(arrays.items()):
                mean, low, high = mean_ci(
                    values, args.seed + direction_index * 10000 + layer * 100 + index,
                    args.bootstrap_draws,
                )
                row[name + "_mean"] = mean
                row[name + "_ci_95_low"] = low
                row[name + "_ci_95_high"] = high
                row[name + "_p_value"] = paired_p_value(values)
            comparison_rows.append(row)
    comparison_names = (
        "ungrounded_minus_grounded", "grounded_minus_irrelevant",
        "ungrounded_minus_irrelevant", "grounded_text_minus_random",
        "ungrounded_text_minus_random", "irrelevant_text_minus_random",
    )
    for direction in DIRECTIONS:
        direction_rows = [row for row in comparison_rows if row["direction"] == direction]
        for name in comparison_names:
            q_values = benjamini_hochberg([row[name + "_p_value"] for row in direction_rows])
            for row, q_value in zip(direction_rows, q_values):
                row[name + "_fdr_q_value"] = q_value
    write_csv(args.comparisons_csv, comparison_rows)

    prediction_rows = []
    text_records = [record for record in report["records"] if record["region"] == "text_region"]
    for variant in VARIANTS:
        for direction in DIRECTIONS:
            for layer in layers:
                records = [
                    record for record in text_records
                    if record["variant"] == variant and record["direction"] == direction
                    and int(record["layer"]) == layer
                ]
                if len(records) != sample_count:
                    raise ValueError("Incomplete prediction group")
                target_flips = sum(bool(record["target_flip"]) for record in records)
                recoveries = sum(bool(record["recovery_to_correct"]) for record in records)
                if direction == "insertion":
                    eligible = sum(
                        record["recipient_prediction"] == report["samples"][record["question_id"]]["correct_letter"]
                        for record in records
                    )
                    desired = sum(
                        record["recipient_prediction"] == report["samples"][record["question_id"]]["correct_letter"]
                        and bool(record["target_flip"])
                        for record in records
                    )
                    transition = "correct_to_target"
                else:
                    eligible = sum(
                        record["recipient_prediction"] == record["target_option_letter"]
                        for record in records
                    )
                    desired = sum(
                        record["recipient_prediction"] == record["target_option_letter"]
                        and bool(record["recovery_to_correct"])
                        for record in records
                    )
                    transition = "target_to_correct"
                prediction_rows.append({
                    "variant": variant, "direction": direction, "layer": layer,
                    "n_all": sample_count, "target_flips_all": target_flips,
                    "target_flip_rate_all": target_flips / sample_count,
                    "recoveries_all": recoveries, "recovery_rate_all": recoveries / sample_count,
                    "conditional_transition": transition, "eligible_n": eligible,
                    "desired_transition_count": desired,
                    "conditional_transition_rate": desired / eligible if eligible else "",
                })
    write_csv(args.prediction_csv, prediction_rows)

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=True, sharey=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            for region in REGIONS:
                rows = [lookup[(variant, direction, layer, region)] for layer in layers]
                means = np.asarray([row["mean_effect"] for row in rows])
                lows = np.asarray([row["ci_95_low"] for row in rows])
                highs = np.asarray([row["ci_95_high"] for row in rows])
                axis.plot(layers, means, color=COLORS[region], linewidth=2, label=LABELS[region])
                axis.fill_between(layers, lows, highs, color=COLORS[region], alpha=0.13)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title("{} — {}".format(VARIANTS[variant], direction.capitalize()))
            axis.set_xticks(range(0, 32, 2))
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Oriented logit-margin effect")
            if row_index == 1:
                axis.set_xlabel("Decoder layer (resid_pre)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Shared-subset single-layer activation patching (n={})".format(sample_count))
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    zoom_regions = ("correct_object_region", "grounded_object_region", "matched_random_mean")
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            for region in zoom_regions:
                rows = [lookup[(variant, direction, layer, region)] for layer in layers]
                means = np.asarray([row["mean_effect"] for row in rows])
                lows = np.asarray([row["ci_95_low"] for row in rows])
                highs = np.asarray([row["ci_95_high"] for row in rows])
                axis.plot(layers, means, color=COLORS[region], linewidth=2, label=LABELS[region])
                axis.fill_between(layers, lows, highs, color=COLORS[region], alpha=0.13)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title("{} — {}".format(VARIANTS[variant], direction.capitalize()))
            axis.set_xticks(range(0, 32, 2))
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Oriented logit-margin effect (zoomed)")
            if row_index == 1:
                axis.set_xlabel("Decoder layer (resid_pre)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Object and matched-random controls (clean object subsets)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    zoom_output = Path(args.object_zoom_output)
    zoom_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(zoom_output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    condition_colors = {
        "misleading_groundable": "#1f77b4",
        "misleading_ungroundable": "#d62728",
        "irrelevant_word": "#2ca02c",
    }
    for axis, direction in zip(axes, DIRECTIONS):
        for variant in VARIANTS:
            rows = [lookup[(variant, direction, layer, "text_region")] for layer in layers]
            means = np.asarray([row["mean_effect"] for row in rows])
            lows = np.asarray([row["ci_95_low"] for row in rows])
            highs = np.asarray([row["ci_95_high"] for row in rows])
            axis.plot(layers, means, linewidth=2, color=condition_colors[variant], label=VARIANTS[variant])
            axis.fill_between(layers, lows, highs, color=condition_colors[variant], alpha=0.13)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(direction.capitalize())
        axis.set_xticks(range(0, 32, 2))
        axis.set_xlabel("Decoder layer (resid_pre)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Oriented text-region logit-margin effect")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Text-region causal effect across overlay conditions")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    condition_output = Path(args.condition_output)
    condition_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(condition_output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE] {}".format(args.summary_csv), flush=True)
    print("[SAVE] {}".format(args.comparisons_csv), flush=True)
    print("[SAVE] {}".format(args.output), flush=True)
    print("[SAVE] {}".format(args.object_zoom_output), flush=True)
    print("[SAVE] {}".format(args.condition_output), flush=True)
    print("[SAVE] {}".format(args.prediction_csv), flush=True)
    print("[COMPLETE] summary_rows={} comparison_rows={}".format(len(summary_rows), len(comparison_rows)), flush=True)


if __name__ == "__main__":
    main()
