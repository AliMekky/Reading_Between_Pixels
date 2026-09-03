#!/usr/bin/env python3
"""Summarize and plot held-out activation-patching window confirmation."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = {
    "misleading_groundable": "Grounded misleading",
    "misleading_ungroundable": "Ungrounded misleading",
}
DIRECTIONS = ("restoration", "insertion")
WINDOWS = ("early_0_5", "middle_10_15", "late_26_31")
WINDOW_LABELS = {
    "early_0_5": "Early\n0–5",
    "middle_10_15": "Middle\n10–15",
    "late_26_31": "Late\n26–31",
}
REGIONS = (
    "text_region", "correct_object_region", "grounded_object_region",
    "matched_random_mean", "all_image_tokens",
)
REGION_LABELS = {
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


def bootstrap_ci(values: List[float], seed: int, draws: int) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--comparisons_csv", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "success":
        raise ValueError("Confirmation report is incomplete")
    sample_count = int(report["configuration"]["sample_count"])
    raw = defaultdict(dict)
    outcomes = defaultdict(list)
    for record in report["records"]:
        key = (
            str(record["question_id"]), str(record["variant"]),
            str(record["direction"]), str(record["window"]),
        )
        region = str(record["region"])
        if region in ("correct_object_region", "grounded_object_region"):
            if record.get("clean_object_control") is not True:
                continue
        raw[key][region] = float(record["effect"])
        outcomes[(str(record["variant"]), str(record["direction"]), str(record["window"]), str(record["region"]))].append(str(record["outcome"]))

    values = defaultdict(list)
    per_sample = {}
    for key, regions in raw.items():
        random_mean = float(np.mean([
            regions["matched_random_region_1"], regions["matched_random_region_2"],
            regions["matched_random_region_3"],
        ]))
        displayed = {
            "text_region": regions["text_region"],
            "matched_random_mean": random_mean,
            "all_image_tokens": regions["all_image_tokens"],
        }
        for object_region in ("correct_object_region", "grounded_object_region"):
            if object_region in regions:
                displayed[object_region] = regions[object_region]
        per_sample[key] = displayed
        _, variant, direction, window = key
        for region, effect in displayed.items():
            values[(variant, direction, window, region)].append(effect)

    summary_rows = []
    lookup = {}
    for variant_index, variant in enumerate(VARIANTS):
        for direction_index, direction in enumerate(DIRECTIONS):
            for window_index, window in enumerate(WINDOWS):
                for region_index, region in enumerate(REGIONS):
                    group = values[(variant, direction, window, region)]
                    if region not in ("correct_object_region", "grounded_object_region") and len(group) != sample_count:
                        raise ValueError("Incomplete group {}".format((variant, direction, window, region)))
                    if not group:
                        raise ValueError("No valid results for {}".format((variant, direction, window, region)))
                    mean, low, high = bootstrap_ci(
                        group,
                        args.seed + variant_index * 10000 + direction_index * 1000 + window_index * 100 + region_index,
                        args.bootstrap_draws,
                    )
                    row = {
                        "variant": variant,
                        "direction": direction,
                        "window": window,
                        "region": region,
                        "n": len(group),
                        "mean_effect": mean,
                        "ci_95_low": low,
                        "ci_95_high": high,
                    }
                    summary_rows.append(row)
                    lookup[(variant, direction, window, region)] = row
    write_csv(Path(args.summary_csv), summary_rows)

    qids = [str(entry["question_id"]) for entry in report["selection"]["selected_samples"]]
    comparison_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for direction_index, direction in enumerate(DIRECTIONS):
            text_by_window = {
                window: [per_sample[(qid, variant, direction, window)]["text_region"] for qid in qids]
                for window in WINDOWS
            }
            random_by_window = {
                window: [per_sample[(qid, variant, direction, window)]["matched_random_mean"] for qid in qids]
                for window in WINDOWS
            }
            row = {"variant": variant, "direction": direction, "n": sample_count}
            comparisons = {
                "early_minus_middle": np.asarray(text_by_window["early_0_5"]) - np.asarray(text_by_window["middle_10_15"]),
                "early_minus_late": np.asarray(text_by_window["early_0_5"]) - np.asarray(text_by_window["late_26_31"]),
            }
            for window in WINDOWS:
                comparisons["{}_text_minus_random".format(window)] = (
                    np.asarray(text_by_window[window]) - np.asarray(random_by_window[window])
                )
                text_mean = lookup[(variant, direction, window, "text_region")]["mean_effect"]
                all_mean = lookup[(variant, direction, window, "all_image_tokens")]["mean_effect"]
                row["{}_text_over_all_ratio".format(window)] = (
                    text_mean / all_mean if abs(all_mean) > 0.05 else ""
                )
            for comparison_index, (name, array) in enumerate(comparisons.items()):
                mean, low, high = bootstrap_ci(
                    array.tolist(),
                    args.seed + variant_index * 1000 + direction_index * 100 + comparison_index,
                    args.bootstrap_draws,
                )
                row["{}_mean".format(name)] = mean
                row["{}_ci_95_low".format(name)] = low
                row["{}_ci_95_high".format(name)] = high
            comparison_rows.append(row)
    write_csv(Path(args.comparisons_csv), comparison_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x = np.arange(len(WINDOWS), dtype=float)
    width = 0.15
    offsets = (np.arange(len(REGIONS)) - (len(REGIONS) - 1) / 2) * width
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            for region_index, region in enumerate(REGIONS):
                rows = [lookup[(variant, direction, window, region)] for window in WINDOWS]
                means = np.asarray([row["mean_effect"] for row in rows])
                lows = np.asarray([row["ci_95_low"] for row in rows])
                highs = np.asarray([row["ci_95_high"] for row in rows])
                axis.bar(
                    x + offsets[region_index], means, width,
                    yerr=np.vstack((means - lows, highs - means)), capsize=3,
                    color=COLORS[region], label=REGION_LABELS[region], alpha=0.9,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(x, [WINDOW_LABELS[window] for window in WINDOWS])
            axis.set_title("{} — {}".format(VARIANTS[variant], direction.capitalize()))
            axis.grid(axis="y", alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Oriented logit-margin effect")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Shared-subset simultaneous layer-window analysis (primary n={})".format(sample_count))
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE] {}".format(args.summary_csv), flush=True)
    print("[SAVE] {}".format(args.comparisons_csv), flush=True)
    print("[SAVE] {}".format(args.output), flush=True)
    print("[COMPLETE] confirmation_summary_rows={} comparison_rows={}".format(len(summary_rows), len(comparison_rows)), flush=True)


if __name__ == "__main__":
    main()
