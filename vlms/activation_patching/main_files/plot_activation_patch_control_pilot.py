#!/usr/bin/env python3
"""Plot and tabulate the ten-sample activation-patching control pilot."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


VARIANT_LABELS = {
    "misleading_groundable": "Grounded misleading",
    "misleading_ungroundable": "Ungrounded misleading",
}
DIRECTIONS = ("restoration", "insertion")
DISPLAY_REGIONS = (
    "text_region",
    "correct_object_region",
    "grounded_object_region",
    "matched_random_mean",
    "all_image_tokens",
)
REGION_LABELS = {
    "text_region": "Text",
    "correct_object_region": "Correct object",
    "grounded_object_region": "Grounded object",
    "matched_random_mean": "Matched random (3-region mean)",
    "all_image_tokens": "All image tokens",
}


def bootstrap_mean_ci(values: List[float], seed: int, draws: int = 10000) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        value = float(array[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def load_sample_values(report: Dict) -> Dict[Tuple[str, str, int, str], List[float]]:
    raw = defaultdict(dict)
    for record in report["records"]:
        key = (
            str(record["question_id"]), str(record["variant"]),
            int(record["layer"]), str(record["direction"]),
        )
        raw[key][str(record["region"])] = float(record["effect"])

    grouped = defaultdict(list)
    for (qid, variant, layer, direction), regions in raw.items():
        del qid
        random_values = [regions["matched_random_region_{}".format(i)] for i in range(1, 4)]
        values = {
            "text_region": regions["text_region"],
            "correct_object_region": regions["correct_object_region"],
            "grounded_object_region": regions["grounded_object_region"],
            "matched_random_mean": float(np.mean(random_values)),
            "all_image_tokens": regions["all_image_tokens"],
        }
        for region, effect in values.items():
            grouped[(variant, direction, layer, region)].append(effect)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "success":
        raise ValueError("Pilot report is not complete: status={}".format(report.get("status")))
    grouped = load_sample_values(report)
    layers = [int(value) for value in report["configuration"]["layers"]]
    sample_count = int(report["configuration"]["sample_count"])

    rows = []
    for variant in VARIANT_LABELS:
        for direction in DIRECTIONS:
            for layer in layers:
                for region_index, region in enumerate(DISPLAY_REGIONS):
                    values = grouped[(variant, direction, layer, region)]
                    if len(values) != sample_count:
                        raise ValueError("Expected {} values for {}, found {}".format(sample_count, (variant, direction, layer, region), len(values)))
                    mean, low, high = bootstrap_mean_ci(
                        values, args.seed + layer * 100 + region_index,
                        draws=args.bootstrap_draws,
                    )
                    rows.append({
                        "variant": variant,
                        "direction": direction,
                        "layer": layer,
                        "region": region,
                        "n": len(values),
                        "mean_effect": mean,
                        "ci_95_low": low,
                        "ci_95_high": high,
                    })

    csv_path = Path(args.summary_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["variant"], row["direction"], row["layer"], row["region"]): row for row in rows}
    colors = {
        "text_region": "#d62728",
        "correct_object_region": "#1f77b4",
        "grounded_object_region": "#2ca02c",
        "matched_random_mean": "#7f7f7f",
        "all_image_tokens": "#9467bd",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row_index, direction in enumerate(DIRECTIONS):
        for column_index, variant in enumerate(VARIANT_LABELS):
            axis = axes[row_index, column_index]
            for region in DISPLAY_REGIONS:
                region_rows = [lookup[(variant, direction, layer, region)] for layer in layers]
                means = np.asarray([row["mean_effect"] for row in region_rows])
                lows = np.asarray([row["ci_95_low"] for row in region_rows])
                highs = np.asarray([row["ci_95_high"] for row in region_rows])
                axis.plot(layers, means, marker="o", linewidth=1.8, color=colors[region], label=REGION_LABELS[region])
                axis.fill_between(layers, lows, highs, color=colors[region], alpha=0.12)
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            axis.set_title("{} — {}".format(VARIANT_LABELS[variant], direction.capitalize()))
            axis.set_xticks(layers)
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Oriented logit-margin effect")
            if row_index == 1:
                axis.set_xlabel("Decoder layer (resid_pre)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    title = args.title or "Activation-patching control pilot (n={}; diagnostic, not final evidence)".format(sample_count)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE] {}".format(csv_path), flush=True)
    print("[SAVE] {}".format(output_path), flush=True)
    print("[COMPLETE] Pilot summary contains {} rows".format(len(rows)), flush=True)


if __name__ == "__main__":
    main()
