#!/usr/bin/env python3
"""Plot the one-sample activation-patching layer sweep."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = ("misleading_groundable", "misleading_ungroundable")
DIRECTIONS = ("restoration", "insertion")
REGION_STYLE = {
    "text_region": ("Text region", "#d62728"),
    "matched_random_region": ("Matched random", "#7f7f7f"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "success":
        raise RuntimeError("Sweep report is not complete")
    records = report["records"]
    expected = report["completion"]["expected_interventions"]
    if len(records) != expected:
        raise RuntimeError("Record count does not match expected interventions")

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for row, direction in enumerate(DIRECTIONS):
        for col, variant in enumerate(VARIANTS):
            axis = axes[row][col]
            for region, (label, color) in REGION_STYLE.items():
                subset = sorted(
                    (
                        record for record in records
                        if record["direction"] == direction
                        and record["variant"] == variant
                        and record["region"] == region
                    ),
                    key=lambda record: record["layer"],
                )
                axis.plot(
                    [record["layer"] for record in subset],
                    [record["effect"] for record in subset],
                    marker="o", markersize=3, linewidth=1.5,
                    color=color, label=label,
                )
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            axis.set_title("{} — {}".format(direction.title(), variant.replace("misleading_", "").title()))
            axis.grid(alpha=0.2)
            if row == 1:
                axis.set_xlabel("Decoder layer")
            if col == 0:
                axis.set_ylabel("Patching effect on logit margin")
            axis.legend(frameon=False)
    figure.suptitle("One-Sample Cleaned-Overlay Activation-Patching Sweep")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print("[PASS] Saved four-panel layer-sweep plot: {}".format(output), flush=True)


if __name__ == "__main__":
    main()

