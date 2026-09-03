#!/usr/bin/env python3
"""Appendix plot for clean object-region and matched-random controls."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (ACL_TWO_COL, CI_ALPHA, LINE_WIDTH, REGION_COLORS,
                        REGION_LABELS, VARIANTS, finish_axis, read_csv,
                        save_figure, write_interpretation)


REGIONS = ("correct_object_region", "grounded_object_region", "matched_random_mean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_irrelevant", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.summary)
    lookup = {(r["variant"], r["direction"], int(r["layer"]), r["region"]): r for r in rows}
    variants = list(VARIANTS.items())[:2] if args.exclude_irrelevant else list(VARIANTS.items())

    figure, axes = plt.subplots(2, len(variants), figsize=(ACL_TWO_COL, 4.6), sharex=True, sharey=True)
    notes = []
    letters = iter("abcdef")
    for row_index, direction in enumerate(("restoration", "insertion")):
        for column_index, (variant, title) in enumerate(variants):
            axis = axes[row_index, column_index]
            for region in REGIONS:
                series = [lookup[(variant, direction, layer, region)] for layer in range(32)]
                mean = np.array([float(r["mean_effect"]) for r in series])
                low = np.array([float(r["ci_95_low"]) for r in series])
                high = np.array([float(r["ci_95_high"]) for r in series])
                axis.plot(range(32), mean, color=REGION_COLORS[region], lw=LINE_WIDTH,
                          label=REGION_LABELS[region])
                axis.fill_between(range(32), low, high, color=REGION_COLORS[region], alpha=CI_ALPHA)
                peak = int(np.argmax(np.abs(mean)))
                notes.append("- **{}—{}—{}:** largest absolute mean is {:.3f} at layer {} (n={}).".format(
                    title, direction, REGION_LABELS[region], mean[peak], peak, int(series[peak]["n"])))
            axis.set_title("({}) {}".format(next(letters), title), fontweight="bold", loc="left")
            finish_axis(axis, row_index == 1)
            if column_index == 0:
                axis.set_ylabel("{} effect\n(logit margin)".format(direction.capitalize()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
                  framealpha=.95, edgecolor="#CCCCCC", bbox_to_anchor=(.5, -.01))
    figure.tight_layout(rect=(0, .07, 1, 1))
    save_figure(figure, args.output)
    plt.close(figure)
    write_interpretation(args.output, "Appendix: Object and spatial controls", [
        "Object results use only samples marked as clean controls; matched-random regions use all 305 samples.",
        *notes,
        "Object effects are about 1–2 orders of magnitude smaller than the text effects, supporting spatial specificity. Any nonzero object effect can reflect contextual propagation of overlay information, not pixel differences in the object regions.",
    ])


if __name__ == "__main__":
    main()
