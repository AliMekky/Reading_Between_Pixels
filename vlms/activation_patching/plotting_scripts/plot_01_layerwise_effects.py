#!/usr/bin/env python3
"""Main 2x3 layer-wise plot: text, matched-random, and all-image patching."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (ACL_TWO_COL, CI_ALPHA, LINE_WIDTH, REGION_COLORS,
                        REGION_LABELS, VARIANTS, finish_axis, read_csv,
                        save_figure, write_interpretation)


REGIONS = ("text_region", "matched_random_mean", "all_image_tokens")
DIRECTIONS = ("restoration", "insertion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_irrelevant", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.summary)
    comparisons = read_csv(args.comparisons)
    lookup = {(r["variant"], r["direction"], int(r["layer"]), r["region"]): r for r in rows}
    variants = list(VARIANTS.items())[:2] if args.exclude_irrelevant else list(VARIANTS.items())

    figure, axes = plt.subplots(2, len(variants), figsize=(ACL_TWO_COL, 4.6), sharex=True, sharey=True)
    interpretations = []
    letters = iter("abcdef")
    for row_index, direction in enumerate(DIRECTIONS):
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
            axis.set_title("({}) {}".format(next(letters), title), fontweight="bold", loc="left")
            finish_axis(axis, row_index == 1)
            if column_index == 0:
                axis.set_ylabel("{} effect\n(logit margin)".format(direction.capitalize()))

            text = np.array([float(lookup[(variant, direction, layer, "text_region")]["mean_effect"]) for layer in range(32)])
            peak = int(np.argmax(text))
            text_row = lookup[(variant, direction, peak, "text_region")]
            random = float(lookup[(variant, direction, peak, "matched_random_mean")]["mean_effect"])
            all_image = float(lookup[(variant, direction, peak, "all_image_tokens")]["mean_effect"])
            comparison_name = {
                "misleading_groundable": "grounded_text_minus_random",
                "misleading_ungroundable": "ungrounded_text_minus_random",
                "irrelevant_word": "irrelevant_text_minus_random",
            }[variant]
            tested = [r for r in comparisons if r["direction"] == direction]
            significant = sum(float(r[comparison_name + "_fdr_q_value"]) < .05 and
                              float(r[comparison_name + "_mean"]) > 0 for r in tested)
            interpretations.append(
                "- **{}—{}:** the text effect peaks at layer {}: {:.3f} (95% CI [{:.3f}, {:.3f}]); "
                "matched random is {:.3f} and all-image patching is {:.3f} at that layer. "
                "The paired text-minus-random effect is positive with FDR q<0.05 at {}/32 layers.".format(
                    title, direction, peak, float(text_row["mean_effect"]),
                    float(text_row["ci_95_low"]), float(text_row["ci_95_high"]), random, all_image,
                    significant))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
                  framealpha=.95, edgecolor="#CCCCCC", bbox_to_anchor=(.5, -.01))
    figure.tight_layout(rect=(0, .07, 1, 1))
    save_figure(figure, args.output)
    plt.close(figure)
    write_interpretation(args.output, "Figure 1: Layer-wise causal effects", [
        "Positive values indicate movement in the intended direction: toward the correct answer for restoration and toward the overlaid option for insertion.",
        *interpretations,
        "Peak layers are descriptive because they were selected from this sweep; the paired-comparison table provides the inferential tests.",
    ])


if __name__ == "__main__":
    main()
