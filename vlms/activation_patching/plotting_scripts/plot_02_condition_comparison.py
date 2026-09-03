#!/usr/bin/env python3
"""Compare text-region effects across grounded, ungrounded, and irrelevant text."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_style import (ACL_TWO_COL, CI_ALPHA, CONDITION_COLORS, LINE_WIDTH,
                        VARIANTS, finish_axis, read_csv, save_figure,
                        write_interpretation)


def strongest(rows, name):
    row = max(rows, key=lambda value: abs(float(value[name + "_mean"])))
    return (int(row["layer"]), float(row[name + "_mean"]),
            float(row[name + "_ci_95_low"]), float(row[name + "_ci_95_high"]),
            float(row[name + "_fdr_q_value"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_irrelevant", action="store_true")
    args = parser.parse_args()
    summary = read_csv(args.summary)
    comparisons = read_csv(args.comparisons)
    lookup = {(r["variant"], r["direction"], int(r["layer"])): r for r in summary if r["region"] == "text_region"}
    variants = list(VARIANTS.items())[:2] if args.exclude_irrelevant else list(VARIANTS.items())

    figure, axes = plt.subplots(1, 2, figsize=(ACL_TWO_COL, 2.55), sharex=True, sharey=True)
    interpretations = []
    for axis, direction, letter in zip(axes, ("restoration", "insertion"), "ab"):
        for variant, label in variants:
            rows = [lookup[(variant, direction, layer)] for layer in range(32)]
            mean = np.array([float(r["mean_effect"]) for r in rows])
            low = np.array([float(r["ci_95_low"]) for r in rows])
            high = np.array([float(r["ci_95_high"]) for r in rows])
            axis.plot(range(32), mean, color=CONDITION_COLORS[variant], lw=LINE_WIDTH, label=label)
            axis.fill_between(range(32), low, high, color=CONDITION_COLORS[variant], alpha=CI_ALPHA)
        axis.set_title("({}) {}".format(letter, direction.capitalize()), fontweight="bold", loc="left")
        finish_axis(axis)
        direction_rows = [r for r in comparisons if r["direction"] == direction]
        tests = [("ungrounded_minus_grounded", "Ungrounded − grounded")]
        if not args.exclude_irrelevant:
            tests += [("grounded_minus_irrelevant", "Grounded − irrelevant"),
                      ("ungrounded_minus_irrelevant", "Ungrounded − irrelevant")]
        for name, label in tests:
            layer, mean, low, high, q = strongest(direction_rows, name)
            relation = "larger" if mean > 0 else "smaller"
            significant = sum(float(row[name + "_fdr_q_value"]) < .05 for row in direction_rows)
            interpretations.append(
                "- **{}; {}:** strongest absolute difference at layer {}: {:.3f} [{:.3f}, {:.3f}], "
                "FDR q={:.3g}; the first condition is {}. Significant at {}/32 layers.".format(
                    direction.capitalize(), label, layer, mean, low, high, q, relation, significant))
    axes[0].set_ylabel("Text-region effect (logit margin)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(variants), frameon=True,
                  framealpha=.95, edgecolor="#CCCCCC", bbox_to_anchor=(.5, -.08))
    figure.tight_layout(rect=(0, .13, 1, 1))
    save_figure(figure, args.output)
    plt.close(figure)
    early = {}
    for direction in ("restoration", "insertion"):
        early[direction] = {
            variant: np.mean([float(lookup[(variant, direction, layer)]["mean_effect"]) for layer in range(7)])
            for variant, _ in variants
        }
    if args.exclude_irrelevant:
        overview = "Across layers 0–6, ungrounded effects exceed grounded effects: restoration means are {:.3f} versus {:.3f}; insertion means are {:.3f} versus {:.3f}.".format(
            early["restoration"]["misleading_ungroundable"], early["restoration"]["misleading_groundable"],
            early["insertion"]["misleading_ungroundable"], early["insertion"]["misleading_groundable"])
    else:
        overview = "Across layers 0–6, the observed ordering is ungrounded > irrelevant > grounded: restoration means are {:.3f}, {:.3f}, and {:.3f}; insertion means are {:.3f}, {:.3f}, and {:.3f}. This does not support a simple claim that only semantically misleading text is causally used; visible option text has a strong effect even in the irrelevant condition.".format(
            early["restoration"]["misleading_ungroundable"], early["restoration"]["irrelevant_word"], early["restoration"]["misleading_groundable"],
            early["insertion"]["misleading_ungroundable"], early["insertion"]["irrelevant_word"], early["insertion"]["misleading_groundable"])
    write_interpretation(args.output, "Figure 2: Comparison across text conditions", [
        "This figure compares the causal influence of the displayed text across the included conditions.",
        overview,
        *interpretations,
        "A difference is treated as statistically reliable only when its layer-wise FDR q-value is below 0.05.",
    ])


if __name__ == "__main__":
    main()
