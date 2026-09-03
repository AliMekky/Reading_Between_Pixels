#!/usr/bin/env python3
"""Plot conditional answer recovery and target-option flip rates."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from plot_style import (ACL_TWO_COL, CI_ALPHA, CONDITION_COLORS, LINE_WIDTH,
                        VARIANTS, read_csv, save_figure, write_interpretation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_irrelevant", action="store_true")
    args = parser.parse_args()
    rows = read_csv(args.predictions)
    lookup = {(r["variant"], r["direction"], int(r["layer"])): r for r in rows}
    variants = list(VARIANTS.items())[:2] if args.exclude_irrelevant else list(VARIANTS.items())

    figure, axes = plt.subplots(1, 2, figsize=(ACL_TWO_COL, 2.55), sharex=True, sharey=True)
    interpretations = []
    titles = {"restoration": "Recovery: target → correct", "insertion": "Transfer: correct → target"}
    for axis, direction, letter in zip(axes, ("restoration", "insertion"), "ab"):
        for variant, label in variants:
            series = [lookup[(variant, direction, layer)] for layer in range(32)]
            rate = np.array([float(r["conditional_transition_rate"]) for r in series])
            low = np.array([float(r["ci_95_low"]) for r in series])
            high = np.array([float(r["ci_95_high"]) for r in series])
            axis.plot(range(32), rate, color=CONDITION_COLORS[variant], lw=LINE_WIDTH, label=label)
            axis.fill_between(range(32), low, high, color=CONDITION_COLORS[variant], alpha=CI_ALPHA)
            peak = int(np.argmax(rate))
            interpretations.append(
                "- **{}—{}:** maximum conditional rate {:.1%} at layer {} ({}/{} eligible cases; 95% CI [{:.1%}, {:.1%}]).".format(
                    titles[direction], label, rate[peak], peak,
                    int(series[peak]["desired_transition_count"]), int(series[peak]["eligible_n"]),
                    low[peak], high[peak]))
        axis.set_title("({}) {}".format(letter, direction.capitalize()), fontweight="bold", loc="left")
        axis.axhline(0, color="#333333", linewidth=.7)
        axis.set_xticks(range(0, 32, 5))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Conditional transition rate")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(variants), frameon=True,
                  framealpha=.95, edgecolor="#CCCCCC", bbox_to_anchor=(.5, -.08))
    figure.supxlabel("Decoder layer (resid_pre)", y=.14)
    figure.tight_layout(rect=(0, .13, 1, 1))
    save_figure(figure, args.output)
    plt.close(figure)
    write_interpretation(args.output, "Figure 3: Prediction transitions", [
        "Restoration asks whether removing an overlay representation recovers an initially target-misled answer. Insertion asks whether adding it transfers an initially correct answer to the overlay option.",
        *interpretations,
        "These peak rates are descriptive. Logit-margin effects remain the primary outcome because many causal shifts do not cross the discrete prediction boundary.",
    ])


if __name__ == "__main__":
    main()
