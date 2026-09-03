"""Shared implementation for one-condition behavioral-breakdown figures."""

import matplotlib.pyplot as plt
import numpy as np

from plot_style import ACL_TWO_COL, CI_ALPHA, LINE_WIDTH, VARIANTS, finish_axis, read_csv, save_figure, write_interpretation


GROUPS = {
    "fooled": ("Fooled", "#D55E00", "-"),
    "robust": ("Robust", "#0072B2", "-"),
}


def make_plot(variant, summary_path, comparisons_path, counts_path, output):
    summary, comparisons, counts = map(read_csv, (summary_path, comparisons_path, counts_path))
    lookup = {(r["variant"], r["direction"], int(r["layer"]), r["group"]): r for r in summary}
    count_lookup = {(r["variant"], r["group"]): r for r in counts}
    figure, axes = plt.subplots(1, 2, figsize=(ACL_TWO_COL, 2.7), sharex=True, sharey=True)
    notes = []
    for axis, direction, letter in zip(axes, ("restoration", "insertion"), "ab"):
        for group, (label, color, style) in GROUPS.items():
            rows = [lookup[(variant, direction, layer, group)] for layer in range(32)]
            mean = np.array([float(r["mean_effect"]) for r in rows])
            low = np.array([float(r["ci_95_low"]) for r in rows])
            high = np.array([float(r["ci_95_high"]) for r in rows])
            axis.plot(range(32), mean, color=color, ls=style, lw=LINE_WIDTH, label="{} (n={})".format(label, rows[0]["n"]))
            axis.fill_between(range(32), low, high, color=color, alpha=CI_ALPHA)
        axis.set_title("({}) {}".format(letter, direction.capitalize()), fontweight="bold", loc="left")
        finish_axis(axis)
        rows = [r for r in comparisons if r["variant"] == variant and r["direction"] == direction]
        strongest = max(rows, key=lambda r: abs(float(r["fooled_minus_robust_mean"])))
        significant = sum(float(r["fdr_q_value"]) < .05 for r in rows)
        notable = sum(float(r["fdr_q_value"]) < .05 and abs(float(r["fooled_minus_robust_mean"])) >= .1 for r in rows)
        early = {
            group: np.mean([float(lookup[(variant, direction, layer, group)]["mean_effect"]) for layer in range(7)])
            for group in GROUPS
        }
        notes.append("- **{}:** mean effects across layers 0–6 are fooled={:.3f} and robust={:.3f}. The largest fooled-minus-robust difference is {:.3f} at layer {} (95% CI [{:.3f}, {:.3f}], FDR q={:.3g}). The difference is FDR-significant at {}/32 layers, with {}/32 also having magnitude at least 0.1 logit.".format(
            direction.capitalize(), early["fooled"], early["robust"],
            float(strongest["fooled_minus_robust_mean"]), strongest["layer"],
            float(strongest["ci_95_low"]), float(strongest["ci_95_high"]), float(strongest["fdr_q_value"]), significant, notable))
    axes[0].set_ylabel("Text-region effect (logit margin)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
                  framealpha=.95, edgecolor="#CCCCCC", bbox_to_anchor=(.5, -.13))
    figure.tight_layout(rect=(0, .2, 1, 1))
    save_figure(figure, output)
    plt.close(figure)

    group_counts = {group: int(count_lookup[(variant, group)]["n"]) for group in GROUPS}
    fooled = count_lookup[(variant, "fooled")]
    target_n = int(fooled["overlay_target_prediction_n"])
    write_interpretation(output, "Behavioral breakdown: {}".format(VARIANTS[variant]), [
        "The plotted baseline groups are fooled n={fooled} and robust n={robust}. The separate primary figure retains the aggregate estimate over all 305 samples.".format(**group_counts),
        "Among fooled examples, {}/{} ({:.1%}) follow the displayed condition-specific option.".format(
            target_n, group_counts["fooled"], target_n / group_counts["fooled"]),
        *notes,
        "Subgroups are defined from unpatched baseline behavior. Because fooled cases are selected by the overlay-induced prediction change, their larger effects are descriptive enrichment rather than an independent causal test. The all-305 result remains primary.",
    ])
