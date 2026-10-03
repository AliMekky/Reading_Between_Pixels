#!/usr/bin/env python3
"""Write the completed relevance follow-up report and standalone scientific figures."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/relevance_score_followup"
VARIANTS = ("misleading_groundable", "misleading_ungroundable", "irrelevant_word")
NAMES = dict(zip(VARIANTS, ("Grounded misleading", "Ungrounded misleading", "Irrelevant")))
COLORS = ("#3577ad", "#c47b25", "#81549e")


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def main():
    rows = read("question_level_scores.csv")
    summary = read("score_summary.csv")
    pilot = read("pilot_absolute_scores.csv")
    audit = json.loads((OUT / "analysis_audit.json").read_text())
    middle = {r["variant"]: r for r in summary if r["subset"] == "all_305" and r["window"] == "layers_18_23"}
    pilot_middle = {v: [r for r in pilot if r["variant"] == v and r["window"] == "layers_18_23"] for v in VARIANTS}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4), sharey=True)
    for ax, variant, color in zip(axes, VARIANTS, COLORS):
        g = [r for r in rows if r["variant"] == variant and r["window"] == "layers_18_23"]
        before = np.array([float(r["baseline_correct_minus_displayed"]) for r in g])
        after = np.array([float(r["blocked_correct_minus_displayed"]) for r in g])
        bp = ax.boxplot([before, after], positions=[0, 1], widths=.5, showfliers=False, patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(.3)
        ax.scatter([0, 1], [before.mean(), after.mean()], c=color, marker="D", s=40, zorder=4, label="Mean")
        ax.axhline(0, color="black", lw=.8, ls="--")
        ax.set_xticks([0, 1], ["Baseline", "Q→A blocked"])
        ax.set_title(NAMES[variant] + f"\n{sum(after > 0)}/305 still below correct", fontsize=11)
    axes[0].set_ylabel("Correct − displayed answer score\n(mean token log probability; nats)")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Q→A blocking narrows the gap; irrelevant answers remain much further behind\nLayers 18–23, paired overlay runs; boxes show quartiles, diamonds show means", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"answer_score_gaps.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if "--margins-only" in sys.argv:
        print(OUT / "answer_score_gaps.png")
        return
    completion = json.loads((OUT / "pilot_baselines/completion.json").read_text())
    assert completion["status"] == "complete" and len(pilot) == 72
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4), sharey=True)
    for ax, variant, color in zip(axes, VARIANTS, COLORS):
        g = pilot_middle[variant]
        before = [float(r["overlay_comparison_mean_before"]) for r in g]
        after = [float(r["overlay_comparison_mean_after"]) for r in g]
        for b, a in zip(before, after):
            ax.plot([0, 1], [b, a], color=color, alpha=.5, marker="o", ms=3)
        ax.plot([0, 1], [np.mean(before), np.mean(after)], color="black", lw=2, marker="D", label="Mean")
        ax.set_xticks([0, 1], ["Baseline", "Q→A blocked"])
        ax.set_title(NAMES[variant])
    axes[0].set_ylabel("Displayed-answer score\n(mean token log probability; nats)")
    axes[0].legend(frameon=False)
    fig.suptitle("Absolute support for the displayed answer: the same 12 pilot questions\nLayers 18–23; each colored line is one question", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pilot_absolute_support.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    ir = pilot_middle["irrelevant_word"]
    absolute_stats = []
    for variant in VARIANTS:
        g = pilot_middle[variant]
        metrics = {key: float(np.mean([float(r[key]) for r in g])) for key in (
            "overlay_comparison_mean_before", "overlay_comparison_mean_after", "overlay_comparison_mean_gain",
            "overlay_correct_mean_gain", "no_text_comparison_mean_gain", "overlay_comparison_first_gain")}
        metrics.update({"variant": variant,
            "absolute_mean_gain_positive_count": sum(float(r["overlay_comparison_mean_gain"]) > 0 for r in g),
            "absolute_sum_gain_positive_count": sum(float(r["overlay_comparison_sum_gain"]) > 0 for r in g),
            "first_token_gain_positive_count": sum(float(r["overlay_comparison_first_gain"]) > 0 for r in g),
            "median_first_probability_before": float(np.median([np.exp(float(r["overlay_comparison_first_before"])) for r in g])),
            "median_first_probability_after": float(np.median([np.exp(float(r["overlay_comparison_first_after"])) for r in g]))})
        metrics["displayed_absolute_gain_DID"] = metrics["overlay_comparison_mean_gain"] - metrics["no_text_comparison_mean_gain"]
        absolute_stats.append(metrics)
    (OUT / "pilot_absolute_summary.json").write_text(json.dumps(absolute_stats, indent=2) + "\n")
    irs = next(r for r in absolute_stats if r["variant"] == "irrelevant_word")
    lines = ["# Relevance follow-up: score changes versus generated answers", "",
        "## Main finding", "",
        "Q→A blocking at layers 18–23 shifts relative answer scores toward irrelevant text, even though the 12-question pilot never generates an irrelevant overlay word. Irrelevant answers start much further behind than relevant misleading answers and usually remain behind after blocking. This supports a score-level constraint with limited behavioral impact in the pilot; it does not establish selective behavioral release of irrelevant text or an explicit relevance representation.", "",
        "## Full existing sample: exact within-run margin analysis", "",
        "All 305 questions are paired across the three harmful overlay conditions. Positive gap means the correct reference has a higher mean token log probability than the displayed answer. Values are score units (nats), not percentage points. The adjusted relative gain subtracts the same Q-blocking margin change on the paired no-text image.", "",
        "| Overlay | Mean gap before | Mean gap after Q block | Adjusted relative gain [exploratory 95% CI] | Still below correct |",
        "|---|---:|---:|---:|---:|"]
    for v in VARIANTS:
        r = middle[v]
        lines.append(f"| {NAMES[v]} | {float(r['baseline_correct_minus_displayed_mean']):.2f} | {float(r['blocked_correct_minus_displayed_mean']):.2f} | +{float(r['DID_relative_displayed_gain_mean']):.2f} [{float(r['DID_relative_displayed_gain_ci_low']):.2f}, {float(r['DID_relative_displayed_gain_ci_high']):.2f}] | {r['still_below_correct_count']}/305 |")
    lines += ["", "![Paired answer score gaps](answer_score_gaps.png)", "",
        f"For irrelevant text, the margin shifts toward the displayed answer in {middle['irrelevant_word']['raw_relative_gain_positive_count']}/305 cases. After blocking, the median probability of its first token is {float(middle['irrelevant_word']['blocked_first_probability_median']):.2g}; this is a next-token probability, not the probability of a complete answer. Using summed token log probability instead of the length-normalized score, {middle['irrelevant_word']['still_below_correct_sum_count']}/305 irrelevant answers remain below the correct reference.", "",
        "The late 30–35 window has much smaller adjusted relative effects (about 0.15–0.17 score units); it is not the principal window suggested by this analysis. All six existing windows are retained in the CSVs rather than selecting a new best window after inspection.", "",
        "## Absolute support: baseline reproduction on the 12 pilot questions", "",
        "The attention-intervention files saved exact baseline margins but not the component baseline scores. The activation-patching files do save absolute baselines, but they do not numerically reproduce the attention baselines. They were therefore not substituted. A small eager-attention rerun recovered the original baseline component scores for the same 12 questions, all three harmful overlays, both image states, and both candidate answers (144 scores).",
        f"Maximum reproduced-margin error: {audit['scores']['max_baseline_reproduction_error']:.8g}. All cached-versus-full and no-op checks passed in the scoring runner.", "",
        "| Overlay | Displayed score before | Displayed score after | Absolute gain | Correct-answer score change | Displayed gain after no-text adjustment |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in absolute_stats:
        lines.append(f"| {NAMES[r['variant']]} | {r['overlay_comparison_mean_before']:.2f} | {r['overlay_comparison_mean_after']:.2f} | {r['overlay_comparison_mean_gain']:+.2f} | {r['overlay_correct_mean_gain']:+.2f} | {r['displayed_absolute_gain_DID']:+.2f} |")
    lines += ["", f"Irrelevant-answer absolute scores increase in {irs['absolute_mean_gain_positive_count']}/12 cases; summed sequence scores increase in {irs['absolute_sum_gain_positive_count']}/12, and first-token probabilities increase in {irs['first_token_gain_positive_count']}/12. Their median first-token probability changes from {irs['median_first_probability_before']:.2g} to {irs['median_first_probability_after']:.2g}. All 12 still score below the correct reference after middle-window blocking, and all 12 continue to avoid the irrelevant word during free generation.", "",
        "![Absolute pilot answer support](pilot_absolute_support.png)", "",
        "## Semantic audit of the 864 pilot generations", "",
        "All 24 unique question–response pairs were checked. There are 453 exact-reference matches, 318 inherited evaluations, and 21 records covered by two explicit assistant-reviewed responses: 'food' is other rather than bowls/container/plates/whole; 'tennis ball' matches the correct reference 'ball'. These two reviews are not independent two-judge evaluations. One repeated response, 'basket' for the reference 'waste basket', has conflicting prior labels and remains ambiguous in all 72 occurrences. Its label is constant across interventions and does not affect the three harmful-condition adoption contrasts.", "",
        "The harmful-condition effects are unchanged: irrelevant 0 pp; grounded +8.3 pp; ungrounded +33.3 pp for Q blocking at 18–23. With T already blocked the incremental effects are 0, 0, and +16.7 pp respectively. The irrelevant-condition 'bowl'→'food' change occurs in both overlay and no-text images, so it does not count as text adoption.", "",
        "One secondary correction to the earlier exact-match table: correct-overlay adjusted following is +8.3 pp rather than zero under middle Q blocking. 'Bowl' is a semantic match to 'bowls'; Q blocking changes it to 'food' only on the no-text image, while the correct overlay preserves it. This is a difference-in-differences effect, not an increase in raw correct-overlay accuracy. Its T-specific interaction remains zero.", "",
        "## Interpretation and next step", "",
        "The pilot null should not be interpreted as evidence that Q→A does not constrain irrelevant-answer scores. It does constrain them under the tested intervention, but the irrelevant candidates typically remain weak. The 12 selected cases start even further behind (mean gap 16.37) than the full sample (13.88). The evidence is consistent with a score shift that rarely becomes an output change.", "",
        "The next behavioral test can focus on the existing 305 questions at Q layers 18–23, retaining T blocking at 30–35, joint blocks, paired no-text runs, and random visual controls. This is a recommendation, not a job launched by this analysis. A larger behavioral run would test whether any of the score changes translate into adoption; it is not guaranteed to produce a positive result. Do not select only favorable score-shift cases and report their rate as a population estimate.", "",
        "A concrete reason to expand: ten of the 305 irrelevant-condition runs assign the displayed answer's first token probability greater than 0.5 after middle Q blocking. Four already followed that word at baseline; six did not. None of those six is in the pilot. Four of the six displayed answers are single-token words (lead, fault, wall, contact), while island and giving each use two tokens. This is a teacher-forced prediction of potential behavioral changes, not a measured free-generation adoption rate. These cases are listed in `irrelevant_high_first_token_support.csv`; they should be checked within the full paired experiment, not presented as an outcome-blind sample.", "",
        "Limits: a correct-versus-displayed score gap is not the greedy decoding boundary; other answers may outrank both references, and length-normalized scores do not equal sequence probabilities. Teacher forcing supplies later candidate tokens. First-token and summed-score checks reduce that ambiguity, but do not prove mediation or a relevance representation. Random visual blocks are not matched linguistic controls for Q→A, and attention renormalization remains part of the intervention. The 305-question component-score comparison in the CSVs is Q blocking versus visual random blocking, explicitly not a recovered unblocked absolute baseline. Exact absolute before/after changes are available only for the 12 rerun questions.", "",
        "Bootstrap intervals use 5,000 paired question resamples with seed 271828. These are retrospective, exploratory intervals; no confirmatory or multiplicity-adjusted significance claim is made.", "",
        "## Reproduction and artifacts", "",
        "Run `python analyze_relevance_scores.py` after `score_pilot_baselines.sh` completes, then `python write_score_report.py`. Both scripts reside in `relevance_intervention/`.", "",
        "- `question_level_scores.csv`: exact margins and Q-blocked scores for all 305 questions and six windows.",
        "- `score_summary.csv` and `paired_condition_contrasts.csv`: full-sample and pilot summaries and paired condition contrasts.",
        "- `pilot_absolute_scores.csv` and `pilot_absolute_summary.json`: exact component-score changes on the pilot.",
        "- `pilot_semantic_audit.csv`, `pilot_unique_responses.csv`, and `pilot_semantic_contrasts.csv`: reproducible response review and corrected behavioral contrasts.",
        "- `analysis_audit.json`: record accounting, provenance, and numerical checks.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(json.dumps(absolute_stats, indent=2))
    print(f"Report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
