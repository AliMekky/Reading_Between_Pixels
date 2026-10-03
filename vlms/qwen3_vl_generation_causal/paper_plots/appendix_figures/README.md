# Appendix figures

Spec: `Reading_Between_Pixels/appendix_figures_tables_instructions.md`. Every figure is produced by the script of the
same name in `../` (run from `paper_plots/`), uses `paper_plot_style.py` (same palette, fonts, line widths, band
style and condition order as the main-paper figures), and is exported as vector PDF (for LaTeX), SVG and PNG preview.
All analyses are Qwen3-VL-8B on the 305-question causal set unless stated otherwise. No model inference is run.

| File | Script | Source analysis | Appendix subsection | Width |
|---|---|---|---|---|
| `appx_a1_activation_restoration` | `appx_a1_activation_restoration.py` | `outputs/step6_8b_analysis/layer_summary.csv` | A1 Complementary activation-patching direction | single (3.3 in) |
| `appx_a3_attention_pathways` | `appx_a3_attention_pathways.py` | `attention_intervention/outputs/analysis/path_summary.csv` | A3 Full attention-pathway analysis | double (6.9 in) |
| `appx_a6_fooled_robust` | `appx_a6_fooled_robust.py` | `free_generation_intervention/outputs/heterogeneity/group_summary.csv` | A6 Fooled vs robust | single (3.3 in) |
| `appx_a8_candidate_trajectory` | `appx_a8_candidate_trajectory.py` | `relevance_intervention/outputs/input_comparison_305/candidate_scores.csv` | A8 Starting plausibility | single (3.3 in) |
| `appx_a10_mcq_vs_open` | `appx_a10_mcq_vs_open.py` | `../format_replication_20260921/comparison/metrics.csv` (seven models, 474 questions) | A10 MCQ behavioral comparison | double (6.9 in) |

A5 has no figure: the table carries the answer-changed result clearly (spec: prefer the table).
A11/A12 (MCQ mechanisms) are intentionally not included (LLaVA-NeXT runs, not same-model comparisons).

## What each figure adds beyond the main paper

- **A1**: the second patching direction; the main paper shows insertion only.
- **A3**: the routes not shown in the main paper (Q→A, object routes, all-visual V routes, text↔object, random control).
- **A6**: which causal measurements distinguish fooled from robust questions (not in the main paper).
- **A8**: why irrelevant text is rarely adopted despite a large internal boost (not in the main paper).
- **A10**: the MCQ format comparison per model (the main paper is open-ended only).

## Draft captions

**A1.** Activation patching in the restoration direction (Qwen3-VL-8B, 305 questions). At each decoder layer, the
clean-image run's residual states at the overlay-text positions are copied into the overlay run; the y-axis is the
resulting change in the correct-versus-displayed complete-answer margin (nats/token), oriented so positive values mean
the overlay's influence is removed. Colored lines: text region; gray dashed lines: the same intervention at three
count-matched random regions (mean), one per overlay. Bands: question-bootstrap 95% CIs. As with insertion
(Figure 3), text-region effects are large through the early and middle layers and vanish toward the last layer,
while random regions stay near zero. Restoration and insertion need not be symmetric because the model is nonlinear.

**A3.** Effects of blocking attention along every measured pathway (Qwen3-VL-8B, 305 questions). Each cell is the
overlay-specific change in the complete-answer margin (overlay minus paired no-text image, nats/token) when attention
from the source to the destination tokens is blocked in one six-layer window; positive (terracotta) means blocking
removes the overlay's influence, negative (blue) means the pathway normally opposes it. T: text region; R: count-matched
random region; C/G: correct/grounded object; V: all non-text visual tokens; Q: question; A: answer positions. Dots:
Holm-adjusted p < 0.05 within overlay. Symmetric-log color scale (linear within ±0.1). Text routes carry the overlay
signal (T→A dominant and late, T→Q weaker and middle); Q→A, C→A and V routes oppose it; random routes stay near zero.
V is a large, unmatched token set, so its magnitude is not comparable to T.

**A6.** Causal measurements in fooled versus robust questions (Qwen3-VL-8B). Fooled: correct on the clean image and
answers the displayed misleading word with the overlay; robust: correct on both (grounded 23/147, ungrounded 38/134).
(a) Early–middle activation-patching restoration effect (text minus random, layers 5–12); (b) middle T→Q and
(c) late T→A attention-blocking effects (text minus random). Solid bars: fooled; light bars: robust; whiskers: bootstrap 95%
CIs of group means. Windows were fixed before the comparison. Fooled questions have much larger early–middle text
effects, and for ungrounded text a larger middle T→Q effect, but late T→A is not reliably stronger. Groups are
small and defined by behavior; this describes associations, not why individual questions are fooled.

**A8.** Support for each candidate answer as input is added (Qwen3-VL-8B, 305 questions). Mean teacher-forced
log-probability per answer token for the four candidate answers, given the question alone, a uniform gray image, the
clean scene, and the clean scene with that candidate's own word overlaid. Bands: question-bootstrap 95% CIs. Irrelevant
candidates start about 10 nats/token below the relevant misleading candidates, receive an overlay boost similar to
grounded misleading text, and still finish far below every other candidate. Teacher-forced scores are not generation
rates.

**A10.** Scene-text following with and without answer options, across seven VLMs (474 questions per model and
format). Additional displayed-answer adoption: adoption of the displayed word with its overlay minus adoption of the
same word on the clean image, within each format (pp). Hollow: open-ended; filled: multiple choice with the four
candidate words as options. Bottom row: equal-weight mean of the seven models with paired question-bootstrap 95% CIs.
MCQ increases irrelevant-text following in every model, while it often reduces following of correct text. Qwen3-VL-32B
is served through an API; the other models run locally.
