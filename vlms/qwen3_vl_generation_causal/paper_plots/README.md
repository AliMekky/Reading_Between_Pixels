# Paper plots (ACL submission)

One script per figure. Scripts only read saved analysis outputs (no inference, no recomputed numbers) and
write `figures/<name>.pdf` (vector), `.svg`, and `.png` (preview) at final print size.

```bash
./make_all.sh                          # all figures
python fig4_attention_routes.py        # one figure
```

All styling lives in `paper_plot_style.py`: palette, condition order, widths (3.3 in single / 6.9 in double),
font sizes, line widths, marker sizes, band alpha, Matplotlib settings, data paths and the export helper.
Fonts: Liberation Sans (metric-compatible with Arial) from `fonts/`, embedded as TrueType.

## Main-paper figures

| Script | Width | Shows | Data |
|---|---|---|---|
| `fig2_behavior.py` | single | Horizontal grouped bars: additional displayed-answer adoption vs. clean image, 7 models × 4 overlays, open-ended, 95% CIs | `format_replication_20260921/comparison/metrics.csv` |
| `fig3_activation_insertion.py` | single | Insertion per layer: text-region effect for 4 overlays (colored) with the matched-random-region control (gray dashed) | `outputs/step6_8b_analysis/layer_summary.csv` |
| `fig4_attention_routes.py` | single | (a) T→Q, (b) T→A blocking per six-layer window (separate y-scales; route names as panel headers) | `attention_intervention/outputs/analysis/primary_text_path_summary.csv` |
| `fig5_free_generation.py` | single | Bars: accuracy change from blocking late T→A during generation | `free_generation_intervention/outputs/evaluation/statistics/contrasts.csv` |
| `fig6_patching_attention_correlation.py` | single | Early–middle restoration vs late T→A per question, grounded and ungrounded, Spearman ρ | `free_generation_intervention/outputs/heterogeneity/` |

Caption notes: Fig. 3 shows insertion (restoration → appendix), while Fig. 6 uses the restoration L5–12 metric; Fig. 4 panels have different y-scales; Fig. 5 text should cite displayed-answer reductions (−6.8 grounded, −7.9 ungrounded);
Fig. 6 partial ρ controlling for text-token count is 0.67 / 0.73 and the figure is not a mediation analysis.

## Other scripts

- `fig1_setup.py`: setup example (clean image and four overlays) and analysis pipeline; needs the extracted dataset.
- `appx_*.py`: appendix figures (`appendix_figures/`, see its README) and LaTeX tables (`appendix_tables/`).
