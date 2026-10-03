"""Fig. 2 — Behavioral effect of scene text across seven VLMs (open-ended answers, 474 questions per model).

(a) Accuracy change: overlay accuracy − clean-image accuracy.
(b) Additional displayed-answer adoption: adoption of the displayed word with its overlay − adoption of the
    same word on the clean image. For the correct overlay the displayed word is the correct answer, so (a)
    and (b) coincide for it.
Bars: estimates; whiskers: paired question-bootstrap 95% CIs (report Secs. 3.2, 17, 18).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 2.75
PANELS = [('open_ended_accuracy_change', 'Accuracy change (pp)'),
          ('open_ended_adjusted_following', 'Additional displayed-answer adoption (pp)')]
BAR_H = 0.19
OFFSETS = (np.arange(len(S.CONDITIONS)) - 1.5) * BAR_H     # vertical position of each condition within a model row

S.apply()
df = pd.read_csv(S.DATA['format_metrics']).set_index(['model', 'condition', 'metric'])
models = [m for m, _ in S.MODELS]
rows = np.arange(len(models))[::-1]                          # first model at the top

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, (metric, xlabel) in zip(axes, PANELS):
    for k in rows[:-1]:
        ax.axhline(k - 0.5, color=S.GRID, lw=0.6, zorder=0)    # thin separators between models
    for off, cond in zip(OFFSETS, S.CONDITIONS):
        color = S.COND[cond]['color']
        r = df.loc[[(m, cond, metric) for m in models]]
        y = rows - off
        ax.barh(y, 100 * r.estimate, BAR_H * 0.92, color=color, lw=0, zorder=2)
        ax.hlines(y, 100 * r.ci_low, 100 * r.ci_high, color=S.INK, lw=S.LW_CI * 0.8, zorder=3)
    S.zero_line(ax, vertical=True, color=S.INK, lw=0.8, zorder=4)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color=S.GRID, lw=0.5)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(rows, [S.MODEL_NAMES[m] for m in models])
axes[0].set_xlim(-22, 29); axes[1].set_xlim(-3, 35)
S.condition_legend(fig, patch=True, ncol=4, loc='lower center', bbox_to_anchor=(0.55, 0.95), handlelength=1.0, handleheight=0.8)
S.panel_label(axes[0], '(a)', x=-0.02, y=1.0)
S.panel_label(axes[1], '(b)', x=-0.02, y=1.0)
S.save(fig, 'appendix_fig2_behavior_two_panels')
