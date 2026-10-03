"""Fig. 2 — Scene text changes what VLMs answer, across seven models (open-ended, 474 questions per model).

Bar = additional displayed-answer adoption: how much more often the model answers with the displayed word
when it is overlaid than on the clean image (pp). For the correct overlay the displayed word is the correct
answer, so its bar is the accuracy gain. Whiskers: paired question-bootstrap 95% CIs (report Secs. 3.2, 17, 18).
Accuracy change: appendix_fig2_behavior_two_panels.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 3.0
METRIC = 'open_ended_adjusted_following'
BAR_H = 0.19
OFFSETS = (np.arange(len(S.CONDITIONS)) - 1.5) * BAR_H      # position of each condition within a model row

S.apply()
df = pd.read_csv(S.DATA['format_metrics']).set_index(['model', 'condition', 'metric'])
models = [m for m, _ in S.MODELS]
rows = np.arange(len(models))[::-1]                          # first model at the top

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
ax.grid(axis='x', color=S.GRID, lw=0.5)
for k in rows[:-1]:
    ax.axhline(k - 0.5, color=S.GRID, lw=0.6, zorder=0)      # thin separators between models
for off, cond in zip(OFFSETS, S.CONDITIONS):
    r = df.loc[[(m, cond, METRIC) for m in models]]
    y = rows - off
    ax.barh(y, 100 * r.estimate, BAR_H * S.BAR_FILL, color=S.COND[cond]['color'], lw=0, zorder=2)
    ax.hlines(y, 100 * r.ci_low, 100 * r.ci_high, color=S.INK, lw=S.LW_CI * 0.8, zorder=3)
S.zero_line(ax, vertical=True, color=S.INK, lw=0.8, zorder=4)
ax.set_yticks(rows, [S.MODEL_NAMES[m] for m in models])
ax.tick_params(axis='y', length=0)
ax.set_ylim(-0.6, len(models) - 0.4)
ax.set_xlim(-1, 35); ax.set_xticks([0, 10, 20, 30])
ax.set_xlabel('Additional displayed-answer adoption (pp)')
S.condition_legend(ax, patch=True, loc='best', handlelength=0.9, handleheight=0.8)   # boxed, emptiest corner
S.save(fig, 'fig2_behavior')
