"""Appendix A10 — Multiple choice vs open-ended scene-text following, seven VLMs (474 questions per model and format).

Additional displayed-answer adoption = adoption of the displayed word with its overlay minus adoption of that
same word on the clean image, within each format (pp). Hollow: open-ended (no options); filled: MCQ (the four
candidate words listed as options, letter answer). Bottom row: equal-weight mean of the seven models with
paired question-bootstrap 95% CIs (10,000 resamples). The 32B model is served by an API; the others run locally.
Source: format_replication_20260921/comparison/metrics.csv (report Secs. 3.3–3.4, 18).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 2.25
MEAN = 'unweighted_seven_model_mean'
ROWS = [m for m, _ in S.MODELS] + [MEAN]
LABELS = [S.MODEL_NAMES[m] for m, _ in S.MODELS] + ['Mean of 7']
FORMATS = [('open_ended_adjusted_following', False), ('mcq_adjusted_following', True)]

S.apply()
df = pd.read_csv(S.DATA['format_metrics']).set_index(['model', 'condition', 'metric'])
y = np.arange(len(ROWS))[::-1].astype(float); y[-1] -= 0.4          # extra gap before the mean row

fig, axes = plt.subplots(1, 4, figsize=(WIDTH, HEIGHT), sharey=True, sharex=True, gridspec_kw={'wspace': 0.08})
for ax, cond in zip(axes, S.CONDITIONS):
    color = S.COND[cond]['color']
    ax.grid(axis='x', color=S.GRID, lw=0.5)
    vals = {m: [100 * df.loc[(m, cond, metric), 'estimate'] for metric, _ in FORMATS] for m in ROWS}
    for yi, m in zip(y, ROWS):
        ax.plot(vals[m], [yi, yi], color=color, lw=S.LW_MAIN, alpha=0.55, zorder=2)
        for (metric, filled), v in zip(FORMATS, vals[m]):
            ax.plot(v, yi, marker='o', ms=S.MARKER, color=color, mfc=color if filled else 'white', mec=color,
                    mew=0.8, ls='none', zorder=3)
    for metric, _ in FORMATS:                                            # CIs on the mean row only
        r = df.loc[(MEAN, cond, metric)]
        ax.hlines(y[-1], 100 * r.ci_low, 100 * r.ci_high, color=S.INK, lw=S.LW_CI, zorder=1)
    ax.axhline((y[-2] + y[-1]) / 2, color=S.GRID, lw=0.8)
    S.zero_line(ax, vertical=True)
    ax.set_title(S.COND[cond]['label'], color=color, loc='center', fontweight='bold')
    ax.set_xlim(-2, 34); ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(y, LABELS)
axes[0].get_yticklabels()[-1].set_fontweight('bold')
fig.supxlabel('Additional displayed-answer adoption (pp)', fontsize=S.FONT_AXIS_LABEL, y=-0.02)
handles = [Line2D([], [], marker='o', ms=S.MARKER, color=S.INK, mfc='white', mew=0.8, ls='none'),
           Line2D([], [], marker='o', ms=S.MARKER, color=S.INK, ls='none')]
fig.legend(handles, ['Open-ended', 'MCQ'], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=6.5, borderpad=0.3)
S.save(fig, 'appx_a10_mcq_vs_open', S.APPENDIX_FIG_DIR)
