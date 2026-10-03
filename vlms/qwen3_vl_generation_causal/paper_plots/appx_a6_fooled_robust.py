"""Appendix A6 — Fooled vs robust questions: which causal measurements differ (Qwen3-VL-8B).

Groups are defined from behavior only: fooled = correct on the clean image and answers the displayed misleading
word with the overlay; robust = correct on both. Grounded: 23 fooled / 147 robust; ungrounded: 38 / 134.
Panels: (a) early–middle activation-patching restoration effect (text − random, mean over layers 5–12);
(b) middle T→Q blocking effect (layers 12–17); (c) late T→A blocking effect (layers 30–35). Layer windows were
fixed before the comparison. Solid bars = fooled, light bars = robust; whiskers = question-bootstrap 95% CIs of group means.
Source: free_generation_intervention/outputs/heterogeneity/group_summary.csv (report Sec. 8).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 1.75
PANELS = [('activation_restoration_L5_12', 'Early–middle\nrestoration (L5–12)', '(a)'),
          ('attention_TQ_L12_17', 'Middle T→Q\n(L12–17)', '(b)'),
          ('attention_TA_L30_35', 'Late T→A\n(L30–35)', '(c)')]
VARIANTS = ['misleading_groundable', 'misleading_ungroundable']
DX = 0.2

S.apply()
df = pd.read_csv(S.CAUSAL / 'free_generation_intervention/outputs/heterogeneity/group_summary.csv')
df = df.set_index(['variant', 'group', 'metric'])

fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace': 0.62})
for ax, (metric, title, panel) in zip(axes, PANELS):
    S.ygrid(ax)
    for i, cond in enumerate(VARIANTS):
        color = S.COND[cond]['color']
        for group, dx, alpha in (('fooled', -DX, 1.0), ('robust', DX, 0.35)):
            r = df.loc[(cond, group, metric)]
            ax.bar(i + dx, r['mean'], 2 * DX * S.BAR_FILL, color=color, alpha=alpha, lw=0, zorder=2)
            ax.vlines(i + dx, r.ci_low, r.ci_high, color=S.INK, lw=S.LW_CI, zorder=3)
    S.zero_line(ax)
    ax.set_xticks(range(len(VARIANTS)), ['Grnd.', 'Ungr.'])
    for tick, cond in zip(ax.get_xticklabels(), VARIANTS):
        tick.set_color(S.COND[cond]['color'])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.55, len(VARIANTS) - 0.45)
    ax.set_title(f'{panel} {title}', loc='center', fontsize=6.6, pad=4)
axes[0].set_ylabel('Effect (nats/token)')
handles = [Patch(color=S.INK, lw=0), Patch(color=S.INK, alpha=0.35, lw=0)]
fig.legend(handles, ['Fooled', 'Robust'], loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)
S.save(fig, 'appx_a6_fooled_robust', S.APPENDIX_FIG_DIR)
