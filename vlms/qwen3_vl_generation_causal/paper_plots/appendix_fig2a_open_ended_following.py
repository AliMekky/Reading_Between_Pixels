"""Fig. 2a — Open-ended scene-text following across seven VLMs (report Sec. 3.2).

Adjusted following = adoption of the displayed word with its overlay minus adoption of that
word on the clean image, open-ended format (no answer options), 474 questions per model.
Points are estimates; whiskers are paired bootstrap 95% CIs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 4.15, 2.05
METRIC = 'open_ended_adjusted_following'

S.apply()
df = pd.read_csv(S.DATA['format_metrics'])
df = df[df.metric == METRIC].set_index(['model', 'condition'])

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
x = np.arange(len(S.MODELS))
offsets = np.linspace(-0.24, 0.24, len(S.CONDITIONS))
for off, cond in zip(offsets, S.CONDITIONS):
    st = S.COND[cond]
    rows = df.loc[[(m, cond) for m, _ in S.MODELS]]
    est, lo, hi = (100 * rows[c].to_numpy() for c in ('estimate', 'ci_low', 'ci_high'))
    ax.vlines(x + off, lo, hi, color=st['color'], lw=0.8, alpha=0.55, zorder=2)
    ax.plot(x + off, est, ls='none', marker=st['marker'], color=st['color'], ms=3.4, zorder=3, label=st['label'])

S.zero_line(ax)
ax.axvspan(3.5, 6.5, color=S.BAND, lw=0, zorder=0)           # Qwen3-VL family (causal analyses use 8B)
ax.text(5, 0.985, 'Qwen3-VL', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6, color='#6E6E6E')
ax.set_xticks(x, [label for _, label in S.MODELS], fontsize=5.9)
ax.tick_params(axis='x', length=0, pad=3)
ax.set_xlim(-0.55, len(S.MODELS) - 0.45)
ax.set_ylabel('Adjusted following (pp)')
ax.set_ylim(-3, 37)
ax.set_title('Open-ended: overlay-induced adoption of the displayed word')
ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.0), handletextpad=0.3, columnspacing=1.0)
S.panel_label(ax, '(a)', x=-0.1)
S.save(fig, 'appendix_fig2a_open_ended_following')
