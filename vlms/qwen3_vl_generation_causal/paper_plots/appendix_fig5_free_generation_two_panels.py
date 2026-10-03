"""Fig. 5 — Free-generation causal validation: blocking late T→A changes generated answers (Qwen3-VL-8B).

During greedy generation, attention from the text region to the current answer query is blocked at layers
30–35. Values are difference-in-differences: text block − mean of three matched-random blocks, on the overlay
image − the same contrast on the paired no-text image. 305 questions per overlay; whiskers are question-
bootstrap 95% CIs (report Sec. 7.3). For the correct overlay the displayed answer is the correct answer, so
(a) and (b) coincide for it. Answer-change rate: appendix.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 2.0
CONTRAST, STATE = 'overlay_minus_no_text_text_minus_random', 'difference_in_differences'
PANELS = [('target_following_change', 'Change in displayed-answer\nfollowing (pp)'),
          ('accuracy_change', 'Change in accuracy (pp)')]

S.apply()
df = pd.read_csv(S.DATA['free_generation'])
df = df[(df.contrast == CONTRAST) & (df.image_state == STATE)].set_index(['variant', 'metric'])

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True, gridspec_kw={'wspace': 0.42})
x = np.arange(len(S.CONDITIONS))
for ax, (metric, ylabel) in zip(axes, PANELS):
    S.ygrid(ax)
    for xi, cond in zip(x, S.CONDITIONS):
        r, color = df.loc[(cond, metric)], S.COND[cond]['color']
        ax.bar(xi, 100 * r.estimate, 0.66, color=color, lw=0, zorder=2)
        ax.vlines(xi, 100 * r.ci_low, 100 * r.ci_high, color=S.INK, lw=S.LW_CI, zorder=3)
    ax.set_xticks(x, [S.COND[c]['short'] for c in S.CONDITIONS], rotation=0)
    ax.tick_params(axis='x', length=0, labelsize=S.FONT_TICK - 0.5)
    ax.set_xlim(-0.5, len(S.CONDITIONS) - 0.5)
    ax.set_ylabel(ylabel)
    S.zero_line(ax, color=S.INK, lw=0.8, zorder=4)
    ax.yaxis.set_tick_params(labelleft=True)
for tick_ax in axes:
    for tick, cond in zip(tick_ax.get_xticklabels(), S.CONDITIONS):
        tick.set_color(S.COND[cond]['color'])
axes[0].set_ylim(-12, 9); axes[0].set_yticks([-10, -5, 0, 5])
S.panel_label(axes[0], '(a)', x=-0.36, y=1.0)
S.panel_label(axes[1], '(b)', x=-0.36, y=1.0)
S.save(fig, 'appendix_fig5_free_generation_two_panels')
