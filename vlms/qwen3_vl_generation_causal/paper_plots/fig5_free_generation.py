"""Fig. 5 — Free-generation causal validation: blocking late T→A changes generated answers (Qwen3-VL-8B).

Accuracy change when attention from the text region to the current answer query is blocked at layers 30–35
during greedy generation. Difference-in-differences: text block − mean of three matched-random blocks, on the
overlay image − the same contrast on the paired no-text image. 305 questions per overlay; whiskers are
question-bootstrap 95% CIs (report Sec. 7.3). Displayed-answer following (−6.8 grounded, −7.9 ungrounded) and
answer-change rate: appendix_fig5_free_generation_two_panels.py and the appendix table.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 1.85
CONTRAST, STATE, METRIC = 'overlay_minus_no_text_text_minus_random', 'difference_in_differences', 'accuracy_change'

S.apply()
df = pd.read_csv(S.DATA['free_generation'])
df = df[(df.contrast == CONTRAST) & (df.image_state == STATE) & (df.metric == METRIC)].set_index('variant')

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
S.ygrid(ax)
x = np.arange(len(S.CONDITIONS))
for xi, cond in zip(x, S.CONDITIONS):
    r = df.loc[cond]
    ax.bar(xi, 100 * r.estimate, 0.55 * S.BAR_FILL, color=S.COND[cond]['color'], lw=0, zorder=2)
    ax.vlines(xi, 100 * r.ci_low, 100 * r.ci_high, color=S.INK, lw=S.LW_CI, zorder=3)
S.zero_line(ax, color=S.INK, lw=0.8, zorder=4)
ax.set_xticks(x, [S.COND[c]['label'] for c in S.CONDITIONS])
ax.tick_params(axis='x', length=0)
for tick, cond in zip(ax.get_xticklabels(), S.CONDITIONS):
    tick.set_color(S.COND[cond]['color'])
ax.set_xlim(-0.5, len(S.CONDITIONS) - 0.5)
ax.set_ylim(-7, 8.5); ax.set_yticks([-6, -3, 0, 3, 6])
ax.set_ylabel('Change in accuracy (pp)')
S.save(fig, 'fig5_free_generation')
