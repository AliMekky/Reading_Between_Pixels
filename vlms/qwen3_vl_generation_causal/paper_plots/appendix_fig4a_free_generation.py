"""Fig. 4a — Blocking late T→A attention changes freely generated answers (report Sec. 7.3).

Qwen3-VL-8B, 305 questions, greedy decoding with the current answer query blocked from the text-region
keys at layers 30–35. Each value is a difference-in-differences: text blocking minus the mean of three
matched-random blocks, on the overlay image minus the same contrast on the paired no-text image.
Whiskers: 95% bootstrap CIs. For the correct overlay, displayed answer = correct answer, so both panels agree.
"""
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 3.35, 1.85
CONTRAST, STATE = 'overlay_minus_no_text_text_minus_random', 'difference_in_differences'
PANELS = [('target_following_change', 'Displayed answer'), ('accuracy_change', 'Accuracy')]

S.apply()
df = pd.read_csv(S.DATA['free_generation'])
df = df[(df.contrast == CONTRAST) & (df.image_state == STATE)].set_index(['variant', 'metric'])

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True, gridspec_kw={'wspace': 0.12})
for ax, (metric, title) in zip(axes, PANELS):
    S.zero_line(ax, color=S.INK, lw=0.8)
    for i, cond in enumerate(S.CONDITIONS):
        st = S.COND[cond]; r = df.loc[(cond, metric)]
        ax.vlines(i, 100 * r.ci_low, 100 * r.ci_high, color=st['color'], lw=1.0)
        ax.plot(i, 100 * r.estimate, marker=st['marker'], ms=4.4, color=st['color'], ls='none')
    ax.set_xticks(range(len(S.CONDITIONS)), [S.COND[c]['short'] for c in S.CONDITIONS])
    ax.tick_params(axis='x', length=0, pad=3)
    ax.set_xlim(-0.6, len(S.CONDITIONS) - 0.4)
    ax.set_title(title)
axes[0].set_ylabel('Change from blocking\nlate T→A (pp)')
axes[1].text(0.97, 0.04, 'misleading text blocked\n→ accuracy recovers', transform=axes[1].transAxes,
             ha='right', va='bottom', fontsize=5.6, color='#6E6E6E')
S.panel_label(axes[0], '(a)', x=-0.2)
S.save(fig, 'appendix_fig4a_free_generation')
