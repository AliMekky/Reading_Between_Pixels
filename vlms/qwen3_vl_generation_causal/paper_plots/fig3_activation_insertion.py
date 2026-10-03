"""Fig. 3 — Activation patching (insertion): where scene text becomes causally important (Qwen3-VL-8B).

At each decoder layer, the overlay run's text-region residual states are inserted into the paired clean-image
run. y = change in the correct-vs-displayed complete-answer margin (nats/token), oriented so positive = the
overlay's influence is transferred. Colored: text region. Gray dashed: the same insertion at three
count-matched random regions (mean of the three), one line per overlay condition — the spatial control.
305 questions; bands are question-bootstrap 95% CIs (report Sec. 4.2). Restoration: appendix.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 2.15

S.apply()
df = pd.read_csv(S.DATA['patching_layers'])
df = df[(df.scope == 'all_305') & (df.direction == 'insertion')]

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
S.ygrid(ax)
for cond in S.CONDITIONS:
    d = df[df.variant == cond].sort_values('layer')
    ax.fill_between(d.layer, d.random_mean_effect_ci95_low, d.random_mean_effect_ci95_high,
                    color=S.CONTROL, alpha=S.BAND_ALPHA, lw=0)
    ax.plot(d.layer, d.random_mean_effect_mean, color=S.CONTROL, lw=S.LW_CONTROL, ls='--', zorder=2)
for cond in S.CONDITIONS:
    d, color = df[df.variant == cond].sort_values('layer'), S.COND[cond]['color']
    ax.fill_between(d.layer, d.text_effect_ci95_low, d.text_effect_ci95_high, color=color, alpha=S.BAND_ALPHA, lw=0)
    ax.plot(d.layer, d.text_effect_mean, color=color, lw=S.LW_MAIN, zorder=3)
S.zero_line(ax)
ax.set_xlim(0, 35); ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
ax.set_ylim(-0.4, 5.9); ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel('Decoder layer')
ax.set_ylabel('Insertion effect (nats/token)')
S.condition_legend(ax, loc='upper right', fontsize=6.5, borderpad=0.3, labelspacing=0.2, extra=[(Line2D([], [], color=S.CONTROL, lw=S.LW_CONTROL, ls='--'), 'Random')])
S.save(fig, 'fig3_activation_insertion')
