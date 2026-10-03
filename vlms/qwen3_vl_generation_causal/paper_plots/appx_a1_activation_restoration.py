"""Appendix A1 — Activation patching, complementary direction (restoration), Qwen3-VL-8B.

Same design as the main insertion figure (fig3_activation_insertion.py). At each decoder layer, the clean-image
run's text-region residual states are put into the overlay run. y = change in the correct-vs-displayed
complete-answer margin (nats/token), oriented so positive = the overlay's influence is removed.
Colored: text region. Gray dashed: the same restoration at three count-matched random regions (mean), one line
per overlay condition. 305 questions; bands are question-bootstrap 95% CIs (report Sec. 4.2).
Source: outputs/step6_8b_analysis/layer_summary.csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 2.15

S.apply()
df = pd.read_csv(S.DATA['patching_layers'])
df = df[(df.scope == 'all_305') & (df.direction == 'restoration')]

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
ax.set_ylim(-0.4, 4.9); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_xlabel('Decoder layer')
ax.set_ylabel('Restoration effect (nats/token)')
S.condition_legend(ax, loc='lower left', bbox_to_anchor=(0.0, 0.07), ncol=2, fontsize=6.5, borderpad=0.3,
                   labelspacing=0.2, columnspacing=0.7,
                   extra=[(Line2D([], [], color=S.CONTROL, lw=S.LW_CONTROL, ls='--'), 'Random')])
S.save(fig, 'appx_a1_activation_restoration', S.APPENDIX_FIG_DIR)
