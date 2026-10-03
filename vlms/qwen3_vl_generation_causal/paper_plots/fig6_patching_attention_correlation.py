"""Fig. 6 — Connecting causal source and causal route across questions (Qwen3-VL-8B, 305 questions per panel).

x: early–middle activation-patching restoration effect (text − random, mean over layers 5–12).
y: late T→A attention-blocking effect (text − random, layers 30–35). One point per question.
Line: least-squares fit; ρ: Spearman. Partial ρ controlling for mapped text-token count (0.67 grounded,
0.73 ungrounded) is reported in the caption. The two measurements are associated across examples; this is
not a mediation analysis (report Sec. 9).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 1.85
X, Y = 'activation_restoration_L5_12', 'attention_TA_L30_35'
PANELS = ['misleading_groundable', 'misleading_ungroundable']

S.apply()
df = pd.read_csv(S.DATA['joined_samples'])
corr = pd.read_csv(S.DATA['correlations'])
corr = corr[(corr.subset == 'all') & (corr.x == X) & (corr.y == Y)].set_index('variant')

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharex=True, sharey=True, gridspec_kw={'wspace': 0.12})
for ax, cond, label in zip(axes, PANELS, ('(a)', '(b)')):
    d, color = df[df.variant == cond], S.COND[cond]['color']
    ax.scatter(d[X], d[Y], s=S.SCATTER ** 2, color=color, alpha=S.SCATTER_ALPHA, lw=0)
    slope, intercept = np.polyfit(d[X], d[Y], 1)
    xs = np.array([-4, 32]); ax.plot(xs, intercept + slope * xs, color=color, lw=S.LW_MAIN)
    ax.text(0.05, 0.96, f"{S.COND[cond]['label']}\nρ = {corr.loc[cond, 'spearman_rho']:.2f}", transform=ax.transAxes,
            ha='left', va='top', fontsize=S.FONT_ANNOT, color=S.INK, linespacing=1.3)
    ax.set_xlim(-4, 32); ax.set_ylim(-1.5, 16)
    ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 5, 10, 15])
    S.panel_label(ax, label, x=-0.04 if cond == PANELS[1] else -0.3, y=1.0)
axes[0].set_ylabel('Late T→A effect')
fig.supxlabel('Early–middle insertion effect', fontsize=S.FONT_AXIS_LABEL, y=-0.1)
S.save(fig, 'fig6_patching_attention_correlation')
