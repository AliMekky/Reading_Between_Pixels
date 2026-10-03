"""Fig. 4b — Formation co-varies with readout across questions (report Sec. 9).

x: early–middle activation-patching restoration effect (text − random, mean over layers 5–12);
y: late T→A attention-blocking effect (layers 30–35). One point per question (305 per panel).
Line: least-squares fit. ρ: Spearman; ρ_partial: after rank-residualizing mapped text-token count.
Both axes are independently causal measurements; their correlation does not establish mediation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 2.95, 1.85
X, Y = 'activation_restoration_L5_12', 'attention_TA_L30_35'
PANELS = ['misleading_groundable', 'misleading_ungroundable']

S.apply()
df = pd.read_csv(S.DATA['joined_samples'])
corr = pd.read_csv(S.DATA['correlations'])
corr = corr[(corr.subset == 'all') & (corr.x == X) & (corr.y == Y)].set_index('variant')

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharex=True, sharey=True, gridspec_kw={'wspace': 0.1})
for ax, cond in zip(axes, PANELS):
    st = S.COND[cond]; d = df[df.variant == cond]
    ax.plot(d[X], d[Y], ls='none', marker='o', ms=2.0, color=st['color'], alpha=0.45)
    slope, intercept = np.polyfit(d[X], d[Y], 1)
    xs = np.linspace(d[X].min(), d[X].max(), 2)
    ax.plot(xs, intercept + slope * xs, color=S.INK, lw=0.9)
    c = corr.loc[cond]
    ax.text(0.04, 0.96, f"ρ = {c.spearman_rho:.2f}\nρ$_{{partial}}$ = {c.partial_rho_controlling_text_tokens:.2f}",
            transform=ax.transAxes, va='top', fontsize=6.2)
    ax.set_title(st['label'])
    ax.grid(axis='x', color=S.GRID, lw=0.5)
fig.supxlabel('Early–middle formation effect (L5–12)', fontsize=7, y=-0.04)
axes[0].set_ylabel('Late T→A readout effect\n(L30–35)')
S.panel_label(axes[0], '(b)', x=-0.2)
S.save(fig, 'appendix_fig4b_formation_readout')
