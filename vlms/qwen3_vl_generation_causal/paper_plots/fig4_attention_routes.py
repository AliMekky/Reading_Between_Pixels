"""Fig. 4 — Attention intervention: how scene-text information reaches the answer (Qwen3-VL-8B).

Attention from overlay-text tokens to (a) question tokens (T→Q) or (b) answer-predicting positions (T→A) is
blocked in six-layer windows. y = overlay-specific change in the complete-answer margin (overlay − paired
no-text image), text route minus the mean of three matched-random source routes, oriented so positive = the
overlay's influence is removed. 305 questions; bands are 95% CIs (report Sec. 5.2). Panels use separate
y-scales: T→Q is an order of magnitude weaker than late T→A. Other pathways and the formal grounded-vs-
ungrounded contrast: appendix.
"""
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 2.8
ROUTES = [('T_minus_R_to_Q', 'T→Q effect\n(nats/token)', '(a)'), ('T_minus_R_to_A', 'T→A effect\n(nats/token)', '(b)')]

S.apply()
df = pd.read_csv(S.DATA['attention_primary'])
df['x'] = df.window.map(S.WINDOW_MID)

fig, axes = plt.subplots(2, 1, figsize=(WIDTH, HEIGHT), sharex=True, gridspec_kw={'hspace': 0.3})
for ax, (path, ylabel, panel) in zip(axes, ROUTES):
    S.ygrid(ax)
    for cond in S.CONDITIONS:
        d, color = df[(df.variant == cond) & (df.path == path)].sort_values('x'), S.COND[cond]['color']
        ax.fill_between(d.x, d.ci95_low, d.ci95_high, color=color, alpha=S.BAND_ALPHA, lw=0)
        ax.plot(d.x, d.mean_oriented_effect, color=color, lw=S.LW_MAIN, marker='o', ms=S.MARKER_LINE)
    S.zero_line(ax)
    S.panel_label(ax, panel, x=-0.22, y=1.0)
    ax.set_ylabel(ylabel)
axes[0].set_ylim(-0.03, 0.32); axes[0].set_yticks([0, 0.1, 0.2, 0.3])
axes[1].set_ylim(-0.15, 2.5); axes[1].set_yticks([0, 1, 2])
axes[1].set_xticks([S.WINDOW_MID[w] for w in S.WINDOWS], [S.WINDOW_LABEL[w] for w in S.WINDOWS])
axes[1].set_xlim(0.5, 34.5)
axes[1].set_xlabel('Decoder layers (blocked window)')
S.condition_legend(axes[1], loc='upper left', ncol=2)   # boxed, in the empty early-layer corner of (b)
S.save(fig, 'fig4_attention_routes')
