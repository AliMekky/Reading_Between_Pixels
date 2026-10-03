"""Fig. 3b — Causal readout: primary text routes under directed attention blocking (report Sec. 5.2).

Blocking text-token → question-token (T→Q) or text-token → answer-position (T→A) attention in
six-layer windows of Qwen3-VL-8B. Effect = overlay-specific change in the complete-answer margin
(overlay minus paired no-text), minus the mean of three matched-random source regions; oriented so
positive = removes the overlay's influence. 305 questions; bands are 95% CIs.
Note the different y-scales: T→Q is an order of magnitude weaker than late T→A.
"""
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 4.2, 1.85
ROUTES = [('T_minus_R_to_Q', 'Text → question (T→Q)'), ('T_minus_R_to_A', 'Text → answer (T→A)')]

S.apply()
df = pd.read_csv(S.DATA['attention_primary'])
df['x'] = df.window.map(S.WINDOW_MID)

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace': 0.32})
for ax, (path, title) in zip(axes, ROUTES):
    S.band(ax, S.LATE[0] - 0.5, S.LATE[1] + 0.5, 'late')
    labels = []
    for cond in S.CONDITIONS:
        st = S.COND[cond]
        d = df[(df.variant == cond) & (df.path == path)].sort_values('x')
        ax.fill_between(d.x, d.ci95_low, d.ci95_high, color=st['color'], alpha=0.13, lw=0)
        ax.plot(d.x, d.mean_oriented_effect, color=st['color'], ls=st['ls'], marker=st['marker'], ms=2.8,
                label=st['label'])
        labels.append((d.mean_oriented_effect.iloc[-1], st['label'], st['color']))
    S.zero_line(ax)
    ax.set_title(title)
    ax.set_xlim(-1, 36)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 35])
    ax.set_xlabel('Decoder layer (6-layer windows)')
axes[0].set_ylabel('Blocking effect\n(margin, nats/token)')
S.end_labels(axes[1], labels, 36.8, min_gap=0.2)
S.panel_label(axes[0], '(b)', x=-0.2)
S.save(fig, 'appendix_fig3b_attention_routes')
