"""Fig. 3c — Groundedness contrast: ungrounded − grounded route effects (report Sec. 5.5).

Paired (within-question) difference between ungrounded and grounded misleading overlays in the
T→Q and T→A blocking effects of Fig. 3b, per six-layer window; 95% CIs. Filled markers: Holm-adjusted
p < 0.05 across the 12 tests; hollow: not significant. The dominant T→A route does not differ;
the weaker T→Q route is stronger for ungrounded text in the middle layers.
"""
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 2.1, 1.85
ROUTES = {'Q': dict(label='T→Q', color=S.CHARCOAL, marker='o', ls='-', dx=-0.45),
          'A': dict(label='T→A', color=S.CONTROL, marker='s', ls='--', dx=0.45)}
ALPHA = 0.05

S.apply()
df = pd.read_csv(S.DATA['groundedness'])
df['x'] = df.window.map(S.WINDOW_MID)

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
for dest, st in ROUTES.items():
    d = df[df.destination == dest].sort_values('x')
    x = d.x + st['dx']
    ax.vlines(x, d.ci95_low, d.ci95_high, color=st['color'], lw=0.8, zorder=2)
    ax.plot(x, d.ungrounded_minus_grounded_mean, color=st['color'], ls=st['ls'], lw=0.9, zorder=3)
    sig = d.p_holm_across_12_tests < ALPHA
    ax.plot(x[sig], d.ungrounded_minus_grounded_mean[sig], ls='none', marker=st['marker'], ms=3.4,
            color=st['color'], zorder=4)
    ax.plot(x[~sig], d.ungrounded_minus_grounded_mean[~sig], ls='none', marker=st['marker'], ms=3.4,
            mfc='white', mec=st['color'], mew=0.8, zorder=4)
    ax.plot([], [], color=st['color'], ls=st['ls'], marker=st['marker'], ms=3.4, label=st['label'])
S.zero_line(ax)
ax.legend(loc='lower left', handlelength=2.2)
ax.set_xlim(-1, 36)
ax.set_xticks([0, 6, 12, 18, 24, 30, 35])
ax.set_xlabel('Decoder layer (6-layer windows)')
ax.set_ylabel('Ungrounded − grounded')
ax.set_title('Groundedness contrast')
ax.text(0.03, 0.97, 'filled: Holm p < .05', transform=ax.transAxes, va='top', fontsize=5.6, color='#6E6E6E')
S.panel_label(ax, '(c)', x=-0.25)
S.save(fig, 'appendix_fig3c_groundedness_contrast')
