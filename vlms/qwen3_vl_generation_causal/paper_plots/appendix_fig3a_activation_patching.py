"""Fig. 3a — Causal formation: activation patching across all 36 Qwen3-VL-8B layers (report Sec. 4.2).

Each line is the text-region effect minus the mean of three count-matched random regions
(change in the correct-vs-displayed complete-answer margin, nats/token, oriented so that positive =
expected direction), 305 questions. Bands are 95% CIs. Restoration: no-text states into the overlay run;
insertion: overlay states into the no-text run. Shaded: the frozen early–middle window used in Fig. 4b.
"""
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 1.85

S.apply()
df = pd.read_csv(S.DATA['patching_layers'])
df = df[df.scope == 'all_305']

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, direction, title in zip(axes, ('restoration', 'insertion'),
                                ('Restoration (remove text state)', 'Insertion (add text state)')):
    S.band(ax, S.EARLY_MIDDLE[0] - 0.5, S.EARLY_MIDDLE[1] + 0.5, 'early–middle')
    labels = []
    for cond in S.CONDITIONS:
        st = S.COND[cond]
        d = df[(df.variant == cond) & (df.direction == direction)].sort_values('layer')
        ax.fill_between(d.layer, d.text_minus_random_ci95_low, d.text_minus_random_ci95_high,
                        color=st['color'], alpha=0.13, lw=0)
        ax.plot(d.layer, d.text_minus_random_mean, color=st['color'], ls=st['ls'], marker=st['marker'],
                ms=2.1, markevery=3, label=st['label'])
        labels.append((d.text_minus_random_mean.iloc[-1], st['label'], st['color']))
    S.zero_line(ax)
    ax.set_title(title)
    ax.set_xlim(-0.8, 35.8)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_xlabel('Decoder layer')
axes[0].set_ylabel('Text − random effect\n(margin, nats/token)')
S.end_labels(axes[1], labels, 36.6, min_gap=0.42)
S.panel_label(axes[0], '(a)', x=-0.1)
S.save(fig, 'appendix_fig3a_activation_patching')
