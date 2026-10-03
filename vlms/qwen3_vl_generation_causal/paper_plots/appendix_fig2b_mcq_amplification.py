"""Fig. 2b — MCQ format amplification (report Sec. 3.4).

Amplification = MCQ adjusted following − open-ended adjusted following, paired within question.
Small dots: the seven models. Large marker + whisker: seven-model mean with paired bootstrap 95% CI.
Irrelevant text is emphasized: it is amplified in every model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = 2.05, 2.05
METRIC = 'mcq_minus_open_adjusted_following'
MEAN_KEY = 'unweighted_seven_model_mean'

S.apply()
df = pd.read_csv(S.DATA['format_metrics'])
df = df[df.metric == METRIC].set_index(['model', 'condition'])

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
rng = np.random.default_rng(0)
for i, cond in enumerate(S.CONDITIONS):
    st = S.COND[cond]
    emph = cond == 'irrelevant_word'
    models = 100 * df.loc[[(m, cond) for m, _ in S.MODELS], 'estimate'].to_numpy()
    jitter = rng.uniform(-0.13, 0.13, len(models))
    ax.plot(i + jitter, models, ls='none', marker=st['marker'], ms=2.6,
            color=st['color'] if emph else S.CONTROL, alpha=0.9 if emph else 0.8, zorder=2)
    m = df.loc[(MEAN_KEY, cond)]
    ax.vlines(i + 0.3, 100 * m.ci_low, 100 * m.ci_high, color=st['color'], lw=1.1, zorder=3)
    ax.plot(i + 0.3, 100 * m.estimate, marker=st['marker'], ms=4.6, color=st['color'], ls='none', zorder=4)

S.zero_line(ax)
ax.set_xticks(range(len(S.CONDITIONS)), [S.COND[c]['short'] for c in S.CONDITIONS])
for tick, cond in zip(ax.get_xticklabels(), S.CONDITIONS):
    if cond == 'irrelevant_word': tick.set_fontweight('bold')
ax.tick_params(axis='x', length=0, pad=3)
ax.set_xlim(-0.5, len(S.CONDITIONS) - 0.3)
ax.set_ylabel('MCQ − open-ended (pp)')
ax.set_title('MCQ amplification')
ax.text(0.98, 0.03, 'dots: 7 models\nlarge: mean, 95% CI', transform=ax.transAxes, va='bottom', ha='right', fontsize=5.8, color='#6E6E6E')
S.panel_label(ax, '(b)', x=-0.22)
S.save(fig, 'appendix_fig2b_mcq_amplification')
