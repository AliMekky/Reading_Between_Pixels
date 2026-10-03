"""Appendix A8 — Candidate support from question-only input to the candidate's own overlay (Qwen3-VL-8B, 305 questions).

Mean teacher-forced log-probability per answer token for each candidate answer (EOS excluded) under four inputs:
question only (no image tokens), a uniform gray image, the clean scene, and the clean scene with that candidate's
own word overlaid. All inputs use the same short-answer instruction. Bands: question-bootstrap 95% CIs
(10,000 resamples, seed 271828). Source: relevance_intervention/outputs/input_comparison_305/candidate_scores.csv
(report Sec. 16).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_plot_style as S

WIDTH, HEIGHT = S.SINGLE, 2.15
STAGES = [('question_only', 'Question\nonly'), ('gray_image', 'Gray\nimage'), ('clean_image', 'Clean\nscene'),
          ('own', 'Own\noverlay')]
B, SEED = 10000, 271828

S.apply()
scores = pd.read_csv(S.CAUSAL / 'relevance_intervention/outputs/input_comparison_305/candidate_scores.csv',
                     dtype={'question_id': str})
rng = np.random.default_rng(SEED)
qids = sorted(scores.question_id.unique())
idx = rng.integers(0, len(qids), (B, len(qids)))

fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
S.ygrid(ax)
x = np.arange(len(STAGES))
for cond in S.CONDITIONS:
    st, d = S.COND[cond], scores[scores.candidate == cond]
    cols = []
    for state, _ in STAGES:
        s = d[d.state == (cond if state == 'own' else state)].set_index('question_id').loc[qids, 'mean_logprob']
        cols.append(s.to_numpy())
    mat = np.column_stack(cols)
    mean = mat.mean(0); draws = mat[idx].mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    ax.fill_between(x, lo, hi, color=st['color'], alpha=S.BAND_ALPHA, lw=0)
    ax.plot(x, mean, color=st['color'], lw=S.LW_MAIN, marker='o', ms=S.MARKER_LINE + 0.6, label=st['label'])
ax.set_xticks(x, [label for _, label in STAGES])
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.2, len(STAGES) - 0.8)
ax.set_ylim(-28, 0); ax.set_yticks([-25, -20, -15, -10, -5, 0])
ax.set_ylabel('Mean candidate log-probability\n(nats/token)')
S.condition_legend(ax, loc='upper left', fontsize=6.5, borderpad=0.3, labelspacing=0.2)
S.save(fig, 'appx_a8_candidate_trajectory', S.APPENDIX_FIG_DIR)
