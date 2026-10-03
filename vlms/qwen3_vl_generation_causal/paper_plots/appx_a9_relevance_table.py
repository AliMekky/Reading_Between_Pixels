"""Appendix A9 — Starting plausibility, image and overlay contributions (LaTeX table), Qwen3-VL-8B, 305 questions.

Question-only score + image boost + overlay boost = final score (own overlay), up to rounding.
Also: number of questions with a positive overlay boost, and the median candidate-minus-correct margin in the clean
scene and under the candidate's own overlay (not defined for the correct candidate).
Source: relevance_intervention/outputs/input_comparison_305/{candidate_scores,contrast_summary}.csv (report Sec. 16).
"""
import pandas as pd
import paper_plot_style as S

NAMES = {'correct_answer': 'Correct', 'misleading_groundable': 'Grounded misleading',
         'misleading_ungroundable': 'Ungrounded misleading', 'irrelevant_word': 'Irrelevant'}
BASE = S.CAUSAL / 'relevance_intervention/outputs/input_comparison_305'

scores = pd.read_csv(BASE / 'candidate_scores.csv')
c = pd.read_csv(BASE / 'contrast_summary.csv')
c = c[c.metric == 'mean_logprob'].set_index(['effect', 'candidate'])
num = lambda v: f'{v + 0.0:.2f}'.replace('-', '$-$')
rows_a, rows_b = [], []
for cond in S.CONDITIONS:
    s = scores[scores.candidate == cond]
    q_only = s[s.state == 'question_only'].mean_logprob.mean()
    final = s[s.state == cond].mean_logprob.mean()
    img, ovl = c.loc[('clean_image_minus_question_only', cond)], c.loc[('own_overlay_minus_clean_image', cond)]
    rows_a.append(f'{NAMES[cond]} & {num(q_only)} & ' + S.fmt_ci(img['mean'], img.ci_low, img.ci_high) + ' & '
                  + S.fmt_ci(ovl['mean'], ovl.ci_low, ovl.ci_high) + f' & {num(final)} \\\\')
    pos = f"{int(ovl.positive)}/{int(ovl.n)}"
    if cond == 'correct_answer':
        margins = r'\multicolumn{1}{c}{--}'
    else:
        before = c.loc[('candidate_minus_correct_in_clean_image', cond), 'median']
        after = c.loc[('candidate_minus_correct_in_own_overlay', cond), 'median']
        margins = f'{num(before)} $\\rightarrow$ {num(after)}'
    rows_b.append(f'{NAMES[cond]} & {pos} & {margins} \\\\')

tex = r'''
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Candidate & Question only & Image boost & Overlay boost & Final score \\
\midrule
''' + '\n'.join(rows_a) + r'''
\bottomrule
\end{tabular}

\vspace{0.6em}
\begin{tabular}{lrr}
\toprule
Candidate & Positive overlay boost & Median margin vs.\ correct (clean $\rightarrow$ overlay) \\
\midrule
''' + '\n'.join(rows_b) + r'''
\bottomrule
\end{tabular}
\caption{Candidate-answer support in Qwen3-VL-8B (305 questions; mean log-probability per answer token, nats/token; EOS excluded). Question only: no image tokens. Image boost: clean scene minus question only. Overlay boost: the candidate's own overlay minus the clean scene. Final score: under the candidate's own overlay (question only + image boost + overlay boost, up to rounding). Brackets: question-bootstrap 95\% CI (10{,}000 resamples). Bottom: number of questions whose overlay boost is positive, and the median of the candidate-minus-correct score margin in the clean scene and under the candidate's own overlay (negative = behind the correct answer).}
\label{tab:relevance-summary}
\end{table*}
'''
S.save_table(tex, 'tab_relevance_summary')
