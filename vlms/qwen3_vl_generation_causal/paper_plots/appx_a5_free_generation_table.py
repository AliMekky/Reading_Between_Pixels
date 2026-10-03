"""Appendix A5 — Free-generation intervention, all outcomes (LaTeX table), Qwen3-VL-8B, 305 questions.

Blocking the current answer query from the text-region keys at layers 30–35 during greedy decoding.
Difference-in-differences: (text block − mean of three matched-random blocks) on the overlay image minus the same
contrast on the paired no-text image. Answer changed is not in the main figure; the other two columns are.
Source: free_generation_intervention/outputs/evaluation/statistics/contrasts.csv.
"""
import pandas as pd
import paper_plot_style as S

NAMES = {'correct_answer': 'Correct', 'misleading_groundable': 'Grounded misleading',
         'misleading_ungroundable': 'Ungrounded misleading', 'irrelevant_word': 'Irrelevant'}
METRICS = ['answer_changed', 'target_following_change', 'accuracy_change']

df = pd.read_csv(S.DATA['free_generation'])
df = df[(df.contrast == 'overlay_minus_no_text_text_minus_random') & (df.image_state == 'difference_in_differences')]
df = df.set_index(['variant', 'metric'])
lines = []
for cond in S.CONDITIONS:
    cells = [S.fmt_ci(*df.loc[(cond, m), ['estimate', 'ci_low', 'ci_high']], d=1, scale=100) for m in METRICS]
    lines.append(f'{NAMES[cond]} & ' + ' & '.join(cells) + r' \\')

tex = r'''
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}
\toprule
Overlay & Answer changed & Displayed-answer following & Accuracy change \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\caption{Effect of blocking late text-to-answer attention during free generation in Qwen3-VL-8B (305 questions per overlay; layers 30--35; greedy decoding). All values are in percentage points and are difference-in-differences: blocking the text region minus the mean of three count-matched random regions, on the overlay image minus the same contrast on the paired no-text image. Answer changed: generated answer differs from the unblocked answer. Displayed-answer following: change in the rate of producing the overlaid word (for the correct overlay this equals the accuracy change). Brackets: question-bootstrap 95\% CI (10{,}000 resamples).}
\label{tab:free-generation-full}
\end{table*}
'''
S.save_table(tex, 'tab_free_generation_full')
