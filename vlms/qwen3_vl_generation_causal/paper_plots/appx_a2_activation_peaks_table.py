"""Appendix A2 — Activation-patching peak effects (LaTeX table), Qwen3-VL-8B, 305 questions.

Peak of the text-region minus matched-random effect across the 36 decoder layers, per direction, with the
question-bootstrap 95% CI at the peak layer. Peaks are post hoc maxima; the full curves are A1 and the main figure.
Source: outputs/step6_8b_analysis/peak_layer_summary.csv (scope all_305).
"""
import pandas as pd
import paper_plot_style as S

NAMES = {'correct_answer': 'Correct', 'misleading_groundable': 'Grounded misleading',
         'misleading_ungroundable': 'Ungrounded misleading', 'irrelevant_word': 'Irrelevant'}

df = pd.read_csv(S.CAUSAL / 'outputs/step6_8b_analysis/peak_layer_summary.csv')
df = df[df.scope == 'all_305'].set_index(['variant', 'direction'])
rows = []
for cond in S.CONDITIONS:
    cells = []
    for direction in ('restoration', 'insertion'):
        r = df.loc[(cond, direction)]
        cells += [S.fmt_ci(r.text_minus_random_mean, r.text_minus_random_ci95_low, r.text_minus_random_ci95_high),
                  str(int(r.layer))]
    rows.append(f'{NAMES[cond]} & ' + ' & '.join(cells) + r' \\')

tex = r'''
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrcrc}
\toprule
 & \multicolumn{2}{c}{Restoration} & \multicolumn{2}{c}{Insertion} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Overlay & Peak [95\% CI] & Layer & Peak [95\% CI] & Layer \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\caption{Peak activation-patching effects in Qwen3-VL-8B (305 questions). Each value is the text-region effect minus the mean of three count-matched random regions, as a change in the correct-versus-displayed complete-answer margin (nats/token), oriented so that positive values mean the intervention removes (restoration) or transfers (insertion) the overlay's influence. Layer is the zero-based decoder layer (of 36) with the largest effect. Brackets: question-bootstrap 95\% CI at that layer.}
\label{tab:activation-peaks}
\end{table}
'''
S.save_table(tex, 'tab_activation_peaks')
