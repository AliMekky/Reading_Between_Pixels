"""Appendix A4 — Key attention-pathway effects (LaTeX table), Qwen3-VL-8B, 305 questions.

Only the routes discussed in the paper, each at the window where it is discussed: T→Q (middle), T→A (late),
Q→A (middle), C→A (late). Full matrix: A3 figure and the source CSV (supplementary data).
Source: attention_intervention/outputs/analysis/path_summary.csv.
"""
import pandas as pd
import paper_plot_style as S

ROWS = [('T_to_Q', r'T$\rightarrow$Q', 'layers_12_17'), ('T_to_Q', r'T$\rightarrow$Q', 'layers_18_23'),
        ('T_to_A', r'T$\rightarrow$A', 'layers_30_35'), ('Q_to_A', r'Q$\rightarrow$A', 'layers_18_23'),
        ('C_to_A', r'C$\rightarrow$A', 'layers_30_35')]
ALPHA = 0.05

df = pd.read_csv(S.DATA['attention_paths']).set_index(['variant', 'window', 'path'])
lines = []
for path, label, window in ROWS:
    cells = []
    for cond in S.CONDITIONS:
        r = df.loc[(cond, window, path)]
        mark = r'$^{*}$' if r.p_holm_within_condition < ALPHA else r'\phantom{$^{*}$}'
        cells.append(S.fmt_ci(r.mean_oriented_effect, r.ci95_low, r.ci95_high) + mark)
    lines.append(f'{label} & {S.WINDOW_LABEL[window].replace("–", "--")} & ' + ' & '.join(cells) + r' \\')

tex = r'''
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Path & Layers & Correct & Grounded & Ungrounded & Irrelevant \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\caption{Effects of blocking selected attention pathways in Qwen3-VL-8B (305 questions; $n{=}279$--$283$ for C$\rightarrow$A, where a clean correct-object region exists). Each value is the change in the correct-versus-displayed complete-answer margin (nats/token) caused by blocking attention from the source to the destination tokens in a six-layer window, with the paired no-text effect subtracted; positive values mean blocking removes the overlay's influence (sign reversed for the correct overlay so that positive again means removing the text's effect). T: text region; Q: question tokens; A: answer positions; C: correct-object region. Brackets: question-bootstrap 95\% CI. $^{*}$Holm-adjusted $p<0.05$ within overlay condition.}
\label{tab:attention-effects}
\end{table*}
'''
S.save_table(tex, 'tab_attention_effects')
