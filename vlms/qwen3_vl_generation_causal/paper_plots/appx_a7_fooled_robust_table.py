"""Appendix A7 — Fooled minus robust differences (LaTeX table), Qwen3-VL-8B.

Same three measurements as A6, plus late Q→A, which the paper discusses (corrective question evidence).
Source: free_generation_intervention/outputs/heterogeneity/fooled_minus_robust.csv (report Sec. 8).
"""
import pandas as pd
import paper_plot_style as S

ROWS = [('activation_restoration_L5_12', r'Early--middle restoration (L5--12)'),
        ('attention_TQ_L12_17', r'Middle T$\rightarrow$Q (L12--17)'),
        ('attention_TA_L30_35', r'Late T$\rightarrow$A (L30--35)'),
        ('attention_QA_L30_35', r'Late Q$\rightarrow$A (L30--35)')]
VARIANTS = ['misleading_groundable', 'misleading_ungroundable']

df = pd.read_csv(S.CAUSAL / 'free_generation_intervention/outputs/heterogeneity/fooled_minus_robust.csv')
df = df.set_index(['variant', 'metric'])
n = {v: (int(df.loc[(v, ROWS[0][0]), 'n_fooled']), int(df.loc[(v, ROWS[0][0]), 'n_robust'])) for v in VARIANTS}
lines = [f'{label} & ' + ' & '.join(S.fmt_ci(*df.loc[(v, metric), ['difference', 'ci_low', 'ci_high']]) for v in VARIANTS)
         + r' \\' for metric, label in ROWS]

tex = r'''
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrr}
\toprule
 & Grounded & Ungrounded \\
Metric & fooled $-$ robust & fooled $-$ robust \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\caption{Differences between fooled and robust questions in Qwen3-VL-8B (nats/token). Fooled: correct on the clean image and answers the displayed misleading word with the overlay; robust: correct on both (grounded: %d fooled, %d robust; ungrounded: %d fooled, %d robust). Restoration is the text-minus-random activation-patching effect averaged over layers 5--12; T$\rightarrow$Q and T$\rightarrow$A are text-minus-random attention-blocking effects and Q$\rightarrow$A is the raw blocking effect, all overlay-specific (overlay minus paired no-text image) in the stated windows; layer windows were fixed before the comparison. Brackets: bootstrap 95\%% CI of the group difference.}
\label{tab:fooled-robust}
\end{table}
''' % (*n['misleading_groundable'], *n['misleading_ungroundable'])
S.save_table(tex, 'tab_fooled_robust')
