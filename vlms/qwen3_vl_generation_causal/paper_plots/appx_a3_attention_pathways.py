"""Appendix A3 — Full attention-pathway map, Qwen3-VL-8B (305 questions).

Each cell: effect of blocking attention from a source token group to a destination group in one six-layer
window, on the correct-vs-displayed complete-answer margin (nats/token); overlay-specific (overlay minus paired
no-text image), oriented so positive = blocking removes the overlay's influence (sign flipped for Correct).
Groups: T text region, R count-matched random region (mean of three), C correct object, G grounded object,
V all non-text visual tokens (large, unmatched set), Q question tokens, A answer positions.
Dot: Holm-adjusted p < .05 within overlay condition. Gray: structurally unavailable (no clean object region).
Color scale is symmetric-log (linear within ±0.1) so small but reliable routes remain visible.
Source: attention_intervention/outputs/analysis/path_summary.csv. Joint (T+C, T+G) blocks are omitted.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 3.15
PATHS = [('T_to_Q', 'T→Q'), ('T_to_A', 'T→A'), ('R_to_Q', 'R→Q'), ('R_to_A', 'R→A'),
         ('C_to_Q', 'C→Q'), ('C_to_A', 'C→A'), ('G_to_Q', 'G→Q'), ('G_to_A', 'G→A'),
         ('Q_to_A', 'Q→A'), ('V_to_Q', 'V→Q'), ('V_to_A', 'V→A'),
         ('T_to_C', 'T→C'), ('C_to_T', 'C→T'), ('T_to_G', 'T→G'), ('G_to_T', 'G→T')]
GROUP_BREAKS = [2, 4, 8, 9, 11]            # thin separators after these rows: text | random | objects | Q→A | V | text↔object
LIMIT, LINTHRESH = 2.4, 0.1
CMAP = LinearSegmentedColormap.from_list('blue_cream_terracotta', ['#4E6E92', '#9FB2C6', '#F6F1E7', '#D9A48C', '#A4553C'])
ALPHA = 0.05

S.apply()
df = pd.read_csv(S.DATA['attention_paths']).set_index(['variant', 'window', 'path'])
norm = SymLogNorm(linthresh=LINTHRESH, linscale=0.6, vmin=-LIMIT, vmax=LIMIT, base=10)

fig, axes = plt.subplots(1, 4, figsize=(WIDTH, HEIGHT), sharey=True, gridspec_kw={'wspace': 0.05})
for ax, cond in zip(axes, S.CONDITIONS):
    est = np.full((len(PATHS), len(S.WINDOWS)), np.nan); sig = np.zeros_like(est, bool)
    for i, (path, _) in enumerate(PATHS):
        for j, w in enumerate(S.WINDOWS):
            if (cond, w, path) in df.index:
                r = df.loc[(cond, w, path)]
                est[i, j], sig[i, j] = r.mean_oriented_effect, r.p_holm_within_condition < ALPHA
    ax.set_facecolor('#E4E2DD')                                   # unavailable cells
    im = ax.imshow(np.ma.masked_invalid(est), cmap=CMAP, norm=norm, aspect='auto', interpolation='nearest')
    ii, jj = np.where(sig)
    ax.scatter(jj, ii, s=2.2, color=S.INK, lw=0, zorder=3)
    for b in GROUP_BREAKS:
        ax.axhline(b - 0.5, color='white', lw=1.6)
    ax.set_xticks(range(len(S.WINDOWS)), [S.WINDOW_LABEL[w] for w in S.WINDOWS], rotation=45, ha='right',
                  rotation_mode='anchor', fontsize=6.0)
    ax.set_yticks(range(len(PATHS)), [label for _, label in PATHS])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(S.COND[cond]['label'], color=S.COND[cond]['color'], loc='center', fontweight='bold')
fig.supxlabel('Decoder layers (blocked window)', fontsize=S.FONT_AXIS_LABEL, y=0.0)
cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.012, ticks=[-2, -1, -0.3, -0.1, 0, 0.1, 0.3, 1, 2])
cbar.ax.set_yticklabels(['−2', '−1', '−0.3', '−0.1', '0', '0.1', '0.3', '1', '2'])
cbar.outline.set_visible(False); cbar.ax.tick_params(length=0, labelsize=6.0)
cbar.set_label('Blocking effect (nats/token)', fontsize=6.5)
S.save(fig, 'appx_a3_attention_pathways', S.APPENDIX_FIG_DIR)
