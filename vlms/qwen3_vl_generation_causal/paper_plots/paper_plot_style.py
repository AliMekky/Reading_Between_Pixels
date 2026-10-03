"""Shared style for all ACL paper figures (spec: ../../../main_paper_figures_redesign_instructions.md).

Palette, condition order, figure sizes, fonts, line/marker sizes, band alpha, Matplotlib settings,
data paths, and the export helper. Every figure script imports this module; no styling constants
are defined anywhere else.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parent                                   # qwen3_vl_generation_causal/
VLMS = CAUSAL.parent                                   # vlms/
FIG_DIR = HERE / 'figures'
FONT_DIR = HERE / 'fonts'                              # Liberation Sans: metric-compatible with Arial
DATA = {
    'format_metrics': VLMS / 'format_replication_20260921/comparison/metrics.csv',
    'patching_layers': CAUSAL / 'outputs/step6_8b_analysis/layer_summary.csv',
    'attention_primary': CAUSAL / 'attention_intervention/outputs/analysis/primary_text_path_summary.csv',
    'attention_paths': CAUSAL / 'attention_intervention/outputs/analysis/path_summary.csv',
    'groundedness': CAUSAL / 'attention_intervention/outputs/analysis/groundedness_contrast.csv',
    'free_generation': CAUSAL / 'free_generation_intervention/outputs/evaluation/statistics/contrasts.csv',
    'joined_samples': CAUSAL / 'free_generation_intervention/outputs/heterogeneity/joined_sample_metrics.csv',
    'correlations': CAUSAL / 'free_generation_intervention/outputs/heterogeneity/correlations.csv',
    'dataset': VLMS / 'activation_patching/hf_dataset_GUIC_cleaned/AHAAM__GUIC',
}

# ----------------------------------------------------------------------------- sizes (inches)
SINGLE = 3.3            # ACL single column (3.25–3.35)
DOUBLE = 6.9            # ACL double column (6.8–7.0)

# ----------------------------------------------------------------------------- palette
CORRECT, GROUNDED, UNGROUNDED, IRRELEVANT = '#4F7D67', '#C8874A', '#B96A72', '#5F7394'
CONTROL = '#A9A6A0'     # random / control
CLEAN = '#2F2F2C'       # clean / no-text
INK = '#2F2F2C'         # text
AXIS = '#7A7873'        # spines, ticks, zero lines
GRID = '#F1F0EE'        # very light grid
BAND = '#F2F1EE'        # light highlight band

CONDITIONS = ['correct_answer', 'misleading_groundable', 'misleading_ungroundable', 'irrelevant_word']   # fixed order
COND = {
    'correct_answer':          dict(label='Correct',    short='Corr.',  color=CORRECT,    marker='o', ls='-'),
    'misleading_groundable':   dict(label='Grounded',   short='Grnd.',  color=GROUNDED,   marker='s', ls='-'),
    'misleading_ungroundable': dict(label='Ungrounded', short='Ungr.',  color=UNGROUNDED, marker='^', ls='-'),
    'irrelevant_word':         dict(label='Irrelevant', short='Irrel.', color=IRRELEVANT, marker='D', ls='-'),
}

# ----------------------------------------------------------------------------- typography (pt)
FONT_AXIS_LABEL, FONT_TICK, FONT_LEGEND, FONT_PANEL, FONT_ANNOT = 7.5, 7.0, 7.0, 8.5, 7.0

# ----------------------------------------------------------------------------- lines and markers
LW_MAIN = 0.9           # main lines (thin, editorial style)
LW_CONTROL = 0.75       # controls
LW_AXIS = 0.6
LW_ZERO = 0.8
LW_CI = 0.6             # interval whiskers
MARKER = 3.0            # marker size for dot/interval plots
MARKER_LINE = 1.8       # markers on lines
SCATTER = 2.4           # scatter points
SCATTER_ALPHA = 0.35
BAND_ALPHA = 0.11       # confidence bands
BAR_FILL = 0.62         # bar thickness as a fraction of its slot (thinner bars, visible gaps)
LEGEND_EDGE = '#CFCDC8' # light frame around legends

# ----------------------------------------------------------------------------- models and layers
MODELS = [  # (key in metrics.csv, label for multi-line tick)
    ('llava-hf__llava-1.5-7b-hf', 'LLaVA\n1.5-7B'),
    ('llava-hf__llava-v1.6-mistral-7b-hf', 'LLaVA\nNeXT-7B'),
    ('Qwen__Qwen2.5-VL-7B-Instruct', 'Qwen2.5\nVL-7B'),
    ('OpenGVLab__InternVL3_5-8B', 'InternVL\n3.5-8B'),
    ('Qwen__Qwen3-VL-2B-Instruct', 'Qwen3-VL\n2B'),
    ('Qwen__Qwen3-VL-8B-Instruct', 'Qwen3-VL\n8B'),
    ('qwen3-vl-32b-instruct', 'Qwen3-VL\n32B'),
]
MODEL_NAMES = {'llava-hf__llava-1.5-7b-hf': 'LLaVA-1.5-7B', 'llava-hf__llava-v1.6-mistral-7b-hf': 'LLaVA-NeXT-7B',
               'Qwen__Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL-7B', 'OpenGVLab__InternVL3_5-8B': 'InternVL3.5-8B',
               'Qwen__Qwen3-VL-2B-Instruct': 'Qwen3-VL-2B', 'Qwen__Qwen3-VL-8B-Instruct': 'Qwen3-VL-8B',
               'qwen3-vl-32b-instruct': 'Qwen3-VL-32B'}
WINDOWS = ['layers_00_05', 'layers_06_11', 'layers_12_17', 'layers_18_23', 'layers_24_29', 'layers_30_35']
WINDOW_MID = {w: (int(w[7:9]) + int(w[10:12])) / 2 for w in WINDOWS}
WINDOW_LABEL = {w: f'{int(w[7:9])}–{int(w[10:12])}' for w in WINDOWS}
EARLY_MIDDLE = (5, 12)  # frozen early–middle window used by the formation metric (report Sec. 8)
LATE = (30, 35)         # late readout window

# Aliases kept for the earlier appendix drafts (appendix_*.py); they map onto the palette above.
BLUE, ORANGE, BURNT, CHARCOAL = CORRECT, GROUNDED, UNGROUNDED, IRRELEVANT


def apply():
    for path in sorted(FONT_DIR.glob('*.ttf')):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': FONT_TICK, 'axes.labelsize': FONT_AXIS_LABEL, 'axes.titlesize': FONT_AXIS_LABEL,
        'xtick.labelsize': FONT_TICK, 'ytick.labelsize': FONT_TICK, 'legend.fontsize': FONT_LEGEND,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
        'axes.edgecolor': AXIS, 'axes.linewidth': LW_AXIS,
        'xtick.major.width': LW_AXIS, 'ytick.major.width': LW_AXIS, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
        'axes.titlelocation': 'left', 'axes.titlepad': 3,
        'lines.linewidth': LW_MAIN, 'lines.markersize': MARKER_LINE, 'lines.markeredgewidth': 0,
        'legend.frameon': True, 'legend.edgecolor': LEGEND_EDGE, 'legend.facecolor': 'white', 'legend.framealpha': 0.95,
        'legend.fancybox': True, 'legend.handlelength': 1.4, 'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.9, 'legend.borderaxespad': 0.4, 'legend.borderpad': 0.4, 'legend.labelspacing': 0.3,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def zero_line(ax, vertical=False, **kw):
    kw = dict(color=AXIS, lw=LW_ZERO, zorder=1) | kw
    (ax.axvline if vertical else ax.axhline)(0, **kw)


def ygrid(ax):
    ax.grid(axis='y', color=GRID, lw=0.5)


def panel_label(ax, text, x=-0.02, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=FONT_PANEL, fontweight='bold', color=INK)


def condition_legend(fig_or_ax, conditions=CONDITIONS, marker=False, patch=False, extra=(), **kw):
    """One shared legend in the fixed condition order: lines (default), markers, or filled patches.
    extra: additional (handle, label) pairs appended at the end (e.g. a random-region control)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if patch:
        handles = [Patch(color=COND[c]['color'], lw=0) for c in conditions]
    else:
        handles = [Line2D([], [], color=COND[c]['color'], lw=0 if marker else LW_MAIN,
                          marker=COND[c]['marker'] if marker else None, ms=MARKER) for c in conditions]
    labels = [COND[c]['label'] for c in conditions]
    for handle, label in extra:
        handles.append(handle); labels.append(label)
    return fig_or_ax.legend(handles, labels, **kw)


def compact_legend_above(ax, **kw):
    """Single-row legend centered just above an axes."""
    kw = dict(ncol=len(CONDITIONS), loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              columnspacing=0.8, fontsize=FONT_LEGEND) | kw
    return condition_legend(ax, **kw)


def band(ax, x0, x1, label=None, y=0.985):
    ax.axvspan(x0, x1, color=BAND, lw=0, zorder=0)
    if label:
        ax.text((x0 + x1) / 2, y, label, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=FONT_ANNOT - 0.5, color=AXIS)


def end_labels(ax, items, x, min_gap, fontsize=FONT_ANNOT):
    """Direct end-of-line labels; items = [(y, text, color)], nudged apart by at least min_gap (data units)."""
    items = sorted(items); ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (y0, text, color), y in zip(items, ys):
        ax.annotate(text, (x, y0), xytext=(x, y), textcoords='data', ha='left', va='center',
                    fontsize=fontsize, color=color, annotation_clip=False)


APPENDIX_FIG_DIR = HERE / 'appendix_figures'
APPENDIX_TAB_DIR = HERE / 'appendix_tables'


def save(fig, name, out_dir=None):
    """Export vector PDF, SVG, and a PNG preview to figures/ (or out_dir, e.g. APPENDIX_FIG_DIR)."""
    out_dir = out_dir or FIG_DIR
    out_dir.mkdir(exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out_dir / f'{name}.{ext}')
    plt.close(fig)
    print(f'saved {out_dir.name}/{name}.pdf/.svg/.png')


def save_table(tex, name):
    """Write a LaTeX table snippet to appendix_tables/<name>.tex."""
    APPENDIX_TAB_DIR.mkdir(exist_ok=True)
    (APPENDIX_TAB_DIR / f'{name}.tex').write_text(tex.strip() + '\n')
    print(f'saved appendix_tables/{name}.tex')


def fmt_ci(est, lo, hi, d=2, scale=1.0):
    """'estimate [low, high]' with LaTeX minus signs and consistent precision."""
    f = lambda v: f'{round(scale * v, d) + 0.0:.{d}f}'.replace('-', '$-$')   # +0.0 drops negative zero
    return f'{f(est)} [{f(lo)}, {f(hi)}]'

