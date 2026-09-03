"""Shared paper style and small I/O helpers for activation-patching plots."""

import csv
from pathlib import Path

import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nunito", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

ACL_TWO_COL = 7.0
LINE_WIDTH = 1.4
CI_ALPHA = 0.18

VARIANTS = {
    "misleading_groundable": "Grounded misleading",
    "misleading_ungroundable": "Ungrounded misleading",
    "irrelevant_word": "Irrelevant option text",
}
CONDITION_COLORS = {
    "misleading_groundable": "#D55E00",
    "misleading_ungroundable": "#0072B2",
    "irrelevant_word": "#009E73",
}
REGION_LABELS = {
    "text_region": "Text region",
    "matched_random_mean": "Matched random",
    "all_image_tokens": "All image tokens",
    "correct_object_region": "Correct object",
    "grounded_object_region": "Grounded object",
}
REGION_COLORS = {
    "text_region": "#0072B2",
    "matched_random_mean": "#7F7F7F",
    "all_image_tokens": "#CC79A7",
    "correct_object_region": "#009E73",
    "grounded_object_region": "#D55E00",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finish_axis(axis, show_xlabel=True):
    axis.axhline(0, color="#333333", linewidth=0.7, zorder=0)
    axis.set_xticks(range(0, 32, 5))
    axis.tick_params(direction="out")
    if show_xlabel:
        axis.set_xlabel("Decoder layer (resid_pre)")


def save_figure(figure, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(output.with_suffix(suffix), bbox_inches="tight", facecolor="white")


def write_interpretation(output, title, lines):
    path = Path(output).with_name(Path(output).stem + "_interpretation.md")
    path.write_text("# {}\n\n{}\n".format(title, "\n\n".join(lines)), encoding="utf-8")
    return path
