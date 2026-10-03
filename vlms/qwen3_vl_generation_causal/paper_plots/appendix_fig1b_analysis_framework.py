"""Fig. 1b — Analysis framework: how the experiments trace scene text from input to answer.

Pipeline of stages with the intervention that tests each link and the figure that reports it.
Edit STAGES / METHODS to change wording; positions are in axes-fraction units.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 1.45
STAGES = ['Scene text\nin the image', 'Text-region\nvisual tokens', 'Decoder states\n(early–middle)',
          'Answer positions\n(late layers)', 'Generated\nanswer']
# (first stage, last stage, method, role · figure, color); spans are inclusive stage indices and must not overlap.
METHODS = [(1, 2, 'Activation patching', 'formation · Fig. 3a', S.ORANGE),
           (3, 3, 'Attention blocking\n(T→Q, T→A)', 'readout · Fig. 3b–c', S.BLUE),
           (4, 4, 'T→A blocking\nduring generation', 'answers · Fig. 4', S.CHARCOAL)]
BOX_W, BOX_H, Y_BOX = 0.15, 0.3, 0.62

S.apply()
fig = plt.figure(figsize=(WIDTH, HEIGHT))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
xs = [0.1 + i * 0.2 for i in range(len(STAGES))]

for x, text in zip(xs, STAGES):
    ax.add_patch(FancyBboxPatch((x - BOX_W / 2, Y_BOX - BOX_H / 2), BOX_W, BOX_H, boxstyle='round,pad=0.005,rounding_size=0.02',
                                fc='white', ec=S.AXIS, lw=0.7))
    ax.text(x, Y_BOX, text, ha='center', va='center', fontsize=6.5)
for x0, x1 in zip(xs, xs[1:]):
    ax.add_patch(FancyArrowPatch((x0 + BOX_W / 2 + 0.005, Y_BOX), (x1 - BOX_W / 2 - 0.005, Y_BOX),
                                 arrowstyle='-|>', mutation_scale=7, lw=0.8, color=S.AXIS))

# Behavior (Fig. 2) spans the whole pipeline, drawn above.
top = Y_BOX + BOX_H / 2 + 0.07
ax.plot([xs[0], xs[0], xs[-1], xs[-1]], [top - 0.03, top, top, top - 0.03], color=S.CONTROL, lw=0.7)
ax.text(0.5, top + 0.02, 'Behavior across 7 VLMs: open-ended and MCQ  (Fig. 2)', ha='center', va='bottom',
        fontsize=6.3, color='#5A5A5A')

# Causal interventions (Qwen3-VL-8B), drawn below at staggered heights.
y = Y_BOX - BOX_H / 2 - 0.06
for a, b, method, role, color in METHODS:
    x0, x1 = xs[a] - BOX_W / 2, xs[b] + BOX_W / 2
    ax.plot([x0, x0, x1, x1], [y + 0.03, y, y, y + 0.03], color=color, lw=1.0)
    ax.text((x0 + x1) / 2, y - 0.03, method, ha='center', va='top', fontsize=6.3, color=color, linespacing=1.1)
    ax.text((x0 + x1) / 2, y - 0.2, role, ha='center', va='top', fontsize=5.9, color='#6E6E6E')
ax.text(xs[0], y - 0.03, 'Causal tests\n(Qwen3-VL-8B)', ha='center', va='top', fontsize=6, color='#6E6E6E')
ax.text(0.0, 0.98, '(b)', ha='left', va='top', fontsize=8, fontweight='bold')
S.save(fig, 'appendix_fig1b_analysis_framework')
