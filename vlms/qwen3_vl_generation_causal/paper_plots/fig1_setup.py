"""Fig. 1 — (a) Controlled setup: one question with the clean image and four text overlays (overlays change only
the pixels inside the text box). (b) How the experiments trace the text from the image to the answer.
Change SAMPLE_INDEX to show another example; edit STAGES / METHODS to change the diagram wording.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from datasets import load_from_disk
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 3.1
SAMPLE_INDEX = 0    # "What is on the bed?" (pillow / flowers / duvet / refute)
PANELS = [('notext', 'No text', S.CONTROL), *[(c, S.COND[c]['label'], S.COND[c]['color']) for c in S.CONDITIONS]]
STAGES = ['Text in\nthe image', 'Text-region\nstates', 'Decoder\nprocessing', 'Answer reads\nthe text', 'Generated\nanswer']
# (first stage, last stage, method, what it shows, color); inclusive, non-overlapping stage spans.
METHODS = [(1, 2, 'Activation patching', 'where the text becomes\ninfluential (Fig. 3a)', S.ORANGE),
           (3, 3, 'Attention\nblocking', 'how it is read\n(Fig. 3b)', S.BLUE),
           (4, 4, 'Blocking during\ngeneration', 'effect on answers\n(Fig. 4)', S.CHARCOAL)]

S.apply()
sample = load_from_disk(str(S.DATA['dataset']))[SAMPLE_INDEX]
fig = plt.figure(figsize=(WIDTH, HEIGHT))
gs = fig.add_gridspec(2, len(PANELS), height_ratios=[1.2, 1.1], hspace=0.2, wspace=0.06)

# (a) images
for j, (key, label, color) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, j])
    image = sample[key]['image'] if key == 'notext' else sample[key]['cleaned_image']
    ax.imshow(image.convert('RGB')); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(1.4)
    word = '\n ' if key == 'notext' else f'\n“{sample[key]["text"]}”'
    ax.set_title(label + word, color=S.INK if key == 'notext' else color, fontsize=7.2, loc='center')
    if j == 0: S.panel_label(ax, '(a)', x=-0.05, y=1.02)
    if j == 2:
        ax.text(0.5, -0.06, f'Q: {sample["question"]}    Correct answer: {sample["correct_answer"]["text"]}',
                transform=ax.transAxes, ha='center', va='top', fontsize=7.5)

# (b) framework
ax = fig.add_subplot(gs[1, :]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
xs = [0.1 + i * 0.2 for i in range(len(STAGES))]; bw, bh, yb = 0.15, 0.36, 0.72
for x, text in zip(xs, STAGES):
    ax.add_patch(FancyBboxPatch((x - bw / 2, yb - bh / 2), bw, bh, boxstyle='round,pad=0.005,rounding_size=0.02',
                                fc='white', ec=S.AXIS, lw=0.7))
    ax.text(x, yb, text, ha='center', va='center', fontsize=7.2)
for x0, x1 in zip(xs, xs[1:]):
    ax.add_patch(FancyArrowPatch((x0 + bw / 2 + 0.005, yb), (x1 - bw / 2 - 0.005, yb), arrowstyle='-|>',
                                 mutation_scale=7, lw=0.8, color=S.AXIS))
y = yb - bh / 2 - 0.08
for a, b, method, what, color in METHODS:
    x0, x1 = xs[a] - bw / 2, xs[b] + bw / 2
    ax.plot([x0, x0, x1, x1], [y + 0.04, y, y, y + 0.04], color=color, lw=1.1)
    ax.text((x0 + x1) / 2, y - 0.05, method, ha='center', va='top', fontsize=7.2, color=color, fontweight='bold')
    ax.text((x0 + x1) / 2, y - 0.08 - 0.14 * (method.count('\n') + 1), what, ha='center', va='top', fontsize=6.6, color='#5A5A5A')
ax.text(xs[0], y - 0.05, 'Behavior, 7 VLMs\n(Fig. 2)', ha='center', va='top', fontsize=6.6, color='#5A5A5A')
S.panel_label(ax, '(b)', x=0.0, y=0.98)
S.save(fig, 'fig1_setup')
