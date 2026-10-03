"""Fig. 1a — Controlled scene-text setup (report Sec. 2.1).

One question shown with its clean image and the four cleaned overlays. Cleaned overlays differ from
the clean image only inside the annotated text box. Change SAMPLE_INDEX to show another example.
"""
import matplotlib.pyplot as plt
from datasets import load_from_disk
import paper_plot_style as S

WIDTH, HEIGHT = S.DOUBLE, 1.45
SAMPLE_INDEX = 0    # dataset row; 0 = "What is on the bed?" (pillow / flowers / duvet / refute)
PANELS = [('notext', 'Clean', S.CONTROL), *[(c, S.COND[c]['label'], S.COND[c]['color']) for c in S.CONDITIONS]]

S.apply()
sample = load_from_disk(str(S.DATA['dataset']))[SAMPLE_INDEX]

fig, axes = plt.subplots(1, len(PANELS), figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace': 0.06})
for ax, (key, label, color) in zip(axes, PANELS):
    image = sample[key]['image'] if key == 'notext' else sample[key]['cleaned_image']
    ax.imshow(image.convert('RGB')); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(1.4)
    word = 'no text' if key == 'notext' else f'“{sample[key]["text"]}”'
    ax.set_title(f'{label}\n{word}', color=color if key != 'notext' else S.INK, fontsize=6.6, loc='center')
axes[2].text(0.5, -0.05, f'Q: {sample["question"]}    correct answer: {sample["correct_answer"]["text"]}', ha='center', va='top',
         fontsize=7, color=S.INK, transform=axes[2].transAxes)
S.panel_label(axes[0], '(a)', x=-0.04, y=1.02)
S.save(fig, 'appendix_fig1a_setup_conditions')
