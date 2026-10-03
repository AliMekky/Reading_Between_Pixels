#!/usr/bin/env python3
"""Integrate finalized seven-model behavioral findings into the main report."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;P=H.parent/'qwen3_vl_generation_causal/qwen3_vl_results_report.md'
MODELS=[('llava-hf__llava-1.5-7b-hf','LLaVA-1.5-7B'),('llava-hf__llava-v1.6-mistral-7b-hf','LLaVA-NeXT-7B'),('Qwen__Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-7B'),('OpenGVLab__InternVL3_5-8B','InternVL3.5-8B'),('Qwen__Qwen3-VL-2B-Instruct','Qwen3-VL-2B'),('Qwen__Qwen3-VL-8B-Instruct','Qwen3-VL-8B'),('qwen3-vl-32b-instruct','Qwen3-VL-32B API')]
K=['notext','correct_answer','misleading_groundable','misleading_ungroundable','irrelevant_word'];L=['Clean','Correct','Grounded misleading','Ungrounded misleading','Irrelevant']
rows=list(csv.DictReader((H/'comparison/metrics.csv').open()));m={(r['model'],r['condition'],r['metric']):r for r in rows}
def value(s,k,metric,ci=False):
 r=m[s,k,metric];a=f"{100*float(r['estimate']):.1f}"
 return a+(f" [{100*float(r['ci_low']):.1f}, {100*float(r['ci_high']):.1f}]" if ci else '')
def table(head,rr):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']+['---:']*(len(head)-1))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for ax,fmt,label in zip(axs,['open_ended','mcq'],['Open-ended generation','MCQ']):
 data=np.array([[100*float(m[s,k,fmt+'_accuracy']['estimate']) for k in K] for s,n in MODELS])
 im=ax.imshow(data,vmin=30,vmax=100,cmap='YlGnBu',aspect='auto')
 ax.set_xticks(range(5),['Clean','Correct','Grounded','Ungrounded','Irrelevant'],rotation=20,ha='right');ax.set_yticks(range(7),[n for s,n in MODELS]);ax.set_title(label)
 for i in range(7):
  for j in range(5):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',color='white' if data[i,j]>70 else 'black',fontsize=9)
fig.colorbar(im,ax=axs,label='Accuracy (%)',shrink=.8)
for ext in ['png','pdf']:fig.savefig(H/'comparison'/f'seven_model_accuracy.{ext}',dpi=180)
plt.close(fig)
b='''## 3. Experiment 1: seven-model behavior in open-ended and MCQ formats

### 3.1 Design and how to read the results

The behavioral comparison now includes **seven models**, including dense **Qwen3-VL-32B-Instruct** alongside Qwen3-VL-2B and 8B. Every model is evaluated on **474 questions × five image conditions** in each format: 2,370 responses per model per format, **16,590 responses per format**, and **33,180 paired-format responses overall**. The older six-model open-ended outputs are reused; MCQ was rerun for all six, and 32B was run in both formats.

Within each model, MCQ and open-ended runs have exactly matched question IDs, image hashes, and reference answers. Each request includes the actual clean or overlay image. MCQ lists the four candidate words in stable seed-42 order and asks for a letter. Open-ended generation exposes no answer options and asks for a short answer. The output caps are the existing protocol's 50 tokens for MCQ and 32 for open-ended generation.

Three quantities answer different questions:

- **Accuracy:** how often the response matches the correct answer.
- **Adjusted following:** adoption of the displayed candidate with its overlay minus adoption of that same candidate on the clean image. It measures additional following beyond the model's baseline answer tendency.
- **MCQ amplification:** adjusted following in MCQ minus adjusted following open-ended, paired within question. Positive means the MCQ format increases the overlay's effect.

Accuracy is shown in **percent**. Changes and adjusted following are shown in **percentage points (pp)**. For correct overlays, following is correctness, so adjusted following equals the accuracy gain. For misleading overlays, additional following can replace either a correct answer or another wrong answer; its magnitude need not equal the accuracy loss.

Open-ended results use normalized exact matching, the fixed 0.90 edit-similarity stage, and two counterbalanced semantic judges for unresolved answers. MCQ uses the existing letter extractor. All 164 non-strict MCQ responses were audited and have clear leading option letters matching the saved extraction; there are zero UNKNOWN parses. Judge disagreements remain ambiguous, rather than being forced into a reference category.

The 32B model is served through QwenCloud; the other six models are local. Prompts, images, and references match, but provider preprocessing and serving can differ. These are therefore model comparisons, not a controlled model-size scaling experiment. All mechanistic interventions later in the report remain specific to 8B unless explicitly stated otherwise.

### 3.2 Open-ended generation: all seven models and all overlays

**Absolute semantic accuracy (%)**

'''
b+=table(['Model']+L,[[n]+[value(s,k,'open_ended_accuracy') for k in K] for s,n in MODELS])
b+='\n\n**Additional adoption caused by the overlay, relative to the clean-image baseline (pp)**\n\n'
b+=table(['Model']+L[1:],[[n]+[value(s,k,'open_ended_adjusted_following') for k in K[1:]] for s,n in MODELS])
b+='''

**Interpretation:** correct overlays increase accuracy for all seven models. Relevant misleading overlays increase production of their displayed answers even without options. Irrelevant following remains much lower, so the scene-text phenomenon survives open-ended generation but depends strongly on overlay type.

For the new 32B model, clean-image accuracy is **59.3%**. Correct overlays raise it to **73.8%**; grounded and ungrounded misleading overlays reduce it to **56.8%** and **53.6%**; irrelevant overlays leave it close to baseline at **59.1%**. Its adjusted following is **+6.5 pp grounded**, **+13.5 pp ungrounded**, and **+1.1 pp irrelevant**. This extends the open-ended taxonomy pattern to the larger dense model. It does not establish that larger models are always more accurate or more robust: 32B's clean accuracy is lower than 8B's, and the deployment differs.

For 32B, paired McNemar tests with Holm correction support the correct-overlay accuracy gain and ungrounded accuracy loss; grounded and irrelevant accuracy changes do not pass that correction. Full uncertainty intervals and stage counts are in Section 17. Correct overlays reveal the answer directly, so their benefit should not be described as improved visual reasoning.

### 3.3 MCQ generation: all seven models and all overlays

**Absolute MCQ accuracy (%)**

'''
b+=table(['Model']+L,[[n]+[value(s,k,'mcq_accuracy') for k in K] for s,n in MODELS])
b+='\n\n**Additional adoption caused by the overlay, relative to the MCQ clean-image baseline (pp)**\n\n'
b+=table(['Model']+L[1:],[[n]+[value(s,k,'mcq_adjusted_following') for k in K[1:]] for s,n in MODELS])
b+='''

**Interpretation:** options change the answer setting. The displayed word is not only visible in the image; it is also explicitly available as a candidate in the prompt. This can encourage an image-text-to-option match. All four overlay types are included, so we retain both the useful effect of correct text and the harmful or distracting effects of the other types.

Raw MCQ and open-ended accuracy cannot isolate overlay amplification because their no-text baselines differ. The following comparison therefore subtracts each format's own baseline before comparing formats.

![Seven-model accuracy in both formats](../format_replication_20260921/comparison/seven_model_accuracy.png)

### 3.4 Paired MCQ versus open-ended comparison

**Adjusted following: MCQ / open-ended (pp)**

'''
b+=table(['Model']+L[1:],[[n]+[value(s,k,'mcq_adjusted_following')+' / '+value(s,k,'open_ended_adjusted_following') for k in K[1:]] for s,n in MODELS+[('unweighted_seven_model_mean','Seven-model mean')]])
b+='\n\n**MCQ amplification: MCQ minus open-ended adjusted following (pp)**\n\n'
b+=table(['Model','Correct','Grounded','Ungrounded','Irrelevant [95% CI]'],[[n]+[value(s,k,'mcq_minus_open_adjusted_following',k=='irrelevant_word') for k in K[1:]] for s,n in MODELS+[('unweighted_seven_model_mean','Seven-model mean')]])
b+='''

**Main finding:** irrelevant-text adoption is amplified by MCQ in **all seven models**. The seven-model mean adjusted irrelevant following is **11.4 pp in MCQ versus 1.8 pp open-ended**, a difference of approximately **9.5 pp** calculated before rounding. For 32B specifically, it is **8.0 versus 1.1 pp**, with amplification **+7.0 pp [4.4, 9.7]**.

The relevant misleading conditions are more model-dependent; MCQ does not uniformly amplify every type of scene-text following. Correct overlays also tend to produce smaller gains in MCQ than open-ended, where there is often more room to improve. These differences are why the full taxonomy belongs in the comparison rather than only the irrelevant row.

All seven irrelevant amplification intervals are above zero. These are descriptive paired bootstrap intervals, not multiplicity-adjusted hypothesis tests. Intervals use 10,000 whole-question resamples with seed 42. The seven-model mean weights the tested models equally; its uncertainty reflects question sampling for this fixed set, not sampling over all possible models.

![Irrelevant following by answer format](../format_replication_20260921/comparison/irrelevant_following.png)

**Accuracy changes: MCQ / open-ended (pp)**

'''
b+=table(['Model']+L[1:],[[n]+[value(s,k,'mcq_accuracy_change')+' / '+value(s,k,'open_ended_accuracy_change') for k in K[1:]] for s,n in MODELS+[('unweighted_seven_model_mean','Seven-model mean')]])
b+='''

### 3.5 Two factors affecting adoption: prompt matching and starting plausibility

The results support a connected explanation with **two interacting factors**, measured by different experiments:

| Factor | What we measured | Simple interpretation | Scope of the evidence |
|---|---|---|---|
| **Matching displayed text to prompt options** | Irrelevant overlay-associated adoption increases in MCQ for all seven models after subtracting each format's clean-image baseline. | An otherwise unlikely answer is explicitly offered, and the word in the image matches that offered candidate. This supports the proposed parrot-bias account. | Format amplification is measured. Literal matching is an interpretation; options also restrict the answer space and the instruction asks for a letter. |
| **Initial question-conditioned language plausibility** | On 305 Qwen3-VL-8B questions, irrelevant candidates already trail the relevant misleading candidates by **10.05 nats/token [9.12, 10.99]** with only the question. | Before seeing an image, the model considers irrelevant words unlikely answers. An overlay can boost them without making them competitive. | The starting gap and subsequent score changes are measured on 8B. This is not a seven-model probability decomposition or a pure word-frequency measurement. |

The 8B score comparison shows why “the text had a strong effect” does not imply “the model followed the text.” Grounded misleading and irrelevant overlays increase their own candidate scores by nearly the same average amount (**+5.61 and +5.59 nats/token**), but irrelevant candidates start much further behind. After adding the overlay, the median irrelevant-minus-correct score margin is still **−13.73**, versus **−2.39** grounded and **−1.60** ungrounded. Section 16 contains all four candidate types, both no-scene controls, and the full distributions.

MCQ adds a second influence: it places those candidate words directly in the prompt. The observed amplification is consistent with the model selecting an explicitly offered option that matches visible text even when it would rarely generate that word unaided. This connects the **parrot-bias / prompt-matching account** to the **starting-plausibility account** rather than treating them as competing explanations.

An intuitive illustration, not an additional measured example: if “memory” is unlikely as an answer to “What is the person standing on?”, seeing “memory” in the image can raise its support while leaving it unlikely in free generation. Listing “memory” among the answer options additionally makes it available for selection and creates a direct match to the displayed text. Our experiments measure the general starting-gap and format-amplification patterns; they do not yet show how much the MCQ format changes that specific word's prior score.

**What we can say:** adoption depends on the text-derived boost, the candidate's initial question-conditioned plausibility, and the answer format. The image itself is also a source of evidence, particularly for the correct answer; these two factors are not an exhaustive model of behavior.

**What we cannot yet say:** that prompt matching and language priors are two independent neural mechanisms, that MCQ completely removes the plausibility gap, or that we have assigned a causal percentage of adoption to each. The MCQ experiment changes both option availability and response instructions. A pure lexical-matching mechanism would require a control that separates wording overlap from merely offering candidate answers. No such new experiment is claimed here.

### 3.6 How this fits the mechanism story and the remaining scope

The behavioral and mechanistic evidence now answer complementary questions:

1. **Does scene text affect actual answers?** Yes, including open-ended answers and the new dense 32B checkpoint.
2. **Why is irrelevant adoption much weaker open-ended?** Its initial question-conditioned support is far lower; a substantial overlay boost usually leaves it behind.
3. **Why does MCQ increase irrelevant adoption?** Explicit options change the competition and provide a match to displayed text, consistent with parrot bias; the measured amplification occurs across all seven tested models.
4. **Where does text exert internal influence?** The later 8B intervention sections identify formation and readout routes. The new behavioral findings do not by themselves identify the pathway that implements either starting plausibility or prompt matching.

The earlier six-model figures and 2B/8B transition analyses remain historical supporting analyses in the [original open-ended report](../open_ended_evaluation/open_ended_evaluation_results_six_models.md). They should not be presented as seven-model figures. The integrated tables and figure above include all seven; the separate 32B semantic audit is in Section 17 and the complete paired-format audit in Section 18.

Artifacts: [all format metrics and intervals](../format_replication_20260921/comparison/metrics.csv), [per-question paired contrasts](../format_replication_20260921/comparison/paired_format_effects.csv), [non-strict MCQ parsing audit](../format_replication_20260921/comparison/non_strict_mcq_audit.json), and [seven-model figure PDF](../format_replication_20260921/comparison/seven_model_accuracy.pdf).

'''
s=P.read_text();start=s.index('## 3. Experiment 1:');end=s.index('## 4. Experiment 2:');s=s[:start]+b+s[end:]
s=s.replace('1. open-ended generation on Qwen3-VL-2B and Qwen3-VL-8B;','1. matched open-ended and MCQ generation across seven models, including Qwen3-VL-2B, 8B, and dense 32B;')
s=s.replace('Together, these experiments ask three progressively more mechanistic questions:','Together, these experiments connect behavior, answer format, starting plausibility, and causal information flow:')
s=s.replace('- **Relevance:** Why can irrelevant text strongly increase internal answer support while rarely becoming the generated answer?','- **Relevance:** Why can irrelevant text strongly increase internal answer support while rarely becoming the generated answer?\n- **Answer format:** How much does explicitly offering the displayed word as an MCQ option amplify adoption?')
s=s.replace('Generation is deterministic, so\npaired changes are attributable to the image condition rather than decoding\nrandomness.','Local generation uses greedy decoding; the hosted 32B run requests temperature\nzero. Provider-side execution is not guaranteed bitwise deterministic. Matched\nimage contrasts reduce avoidable prompt and input differences.')
s=s.replace('No Qwen3 response was resolved by edit similarity.','Qwen3-VL-32B resolved 1,397 by exact match and sent 973 to the same two judges;\n894 agreed and 79 remained ambiguous. All 973 decisions from each provider\nvalidated after targeted retries. No Qwen3 response was resolved by edit similarity.')
needle='## 3. Experiment 1: seven-model behavior'
s=s.replace(needle,'### 2.4 MCQ format and matched comparison\n\nMCQ lists the same four reference candidates with a fixed per-question shuffled\norder across all image conditions, then requests an answer letter. Both formats\nuse the same image and question. We compare overlay effects relative to each\nformat’s own clean-image baseline, rather than interpreting raw accuracy or\nfollowing differences as overlay amplification. Section 3 includes all seven\nmodels and all four overlays in both formats.\n\n'+needle)
s=s.replace('The five experiments now support one connected account:','The behavioral comparisons and causal experiments support one connected account:')
needle='Grounded and ungrounded text should not be described as using separate complete'
s=s.replace(needle,'The adoption account also has two measured components (Section 3.5): low\nquestion-conditioned starting support helps explain weak irrelevant following\nopen-ended, while MCQ reliably amplifies irrelevant following across seven\nmodels. Matching visible text to offered prompt options is a plausible parrot-bias\nexplanation of the format effect; it is not yet isolated from answer-space\nrestriction or localized to a particular attention route. The starting-score\nmeasurement is specific to 8B; the format comparison includes dense 32B.\n\n'+needle)
s=s.replace('The earlier format comparison still shows additional model-dependent MCQ amplification, especially for irrelevant option text.','The completed seven-model paired comparison shows positive irrelevant-adoption amplification in every model, including dense 32B; Section 3 reports all overlay types and distinguishes the measured effect from the prompt-matching interpretation.')
s=s.replace('Open-ended behavior covers six models including Qwen3-VL-2B/8B;','Open-ended and MCQ behavior cover seven models including Qwen3-VL-2B/8B/32B;')
s=s.replace('### Supported now\n','### Supported now\n\n- MCQ increases baseline-adjusted irrelevant-answer adoption in all seven tested\n  models; the effect is not uniform across other overlay types.\n- On 8B, irrelevant candidates already have much lower question-conditioned\n  support without an image, while their own-overlay score boost is similar in\n  average magnitude to grounded misleading text.\n')
s=s.replace('- that two checkpoint sizes establish a general scaling law;','- that the available size comparisons, mixing local and hosted serving, establish a general scaling law;\n- that MCQ amplification isolates literal prompt matching from answer-space restriction;\n- that starting plausibility and option matching are independently identified neural mechanisms;')
s=s.replace('the six-model open-ended behavioral figure, establishing that the phenomenon\n   survives removal of MCQ choices;','the seven-model open-ended/MCQ behavioral figure and paired format contrast,\n   establishing both survival without choices and amplification from the format;')
P.write_text(s)
print('Integrated seven-model behavior and two-factor discussion in Sections 1–3, 10–12, and 14.')
