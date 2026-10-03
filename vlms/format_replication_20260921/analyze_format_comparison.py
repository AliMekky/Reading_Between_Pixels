#!/usr/bin/env python3
"""Matched seven-model MCQ/open-ended comparison; requires final semantic labels."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;V=H.parent;OUT=H/'comparison';OUT.mkdir(exist_ok=True)
MODELS=[('llava-hf__llava-1.5-7b-hf','LLaVA-1.5-7B'),('llava-hf__llava-v1.6-mistral-7b-hf','LLaVA-NeXT-7B'),('Qwen__Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-7B'),('OpenGVLab__InternVL3_5-8B','InternVL3.5-8B'),('Qwen__Qwen3-VL-2B-Instruct','Qwen3-VL-2B'),('Qwen__Qwen3-VL-8B-Instruct','Qwen3-VL-8B'),('qwen3-vl-32b-instruct','Qwen3-VL-32B API')]
K=['notext','correct_answer','misleading_groundable','misleading_ungroundable','irrelevant_word'];L=['Clean','Correct','Grounded','Ungrounded','Irrelevant']
def read(p):
 r=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];assert len(r)==474
 d={str(x['question_id']):x for x in r};assert len(d)==474;return d
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']+['---:']*(len(h)-1))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 metrics=[];audit=[];paired=[];arrays={};rng=np.random.default_rng(42);indices=rng.integers(0,474,size=(10000,474))
 def add(model,condition,name,a):
  boot=a[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);metrics.append(dict(model=model,condition=condition,metric=name,n=474,estimate=float(a.mean()),ci_low=float(lo),ci_high=float(hi)))
 for slug,name in MODELS:
  m={k:read(H/'outputs/full'/slug/f'mcq_{k}.jsonl') for k in K}
  oe=H/'evaluation_32b/final'/slug if slug=='qwen3-vl-32b-instruct' else V/'open_ended_evaluation/outputs/final_classification_six_models'/slug
  o={k:read(oe/f'{k}.jsonl') for k in K};qids=sorted(m['notext']);cats={};amb={}
  for k in K:
   assert set(m[k])==set(o[k])==set(qids)
   mc=[];oc=[]
   for q in qids:
    r,t=m[k][q],o[k][q]
    assert r['reference_answers']==t['reference_answers'] and r['image_sha256']==t['image_sha256']
    assert r['options']==m['notext'][q]['options'] and r['option_meta']==m['notext'][q]['option_meta']
    mc.append(r['option_meta']['label_to_key'].get(r['predicted_answer'],'invalid'));oc.append(t['final_category'])
   cats['mcq',k]=np.array(mc);cats['open_ended',k]=np.array(oc)
   audit.append(dict(model=slug,condition=k,records=474,non_strict_mcq=sum(r.get('strict_letter') is None for r in m[k].values()),unknown_mcq=mc.count('invalid'),ambiguous_open=oc.count('ambiguous')))
  for fmt in ['mcq','open_ended']:
   for k in K:
    accuracy=(cats[fmt,k]=='correct_answer').astype(float);add(slug,k,fmt+'_accuracy',accuracy)
    if k=='notext':continue
    delta=accuracy-(cats[fmt,'notext']=='correct_answer')
    following=(cats[fmt,k]==k).astype(float);adjusted=following-(cats[fmt,'notext']==k)
    for metric,a in [('accuracy_change',delta),('following',following),('adjusted_following',adjusted)]:
     add(slug,k,fmt+'_'+metric,a);arrays[slug,k,fmt+'_'+metric]=a
  for k in K[1:]:
   for metric in ['adjusted_following','accuracy_change']:
    a=arrays[slug,k,'mcq_'+metric]-arrays[slug,k,'open_ended_'+metric]
    add(slug,k,'mcq_minus_open_'+metric,a);arrays[slug,k,'mcq_minus_open_'+metric]=a
    paired.extend(dict(model=slug,question_id=q,condition=k,metric=metric,mcq_minus_open=float(x)) for q,x in zip(qids,a))
 for k in K[1:]:
  for metric in ['mcq_adjusted_following','open_ended_adjusted_following','mcq_minus_open_adjusted_following','mcq_accuracy_change','open_ended_accuracy_change','mcq_minus_open_accuracy_change']:
   a=np.mean([arrays[s,k,metric] for s,n in MODELS],axis=0);add('unweighted_seven_model_mean',k,metric,a)
 for filename,rows in [('metrics.csv',metrics),('generation_audit.csv',audit),('paired_format_effects.csv',paired)]:
  with (OUT/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lookup={(r['model'],r['condition'],r['metric']):r for r in metrics}
 def value(s,k,metric,ci=False):
  r=lookup[s,k,metric];x=f"{100*r['estimate']:.1f}"
  return x+(f" [{100*r['ci_low']:.1f}, {100*r['ci_high']:.1f}]" if ci else '')
 lines=['# Seven-model MCQ versus open-ended results','',
 'All seven MCQ runs contain 474 questions × five image conditions (16,590 responses). They are paired with 16,590 open-ended responses: the six existing locally generated, semantically evaluated models plus the new hosted dense Qwen3-VL-32B-Instruct. Each paired condition has identical question IDs, reference answers, and image hashes. MCQ options use stable seed-42 ordering across image conditions.','',
 '## Main comparison: does MCQ amplify following beyond the clean-image baseline?','',
 'Adjusted following = adoption under an overlay minus adoption of that same candidate on the clean image. Amplification = MCQ adjusted following minus open-ended adjusted following, calculated per question. Positive means options amplify overlay-associated adoption. These are percentage-point changes, not relative percentage increases.','',
 table(['Model','Grounded MCQ / open','Ungrounded MCQ / open','Irrelevant MCQ / open','Irrelevant amplification [95% CI]'],[[n]+[value(s,k,'mcq_adjusted_following')+' / '+value(s,k,'open_ended_adjusted_following') for k in K[2:]]+[value(s,'irrelevant_word','mcq_minus_open_adjusted_following',True)] for s,n in MODELS+[('unweighted_seven_model_mean','Seven-model mean')]]),'',
 '## Correct overlays and accuracy changes','',
 table(['Model','Correct MCQ / open','Grounded MCQ / open','Ungrounded MCQ / open','Irrelevant MCQ / open'],[[n]+[value(s,k,'mcq_accuracy_change')+' / '+value(s,k,'open_ended_accuracy_change') for k in K[1:]] for s,n in MODELS+[('unweighted_seven_model_mean','Seven-model mean')]]),'',
 '## Absolute accuracy by format','',
 table(['Model / format']+L,[[n+' / '+fmt]+[value(s,k,fmt+'_accuracy') for k in K] for s,n in MODELS for fmt in ['mcq','open_ended']]),'',
 '## Interpretation and connection to the paper','',
 'This comparison measures the effect of answer format on overlay-associated behavior. Its interpretation must follow the per-model amplification estimates rather than assuming every model responds identically. Explicit options can make a displayed word available as an answer, consistent with the proposed parrot-bias account; this format contrast does not isolate copying from all other changes caused by MCQ instructions and the letter-answer requirement.','',
 'All seven models have a positive irrelevant-adoption amplification estimate, with the descriptive paired 95% intervals above zero. The seven-model average adjusted irrelevant following is '+value('unweighted_seven_model_mean','irrelevant_word','mcq_adjusted_following')+' pp in MCQ versus '+value('unweighted_seven_model_mean','irrelevant_word','open_ended_adjusted_following')+' pp open-ended. For dense 32B, amplification is '+value('qwen3-vl-32b-instruct','irrelevant_word','mcq_minus_open_adjusted_following',True)+' pp. This consistently supports format amplification in this model set, without proving that copying is its sole mechanism.','',
 'Together with the 8B starting-plausibility experiment, the evidence addresses two factors: the candidate’s question-conditioned plausibility and the answer format. The starting-probability decomposition was run only on 8B; neither these behavioral rates nor a larger model alone show that the same internal mechanism implements both factors. Correct overlays remain in the accuracy table because their beneficial effect is part of the taxonomy.','',
 '## Evaluation, uncertainty, and limits','',
 'Open-ended scoring uses the established deterministic stage followed by two semantic judges for unresolved responses; disagreements remain ambiguous and do not count as correct or adoption. MCQ uses the existing letter extractor; strict-letter and invalid-output counts are exported separately. There are zero UNKNOWN parses. The 133 non-strict 2B responses and 31 non-strict InternVL responses were audited: all 164 start with an unambiguous option letter matching the saved extraction. No parsing corrections were needed; the audit is saved in `comparison/non_strict_mcq_audit.json`. The API 32B model is served remotely while the other six are local, so model-size differences are not isolated from serving and preprocessing differences. MCQ has a 50-token generation cap and open-ended 32, preserving the existing format protocols.','',
 'Intervals use 10,000 paired question bootstrap resamples (seed 42). The seven-model mean weights models equally and resamples the same questions jointly; its interval describes question variation for these fixed models, not uncertainty over all VLMs. These are descriptive intervals, not multiplicity-adjusted significance tests.','',
 '![Irrelevant following by format](irrelevant_following.png)','',
 '[All metrics and intervals](metrics.csv) · [Per-question contrasts](paired_format_effects.csv) · [Parsing and ambiguity audit](generation_audit.csv) · [32B semantic evaluation](../evaluation_32b/results.md)']
 x=np.arange(7);fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
 for offset,fmt,color in [(-.18,'mcq','#b94a48'),(.18,'open_ended','#2878b5')]:
  y=[100*lookup[s,'irrelevant_word',fmt+'_adjusted_following']['estimate'] for s,n in MODELS]
  ax.bar(x+offset,y,width=.36,label=fmt,color=color)
 ax.set_xticks(x,[n for s,n in MODELS],rotation=25,ha='right');ax.set_ylabel('Irrelevant adoption above clean baseline (pp)');ax.axhline(0,color='black',lw=.7);ax.legend();ax.set_title('MCQ versus open-ended: irrelevant-text following')
 for ext in ['png','pdf']:fig.savefig(OUT/f'irrelevant_following.{ext}',dpi=180)
 plt.close(fig);(OUT/'results.md').write_text('\n'.join(lines)+'\n')
 main=V/'qwen3_vl_generation_causal/qwen3_vl_results_report.md';s=main.read_text();marker='\n## 18. Seven-model MCQ versus open-ended comparison'
 if marker in s:s=s[:s.index(marker)]
 import re
 body='\n'.join(lines[2:]).replace('\n## ','\n### ')
 body=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:'](../format_replication_20260921/comparison/'+m.group(1)+')',body)
 main.write_text(s.rstrip()+marker+'\n\n'+body+'\n')
 print(table(['Model','Irrelevant MCQ','Irrelevant open','Amplification'],[[n,value(s,'irrelevant_word','mcq_adjusted_following'),value(s,'irrelevant_word','open_ended_adjusted_following'),value(s,'irrelevant_word','mcq_minus_open_adjusted_following',True)] for s,n in MODELS]))
if __name__=='__main__':main()
