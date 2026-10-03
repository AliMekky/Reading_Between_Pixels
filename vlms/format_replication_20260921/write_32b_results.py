#!/usr/bin/env python3
"""Write explicitly provisional or finalized 32B results without changing old aggregates."""
import csv,json
from pathlib import Path
H=Path(__file__).resolve().parent;V=H.parent;E=H/'evaluation_32b';K=['notext','correct_answer','misleading_groundable','misleading_ungroundable','irrelevant_word']
L=['Clean image','Correct overlay','Grounded misleading','Ungrounded misleading','Irrelevant']
S='qwen3-vl-32b-instruct'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']+['---:']*(len(head)-1))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 final=(E/'statistics/summary.json').exists() and (E/'final/judge_validation.json').exists()
 ds=json.loads((E/'deterministic'/S/'summary.json').read_text())
 raw={k:[json.loads(l) for l in (H/'outputs/full'/S/f'open_ended_{k}.jsonl').read_text().splitlines()] for k in K}
 assert all(len(r)==474 for r in raw.values())
 ins=sum(r['usage']['prompt_tokens'] for rr in raw.values() for r in rr);outs=sum(r['usage']['completion_tokens'] for rr in raw.values() for r in rr)
 lines=['# Qwen3-VL-32B-Instruct: open-ended evaluation','',
 'All **474 questions × five image conditions = 2,370 responses** are complete. Every request contained the actual image (lossless PNG) and the existing open-ended question prompt, without MCQ options. The API returned `qwen3-vl-32b-instruct` for every saved response. This is the dense 32B counterpart of the local 2B/8B models; the interrupted 30B-A3B smoke test is excluded.','',
 '**Evaluation status: '+('complete, including both semantic judges.' if final else 'provisional; semantic judges are pending. Do not treat exact matches as final accuracy.')+'**','',
 '## Evaluation procedure','',
 f'The unchanged deterministic evaluator resolved {ds["normalized_exact"]:,} responses by normalized exact matching and {ds["edit_similarity"]} by the fixed 0.90 edit-similarity rule; {ds["invalid"]} were invalid. All {ds["unresolved"]} unresolved responses were submitted to the same two judges as the previous six-model evaluation (`gpt-5.6-luna` and `gemini-3.5-flash`), with reversed reference order. Agreement determines the semantic category; disagreement remains ambiguous. No inherited labels or assistant-only judgments replace that procedure. Malformed or truncated judge decisions are retried with a larger output budget and concise reasons; Gemini retries disable thinking, following the established retry protocol. The final two Gemini retries constrained the brief reason to an enum to prevent a repeated-text loop; the semantic classifier, model, and reference order were unchanged. Raw requests and responses are preserved. Final metrics require valid decisions from both judges for every unresolved response.','']
 if final:
  rows=list(csv.DictReader((E/'statistics/metrics.csv').open()));m={(r['condition'],r['metric']):r for r in rows if r['model']==S}
  def val(c,metric,ci=False):
   r=m[c,metric];s=f"{float(r['estimate'])*100:.1f}"
   return s+(f" [{float(r['ci_low'])*100:.1f}, {float(r['ci_high'])*100:.1f}]" if ci else '')
  lines+=['## Final semantic results','',table(['Condition','Accuracy % [95% CI]','Accuracy change (pp)','Displayed-answer adoption %','Adjusted adoption (pp)'],[[l,val(k,'accuracy',True),val(k,'accuracy_change',True) if k!='notext' else 'Reference',val(k,'target_following') if k!='notext' else 'Not applicable',val(k,'adjusted_target_following',True) if k!='notext' else 'Reference'] for k,l in zip(K,L)]),'',
  'Accuracy changes compare each overlay with its paired clean-image baseline. Adjusted adoption subtracts the tendency to produce that same candidate on the clean image. Correct-overlay adoption equals accuracy. Accuracy intervals are Wilson intervals; changes and following intervals use 10,000 question bootstrap resamples, seed 42. Paired accuracy tests and Holm corrections are exported in `statistics/mcnemar_holm.csv`. Ambiguous answers are retained in the denominator but not counted as correct or target adoption.','',
  '## Interpretation','',
  f'The clean-image accuracy is {val("notext","accuracy")}%. Correct overlays change accuracy by {val("correct_answer","accuracy_change")} percentage points. Grounded and ungrounded misleading overlays change accuracy by {val("misleading_groundable","accuracy_change")} and {val("misleading_ungroundable","accuracy_change")} points, respectively. Irrelevant overlays change it by {val("irrelevant_word","accuracy_change")} points. These directions and magnitudes describe the observed 32B behavior; a size trend requires paired comparisons rather than comparing point estimates alone.','',
  f'After removing clean-image following, displayed-answer adoption changes by {val("misleading_groundable","adjusted_target_following")} points for grounded misleading text, {val("misleading_ungroundable","adjusted_target_following")} for ungrounded misleading text, and {val("irrelevant_word","adjusted_target_following")} for irrelevant text. This directly measures following beyond the model’s baseline tendency to answer with the same word.','',
  '## Comparison with the six completed local models','']
  old=list(csv.DictReader((V/'open_ended_evaluation/outputs/statistics_six_models/metrics.csv').open()))
  joined=old+[r for r in rows if r['model']==S];models=sorted({r['model'] for r in joined if r['model'] not in ['aggregate','unweighted_model_average']})
  models=[x for x in models if any(r['model']==x and r['condition']=='notext' and r['metric']=='accuracy' for r in joined)]
  tab=[]
  for model in models:
   z={(r['condition'],r['metric']):r for r in joined if r['model']==model}
   if ('notext','accuracy') not in z:continue
   tab.append([model]+[f"{100*float(z[k,'accuracy']['estimate']):.1f}" for k in K])
  lines += [table(['Model','Clean %','Correct %','Grounded %','Ungrounded %','Irrelevant %'],tab),'',
   'The earlier six-model aggregates remain unchanged. This table adds the separately evaluated hosted 32B run. The models share questions, image conditions, and answer instructions, but 32B uses provider-side preprocessing and serving while the smaller models run locally. Its weights revision and output token IDs are not supplied by the API. Consequently, differences cannot be attributed exclusively to model size.','']
  summary=json.loads((E/'final'/S/'summary.json').read_text())
  lines+=['## Semantic audit','', 'Final stage counts: `'+json.dumps(summary['stage_counts'],sort_keys=True)+'`. Final category counts: `'+json.dumps(summary['category_counts'],sort_keys=True)+'`.','']
 else:
  tab=[]
  for k,l in zip(K,L):
   rr=[json.loads(x) for x in (E/'deterministic'/S/f'{k}.jsonl').read_text().splitlines()]
   correct=sum(r['deterministic_category']=='correct_answer' for r in rr)
   target=sum(r['deterministic_category']==k for r in rr) if k!='notext' else None
   tab.append([l,correct,f'{correct/474*100:.1f}',target if target is not None else 'Not applicable',sum(r['needs_llm_judge'] for r in rr)])
  lines+=['## Provisional deterministic counts — not final semantic accuracy','',table(['Condition','Matched correct / 474','Matched correct %','Matched displayed word / 474','Pending semantic review'],tab),'',
   'These counts are intentionally not compared with the earlier semantically judged accuracy. The missing semantic decisions can change both accuracy and adoption estimates. Final interpretation and cross-model comparisons will replace this provisional table after both judge outputs validate.','']
 lines+=['## Relation to the relevance explanation','',
 'This extends the open-ended behavioral evaluation to a larger dense Qwen3-VL model. It does not repeat the 8B candidate-probability decomposition or attention interventions. The claim about low question-conditioned starting support remains based on the 8B experiment. The 32B API MCQ run is complete. The matched seven-model format comparison is reported separately in the main report’s Section 18; it uses each format’s own clean-image baseline.','',
 '## Generation audit and recorded usage','',
 f'Full open-ended usage: **{ins:,} input tokens and {outs:,} output tokens**. At the checked list rates of $0.16/$0.64 per million input/output tokens, the saved full-run responses cost approximately **${(ins*.16+outs*.64)/1e6:.5f}**, excluding smoke tests, taxes, credits, and any unsaved provider-side request. [Pricing source](https://www.qwencloud.com/models/qwen3-vl-32b-instruct).','',
 'Generation was resumed from checkpoints after rate limiting; no successful saved answer was regenerated. Requests used temperature zero and a 32-token answer cap. All conditions have exactly the same 474 question IDs. All 2,370 prompts, image hashes, and reference sets match the earlier Qwen3-VL-8B open-ended inputs exactly; provider-side image processing remains outside our control.','',
 'Artifacts: [deterministic summary](deterministic/qwen3-vl-32b-instruct/summary.json), [judge submission state](judges/batch_state.json).'+(' [Final judge validation](final/judge_validation.json), [metrics and confidence intervals](statistics/metrics.csv), [paired tests](statistics/mcnemar_holm.csv), [transitions](statistics/transition_summary.csv).' if final else ''),'']
 (E/'results.md').write_text('\n'.join(lines))
 # Embed in the main report while making local artifact links relative to it.
 main=V/'qwen3_vl_generation_causal/qwen3_vl_results_report.md';text=main.read_text();marker='\n## 17. Dense Qwen3-VL-32B open-ended extension'
 suffix=''
 if marker in text:
  start=text.index(marker)
  next_section=text.find('\n## 18.',start)
  suffix=text[next_section:] if next_section!=-1 else ''
  text=text[:start]
 body='\n'.join(lines[2:]).replace('\n## ','\n### ')
 import re
 body=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:'](../format_replication_20260921/evaluation_32b/'+m.group(1)+')',body)
 main.write_text(text.rstrip()+marker+'\n\n'+body+'\n'+suffix)
 oldreport=V/'open_ended_evaluation/open_ended_evaluation_results_six_models.md';s=oldreport.read_text();start='<!-- 32B extension status -->';end='<!-- end 32B extension status -->'
 note=start+'\n\n**32B extension:** '+('Semantic evaluation is complete.' if final else 'Generation is complete; semantic evaluation is pending.')+' See the [separate 32B results](../format_replication_20260921/evaluation_32b/results.md). The six-model results and aggregates below retain their original scope.\n\n'+end
 if start in s:s=s[:s.index(start)]+note+s[s.index(end)+len(end):]
 else:s=s.replace('## Scope',note+'\n\n## Scope',1)
 oldreport.write_text(s)
 print('Updated results and main Section 17; final_semantic=',final)
if __name__=='__main__':main()
