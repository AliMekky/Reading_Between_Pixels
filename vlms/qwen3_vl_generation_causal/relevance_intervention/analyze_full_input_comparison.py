#!/usr/bin/env python3
"""Paired descriptive analysis of all input states, with question bootstrap CIs."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_input_comparison import KEYS, STATES, LABELS, norm

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE/'outputs/input_comparison_305')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected', type=int, default=305)
    args = parser.parse_args()
    root, out = args.root, args.output or args.root
    shards = sorted(root.glob('shard_*')) or [root]
    records, configurations, checks = {}, [], []
    for shard in shards:
        done = json.loads((shard/'completion.json').read_text())
        assert done['status'] == 'complete'
        config = json.loads((shard/'configuration.json').read_text())
        configurations.append(config)
        expected = {(e['question_id'], s) for e in config['selected_samples'] for s in STATES}
        local = {}
        for p in sorted((shard/'samples').glob('*.json')):
            r = json.loads(p.read_text())
            key = (r['question_id'], r['state'])
            assert key not in records and key not in local and r['status'] == 'complete'
            local[key] = r
            if 'cache_validation' in r:
                checks.append(r['cache_validation'])
        assert set(local) == expected
        records.update(local)
    qids = sorted({q for q,s in records})
    n = len(qids)
    assert n == args.expected and len(records) == n*7
    assert len(checks) == 6*len(shards)
    assert all(c['generation_tokens_identical'] and c['max_token_logprob_error'] <= .001 for c in checks)
    for c in configurations[1:]:
        for k in ['model','revision','instruction','states','candidate_keys','backend','dtype','gray_rgb']:
            assert c[k] == configurations[0][k]
    if n == 305:
        manifest = json.loads((HERE.parent.parent/'activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json').read_text())
        assert set(qids) == {str(e['question_id']) for e in manifest['selected_samples']}
        assert len(shards) == configurations[0]['shards']
    out.mkdir(parents=True, exist_ok=True)
    def write(name, rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    raw, generations = [], []
    for q in qids:
        gray = records[q,'gray_image']
        for s in STATES:
            r=records[q,s]
            assert r['references'] == gray['references']
            if s != 'question_only':
                assert r['prompt_token_ids'] == gray['prompt_token_ids'] and r['image_grid_thw'] == gray['image_grid_thw']
            else:
                assert r['image_token_count'] == 0
            for k in KEYS:
                v=r['scores'][k]
                assert v['token_ids'] == records[q,'question_only']['scores'][k]['token_ids']
                assert all(math.isfinite(x) for x in v['token_logprobs'])
                raw.append(dict(question_id=q,state=s,candidate=k,answer=v['answer'],token_count=v['token_count'],
                    mean_logprob=v['mean_logprob'],sum_logprob=v['sum_logprob'],first_logprob=v['token_logprobs'][0],first_probability=math.exp(v['token_logprobs'][0])))
            response=r['generation']['raw_response']
            matches=[k for k in KEYS if norm(response)==norm(r['references'][k])]
            generations.append(dict(question_id=q,state=s,question=r['question'],response=response,
                exact_category=matches[0] if len(matches)==1 else 'unmatched_or_ambiguous',
                termination=r['generation']['termination_reason'],references=json.dumps(r['references'])))
    write('candidate_scores.csv',raw);write('generation_exact_audit.csv',generations)
    rng=np.random.default_rng(271828)
    draws=rng.integers(0,n,size=(10000,n))
    contrasts, perquestion = [], []
    def arr(s,k,metric):
        return np.array([records[q,k if s=='own_overlay' else s]['scores'][k][metric] for q in qids])
    def add(metric,effect,candidate,x):
        lo,hi=np.quantile(x[draws].mean(axis=1),[.025,.975])
        contrasts.append(dict(metric=metric,effect=effect,candidate=candidate,n=n,mean=float(x.mean()),median=float(np.median(x)),ci_low=float(lo),ci_high=float(hi),positive=int((x>0).sum())))
        perquestion.extend(dict(question_id=q,metric=metric,effect=effect,candidate=candidate,value=float(v)) for q,v in zip(qids,x))
    pairs=[('gray_image','question_only'),('clean_image','question_only'),('clean_image','gray_image'),('own_overlay','clean_image')]
    for metric in ['mean_logprob','sum_logprob']:
        gains={}
        for k in KEYS:
            for target,base in pairs:
                effect=target+'_minus_'+base
                x=arr(target,k,metric)-arr(base,k,metric)
                add(metric,effect,k,x)
                gains[k,effect]=x
            for s in [*STATES[:3],'own_overlay']:
                # Correct score must come from the SAME image as this candidate.
                c=np.array([records[q,k if s=='own_overlay' else s]['scores'][KEYS[0]][metric] for q in qids])
                add(metric,'candidate_minus_correct_in_'+s,k,arr(s,k,metric)-c)
            for overlay in KEYS:
                add(metric,'overlay_'+overlay+'_minus_clean',k,arr(overlay,k,metric)-arr('clean_image',k,metric))
            own=arr('own_overlay',k,metric)-arr('clean_image',k,metric)
            correct_shift=arr(k,KEYS[0],metric)-arr('clean_image',KEYS[0],metric)
            add(metric,'own_overlay_relative_gain_vs_correct',k,own-correct_shift)
        for a,b in itertools.combinations(KEYS,2):
            for target,base in pairs:
                effect=target+'_minus_'+base
                add(metric,'paired_condition_difference_'+effect,a+'_minus_'+b,gains[a,effect]-gains[b,effect])
        for s in STATES[:3]:
            gap=(arr(s,KEYS[1],metric)+arr(s,KEYS[2],metric))/2-arr(s,KEYS[3],metric)
            add(metric,'relevant_misleading_minus_irrelevant_in_'+s,'paired',gap)
    write('contrast_summary.csv',contrasts);write('paired_question_contrasts.csv',perquestion)
    lookup={(r['effect'],r['candidate']):r for r in contrasts if r['metric']=='mean_logprob'}
    def estimate(effect,k):
        r=lookup[effect,k]
        return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    display=[*STATES[:3],'own_overlay']
    for k,label in zip(KEYS,LABELS):
        axs[0].plot(range(4),[arr(s,k,'mean_logprob').mean() for s in display],'o-',label=label)
    axs[0].set_xticks(range(4),['Question only','Gray image','Clean scene','Own overlay'])
    axs[0].set_ylabel('Mean candidate log probability per token (nats)')
    axs[0].set_title(f'Candidate support ({n} paired questions)');axs[0].legend(fontsize=8)
    matrix=np.array([[(arr(o,k,'mean_logprob')-arr('clean_image',k,'mean_logprob')).mean() for k in KEYS] for o in KEYS])
    limit=max(abs(matrix.min()),abs(matrix.max()))
    im=axs[1].imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit)
    axs[1].set_xticks(range(4),['Correct','Grounded','Ungrounded','Irrelevant'],rotation=25,ha='right')
    axs[1].set_yticks(range(4),['Correct','Grounded','Ungrounded','Irrelevant'])
    axs[1].set_xlabel('Candidate scored');axs[1].set_ylabel('Overlay displayed')
    axs[1].set_title('Overlay contribution relative to clean scene')
    for i in range(4):
        for j in range(4):axs[1].text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.6*limit else 'black')
    fig.colorbar(im,ax=axs[1],label='Change in nats/token',shrink=.8)
    for ext in ['png','pdf']:fig.savefig(out/f'input_comparison.{ext}',dpi=180)
    plt.close(fig)
    lines=[f'# Full input comparison: {n} paired questions','',f'Completed {n*7:,} generations and {n*28:,} candidate scores using question-only, matched gray image, clean scene, and four overlays. The evaluation set is the shared 305-question manifest, not every row of the source dataset.','',
        'Scores are teacher-forced mean log probabilities per answer token (nats; higher is better), excluding EOS. These are not answer probabilities. Brackets give descriptive 95% percentile bootstrap intervals from 10,000 resamples of whole questions; all condition comparisons preserve pairing. Intervals are not multiplicity-adjusted significance tests.','',
        '## How much do the scene and overlay change support?','',
        '| Candidate | Scene − question only | Scene − gray | Own overlay − scene | Own overlay gain relative to correct |','|---|---:|---:|---:|---:|']
    for k,label in zip(KEYS,LABELS):
        lines.append('| '+label+' | '+' | '.join(estimate(e,k) for e in ['clean_image_minus_question_only','clean_image_minus_gray_image','own_overlay_minus_clean_image','own_overlay_relative_gain_vs_correct'])+' |')
    lines += ['', 'The final column subtracts the correct-answer score change under the same overlay from the displayed-candidate score change. Its value is identically zero for correct overlays. A positive value for a harmful overlay means its candidate gained ground relative to the correct answer; it does not imply that the candidate wins generation.','',
        '## Does the gap exist without scene information?','','| Input | Relevant misleading − irrelevant | Positive paired gaps |','|---|---:|---:|']
    for s in STATES[:3]:
        e='relevant_misleading_minus_irrelevant_in_'+s;r=lookup[e,'paired']
        lines.append(f'| {s} | {estimate(e,"paired")} | {r["positive"]}/{n} |')
    lines += ['', 'Relevant misleading is the within-question average of grounded and ungrounded misleading candidates.','',
        '## Direct comparisons between overlay conditions','','These compare each condition’s own-overlay gain over the clean scene on the same questions. Positive means the first condition receives the larger boost.','','| Contrast | Difference in overlay boost |','|---|---:|']
    for a,b in itertools.combinations(KEYS,2):
        lines.append(f'| {a} − {b} | {estimate("paired_condition_difference_own_overlay_minus_clean_image",a+"_minus_"+b)} |')
    lines += ['', '## Generated answers: normalized exact matches only','','| Input | Exact correct | Exact irrelevant | Unmatched/ambiguous | Length-limit terminations |','|---|---:|---:|---:|---:|']
    for s in STATES:
        rows=[r for r in generations if r['state']==s]
        counts=[sum(r['exact_category']==k for r in rows) for k in ['correct_answer','irrelevant_word','unmatched_or_ambiguous']]
        trunc=sum(r['termination']!='eos' for r in rows)
        lines.append(f'| {s} | {counts[0]}/{n} | {counts[1]}/{n} | {counts[2]}/{n} | {trunc}/{n} |')
    lines += ['', 'These are exact-match counts, not final semantic accuracy. Unmatched responses can include correct paraphrases, other answers, and abstentions; no paid judge or unreviewed semantic labels are used. Correctness refers to the original scene answer even when the scene is absent.','',
        '## Interpretation limits and checks','','The contrasts measure input-associated score changes under a fixed prompt and model. They do not isolate pure language priors, prove an abstract relevance representation, or identify a Q→A mechanism. A gray image retains visual tokens but may elicit blank-image answers. Mean and summed log-probability analyses are both exported; candidate spelling and answer formatting can affect scores. The same modality-neutral prompt is used everywhere; earlier intervention scores used a different instruction and are not substituted.','',
        f'All {len(checks)} native-versus-cached visual checks passed; maximum token-logprob error {max(c["max_token_logprob_error"] for c in checks):.6g}. Candidate tokenization matches across states; visual prompt IDs and image grids match within questions. Each shard must complete and all question/state pairs must be present before this report is written.','',
        '![Comparison](input_comparison.png)','',
        'Artifacts: [all candidate scores](candidate_scores.csv), [paired effects and intervals](contrast_summary.csv), [per-question changes](paired_question_contrasts.csv), [raw generations and exact audit](generation_exact_audit.csv), [PDF figure](input_comparison.pdf).']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'analysis_completion.json').write_text(json.dumps(dict(status='complete',questions=n,records=len(records),scores=len(raw),bootstrap_seed=271828,bootstrap_resamples=10000),indent=2)+'\n')
    print(f'PASS: analyzed {n} paired questions; report: {out / "report.md"}')

if __name__=='__main__':main()
