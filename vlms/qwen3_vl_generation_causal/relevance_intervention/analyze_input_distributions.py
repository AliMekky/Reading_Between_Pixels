#!/usr/bin/env python3
"""Distribution plots for the completed paired 305-question input comparison."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/input_comparison_305'
KEYS=['correct_answer','misleading_groundable','misleading_ungroundable','irrelevant_word']
LABELS=['Correct','Grounded misleading','Ungrounded misleading','Irrelevant']
COLORS=['#2878b5','#de9132','#8b61b0','#c94b49']

def main():
    assert json.loads((OUT/'analysis_completion.json').read_text())['questions']==305
    rows=list(csv.DictReader((OUT/'candidate_scores.csv').open()))
    data={(r['question_id'],r['state'],r['candidate']):r for r in rows}
    qids=sorted({r['question_id'] for r in rows}); assert len(data)==8540
    def x(state,k,metric='mean_logprob'):
        return np.array([float(data[q,state,k][metric]) for q in qids])
    summary=[];crossings=[]
    fig,axs=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    titles=['Question-only candidate support','Scene contribution: clean − question only','Overlay contribution: own overlay − clean',
            'Gray-image candidate support','Margin before overlay: candidate − correct','Margin after own overlay: candidate − correct']
    for metric in ['mean_logprob','sum_logprob']:
        for k,label,color in zip(KEYS,LABELS,COLORS):
            before=x('clean_image',k,metric)-x('clean_image',KEYS[0],metric)
            after=x(k,k,metric)-x(k,KEYS[0],metric)
            arrays=[x('question_only',k,metric),x('clean_image',k,metric)-x('question_only',k,metric),
                    x(k,k,metric)-x('clean_image',k,metric),x('gray_image',k,metric),before,after]
            for j,(title,v) in enumerate(zip(titles,arrays)):
                if k==KEYS[0] and j>=4:continue
                quant=np.quantile(v,[.1,.25,.5,.75,.9])
                summary.append(dict(metric=metric,distribution=title,candidate=k,n=len(v),mean=float(v.mean()),
                    p10=quant[0],p25=quant[1],median=quant[2],p75=quant[3],p90=quant[4],positive=int((v>0).sum())))
                if metric=='mean_logprob':
                    axs.flat[j].step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',label=label,color=color)
            if k!=KEYS[0]:
                crossings.append(dict(metric=metric,candidate=k,above_correct_before=int((before>0).sum()),
                    above_correct_after=int((after>0).sum()),below_to_above=int(((before<0)&(after>0)).sum()),
                    above_to_below=int(((before>0)&(after<0)).sum()),ties_before=int((before==0).sum()),ties_after=int((after==0).sum()),
                    within_two_nats_before=int((abs(before)<=2).sum()),within_two_nats_after=int((abs(after)<=2).sum())))
    for ax,title in zip(axs.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Score or change (nats/token)');ax.set_ylabel('Fraction of questions ≤ value')
        ax.axvline(0,color='black',ls=':',lw=.8);ax.set_ylim(0,1.02);ax.grid(alpha=.15);ax.legend(fontsize=7,loc='best')
    for ext in ['png','pdf']:fig.savefig(OUT/f'distributions.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    allvalues=[]
    for ax,k,label,color in zip(axs,KEYS[1:],LABELS[1:],COLORS[1:]):
        before=x('clean_image',k)-x('clean_image',KEYS[0]);after=x(k,k)-x(k,KEYS[0])
        ax.scatter(before,after,s=13,alpha=.45,color=color);allvalues.extend([*before,*after])
        ax.set_title(label);ax.set_xlabel('Before overlay: candidate − correct');ax.set_ylabel('After overlay: candidate − correct')
    lo,hi=min(allvalues)-1,max(allvalues)+1
    for ax in axs:
        ax.plot([lo,hi],[lo,hi],color='gray',ls='--',lw=.8);ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7)
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal');ax.grid(alpha=.15)
    for ext in ['png','pdf']:fig.savefig(OUT/f'paired_margins.{ext}',dpi=180)
    plt.close(fig)
    for name,values in [('distribution_summary.csv',summary),('margin_crossings.csv',crossings)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    lines=['# Distribution comparison: 305 paired questions','','All distributions describe variation across questions, not the model’s full vocabulary distribution. Higher candidate scores and more positive margins favor the candidate. A mean-token score margin above zero does not guarantee free-generation adoption.','','| Candidate | Median question-only score | Median overlay boost | Overlay boosts > 0 | Median margin before → after | Above correct before → after |','|---|---:|---:|---:|---:|---:|']
    for k,label in zip(KEYS[1:],LABELS[1:]):
        boost=x(k,k)-x('clean_image',k);before=x('clean_image',k)-x('clean_image',KEYS[0]);after=x(k,k)-x(k,KEYS[0])
        lines.append(f'| {label} | {np.median(x("question_only",k)):.2f} | {np.median(boost):.2f} | {(boost>0).sum()}/305 | {np.median(before):.2f} → {np.median(after):.2f} | {(before>0).sum()} → {(after>0).sum()} |')
    lines += ['','The correct score is always taken from the same image as the candidate. Own-overlay changes compare each displayed candidate with itself on the clean scene. Crossing counts and summed-logprob sensitivity are exported separately. ECDFs preserve outliers; paired scatter plots show each question’s movement. No unpaired distribution test or multiplicity-adjusted significance claim is made.','','![Distributions](distributions.png)','','![Paired margins](paired_margins.png)','','[Quantiles and positive-effect counts](distribution_summary.csv) · [Margin crossings and summed-score sensitivity](margin_crossings.csv) · [ECDF PDF](distributions.pdf) · [Paired-margin PDF](paired_margins.pdf)']
    (OUT/'distribution_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:9]))

if __name__=='__main__':main()
