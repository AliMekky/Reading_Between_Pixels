#!/usr/bin/env python3
"""Checkpointed MCQ rerun and QwenCloud dense 32B paired-format evaluation."""
import argparse,base64,hashlib,io,json,os,re,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
VLMS=HERE.parent
sys.path.insert(0,str(VLMS/'inference/main_files'))
from infere_vlms import get_or_download_hf_dataset,build_questions_from_hf_dataset,get_evaluator,BaseVLMEvaluator
MODELS=[('llava','llava-hf/llava-1.5-7b-hf'),('llava-next','llava-hf/llava-v1.6-mistral-7b-hf'),('qwen-vl','Qwen/Qwen2.5-VL-7B-Instruct'),('internvl','OpenGVLab/InternVL3_5-8B'),('qwen3-vl','Qwen/Qwen3-VL-2B-Instruct'),('qwen3-vl','Qwen/Qwen3-VL-8B-Instruct')]
STATES=['notext','correct_answer','misleading_groundable','misleading_ungroundable','irrelevant_word']
URL='https://maas.qwencloudapi.com/compatible-mode/v1'
API_MODEL='qwen3-vl-32b-instruct'

def key():
    for line in Path(os.environ['API_ENV_FILE']).read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if line.startswith('QWEN_API_KEY='):
            value=line.split('=',1)[1].strip().strip('"\'')
            if value:return value
    raise RuntimeError('QWEN_API_KEY not found')

class Cloud:
    model_id=API_MODEL
    model_revision=None
    def __init__(self):
        import requests
        self.session=requests.Session();self.session.headers.update(Authorization='Bearer '+key())
        self.next_request=0.0
    def generate(self,image,prompt,limit):
        image=image.convert('RGB');buf=io.BytesIO();image.save(buf,format='PNG')
        payload=dict(model=API_MODEL,temperature=0,max_tokens=limit,messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}},{'type':'text','text':prompt}]}])
        for attempt in range(8):
            time.sleep(max(0.0,self.next_request-time.monotonic()))
            self.next_request=time.monotonic()+3.0
            try:
                response=self.session.post(URL+'/chat/completions',json=payload,timeout=(20,120))
            except Exception as e:
                if attempt==7:raise RuntimeError('API transport failure: '+type(e).__name__) from None
                time.sleep(min(60,2**attempt));continue
            if response.status_code in [429,500,502,503,504] and attempt<7:
                delay=min(120,60*2**attempt) if response.status_code==429 else min(60,2**attempt)
                retry_after=response.headers.get('Retry-After')
                if retry_after:
                    try:
                        delay=max(delay,float(retry_after))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        try:delay=max(delay,parsedate_to_datetime(retry_after).timestamp()-time.time())
                        except (ValueError,TypeError,OverflowError):pass
                # Bound a single pause; persistent limits stop for later resume.
                delay=min(300,max(0,delay))
                print(f'[API RETRY] HTTP {response.status_code}; attempt={attempt+1}/8; wait={delay:.0f}s',flush=True)
                time.sleep(delay);continue
            if response.status_code!=200:
                try:code=response.json().get('error',{}).get('code','unknown')
                except Exception:code='unparseable'
                raise RuntimeError(f'API HTTP {response.status_code}; code={str(code)[:80]}')
            body=response.json();choice=body['choices'][0];usage=body.get('usage',{})
            assert body.get('model')==API_MODEL, 'Unexpected API response model; refusing model substitution'
            text=choice['message'].get('content');assert isinstance(text,str) and text.strip()
            return dict(raw_response=text.strip(),output_token_ids=None,output_token_count=usage.get('completion_tokens'),termination_reason=choice['finish_reason'],usage=usage,provider_response_id=body.get('id'),provider_model=body.get('model'),system_fingerprint=body.get('system_fingerprint'),image_sha256=hashlib.sha256(f'{image.mode}:{image.size}'.encode()+image.tobytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-index',type=int);p.add_argument('--api',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--only-format',choices=['open_ended','mcq']);a=p.parse_args()
    assert a.api != (a.model_index is not None)
    model=API_MODEL if a.api else MODELS[a.model_index][1]
    formats=['open_ended','mcq'] if a.api else ['mcq']
    out=HERE/'outputs'/('smoke' if a.smoke else 'full')/model.replace('/','__');out.mkdir(parents=True,exist_ok=True)
    config=dict(model_id=model,formats=formats,seed=42,image_field='cleaned_image',questions=2 if a.smoke else 474,max_tokens={'open_ended':32,'mcq':50},backend=URL if a.api else 'local_transformers',version=1)
    cp=out/'configuration.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    ds=get_or_download_hf_dataset('anonymous/GUIC',local_cache_root=str(VLMS/'activation_patching/hf_dataset_GUIC_cleaned'),split='test')
    assert len(ds)==474
    evaluator=Cloud() if a.api else get_evaluator(model_type=MODELS[a.model_index][0],model_id=model,device='cuda')
    # Validate all inputs and mappings before sending requests for this model.
    groups={s:build_questions_from_hf_dataset(ds,variant=s,image_field='cleaned_image',shuffle_options=True,seed=42,max_samples=2 if a.smoke else None) for s in STATES}
    n=config['questions'];assert all(len(g)==n for g in groups.values())
    baseline={str(i['question_id']):i for i in groups['notext']}
    for s,items in groups.items():
        for i in items:
            b=baseline[str(i['question_id'])];assert i['options']==b['options'] and i['option_meta']==b['option_meta']
    selected_formats=[a.only_format] if a.only_format else formats
    assert set(selected_formats).issubset(formats)
    for fmt in selected_formats:
        for state,items in groups.items():
            path=out/f'{fmt}_{state}.jsonl'
            existing={r['question_id']:r for line in path.read_text().splitlines() if line for r in [json.loads(line)]} if path.exists() else {}
            assert set(existing).issubset(baseline)
            with path.open('a') as f:
                for i in items:
                    q=str(i['question_id'])
                    prompt=BaseVLMEvaluator.format_mcq_prompt(None,i['question'],i['options']) if fmt=='mcq' else BaseVLMEvaluator.format_open_ended_prompt(None,i['question'])
                    image=i['image_input'].convert('RGB');digest=hashlib.sha256(f'{image.mode}:{image.size}'.encode()+image.tobytes()).hexdigest()
                    if q in existing:
                        assert existing[q]['prompt']==prompt and existing[q]['image_sha256']==digest and existing[q]['status']=='ok'
                        continue
                    if fmt=='open_ended':assert 'Options:' not in prompt
                    limit=config['max_tokens'][fmt]
                    generation=evaluator.generate(image,prompt,limit) if a.api else evaluator.process_open_ended_single(image,prompt,max_new_tokens=limit)
                    assert generation['image_sha256']==digest
                    record=dict(record_id=f'{model}|{fmt}|{state}|{q}',question_id=q,source_image_id=i.get('source_image_id'),model_id=model,model_revision=evaluator.model_revision,evaluation_format=fmt,variant=state,question=i['question'],prompt=prompt,reference_answers=i['reference_answers'],options=i['options'],option_meta=i['option_meta'],correct_answer=i['answer'],status='ok',**generation)
                    if fmt=='mcq':
                        answer=BaseVLMEvaluator.extract_answer(None,generation['raw_response'])
                        strict=re.fullmatch(r'\s*([ABCD])[.)]?\s*',generation['raw_response'])
                        record.update(predicted_answer=answer,is_correct=answer==i['answer'],strict_letter=strict.group(1) if strict else None)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();existing[q]=record
                    print(f'[RESULT] {model} {fmt} {state} {len(existing)}/{n} {generation["raw_response"]!r}',flush=True)
            assert len(existing)==n
    (out/('completion_'+a.only_format+'.json' if a.only_format else 'completion.json')).write_text(json.dumps(dict(status='complete',model=model,questions=n,formats=selected_formats,records=n*5*len(selected_formats)),indent=2)+'\n')
    print('[PASS] complete',model,flush=True)
if __name__=='__main__':main()
