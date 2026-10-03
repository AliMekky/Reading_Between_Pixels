#!/usr/bin/env python3
"""Wait for both submitted judges, validate, calculate metrics, and update reports."""
import json,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;E=H/'evaluation_32b';S=H.parent/'open_ended_evaluation/main_files'
if (E/'PAUSED_BY_USER').exists():raise SystemExit('Evaluation paused by user; explicit resume required.')
def run(script,*args):subprocess.run([sys.executable,str(script),*map(str,args)],check=True)
# Cancelled OpenAI batches may contain valid partial results; retrieve and keep them.
run(S/'check_judge_batches.py','--batch_dir',E/'judges')
state=json.loads((E/'judges/batch_state.json').read_text())
assert state['openai']['status'] in ['completed','cancelled','cancelling','expired','failed'], 'Original OpenAI batch still active'
assert any(x in state['gemini']['status'] for x in ['SUCCEEDED','CANCELLED','EXPIRED','FAILED']), 'Original Gemini batch still active'
# Validate before merging. Retry only malformed decisions with the established
# larger response budget, short reasons, and Gemini thinking disabled.
sys.path.insert(0,str(S))
from merge_judge_results import load_manifest,load_openai,load_gemini
from openai import OpenAI
from google import genai
from google.genai import types
import os
manifest=load_manifest(E/'judges/judge_manifest.jsonl')
def retry_provider(provider,loader):
    source=E/'judges'/f'{provider}_results.jsonl'
    if provider=='openai' and not source.exists():
        client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        for poll in range(690):
            original=client.batches.retrieve(json.loads((E/'judges/batch_state.json').read_text())['openai']['batch_id'])
            if original.status in ['completed','cancelled','expired','failed']:
                if original.output_file_id:
                    source.write_bytes(client.files.content(original.output_file_id).content)
                elif original.request_counts and original.request_counts.completed:
                    raise RuntimeError('Provider reports completed decisions but has not released output; do not duplicate them')
                break
            if poll%5==0:print('[WAIT] OpenAI cancellation pending; preserving completed work before retry.',flush=True)
            time.sleep(120)
        else:raise RuntimeError('OpenAI cancellation still pending; resume later')
    combined=E/'judges'/f'{provider}_results_combined.jsonl'
    if not combined.exists():combined.write_bytes(source.read_bytes() if source.exists() else b'')
    for retry in range(1,3):
        valid,errors=loader(combined,manifest)
        bad=(set(manifest)-set(valid))|set(errors)
        if not bad:break
        import hashlib
        signature=hashlib.sha256('\n'.join(sorted(bad)).encode()).hexdigest()[:12]
        folder=E/f'resume_{provider}_{retry}_{signature}';folder.mkdir(exist_ok=True)
        requests=[json.loads(line) for line in (E/'judges'/f'{provider}_requests.jsonl').read_text().splitlines()]
        ident='custom_id' if provider=='openai' else 'key'
        selected=[r for r in requests if r[ident] in bad]
        assert {r[ident] for r in selected}==bad
        for r in selected:
            if provider=='openai':
                r['body']['max_output_tokens']=512*retry
                r['body']['input'][0]['content']+=' Keep the reason under eight words.'
            else:
                r['request']['generationConfig'].update(maxOutputTokens=512*retry,thinkingConfig={'thinkingBudget':0})
                r['request']['systemInstruction']['parts'][0]['text']+=' Keep the reason under eight words.'
        requestfile=folder/'requests.jsonl'
        requestfile.write_text(''.join(json.dumps(r)+'\n' for r in selected))
        statefile=folder/'state.json'
        state=json.loads(statefile.read_text()) if statefile.exists() else {}
        resultfile=folder/'results.jsonl'
        print(f'[RETRY] provider={provider} malformed_or_missing={len(bad)} round={retry}',flush=True)
        if provider=='openai':
            client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            if 'id' not in state:
                with requestfile.open('rb') as f:uploaded=client.files.create(file=f,purpose='batch')
                b=client.batches.create(input_file_id=uploaded.id,endpoint='/v1/responses',completion_window='24h')
                state={'id':b.id};statefile.write_text(json.dumps(state))
            for poll in range(690):
                b=client.batches.retrieve(state['id'])
                if b.status=='completed':
                    assert b.output_file_id
                    resultfile.write_bytes(client.files.content(b.output_file_id).content);break
                if b.status in ['failed','expired','cancelled']:raise RuntimeError('OpenAI retry batch failed')
                time.sleep(120)
            else:raise RuntimeError('OpenAI retry timeout')
        else:
            client=genai.Client(api_key=os.environ['GEMINI_API_KEY'])
            if 'id' not in state:
                uploaded=client.files.upload(file=requestfile,config=types.UploadFileConfig(mime_type='jsonl'))
                model=json.loads((E/'judges/request_metadata.json').read_text())['gemini_model']
                b=client.batches.create(model=model,src=uploaded.name)
                state={'id':b.name};statefile.write_text(json.dumps(state))
            for poll in range(690):
                b=client.batches.get(name=state['id'])
                if 'SUCCEEDED' in str(b.state):
                    client.files.download(file=b.dest.file_name,destination=resultfile);break
                if any(x in str(b.state) for x in ['FAILED','CANCELLED','EXPIRED']):raise RuntimeError('Gemini retry batch failed')
                time.sleep(120)
            else:raise RuntimeError('Gemini retry timeout')
        originals={r[ident]:r for r in map(json.loads,combined.read_text().splitlines())}
        replacements={r[ident]:r for r in map(json.loads,resultfile.read_text().splitlines())}
        assert set(replacements).issubset(bad)
        originals.update(replacements)
        combined.write_text(''.join(json.dumps(r)+'\n' for r in originals.values()))
    valid,errors=loader(combined,manifest)
    assert set(valid)==set(manifest) and not errors, 'Judge decisions remain invalid after retries'
# Both providers submit and progress independently, preventing serial batch delays.
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(retry_provider,p,l) for p,l in [('openai',load_openai),('gemini',load_gemini)]]
    for job in jobs:job.result()
run(S/'merge_judge_results.py','--evaluation_dir',E/'deterministic','--batch_dir',E/'judges','--output_dir',E/'final','--result_suffix','_combined')
run(S/'compute_open_ended_statistics.py','--classified_dir',E/'final','--output_dir',E/'statistics','--resamples',10000,'--seed',42)
run(H/'write_32b_results.py')
print('[PASS] Final semantic metrics and both Markdown reports updated.',flush=True)
