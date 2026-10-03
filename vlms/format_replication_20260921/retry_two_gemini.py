import json,os,sys
from pathlib import Path
from google import genai
H=Path(__file__).resolve().parent;E=H/'evaluation_32b';sys.path.insert(0,str(H.parent/'open_ended_evaluation/main_files'))
from merge_judge_results import load_manifest,load_gemini,parse_decision
manifest=load_manifest(E/'judges/judge_manifest.jsonl');combined=E/'judges/gemini_results_combined.jsonl'
valid,errors=load_gemini(combined,manifest)
assert set(errors).issubset({'oej-b19588771e5f30ba57710994','oej-305b495186cf15097b14e5f7'})
folder=E/'gemini_final_two';folder.mkdir(exist_ok=True)
requests={r['key']:r['request'] for r in map(json.loads,(E/'judges/gemini_requests.jsonl').read_text().splitlines())}
client=genai.Client(api_key=os.environ['GEMINI_API_KEY'])
for key in errors:
 dest=folder/(key+'.json');req=requests[key]
 schema=req['generationConfig']['responseJsonSchema'];schema['properties']['reason']['enum']=['semantic match','no match','uncertain']
 config=dict(system_instruction=req['systemInstruction']['parts'][0]['text']+' Use one of the permitted brief reason strings; do not repeat words.',temperature=0,max_output_tokens=512,response_mime_type='application/json',response_json_schema=schema,thinking_config={'thinking_budget':0})
 if dest.exists():r=json.loads(dest.read_text())
 else:
  response=client.models.generate_content(model='gemini-3.5-flash',contents=req['contents'],config=config)
  r={'key':key,'response':response.model_dump(mode='json'),'retry_provenance':'synchronous; unchanged classifier and reference order; reason enum prevents repeated-rationale loop'}
  dest.write_text(json.dumps(r)+'\n')
 parsed,bad=load_gemini(dest,manifest)
 assert not bad and key in parsed
 print(key,parsed[key]['mapped_category'],flush=True)
originals={r['key']:r for r in map(json.loads,combined.read_text().splitlines())}
for f in folder.glob('oej-*.json'):
 r=json.loads(f.read_text());originals[r['key']]=r
combined.write_text(''.join(json.dumps(r)+'\n' for r in originals.values()))
valid,bad=load_gemini(combined,manifest);assert len(valid)==973 and not bad
print('PASS: 973 valid Gemini judgments')
