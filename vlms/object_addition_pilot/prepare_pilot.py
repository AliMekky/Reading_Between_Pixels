"""Select pilot by metadata only; no behavioral outcomes used."""
import json, random, hashlib
from pathlib import Path
from datasets import load_from_disk
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
paths=sorted((ROOT/'vlms/qwen3_vl_generation_causal/relevance_intervention/outputs/input_comparison_305').glob('shard_*/samples/*_clean_image.json'))
random.Random(42).shuffle(paths)
allowed={'book','hat','vase','pan','plates'}
selected=[]; seen=set()
for p in paths:
 d=json.loads(p.read_text()); word=d['references']['misleading_ungroundable']
 if word not in allowed or word in seen: continue
 seen.add(word)
 selected.append({k:d[k] for k in ('question_id','dataset_index','question','references')})
 if len(selected)==5: break
assert len(selected)==5
ds=load_from_disk(str(ROOT/'vlms/activation_patching/hf_dataset_GUIC_cleaned/AHAAM__GUIC'))
for d in selected:
 p=OUT/'output/imagegen'/f"{d['question_id']}_original.png"
 ds[d['dataset_index']]['notext']['image'].convert('RGB').save(p)
 d['original_path']=str(p); d['original_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 word=d['references']['misleading_ungroundable']; answer=d['references']['correct_answer']
 d['prompt']=f'''Use case: precise-object-edit. Input image is the edit target: an existing dataset photograph. Add a realistic physical object named "{word}" to this image (a small group if the noun is plural). Choose its position automatically: a natural, clearly visible location away from the subject/relation queried below. Match the original perspective, lighting, texture, resolution, and photographic quality.\nQuestion that MUST retain its original answer: {d['question']}\nOriginal answer that MUST remain correct: {answer}.\nThe inserted {word} MUST NOT become another valid answer to that question. Do not place it in the queried spatial relation, on the queried surface, or worn/held by the queried person. Preserve all existing objects, people, poses, spatial relations, camera framing, and background. Make only the minimum local addition and its natural contact shadow. Do not beautify, restyle, crop, zoom, or add text, labels, borders, or watermarks. Return one edited photograph only, with the same aspect ratio. If necessary use a different unqueried surface or background location. No text overlay is present or should be generated.'''
manifest={'selection':'seed42 shuffled shared305; first distinct nouns in predefined small-object set book,hat,vase,pan,plates; no outcome-based selection','purpose':'five-image automatic placement quality pilot, not behavioral evidence','mode':'OpenAI built-in image_gen; exact model identifier not exposed','samples':selected}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for d in selected: print(d['question_id'],d['question'],d['references']['misleading_ungroundable'],d['original_path'])
