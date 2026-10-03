"""Pilot 4: gpt-image-2 via the Batch API; forbidden areas shown as a red reference image (no API mask).

Usage:  python run_pilot_refimage_batch.py submit    # build inputs, upload JSONL, create batch
        python run_pilot_refimage_batch.py status
        python run_pilot_refimage_batch.py collect   # download results, check, paste, build gallery
Image 1 = no-text photo to edit; image 2 = same photo with automatic forbidden areas shaded red.
Enforcement is ours: fidelity gate, object localization, forbidden-overlap check, paste into the
no-text and ungrounded-overlay images, and pixel checks.
"""
import base64, hashlib, html, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datasets import load_from_disk
from openai import OpenAI
from edit_utils import (MAX_MEDIAN_DIFF, RATES, api_key, composite, forbidden_mask, locate_object,
                        median_diff, price)

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
TAG = 'refimage_batch'
OUT = P / f'output/{TAG}'; OUT.mkdir(parents=True, exist_ok=True)
MODEL, QUALITY, SIZE = 'gpt-image-2', 'medium', 'auto'
STATE = OUT / 'batch_state.json'
HINT = {'plates': 'a short stack of plain plates', 'vase': 'an empty vase with nothing in it'}
PROMPT = '''Image 1 is the photograph to edit. Image 2 is the same photograph for reference only: its RED-SHADED areas are FORBIDDEN.
Add exactly one realistic "{word}" ({hint}) to image 1. Place it in a natural, clearly visible location that is NOT inside, on, or touching any red-shaded area of image 2. It must be fully visible, not hidden, and large enough to be clearly recognizable, resting naturally on a surface with a natural contact shadow.
Reason: the question "{question}" must still have exactly one correct answer, "{answer}"; the added {word} must not be a possible answer to it.
Do not remove, cover, move, or change any existing object or person. Preserve the camera framing, composition, and every other detail of image 1 exactly; change nothing except adding the {word} and its shadow. Match the perspective, lighting, color grading, grain, and photographic quality. Return only the edited image 1: no red shading, no text, labels, borders, or watermarks. Do not restyle, crop, or zoom.'''


def data_url(image):
    buf = io.BytesIO(); image.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def load_inputs():
    ds = load_from_disk(str(ROOT / 'vlms/activation_patching/hf_dataset_GUIC_cleaned/AHAAM__GUIC'))
    gqa = json.load(open(ROOT / 'data_filteration/gqa/passed_questions.json'))
    items = []
    for s in json.loads((P / 'manifest.json').read_text())['samples']:
        q = s['question_id']; sample = ds[int(s['dataset_index'])]
        notext = sample['notext']['image'].convert('RGB'); W, H = notext.size
        forbid, boxes = forbidden_mask(sample, gqa[q], W, H)
        red = Image.new('RGB', (W, H), (230, 20, 20))
        ref = Image.composite(Image.blend(notext, red, .6), notext, Image.fromarray((forbid * 255).astype(np.uint8)))
        items.append(dict(q=q, s=s, sample=sample, notext=notext, forbid=forbid, boxes=boxes, ref=ref,
                          word=s['references']['misleading_ungroundable']))
    return items


def submit():
    if STATE.exists(): sys.exit(f'batch already submitted: {json.loads(STATE.read_text())["batch_id"]}')
    lines = []
    for it in load_inputs():
        it['ref'].save(OUT / f"{it['q']}_reference.png")
        prompt = PROMPT.format(word=it['word'], hint=HINT.get(it['word'], f"one single {it['word']}"),
                               question=it['s']['question'], answer=it['s']['references']['correct_answer'])
        (OUT / f"{it['q']}_prompt.txt").write_text(prompt + '\n')
        lines.append(json.dumps({'custom_id': it['q'], 'method': 'POST', 'url': '/v1/images/edits',
                                 'body': {'model': MODEL, 'prompt': prompt, 'quality': QUALITY, 'size': SIZE, 'n': 1,
                                          'images': [{'image_url': data_url(it['notext'])}, {'image_url': data_url(it['ref'])}]}}))
    jsonl = OUT / 'batch_input.jsonl'; jsonl.write_text('\n'.join(lines) + '\n')
    client = OpenAI(api_key=api_key())
    upload = client.files.create(file=open(jsonl, 'rb'), purpose='batch')
    batch = client.batches.create(input_file_id=upload.id, endpoint='/v1/images/edits', completion_window='24h',
                                  metadata={'purpose': 'object addition pilot 4'})
    STATE.write_text(json.dumps({'batch_id': batch.id, 'input_file_id': upload.id, 'requests': len(lines)}, indent=2) + '\n')
    print(f'[SUBMITTED] batch={batch.id} requests={len(lines)} status={batch.status}')


def status():
    b = OpenAI(api_key=api_key()).batches.retrieve(json.loads(STATE.read_text())['batch_id'])
    print(json.dumps({'status': b.status, 'counts': b.request_counts.model_dump() if b.request_counts else None,
                      'output_file_id': b.output_file_id, 'error_file_id': b.error_file_id}))
    return b


def collect():
    client = OpenAI(api_key=api_key()); state = json.loads(STATE.read_text())
    b = client.batches.retrieve(state['batch_id'])
    if b.status != 'completed': sys.exit(f'batch not completed: {b.status}')
    results = {}
    if b.output_file_id:
        for line in client.files.content(b.output_file_id).text.splitlines():
            r = json.loads(line); results[r['custom_id']] = r
    if b.error_file_id:
        (OUT / 'batch_errors.jsonl').write_text(client.files.content(b.error_file_id).text)
    rates = RATES[(MODEL, 'batch')]; records = []
    for it in load_inputs():
        q = it['q']; notext = it['notext']; W, H = notext.size
        rec = {'question_id': q, 'question': it['s']['question'], 'word': it['word'],
               'correct_answer': it['s']['references']['correct_answer'], 'prompt': (OUT / f'{q}_prompt.txt').read_text().strip(),
               'forbidden_boxes': [[b_[0]] + [round(v, 1) for v in b_[1:]] for b_ in it['boxes']],
               'forbidden_fraction': round(float(it['forbid'].mean()), 3)}
        r = results.get(q)
        if not r or r['response']['status_code'] != 200:
            rec.update(status='reject', reason=f'request failed: {json.dumps(r)[:300] if r else "missing"}',
                       usage={}, cost_usd=price({}, rates)); records.append(rec); continue
        body = r['response']['body']; raw = OUT / f'{q}_raw.png'
        raw.write_bytes(base64.b64decode(body['data'][0]['b64_json']))
        rec.update(request_id=r['response'].get('request_id'), usage=body.get('usage', {}),
                   cost_usd=price(body.get('usage', {}), rates), raw_size_wh=list(Image.open(raw).size))
        edited = Image.open(raw).convert('RGB').resize((W, H), Image.LANCZOS)
        med, diff = median_diff(notext, edited); rec['median_global_diff'] = round(med, 2)
        if med > MAX_MEDIAN_DIFF:
            rec.update(status='reject', reason=f'not a faithful edit: median pixel difference {med:.1f} > {MAX_MEDIAN_DIFF}')
            records.append(rec); continue
        region = locate_object(diff)
        if region is None:
            rec.update(status='reject', reason='no added object detected'); records.append(rec); continue
        ys, xs = np.where(region); overlap = int((region & it['forbid']).sum())
        rec.update(object_box_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                   object_region_fraction=round(float(region.mean()), 4), forbidden_overlap_px=overlap)
        overlay = it['sample']['misleading_ungroundable']['cleaned_image'].convert('RGB'); outs = {}
        for name, base in (('notext_plus_object', notext), ('ungrounded_overlay_plus_object', overlay)):
            img = composite(base, edited, region); path = OUT / f'{q}_{name}.png'; img.save(path)
            changed = np.any(np.asarray(img) != np.asarray(base), 2)
            outs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'changed_px': int(changed.sum()), 'changed_outside_region_px': int((changed & ~region).sum())}
        tx0, ty0, tx1, ty1 = [int(v) for v in it['sample']['misleading_ungroundable']['bbox']]
        tb = np.zeros((H, W), bool); tb[ty0:ty1 + 1, tx0:tx1 + 1] = True
        pair = np.any(np.asarray(Image.open(outs['notext_plus_object']['path'])) !=
                      np.asarray(Image.open(outs['ungrounded_overlay_plus_object']['path'])), 2)
        ok = overlap == 0 and all(o['changed_outside_region_px'] == 0 for o in outs.values())
        rec.update(outputs=outs, pair_diff_outside_text_box_px=int((pair & ~tb).sum()),
                   status='pass_auto_checks' if ok else 'reject',
                   reason='' if overlap == 0 else f'object touches forbidden region ({overlap} px)')
        records.append(rec)
    total = {k: sum(r['cost_usd'][k] for r in records) for k in ('text_input', 'image_input', 'image_output', 'total')}
    (P / f'manifest_{TAG}.json').write_text(json.dumps({
        'purpose': 'pilot 4: gpt-image-2 batch; forbidden areas as red reference image; generic prompt; check-and-paste',
        'model': MODEL, 'quality': QUALITY, 'size': SIZE, 'pricing': 'batch', 'rates_usd_per_1m_tokens': rates,
        'batch': state, 'cost_usd_total': total, 'samples': records}, indent=2) + '\n')
    gallery(records, total)
    print(json.dumps({'cost_usd_total': total, 'results': {r['question_id']: {k: r.get(k) for k in (
        'status', 'reason', 'median_global_diff', 'object_box_xyxy', 'forbidden_overlap_px', 'pair_diff_outside_text_box_px')}
        for r in records}}, indent=1))


def gallery(records, total):
    val = P / f'validation_{TAG}'; val.mkdir(exist_ok=True)
    fp = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font, small = ImageFont.truetype(fp, 22), ImageFont.truetype(fp, 18); sections = []
    for r in records:
        q = r['question_id']
        panels = [('Original', OUT / f'{q}_reference.png'), ('Reference sent (red = forbidden)', OUT / f'{q}_reference.png'),
                  ('Raw gpt-image-2 output', OUT / f'{q}_raw.png')]
        for name in ('notext_plus_object', 'ungrounded_overlay_plus_object'):
            if 'outputs' in r: panels.append((name.replace('_', ' '), Path(r['outputs'][name]['path'])))
        imgs = []
        for label, path in panels:
            if not path.exists(): continue
            im = Image.open(path).convert('RGB')
            if label == 'Original': im = Image.open(P / 'output/imagegen' / f'{q}_original.png').convert('RGB')
            if label.startswith('Reference') and 'object_box_xyxy' in r:
                im = im.copy(); ImageDraw.Draw(im).rectangle(r['object_box_xyxy'], outline=(0, 255, 0), width=3)
            imgs.append((label, im))
        page = Image.new('RGB', (2400, 640), 'white'); d = ImageDraw.Draw(page)
        d.text((20, 12), f"{q} | add: {r['word']} | Q: {r['question']} | keep: {r['correct_answer']} | {r['status']} "
                         f"{r.get('reason', '')} | ${r['cost_usd']['total']:.4f}", font=font, fill='black')
        figs = []
        for j, (label, im) in enumerate(imgs):
            x = 20 + j * 475; d.text((x, 60), label, font=small, fill='black')
            page.paste(ImageOps.contain(im, (460, 540)), (x, 90))
            b = io.BytesIO(); im.save(b, 'PNG'); src = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
            figs.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{src}" target="_blank"><img src="{src}"></a></figure>')
        page.save(val / f'{q}_comparison.png')
        sections.append(f'<section><h2>{q} — add {html.escape(r["word"])}: {html.escape(r["status"])} {html.escape(r.get("reason", ""))}</h2>'
                        f'<p>{html.escape(r["question"])} Keep answer: <b>{html.escape(r["correct_answer"])}</b>. Cost ${r["cost_usd"]["total"]:.4f}</p>'
                        f'<div class="pair">{"".join(figs)}</div><details><summary>Prompt and checks</summary><pre>'
                        f'{html.escape(json.dumps({k: r.get(k) for k in ("prompt", "forbidden_boxes", "median_global_diff", "object_box_xyxy", "forbidden_overlap_px", "outputs", "pair_diff_outside_text_box_px")}, indent=1))}</pre></details></section>')
    style = ('body{font:16px system-ui;max-width:2400px;margin:auto;padding:20px;background:#f5f5f5}section{background:white;padding:18px;margin:20px 0;border:1px solid #ccc}'
             '.pair{display:flex;gap:12px}figure{flex:1;margin:0;min-width:0}img{width:100%;height:420px;object-fit:contain;object-position:top}'
             'figcaption{font-weight:bold;margin:6px 0}pre{white-space:pre-wrap;font-size:13px}@media(max-width:750px){.pair{display:block}img{height:auto}}')
    (P / f'gallery_{TAG}.html').write_text(f'<!doctype html><html><meta charset="utf-8"><title>Object addition pilot 4</title><style>{style}</style>'
                                          f'<h1>Pilot 4: gpt-image-2 batch + red reference image + paste</h1><p>Total measured cost ${total["total"]:.4f} (batch rates).</p>'
                                          + ''.join(sections) + '</html>')


if __name__ == '__main__':
    {'submit': submit, 'status': status, 'collect': collect}[sys.argv[1]]()
