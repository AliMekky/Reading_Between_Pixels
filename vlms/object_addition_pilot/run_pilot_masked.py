"""Pilot 3: masked edits (argv[1]: gpt-image-1-mini | gpt-image-2) with automatic FORBIDDEN-region masks, then check-and-paste.

1. Forbidden region (automatic): scene-graph boxes of every object named in the question's
   semantic program or its answer annotations (queried object, queried surface, answer object),
   all people, and the text boxes of all four overlays; each dilated by a margin.
2. Edit: the forbidden region is protected in the mask (opaque); everything else is editable.
   The prompt is generic (no hand-picked placement), so placement is chosen by the editor.
3. Localize the added object by differencing the resized output with the original.
4. Reject if the object region touches the forbidden region.
5. Paste only the object region into (a) the no-text image and (b) the ungrounded-word overlay;
   verify that no pixel outside the pasted region changed.
"""
import base64, hashlib, html, io, json, re, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy import ndimage
from datasets import load_from_disk
from openai import OpenAI

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
import sys
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-image-1-mini'
QUALITY, SIZE = 'medium', 'auto'
# USD / 1M tokens, https://developers.openai.com/api/docs/pricing (2026-09-28)
RATES = {'gpt-image-1-mini': {'text_input': 2.00, 'image_input': 2.50, 'image_output': 8.00},
         'gpt-image-2': {'text_input': 5.00, 'image_input': 8.00, 'image_output': 30.00}}[MODEL]
TAG = {'gpt-image-1-mini': 'mini_masked', 'gpt-image-2': 'gpt2_masked'}[MODEL]
OUT = P / f'output/{TAG}'; OUT.mkdir(parents=True, exist_ok=True)
VAL = P / f'validation_{TAG}'; VAL.mkdir(exist_ok=True)
PEOPLE = {'man', 'woman', 'person', 'people', 'girl', 'boy', 'lady', 'child', 'kid', 'guy', 'men', 'women',
          'player', 'skier', 'surfer', 'skateboarder', 'baby', 'gentleman', 'children', 'rider', 'chef'}
MARGIN = 0.03   # dilation, fraction of the longer image side
MAX_MEDIAN_DIFF = 10  # fidelity gate: a faithful edit leaves most pixels nearly unchanged (gpt-image-2 pilot 2: 1.3-5.7)
VARIANTS = ('correct_answer', 'misleading_groundable', 'misleading_ungroundable', 'irrelevant_word')
PROMPT = '''Add exactly one realistic "{word}" to this photograph ({hint}). Only the transparent area of the mask may be edited; the opaque area is protected and must stay exactly as it is. Place the {word} in a natural, clearly visible location inside the editable area, fully visible and large enough to be clearly recognizable, resting naturally on a surface with a natural contact shadow. Do not remove, cover, or change any existing object or person. Match the perspective, lighting, color grading, grain, and photographic quality. Do not add text, labels, borders, or watermarks. Do not restyle, crop, or zoom.'''
HINT = {'plates': 'a short stack of plain plates', 'vase': 'an empty vase with nothing in it'}


def api_key():
    for line in Path('/l/users/ali.mekky/.secrets/open_ended_eval.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('OPENAI_API_KEY='):
            value = line.split('=', 1)[1].strip().strip('"\'')
            if value: return value
    raise RuntimeError('OPENAI_API_KEY not found')


def price(usage):
    d = usage.get('input_tokens_details') or {}
    c = {'text_input': d.get('text_tokens', 0) * RATES['text_input'] / 1e6,
         'image_input': d.get('image_tokens', 0) * RATES['image_input'] / 1e6,
         'image_output': usage.get('output_tokens', 0) * RATES['image_output'] / 1e6}
    c['total'] = sum(c.values()); return c


def forbidden_mask(sample, gqa, W, H):
    sg = gqa['scene_graph']; sx, sy = W / sg['width'], H / sg['height']
    ids = set(re.findall(r'\((\d+)\)', gqa['semanticStr']))
    for part in gqa['annotations'].values(): ids |= set(part.values())
    boxes = []  # (label, x0, y0, x1, y1) in image pixels
    for oid, o in sg['objects'].items():
        if oid in ids or o['name'].lower() in PEOPLE:
            boxes.append((o['name'], o['x'] * sx, o['y'] * sy, (o['x'] + o['w']) * sx, (o['y'] + o['h']) * sy))
    for v in VARIANTS:
        x0, y0, x1, y1 = sample[v]['bbox']; boxes.append((f'text:{v}', x0, y0, x1, y1))
    pad = MARGIN * max(W, H); mask = np.zeros((H, W), bool)
    for _, x0, y0, x1, y1 in boxes:
        mask[max(0, int(y0 - pad)):min(H, int(y1 + pad) + 1), max(0, int(x0 - pad)):min(W, int(x1 + pad) + 1)] = True
    return mask, boxes


def locate_object(original, edited):
    a = np.asarray(original.filter(ImageFilter.GaussianBlur(2)), float)
    b = np.asarray(edited.filter(ImageFilter.GaussianBlur(2)), float)
    d = np.abs(a - b).mean(2)
    lab, n = ndimage.label(ndimage.binary_closing(d > max(np.quantile(d, .99), 25), iterations=3))
    if n == 0: return None, d
    sizes = ndimage.sum(np.ones_like(d), lab, range(1, n + 1))
    blob = lab == int(np.argmax(sizes)) + 1
    # Paste region: blob filled and dilated so edges and contact shadow are kept.
    region = ndimage.binary_dilation(ndimage.binary_fill_holes(blob), iterations=4)
    return region, d


def composite(base, edited, region):
    alpha = Image.fromarray((region * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    alpha = Image.fromarray((np.asarray(alpha) * region).astype(np.uint8))  # feather inward only
    return Image.composite(edited, base, alpha)


ds = load_from_disk(str(ROOT / 'vlms/activation_patching/hf_dataset_GUIC_cleaned/AHAAM__GUIC'))
gqa_all = json.load(open(ROOT / 'data_filteration/gqa/passed_questions.json'))
pilot1 = json.loads((P / 'manifest.json').read_text())
client, records = None, []
for s in pilot1['samples']:
    q = s['question_id']; sample = ds[int(s['dataset_index'])]; word = s['references']['misleading_ungroundable']
    notext = sample['notext']['image'].convert('RGB'); overlay = sample['misleading_ungroundable']['cleaned_image'].convert('RGB')
    W, H = notext.size
    forbid, boxes = forbidden_mask(sample, gqa_all[q], W, H)
    rgba = notext.convert('RGBA'); rgba.putalpha(Image.fromarray(np.where(forbid, 255, 0).astype(np.uint8)))
    mask_path = OUT / f'{q}_mask.png'; rgba.save(mask_path)
    prompt = PROMPT.format(word=word, hint=HINT.get(word, f'one single {word}'))
    raw_path, meta_path = OUT / f'{q}_raw.png', OUT / f'{q}_call.json'
    if not (raw_path.exists() and meta_path.exists()):
        client = client or OpenAI(api_key=api_key())
        buf = io.BytesIO(); notext.save(buf, 'PNG'); buf.name = 'image.png'; buf.seek(0)
        mbuf = open(mask_path, 'rb'); start = time.time()
        result = client.images.edit(model=MODEL, image=buf, mask=mbuf, prompt=prompt, quality=QUALITY, size=SIZE, n=1)
        raw_path.write_bytes(base64.b64decode(result.data[0].b64_json))
        usage = result.usage.model_dump() if result.usage else {}
        meta_path.write_text(json.dumps({'request_id': getattr(result, '_request_id', None), 'usage': usage,
                                         'elapsed_seconds': round(time.time() - start, 1)}, indent=2) + '\n')
        print(f'[CALL] {q} {time.time() - start:.1f}s usage={usage}', flush=True)
    meta = json.loads(meta_path.read_text())
    edited = Image.open(raw_path).convert('RGB').resize((W, H), Image.LANCZOS)
    region, diff = locate_object(notext, edited)
    rec = {'question_id': q, 'question': s['question'], 'word': word, 'correct_answer': s['references']['correct_answer'],
           'prompt': prompt, 'forbidden_boxes': [[b[0]] + [round(v, 1) for v in b[1:]] for b in boxes],
           'forbidden_fraction': round(float(forbid.mean()), 3), 'raw_size_wh': list(Image.open(raw_path).size),
           'median_global_diff': round(float(np.median(diff)), 2), **meta, 'cost_usd': price(meta['usage'])}
    if rec['median_global_diff'] > MAX_MEDIAN_DIFF:
        rec.update(status='reject', reason=f"not a faithful edit: median pixel difference {rec['median_global_diff']} > {MAX_MEDIAN_DIFF}")
        records.append(rec); continue
    if region is None:
        rec.update(status='reject', reason='no added object detected'); records.append(rec); continue
    ys, xs = np.where(region)
    overlap = int((region & forbid).sum())
    rec.update(object_box_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
               object_region_fraction=round(float(region.mean()), 4), forbidden_overlap_px=overlap)
    outs = {}
    for name, base in (('notext_plus_object', notext), ('ungrounded_overlay_plus_object', overlay)):
        img = composite(base, edited, region); path = OUT / f'{q}_{name}.png'; img.save(path)
        changed = np.any(np.asarray(img) != np.asarray(base), 2)
        outs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'changed_px': int(changed.sum()), 'changed_outside_region_px': int((changed & ~region).sum())}
    # The two composites must differ only inside the text box (as the original cleaned pair does).
    tx0, ty0, tx1, ty1 = [int(v) for v in sample['misleading_ungroundable']['bbox']]
    tb = np.zeros((H, W), bool); tb[ty0:ty1 + 1, tx0:tx1 + 1] = True
    pair_diff = np.any(np.asarray(Image.open(outs['notext_plus_object']['path'])) != np.asarray(Image.open(outs['ungrounded_overlay_plus_object']['path'])), 2)
    rec.update(outputs=outs, pair_diff_outside_text_box_px=int((pair_diff & ~tb).sum()),
               status='pass_auto_checks' if overlap == 0 and all(o['changed_outside_region_px'] == 0 for o in outs.values()) else 'reject',
               reason='' if overlap == 0 else f'object touches forbidden region ({overlap} px)')
    records.append(rec)

total = {k: sum(r['cost_usd'][k] for r in records) for k in ('text_input', 'image_input', 'image_output', 'total')}
(P / f'manifest_{TAG}.json').write_text(json.dumps({
    'purpose': f'pilot 3: {MODEL}, automatic forbidden-region masks, generic prompt, check-and-paste',
    'model': MODEL, 'quality': QUALITY, 'size': SIZE, 'margin_fraction': MARGIN, 'rates_usd_per_1m_tokens': RATES,
    'cost_usd_total': total, 'samples': records}, indent=2) + '\n')

# Review page: original | forbidden mask | raw mini output | no-text+object | overlay+object.
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font, small = ImageFont.truetype(font_path, 22), ImageFont.truetype(font_path, 18)
sections, pages = [], []
for r in records:
    q = r['question_id']; sample = ds[int(next(s for s in pilot1['samples'] if s['question_id'] == q)['dataset_index'])]
    notext = sample['notext']['image'].convert('RGB')
    shade = notext.copy(); m = Image.open(OUT / f'{q}_mask.png').getchannel('A')
    shade = Image.composite(Image.blend(notext, Image.new('RGB', notext.size, (220, 30, 30)), .55), notext, m)
    if 'object_box_xyxy' in r: ImageDraw.Draw(shade).rectangle(r['object_box_xyxy'], outline=(0, 255, 0), width=3)
    panels = [('Original', notext), ('Forbidden (red), object (green)', shade),
              ('Raw mini output', Image.open(OUT / f'{q}_raw.png').convert('RGB'))]
    for name in ('notext_plus_object', 'ungrounded_overlay_plus_object'):
        if 'outputs' in r: panels.append((name.replace('_', ' '), Image.open(r['outputs'][name]['path']).convert('RGB')))
    page = Image.new('RGB', (2400, 640), 'white'); d = ImageDraw.Draw(page)
    d.text((20, 12), f"{q} | add: {r['word']} | Q: {r['question']} | keep answer: {r['correct_answer']} | "
                     f"{r['status']} {r.get('reason', '')} | ${r['cost_usd']['total']:.4f}", font=font, fill='black')
    figs = []
    for j, (label, im) in enumerate(panels):
        x = 20 + j * 475; d.text((x, 60), label, font=small, fill='black')
        th = ImageOps.contain(im, (460, 540)); page.paste(th, (x, 90))
        b = io.BytesIO(); im.save(b, 'PNG'); src = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
        figs.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{src}" target="_blank"><img src="{src}"></a></figure>')
    page.save(VAL / f'{q}_comparison.png'); pages.append(page)
    sections.append(f'<section><h2>{q} — add {html.escape(r["word"])}: {html.escape(r["status"])} {html.escape(r.get("reason", ""))}</h2>'
                    f'<p>{html.escape(r["question"])} Keep answer: <b>{html.escape(r["correct_answer"])}</b>. Cost ${r["cost_usd"]["total"]:.4f}</p>'
                    f'<div class="pair">{"".join(figs)}</div><details><summary>Prompt and checks</summary><pre>'
                    f'{html.escape(json.dumps({k: r.get(k) for k in ("prompt", "forbidden_boxes", "object_box_xyxy", "forbidden_overlap_px", "outputs", "pair_diff_outside_text_box_px")}, indent=1))}</pre></details></section>')
style = ('body{font:16px system-ui;max-width:2400px;margin:auto;padding:20px;background:#f5f5f5}section{background:white;padding:18px;margin:20px 0;border:1px solid #ccc}'
         '.pair{display:flex;gap:12px}figure{flex:1;margin:0;min-width:0}img{width:100%;height:420px;object-fit:contain;object-position:top}'
         'figcaption{font-weight:bold;margin:6px 0}pre{white-space:pre-wrap;font-size:13px}@media(max-width:750px){.pair{display:block}img{height:auto}}')
(P / f'gallery_{TAG}.html').write_text(f'<!doctype html><html><meta charset="utf-8"><title>Object addition pilot 3</title><style>{style}</style>'
                                            f'<h1>Pilot 3: {MODEL} + automatic forbidden masks + paste</h1><p>Total measured cost ${total["total"]:.4f}.</p>'
                                            + ''.join(sections) + '</html>')
print(json.dumps({'cost_usd_total': total, 'results': {r['question_id']: {k: r.get(k) for k in (
    'status', 'reason', 'object_box_xyxy', 'forbidden_fraction', 'forbidden_overlap_px', 'pair_diff_outside_text_box_px')} for r in records}}, indent=1))
