"""Pilot 2: same five images and API settings as pilot 1, with prompts_v2 (explicit placement).

Calls the OpenAI Images edit endpoint directly so that per-call token usage and request IDs are
saved, then prices each call with the official gpt-image-2 token rates. Existing outputs are
skipped, so the script can be re-run to resume or to rebuild the review pages without new calls.
"""
import base64, hashlib, html, json, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from openai import OpenAI

P = Path(__file__).resolve().parent
OUT = P / 'output/imagegen_v2'; OUT.mkdir(parents=True, exist_ok=True)
VAL = P / 'validation_v2'; VAL.mkdir(exist_ok=True)
MODEL, QUALITY, SIZE = 'gpt-image-2', 'medium', 'auto'
# USD per 1M tokens, https://developers.openai.com/api/docs/pricing (checked 2026-09-28).
RATES = {'text_input': 5.00, 'image_input': 8.00, 'image_output': 30.00}


def api_key():
    for line in Path('/l/users/ali.mekky/.secrets/open_ended_eval.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('OPENAI_API_KEY='):
            value = line.split('=', 1)[1].strip().strip('"\'')
            if value: return value
    raise RuntimeError('OPENAI_API_KEY not found')


def price(usage):
    details = usage.get('input_tokens_details') or {}
    text_in, image_in = details.get('text_tokens', 0), details.get('image_tokens', 0)
    out = usage.get('output_tokens', 0)
    cost = {'text_input': text_in * RATES['text_input'] / 1e6,
            'image_input': image_in * RATES['image_input'] / 1e6,
            'image_output': out * RATES['image_output'] / 1e6}
    cost['total'] = sum(cost.values())
    return cost


v1 = json.loads((P / 'manifest.json').read_text())
records = []
client = None
for s in v1['samples']:
    q = s['question_id']; edited = OUT / f'{q}_edited.png'; meta_path = OUT / f'{q}_call.json'
    prompt = (P / 'prompts_v2' / f'{q}.txt').read_text().strip()
    if not (edited.exists() and meta_path.exists()):
        client = client or OpenAI(api_key=api_key())
        start = time.time()
        with open(s['original_path'], 'rb') as image:
            result = client.images.edit(model=MODEL, image=image, prompt=prompt, quality=QUALITY, size=SIZE, n=1)
        edited.write_bytes(base64.b64decode(result.data[0].b64_json))
        usage = result.usage.model_dump() if result.usage else {}
        meta_path.write_text(json.dumps({'request_id': getattr(result, '_request_id', None), 'usage': usage,
                                         'elapsed_seconds': round(time.time() - start, 1)}, indent=2) + '\n')
        print(f'[CALL] {q} saved in {time.time() - start:.1f}s usage={usage}', flush=True)
    meta = json.loads(meta_path.read_text())
    with Image.open(edited) as im: size = list(im.size)
    records.append({**{k: s[k] for k in ('question_id', 'question', 'references', 'original_path', 'original_sha256',
                                         'mcq_options', 'mcq_correct_letter')},
                    'v1_edited_path': s['edited_path'], 'v1_review': s.get('visual_review'),
                    'prompt': prompt, 'edited_path': str(edited), 'edited_size_wh': size,
                    'edited_sha256': hashlib.sha256(edited.read_bytes()).hexdigest(),
                    **meta, 'cost_usd': price(meta['usage'])})

total = {k: sum(r['cost_usd'][k] for r in records) for k in ('text_input', 'image_input', 'image_output', 'total')}
tokens = {'text_input': sum((r['usage'].get('input_tokens_details') or {}).get('text_tokens', 0) for r in records),
          'image_input': sum((r['usage'].get('input_tokens_details') or {}).get('image_tokens', 0) for r in records),
          'output': sum(r['usage'].get('output_tokens', 0) for r in records)}
manifest = {'purpose': 'pilot 2: explicit-placement prompts on the pilot-1 images; placement-quality check, not behavioral evidence',
            'model': MODEL, 'quality': QUALITY, 'size': SIZE, 'n_per_image': 1, 'masks': None,
            'rates_usd_per_1m_tokens': RATES, 'rates_source': 'https://developers.openai.com/api/docs/pricing (2026-09-28)',
            'tokens_total': tokens, 'cost_usd_total': total, 'samples': records}
(P / 'manifest_v2.json').write_text(json.dumps(manifest, indent=2) + '\n')

# Review pages: original | pilot 1 | pilot 2.
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font, small, title = (ImageFont.truetype(font_path, n) for n in (25, 21, 30))
pages, sections = [], []
for i, r in enumerate(records, 1):
    q, word = r['question_id'], r['references']['misleading_ungroundable']
    options = '    '.join(f'{k}. {v}' for k, v in r['mcq_options'].items())
    key = f"Correct answer to preserve: {r['mcq_correct_letter']}. {r['references']['correct_answer']}    |    Object added: {word}"
    panels_img = [('ORIGINAL', r['original_path']), ('PILOT 1 (v1 prompt)', r['v1_edited_path']), ('PILOT 2 (v2 prompt)', r['edited_path'])]
    page = Image.new('RGB', (2340, 1060), 'white'); draw = ImageDraw.Draw(page)
    draw.text((35, 22), f'{i}/5  |  Question {q}  |  Add: {word}', font=title, fill='black')
    draw.text((35, 72), r['question'], font=font, fill='black')
    draw.text((35, 112), options, font=font, fill='black'); draw.text((35, 155), key, font=small, fill='#174a37')
    figures = []
    for j, (label, path) in enumerate(panels_img):
        x = 35 + j * 765
        draw.text((x, 210), label, font=font, fill='black')
        thumb = ImageOps.contain(Image.open(path).convert('RGB'), (740, 780))
        page.paste(thumb, (x + (740 - thumb.width) // 2, 250))
        src = 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()
        figures.append(f'<figure><figcaption>{label}</figcaption><a href="{src}" target="_blank"><img src="{src}"></a></figure>')
    page.save(VAL / f'{q}_comparison.png'); pages.append(page)
    v1_note = (r['v1_review'] or {}).get('note', '')
    sections.append(f'<section><h2>{i}/5 — {html.escape(word)} · {q}</h2><h3>{html.escape(r["question"])}</h3>'
                    f'<p class="options">{html.escape(options)}</p><p>{html.escape(key)}</p><div class="pair">{"".join(figures)}</div>'
                    f'<p><b>Pilot 1 review:</b> {html.escape(v1_note)}</p><p><b>Pilot 2 cost:</b> ${r["cost_usd"]["total"]:.4f}</p>'
                    f'<details><summary>Pilot 2 prompt</summary><pre>{html.escape(r["prompt"])}</pre></details></section>')
pages[0].save(VAL / 'five_image_validation_v2.pdf', save_all=True, append_images=pages[1:], resolution=130)
style = ('body{font:17px system-ui;max-width:2300px;margin:auto;padding:24px;background:#f5f5f5}section{background:white;padding:22px;'
         'margin:25px 0;border:1px solid #ccc}.options{white-space:pre-wrap}.pair{display:flex;gap:16px}figure{flex:1;margin:0;min-width:0}'
         'img{width:100%;height:620px;object-fit:contain;object-position:top}figcaption{font-weight:bold;margin:10px 0}pre{white-space:pre-wrap}'
         '@media(max-width:750px){.pair{display:block}img{height:auto}}')
(P / 'gallery_v2.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Object addition pilot 2</title><style>' + style +
                                   f'</style><h1>Object addition pilot 2: explicit placement prompts</h1><p>{MODEL} · {QUALITY} quality · size {SIZE} · '
                                   f'no masks. Total measured cost: ${total["total"]:.4f}.</p>' + ''.join(sections) + '</html>')
print(json.dumps({'tokens_total': tokens, 'cost_usd_total': total,
                  'per_image': {r['question_id']: round(r['cost_usd']['total'], 4) for r in records}}, indent=2))
