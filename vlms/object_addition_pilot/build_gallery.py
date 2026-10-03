"""Build a local, self-contained review gallery; does not modify image assets."""
import base64, hashlib, html, json
from pathlib import Path
from PIL import Image
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text())
rows=[]
for s in m['samples']:
 q=s['question_id']; original=Path(s['original_path']); edited=P/'output/imagegen'/f'{q}_edited.png'
 s['edited_path']=str(edited)
 panels=[]
 for label,path in [('Original',original),('Edited',edited)]:
  if path.exists():
   with Image.open(path) as im: size=im.size
   if label=='Edited': s['edited_size_wh']=list(size); s['edited_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
   else: s['original_size_wh']=list(size)
   src='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
   panels.append(f'<figure><figcaption>{label} ({size[0]} × {size[1]})</figcaption><img src="{src}"></figure>')
  else: panels.append('<figure><figcaption>Pending</figcaption></figure>')
 review=s.get('visual_review',{}); review_html='<p><b>'+html.escape(review.get('status','pending').upper())+'</b>: '+html.escape(review.get('note','Not yet reviewed'))+'</p>'
 rows.append(f'<section><h2>{html.escape(q)} — add {html.escape(s["references"]["misleading_ungroundable"])}</h2><p>{html.escape(s["question"])} <b>Answer to preserve: {html.escape(s["references"]["correct_answer"])}</b></p><div class="pair">'+''.join(panels)+f'</div>{review_html}<details><summary>Exact edit prompt</summary><pre>{html.escape(s["prompt"])}</pre></details></section>')
n=sum(Path(s['edited_path']).exists() for s in m['samples'])
m['completed_images']=n; m['status']=m.get('status','generated_pending_review') if n==5 else 'generation_in_progress'
(P/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(P/'gallery.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Object addition pilot</title><style>body{font:16px system-ui;max-width:1300px;margin:auto;padding:24px;background:#fafafa}section{background:white;padding:20px;margin:24px 0;border:1px solid #ddd}.pair{display:flex;gap:20px}figure{flex:1;margin:0;min-width:0}img{width:100%;max-height:650px;object-fit:contain;object-position:top}figcaption{font-weight:bold;margin-bottom:8px}pre{white-space:pre-wrap}@media(max-width:700px){.pair{display:block}}</style><h1>Automatic object addition: five-image pilot</h1><p>OpenAI gpt-image-2 · medium quality · automatic placement · no hand-drawn masks. Originals and API outputs shown without retouching. This is a quality check, not a behavioral experiment.</p>'+''.join(rows)+'</html>')
print(f'Gallery saved; {n}/5 edits present.')
