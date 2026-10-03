"""Render existing images with original MCQ metadata; no inference or image retouching."""
import base64,html,json,textwrap
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text())
source=P.parent/'format_replication_20260921/outputs/full/Qwen__Qwen3-VL-8B-Instruct/mcq_notext.jsonl'
records={str(r['question_id']):r for r in map(json.loads,source.read_text().splitlines())}
V=P/'validation';V.mkdir(exist_ok=True)
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(fontpath,25);small=ImageFont.truetype(fontpath,21);title=ImageFont.truetype(fontpath,30)
pages=[];sections=[]
for i,s in enumerate(m['samples'],1):
 q=s['question_id'];r=records[q];assert r['question']==s['question']
 assert r['options'][r['correct_answer']]==s['references']['correct_answer']
 s['mcq_options']=r['options'];s['mcq_correct_letter']=r['correct_answer'];s['mcq_option_source']=str(source)
 word=s['references']['misleading_ungroundable'];options='    '.join(f'{k}. {v}' for k,v in r['options'].items())
 key=f"Original correct answer: {r['correct_answer']}. {r['options'][r['correct_answer']]}    |    Object added: {word}"
 originals=[Path(s['original_path']),Path(s['edited_path'])]
 page=Image.new('RGB',(1600,1180),'white');draw=ImageDraw.Draw(page)
 draw.text((35,22),f'{i}/5  |  Question {q}  |  Add: {word}',font=title,fill='black')
 y=72
 for line in textwrap.wrap(s['question'],width=100):draw.text((35,y),line,font=font,fill='black');y+=34
 draw.text((35,y+5),options,font=font,fill='black');draw.text((35,y+48),key,font=small,fill='#174a37')
 top=230
 panels=[]
 for x,label,path in [(35,'ORIGINAL',originals[0]),(825,'GENERATED',originals[1])]:
  draw.text((x,top),label,font=font,fill='black')
  im=Image.open(path).convert('RGB');thumb=ImageOps.contain(im,(740,790))
  page.paste(thumb,(x+(740-thumb.width)//2,top+40))
  src='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
  panels.append(f'<figure><figcaption>{label}</figcaption><a href="{src}" target="_blank"><img src="{src}"></a></figure>')
 draw.text((35,1080),'Check: object recognizable? Original answer preserved? Added object also a valid answer?',font=small,fill='black')
 draw.text((35,1115),'Check: existing objects, framing, and background preserved? Any visible editing artifacts?',font=small,fill='black')
 page.save(V/f'{q}_comparison.png');pages.append(page)
 review=s.get('visual_review',{})
 sections.append(f'''<section data-id="{q}"><h2>{i}/5 — {html.escape(word)} · {q}</h2><h3>{html.escape(s['question'])}</h3><p class="options">{html.escape(options)}</p><p>{html.escape(key)}</p><div class="pair">{''.join(panels)}</div><fieldset><legend>Your validation</legend><label>Decision <select class="decision"><option>Not reviewed</option><option>Accept</option><option>Reject</option><option>Uncertain</option></select></label><p>Check object recognizability, whether the original answer remains valid, whether the new object also answers the question, and unintended changes elsewhere.</p><textarea placeholder="Your notes"></textarea></fieldset><details><summary>Show assistant's earlier assessment</summary><p>{html.escape(review.get('status',''))}: {html.escape(review.get('note',''))}</p></details></section>''')
pages[0].save(V/'five_image_validation.pdf',save_all=True,append_images=pages[1:],resolution=130)
style='body{font:17px system-ui;max-width:1550px;margin:auto;padding:24px;background:#f5f5f5}section{background:white;padding:22px;margin:25px 0;border:1px solid #ccc}h3{font-size:23px}.options{font-size:22px;white-space:pre-wrap}.pair{display:flex;gap:20px}figure{flex:1;margin:0;min-width:0}img{width:100%;height:740px;object-fit:contain;object-position:top}figcaption{font-weight:bold;margin:10px 0}textarea{width:98%;min-height:65px}fieldset{margin:16px 0}button,select{font:inherit;padding:7px}details{margin-top:15px}@media(max-width:750px){.pair{display:block}img{height:auto}}@media print{button,fieldset,details{display:none}section{break-after:page}}'
js='''document.getElementById('export').onclick=()=>{const rows=[...document.querySelectorAll('section')].map(s=>({question_id:s.dataset.id,decision:s.querySelector('select').value,notes:s.querySelector('textarea').value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='object_addition_manual_validation.json';a.click();URL.revokeObjectURL(a.href);};'''
(P/'gallery.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Five-image manual validation</title><style>'+style+'</style><h1>Five-image manual validation</h1><p>Original photograph on the left; generated photograph on the right. Questions and A–D options retain the exact order from the original MCQ run. The correct answer shown is the original dataset label, not a claim that the edit preserves it. No text overlays have been added.</p><p>Click an image to inspect it. Record decisions below and download them before closing this page; entries are not automatically saved.</p><button id="export">Download validation notes (JSON)</button>'+''.join(sections)+'<script>'+js+'</script></html>')
(P/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('Created gallery.html, validation/five_image_validation.pdf, and five comparison PNGs.')
