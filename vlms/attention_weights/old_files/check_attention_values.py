import numpy as np
from pathlib import Path
import json

npz_root = Path("./llava-next_attentions")
qid = "122313"
variants = ["notext", "correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"]

def load_attn(v):
    p = npz_root / v / v / qid / "gen_attn_gen_token.npz"
    d = np.load(p, allow_pickle=True)
    attn = d["attn"].astype(np.float32)
    meta = json.loads(str(d["meta"]))
    return attn, meta, str(p)

attns = {}
metas = {}
paths = {}

for v in variants:
    a, m, p = load_attn(v)
    attns[v] = a
    metas[v] = m
    paths[v] = p
    print(v, "path=", p, "shape=", a.shape, "gen_token=", m.get("generated_token_text"))

# pairwise differences
base = "notext"
for v in variants[1:]:
    diff = np.max(np.abs(attns[base] - attns[v]))
    print(f"max|attn({base})-attn({v})| = {diff:.6g}")