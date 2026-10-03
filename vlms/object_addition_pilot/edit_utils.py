"""Shared helpers for the object-addition pilots: forbidden regions, object localization, pasting, cost."""
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

PEOPLE = {'man', 'woman', 'person', 'people', 'girl', 'boy', 'lady', 'child', 'kid', 'guy', 'men', 'women',
          'player', 'skier', 'surfer', 'skateboarder', 'baby', 'gentleman', 'children', 'rider', 'chef'}
VARIANTS = ('correct_answer', 'misleading_groundable', 'misleading_ungroundable', 'irrelevant_word')
MARGIN = 0.03          # forbidden-box dilation, fraction of the longer image side
MAX_MEDIAN_DIFF = 10   # fidelity gate: faithful gpt-image-2 edits in pilot 2 had median diff 1.3-5.7
# USD per 1M tokens, https://developers.openai.com/api/docs/pricing (2026-09-28)
RATES = {('gpt-image-2', 'standard'): {'text_input': 5.00, 'image_input': 8.00, 'image_output': 30.00},
         ('gpt-image-2', 'batch'): {'text_input': 2.50, 'image_input': 4.00, 'image_output': 15.00},
         ('gpt-image-1-mini', 'standard'): {'text_input': 2.00, 'image_input': 2.50, 'image_output': 8.00},
         ('gpt-image-1-mini', 'batch'): {'text_input': 1.00, 'image_input': 1.25, 'image_output': 4.00}}


def api_key():
    for line in Path('/l/users/ali.mekky/.secrets/open_ended_eval.env').read_text().splitlines():
        line = line.strip().removeprefix('export ')
        if line.startswith('OPENAI_API_KEY='):
            value = line.split('=', 1)[1].strip().strip('"\'')
            if value: return value
    raise RuntimeError('OPENAI_API_KEY not found')


def price(usage, rates):
    d = usage.get('input_tokens_details') or {}
    c = {'text_input': d.get('text_tokens', 0) * rates['text_input'] / 1e6,
         'image_input': d.get('image_tokens', 0) * rates['image_input'] / 1e6,
         'image_output': usage.get('output_tokens', 0) * rates['image_output'] / 1e6}
    c['total'] = sum(c.values()); return c


def forbidden_mask(sample, gqa, W, H):
    """Queried/answer objects from the question program, all people, and all four text boxes, dilated."""
    sg = gqa['scene_graph']; sx, sy = W / sg['width'], H / sg['height']
    ids = set(re.findall(r'\((\d+)\)', gqa['semanticStr']))
    for part in gqa['annotations'].values(): ids |= set(part.values())
    boxes = []
    for oid, o in sg['objects'].items():
        if oid in ids or o['name'].lower() in PEOPLE:
            boxes.append((o['name'], o['x'] * sx, o['y'] * sy, (o['x'] + o['w']) * sx, (o['y'] + o['h']) * sy))
    for v in VARIANTS:
        x0, y0, x1, y1 = sample[v]['bbox']; boxes.append((f'text:{v}', x0, y0, x1, y1))
    pad = MARGIN * max(W, H); mask = np.zeros((H, W), bool)
    for _, x0, y0, x1, y1 in boxes:
        mask[max(0, int(y0 - pad)):min(H, int(y1 + pad) + 1), max(0, int(x0 - pad)):min(W, int(x1 + pad) + 1)] = True
    return mask, boxes


def median_diff(original, edited):
    a = np.asarray(original.filter(ImageFilter.GaussianBlur(2)), float)
    b = np.asarray(edited.filter(ImageFilter.GaussianBlur(2)), float)
    d = np.abs(a - b).mean(2); return float(np.median(d)), d


def locate_object(diff):
    lab, n = ndimage.label(ndimage.binary_closing(diff > max(np.quantile(diff, .99), 25), iterations=3))
    if n == 0: return None
    sizes = ndimage.sum(np.ones_like(diff), lab, range(1, n + 1))
    blob = lab == int(np.argmax(sizes)) + 1
    return ndimage.binary_dilation(ndimage.binary_fill_holes(blob), iterations=4)  # keep edges and contact shadow


def composite(base, edited, region):
    alpha = np.asarray(Image.fromarray((region * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)))
    return Image.composite(edited, base, Image.fromarray((alpha * region).astype(np.uint8)))  # feather inward only
