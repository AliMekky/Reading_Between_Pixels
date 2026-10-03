"""Write v2 edit prompts: explicit forbidden region and explicit target location per image.

v1 asked the editor to "choose a natural position away from the queried relation". Because the
inserted word is, by construction, a plausible answer to the question, its most natural position
is the queried surface itself (book on the table, plates on the shelf). v2 names the forbidden
surface concretely, fixes the placement with visual anchors, and specifies a recognizable size.
Slots were written by visual inspection of the five originals; at scale they would come from an
automatic placement planner.
"""
import json
from pathlib import Path
OUT = Path(__file__).resolve().parent
SLOTS = {
    '19358422': dict(
        count='one single closed hardcover book',
        forbidden='the small bedside table with the lamp and bottle (between the beds, left of center), any other table or nightstand, and the beds',
        target='lying flat on the empty wooden floorboards in the lower-right part of the image, between the base of the black standing fan and the white power cord, not touching any furniture',
        size='about the size of a real book at that distance, large enough that its cover and page edges are clearly recognizable'),
    '10560336': dict(
        count='one short stack of three or four plain white plates',
        forbidden='the long shelf across the top of the image (with the glasses, cups and coffee pots), the wall above it, and the lower shelf holding the microwave',
        target='lying flat on the empty grey bar countertop in the foreground, lower-middle of the image, in front of the microwave and well below the top shelf',
        size='normal dinner-plate size at that distance, clearly recognizable as plates'),
    '10720200': dict(
        count='one single straw hat with a dark band',
        forbidden='any person: it must not be worn, held, or touching anyone, and must not be near the older woman in the lower-left',
        target='resting on the wooden dining table in the lower-right, in the empty space between the black camera and the green folded napkin',
        size='normal hat size at that distance, clearly recognizable as a hat'),
    '05580253': dict(
        count='one single EMPTY ceramic vase with nothing in it: no flowers, stems, leaves, or plants',
        forbidden='the white planter holding the flowers, the whole dining table at the left where it stands, and the area around the existing flowers',
        target='standing upright on an empty part of the dark wooden coffee table in the center of the room',
        size='clearly recognizable as a vase, roughly as tall as the lampshade on the side table behind the armchair'),
    '10872518': dict(
        count='one single empty frying pan',
        forbidden='the stove top and every burner: do not place it on, above, or overlapping the stove, and do not move or alter the existing pot',
        target='lying flat on the empty countertop to the right of the stove, between the glass jar and the black speaker, handle pointing right',
        size='similar in width to the existing pot, clearly recognizable as a pan'),
}
TEMPLATE = '''Use case: precise-object-edit. Input image is the edit target: an existing dataset photograph. Add exactly {count} ("{word}") to this image, and nothing else.
PLACEMENT (mandatory): {target}.
FORBIDDEN placement: {forbidden}. The {word} must not be placed on, inside, next to, or touching anything in this forbidden region.
Reason: the question "{question}" must still have exactly one correct answer, "{answer}". The added {word} must not be a possible answer to that question.
SIZE: {size}. It must be fully visible, not cropped, not hidden behind other objects.
Match the original perspective, lighting, color grading, grain, and photographic quality, with a natural contact shadow. Preserve every existing object, person, pose, spatial relation, the camera framing, and the background exactly. Change nothing outside the added object and its shadow. Do not beautify, restyle, crop, zoom, or add text, labels, borders, or watermarks. Return one edited photograph with the same aspect ratio. No text overlay is present or should be generated.'''
manifest = json.loads((OUT / 'manifest.json').read_text())
(OUT / 'prompts_v2').mkdir(exist_ok=True)
for s in manifest['samples']:
    qid = s['question_id']; ref = s['references']
    prompt = TEMPLATE.format(word=ref['misleading_ungroundable'], question=s['question'],
                             answer=ref['correct_answer'], **SLOTS[qid])
    (OUT / 'prompts_v2' / f'{qid}.txt').write_text(prompt + '\n')
(OUT / 'prompts_v2' / 'slots.json').write_text(json.dumps(SLOTS, indent=2) + '\n')
print('\n\n'.join((OUT / 'prompts_v2' / f"{s['question_id']}.txt").read_text() for s in manifest['samples'][:1]))
