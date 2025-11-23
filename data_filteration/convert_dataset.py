from datasets import load_from_disk
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO

######## THIS SCRIPT CONVERTS THE FILTERED DATASET INTO IMAGES + JSON FILE ########

# Load and filter
ds = load_from_disk("seed_bench_filtered")
ds = ds.filter(lambda x: x['quality'] == "Unrated")

# Save
Path("data/images").mkdir(parents=True, exist_ok=True)
questions = []
for i, d in enumerate(ds):
    img_name = f"{d['question_id']}.jpg"

    img_data = d['image']
    
    if isinstance(img_data, str):
        # Base64 encoded string - decode it
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
    elif isinstance(img_data, Image.Image):
        # Already a PIL Image
        img = img_data
    else:
        # Try to open as PIL Image directly
        img = Image.open(img_data)
    
    # Convert to RGB and save
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(f"data/images/{img_name}")
    questions.append({
        "image": img_name,
        "question": d['question'],
        "A": d['A'], "B": d['B'], "C": d['C'], "D": d['D'],
        "answer": d['answer'],
        "question_id": d['question_id']
    })

with open("data/questions.json", 'w') as f:
    json.dump(questions, f, indent=2)

print(f"✓ Converted {len(questions)} samples")