import os
import json
import copy
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image

# ----------------------------
# Config
# ----------------------------
DATASET_JSON_PATH = "../scenetap/questions_v1.json"        # your big JSON (dict keyed by question_id)
PROMPT_TXT_PATH = "./prompt/validation_prompt.txt"            # your prompt template file

ORIG_IMG_DIR = "../data_filteration/passed_images_gqa/"          # directory with original images
# SEG_IMG_DIR = "./som_images_gqa/typo_base_complex/slider_3/seed_42/filter_10.0/"          # directory with segmented images

OUT_DIR = "batches_files_validation"
BATCH_SIZE = 4500

MODEL_NAME = "gpt-5.2"
SYSTEM_PROMPT = """
        You are a strict visual validation assistant.

        Your role is to evaluate whether a multiple-choice example is logically and visually valid.

        Rules:
        - Base your judgment only on the provided image and inputs.
        - Do not assume unseen objects exist.
        - If uncertain about object presence or correctness, reject the example.
        - Be conservative in approval.
        - Output exactly one short sentence.
        - End the sentence with "true" if accepted or "false" if rejected.
        - Do not output anything else.

        """

MAX_TOKENS = 2048
TEMPERATURE = 0

# If your images are jpg and filenames equal to imageId.jpg, set this True.
# Otherwise adjust `resolve_image_path()` below.
USE_IMAGEID_AS_FILENAME = True
ORIG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SEG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ----------------------------
# Helpers
# ----------------------------
def encode_image_from_pil(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    """Return base64 string (no prefix) for a PIL image."""
    buffered = BytesIO()
    pil_image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def pil_to_data_url(pil_image: Image.Image, mime: str = "image/jpeg", fmt: str = "JPEG") -> str:
    b64 = encode_image_from_pil(pil_image, fmt=fmt)
    return f"data:{mime};base64,{b64}"

def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Failed to open image: {path} ({e})")
        return None

def find_file_by_stem(directory: Path, stem: str, exts: tuple) -> Optional[Path]:
    """Find a file in `directory` with basename `stem` and one of `exts`."""
    for ext in exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def resolve_image_path(img_dir: Path, image_id: str, exts: tuple) -> Optional[Path]:
    """
    Adjust this function if your naming differs.
    By default: looks for {imageId}.jpg/png/... in the given directory.
    """
    # Common case: image file named as imageId.{ext}
    p = find_file_by_stem(img_dir, image_id, exts)
    if p:
        return p

    # Fallback: maybe directory contains filenames like 02986944.jpg (question id)
    # You can add other logic here if needed.
    return None

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fill_prompt(template: str, *, question: str, correct_answer: str, misleading_groundable: str, misleading_ungroundable: str) -> str:
    """
    Assumes your prompt.txt has placeholders like:
      {{QUESTION}}, {{CORRECT_ANSWER}}, {{OBJECTS_LIST}}
    You can add more as needed.
    """
    return (
        template
        .replace("{{QUESTION}}", question)
        .replace("{{CORRECT_ANSWER}}", correct_answer)
        .replace("{{MISLEADING_GROUNDABLE}}", misleading_groundable)
        .replace("{{MISLEADING_UNGROUNDABLE}}", misleading_ungroundable)
    )

def build_user_content(prompt_text: str, orig_img: Image.Image) -> List[Dict[str, Any]]:
    """
    Produces a multimodal user content list: text + 1 image
    """
    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": pil_to_data_url(orig_img, mime="image/jpeg", fmt="JPEG")}},
    ]

# ----------------------------
# Base template for each JSONL line
# ----------------------------
base_template_dic = {
    "custom_id": None,
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
        ],
    }
}

# ----------------------------
# Main
# ----------------------------
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = Path(ORIG_IMG_DIR)

    prompt_template = load_prompt_template(PROMPT_TXT_PATH)

    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f) 

    total = len(data)
    print(f"Loaded {total} samples from {DATASET_JSON_PATH}")

    batch: List[Dict[str, Any]] = []
    batch_idx = 0
    written = 0
    skipped = 0
    start = 3000
    end = 4000

    for i, entry in enumerate(data):
        if i >= end:
            break
        if i <start:
            continue
        rec = entry

        qid = rec['question_id']
        question = rec['question']
        answer = rec["correct_answer"]['text']
        misleading_groundable = rec['misleading_groundable']['text']
        misleading_ungroundable = rec['misleading_ungroundable']
        image_id = rec['image_id']


        orig_path = resolve_image_path(orig_dir, image_id, ORIG_EXTS)

        if not orig_path:
            print(f"[WARN] Missing files for imageId={image_id} (qid={qid}) "
                  f"orig={orig_path}, skipping")
            skipped += 1
            continue

        orig_img = safe_open_image(orig_path)
        if orig_img is None:
            skipped += 1
            continue

        prompt_text = fill_prompt(
            prompt_template,
            question=question,
            correct_answer=answer,
            misleading_groundable=misleading_groundable,
            misleading_ungroundable=misleading_ungroundable
        )

        temp = copy.deepcopy(base_template_dic)
        temp["custom_id"] = f"{qid}_{image_id}"  # keep qid as custom_id (best for traceability)
        temp["body"]["messages"][0]["content"] = SYSTEM_PROMPT
        temp["body"]["messages"][1]["content"] = build_user_content(prompt_text, orig_img)

        batch.append(temp)

        # Write batch file every BATCH_SIZE
        # if len(batch) >= BATCH_SIZE:
        #     out_path = out_dir / f"batch_file_{start}_{end}.jsonl"
        #     with open(out_path, "w", encoding="utf-8") as wf:
        #         for entry in batch:
        #             json.dump(entry, wf, ensure_ascii=False)
        #             wf.write("\n")
        #     written += len(batch)
        #     print(f"Wrote {len(batch)} requests -> {out_path}")
        #     batch = []
        #     batch_idx += 1

    # Write remaining
    if batch:
        out_path = out_dir / f"batch_file_validation_{start}_{end}.jsonl"
        with open(out_path, "w", encoding="utf-8") as wf:
            for entry in batch:
                json.dump(entry, wf, ensure_ascii=False)
                wf.write("\n")
        written += len(batch)
        print(f"Wrote {len(batch)} requests -> {out_path}")

    print(f"\nDone.")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")

if __name__ == "__main__":
    main()
