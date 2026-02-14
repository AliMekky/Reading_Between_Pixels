import json
from pathlib import Path

# -------- CONFIG --------
INPUT_DIR = "./batches_outputs"     # folder containing batch output .jsonl files
OUTPUT_JSON = "processed_questions.json"    # output JSON file
# ------------------------


def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def to_bool(val):
    """Convert value to boolean, handling string 'true'/'false'"""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return bool(val)


def extract_model_json(entry: dict):
    resp = entry.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        return None
    return safe_json_loads(content)


def main():
    input_dir = Path(INPUT_DIR)
    files = sorted(input_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir.resolve()}")

    passed_objects = []

    total = 0
    kept = 0

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                total += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if (entry.get("response") or {}).get("status_code") != 200:
                    continue

                model_obj = extract_model_json(entry)
                if not model_obj:
                    continue

                if not to_bool(model_obj.get("example_accepted")):
                    continue

                custom_id = entry.get("custom_id", "")
                parts = str(custom_id).split("_")

                q_id = parts[0] if len(parts) > 0 else ""
                image_id = parts[0] if len(parts) > 0 else ""  # change to parts[1] if needed

                mapped = {
                    "q_id": q_id,
                    "imageId": image_id,
                    "misleading_groundable": model_obj.get("distractor_in_image", ""),
                    "misleading_ungroundable": model_obj.get("distractor_not_in_image", ""),
                    "segment_id": model_obj.get("segment_id", None),
                    "caption": model_obj.get("caption", "")
                }

                passed_objects.append(mapped)
                kept += 1

    # Save as proper JSON file (pretty formatted)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(passed_objects, f, ensure_ascii=False, indent=2)

    print(f"Total processed: {total}")
    print(f"Total kept (example_accepted=true): {kept}")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
