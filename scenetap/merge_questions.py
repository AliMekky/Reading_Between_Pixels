import json
from pathlib import Path

# -------- CONFIG --------
PASSED_QUESTIONS = "../data_filteration/gqa/passed_questions_2.json"
PROCESSED_QUESTIONS = "./processed_questions.json"
OUTPUT_FILE = "questions_v1.json"
# ------------------------


def get_bbox_for_object(scene_graph, obj_id):
    """Extract bounding box for an object ID from scene graph"""
    if not scene_graph or "objects" not in scene_graph:
        return None

    obj = scene_graph["objects"].get(str(obj_id))
    if not obj:
        return None

    return {
        "x": obj.get("x"),
        "y": obj.get("y"),
        "w": obj.get("w"),
        "h": obj.get("h")
    }


def find_object_by_name(scene_graph, name):
    """Find first object in scene graph matching the given name"""
    if not scene_graph or "objects" not in scene_graph:
        return None

    name_lower = name.lower().strip()
    for obj_id, obj in scene_graph["objects"].items():
        if obj.get("name", "").lower().strip() == name_lower:
            bbox = {
                "text": name,
                "x": obj.get("x"),
                "y": obj.get("y"),
                "w": obj.get("w"),
                "h": obj.get("h")
            }
            return bbox
    return None


def merge_questions():
    # Load both files
    with open(PASSED_QUESTIONS, "r", encoding="utf-8") as f:
        passed_questions = json.load(f)

    with open(PROCESSED_QUESTIONS, "r", encoding="utf-8") as f:
        processed_questions = json.load(f)

    merged_results = []

    for processed in processed_questions:
        q_id = processed.get("q_id")

        # Find matching question in passed_questions
        if q_id not in passed_questions:
            print(f"Warning: q_id {q_id} not found in passed_questions_2.json")
            continue

        passed_q = passed_questions[q_id]
        scene_graph = passed_q.get("scene_graph", {})

        # Get correct answer object ID from annotations
        # Handle different annotation key formats (e.g., "0", "0:2", etc.)
        answer_annotations = passed_q.get("annotations", {}).get("answer", {})
        answer_obj_id = None
        if answer_annotations:
            # Get the first value from the annotations dict, regardless of key
            answer_obj_id = next(iter(answer_annotations.values()), None)

        # Get bounding box for correct answer
        correct_answer_bbox = None
        if answer_obj_id:
            bbox = get_bbox_for_object(scene_graph, answer_obj_id)
            if bbox:
                correct_answer_bbox = {
                    "text": passed_q.get("answer", ""),
                    **bbox
                }

        # Find misleading options in scene graph
        misleading_groundable = None
        misleading_groundable_text = processed.get("misleading_groundable", "")
        if misleading_groundable_text:
            misleading_groundable = find_object_by_name(scene_graph, misleading_groundable_text)

        # misleading_ungroundable is not in the image, so no bbox
        misleading_ungroundable = processed.get("misleading_ungroundable", "")

        # Create merged object
        merged_obj = {
            "question_id": q_id,
            "image_id": passed_q.get("imageId", ""),
            "misleading_groundable": misleading_groundable,
            "misleading_ungroundable": misleading_ungroundable,
            "seg_id": processed.get("segment_id"),
            "correct_answer": correct_answer_bbox,
            "question": passed_q.get("question", ""),
            "caption": processed.get("caption")
        }

        merged_results.append(merged_obj)

    # Save merged results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(merged_results)} questions")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_questions()
