#!/usr/bin/env python3
"""Create validated, counterbalanced JSONL requests for both LLM judges."""

import argparse
import json
from pathlib import Path


LABELS = ["reference_1", "reference_2", "reference_3", "reference_4", "other", "ambiguous"]
SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = (
    "You are a strict semantic-equivalence classifier for VQA evaluation. "
    "Do not answer the question yourself. Compare only the generated response with the four references. "
    "Choose reference_1 through reference_4 only when the response has the same meaning as that reference. "
    "Choose other when it clearly matches none. Choose ambiguous when multiple references could match or the meaning is uncertain. "
    "Return the required JSON with a brief reason."
)


def user_prompt(case, order):
    lines = [
        f"Question: {case['question']}",
        f"Generated response: {case['raw_response']}",
        "References:",
    ]
    for index, semantic_id in enumerate(order, start=1):
        lines.append(f"reference_{index}: {case['references'][semantic_id]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--openai_model", default="gpt-5.6-luna")
    parser.add_argument("--gemini_model", default="gemini-3.5-flash")
    args = parser.parse_args()

    cases = []
    for path in sorted(args.evaluation_dir.glob("*/unresolved_for_judges.jsonl")):
        with path.open() as handle:
            cases.extend(json.loads(line) for line in handle if line.strip())
    if not cases:
        raise RuntimeError("No unresolved cases found")
    ids = [case["judge_case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("Judge case IDs are not globally unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    openai_path = args.output_dir / "openai_requests.jsonl"
    gemini_path = args.output_dir / "gemini_requests.jsonl"
    manifest_path = args.output_dir / "judge_manifest.jsonl"

    with openai_path.open("w") as openai_file, gemini_path.open("w") as gemini_file, manifest_path.open("w") as manifest:
        for case in cases:
            order_1 = case["judge_1_reference_order"]
            order_2 = case["judge_2_reference_order"]
            if order_2 != list(reversed(order_1)):
                raise AssertionError(f"Reference order is not reversed: {case['judge_case_id']}")
            case_id = case["judge_case_id"]
            openai_file.write(json.dumps({
                "custom_id": case_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.openai_model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(case, order_1)},
                    ],
                    "max_output_tokens": 128,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "vqa_semantic_equivalence",
                            "strict": True,
                            "schema": SCHEMA,
                        }
                    },
                },
            }, ensure_ascii=False) + "\n")
            gemini_file.write(json.dumps({
                "key": case_id,
                "request": {
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt(case, order_2)}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": 128,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": SCHEMA,
                    },
                },
            }, ensure_ascii=False) + "\n")
            manifest.write(json.dumps({
                **case,
                "openai_reference_map": {f"reference_{i}": value for i, value in enumerate(order_1, 1)},
                "gemini_reference_map": {f"reference_{i}": value for i, value in enumerate(order_2, 1)},
            }, ensure_ascii=False) + "\n")

    metadata = {
        "num_cases": len(cases),
        "openai_model": args.openai_model,
        "gemini_model": args.gemini_model,
        "openai_file": str(openai_path),
        "gemini_file": str(gemini_path),
        "manifest_file": str(manifest_path),
    }
    with (args.output_dir / "request_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[PASS] cases={len(cases)} unique_ids={len(set(ids))} reversed_orders={len(cases)}")
    print(f"[OUTPUT] openai={openai_path} gemini={gemini_path} manifest={manifest_path}")


if __name__ == "__main__":
    main()
