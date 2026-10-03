#!/usr/bin/env python3
"""Validate both judge outputs and merge them with deterministic classifications."""

import argparse
import json
from collections import Counter
from pathlib import Path


REFERENCE_LABELS = {f"reference_{index}" for index in range(1, 5)}
SPECIAL_LABELS = {"other", "ambiguous"}
CONDITIONS = (
    "notext", "correct_answer", "misleading_groundable",
    "misleading_ungroundable", "irrelevant_word",
)


def json_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("output_text"), str):
            return value["output_text"]
        for item in value.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    return content["text"]
        for candidate in value.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    return part["text"]
    raise ValueError("No textual judge response found")


def parse_decision(text, reference_map):
    decision = json.loads(text)
    if set(decision) != {"label", "reason"} or not isinstance(decision["reason"], str):
        raise ValueError("Judge response does not match the required schema")
    label = decision["label"]
    if label in REFERENCE_LABELS:
        mapped = reference_map[label]
    elif label in SPECIAL_LABELS:
        mapped = label
    else:
        raise ValueError(f"Invalid judge label: {label}")
    return {"raw": decision, "mapped_category": mapped}


def load_manifest(path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["judge_case_id"]: row for row in rows}


def load_openai(path, manifest):
    parsed, errors = {}, {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            case_id = row["custom_id"]
            try:
                if row.get("error"):
                    raise ValueError(str(row["error"]))
                body = row["response"]["body"]
                parsed[case_id] = parse_decision(
                    json_text(body), manifest[case_id]["openai_reference_map"]
                )
            except Exception as exc:
                errors[case_id] = f"{type(exc).__name__}: {exc}"
    return parsed, errors


def load_gemini(path, manifest):
    parsed, errors = {}, {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            case_id = row.get("key") or row.get("metadata", {}).get("key")
            try:
                if not case_id:
                    raise ValueError("Missing Gemini case key")
                if row.get("error"):
                    raise ValueError(str(row["error"]))
                response = row.get("response", row)
                parsed[case_id] = parse_decision(
                    json_text(response), manifest[case_id]["gemini_reference_map"]
                )
            except Exception as exc:
                errors[case_id or f"line_{len(parsed) + len(errors)}"] = f"{type(exc).__name__}: {exc}"
    return parsed, errors


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--result_suffix", default="")
    args = parser.parse_args()

    manifest = load_manifest(args.batch_dir / "judge_manifest.jsonl")
    openai, openai_errors = load_openai(args.batch_dir / f"openai_results{args.result_suffix}.jsonl", manifest)
    gemini, gemini_errors = load_gemini(args.batch_dir / f"gemini_results{args.result_suffix}.jsonl", manifest)
    expected = set(manifest)
    missing_openai = expected - set(openai)
    missing_gemini = expected - set(gemini)
    validation = {
        "expected_cases": len(expected),
        "openai_valid": len(openai), "openai_missing_or_invalid": len(missing_openai | set(openai_errors)),
        "gemini_valid": len(gemini), "gemini_missing_or_invalid": len(missing_gemini | set(gemini_errors)),
        "openai_errors": openai_errors, "gemini_errors": gemini_errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judge_validation.json").write_text(json.dumps(validation, indent=2))
    if missing_openai or missing_gemini or openai_errors or gemini_errors:
        raise RuntimeError(f"Judge outputs incomplete or invalid; see {args.output_dir / 'judge_validation.json'}")

    overall = Counter()
    summaries = []
    for model_dir in sorted(path for path in args.evaluation_dir.iterdir() if path.is_dir()):
        model_output = args.output_dir / model_dir.name
        model_output.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for condition in CONDITIONS:
            destination = model_output / f"{condition}.jsonl"
            with destination.open("w") as handle:
                for row in read_jsonl(model_dir / f"{condition}.jsonl"):
                    if not row["needs_llm_judge"]:
                        final_category = row["deterministic_category"]
                        final_stage = row["deterministic_stage"]
                        agreement = None
                    else:
                        case_id = "oej-" + __import__("hashlib").sha256(row["record_id"].encode()).hexdigest()[:24]
                        judge_1, judge_2 = openai[case_id], gemini[case_id]
                        agreement = judge_1["mapped_category"] == judge_2["mapped_category"]
                        final_category = judge_1["mapped_category"] if agreement else "ambiguous"
                        final_stage = "llm_agreement" if agreement else "llm_disagreement"
                        row["openai_judgment"] = judge_1
                        row["gemini_judgment"] = judge_2
                        row["judge_agreement"] = agreement
                    row["final_category"] = final_category
                    row["final_stage"] = final_stage
                    counts[("stage", final_stage)] += 1
                    counts[("category", final_category)] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        total = sum(value for (kind, _), value in counts.items() if kind == "stage")
        summary = {
            "model_slug": model_dir.name,
            "total_records": total,
            "stage_counts": {key: value for (kind, key), value in counts.items() if kind == "stage"},
            "category_counts": {key: value for (kind, key), value in counts.items() if kind == "category"},
        }
        summaries.append(summary)
        overall.update(counts)
        (model_output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[COMPLETE] model={model_dir.name} records={total} stages={summary['stage_counts']}")
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"[PASS] models={len(summaries)} records={sum(item['total_records'] for item in summaries)}")


if __name__ == "__main__":
    main()
