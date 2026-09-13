#!/usr/bin/env python3
"""Validate completeness and invariants of one model's generation outputs."""

import argparse
import json
from pathlib import Path


CONDITIONS = (
    "notext",
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--model_slug", required=True)
    parser.add_argument("--expected_questions", type=int, required=True)
    args = parser.parse_args()

    question_sets = {}
    total = errors = truncated = 0
    for condition in CONDITIONS:
        path = args.input_dir / f"{args.model_slug}_{condition}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing output: {path}")
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        latest = {str(row["question_id"]): row for row in rows}
        ok = sum(row.get("status") == "ok" for row in latest.values())
        condition_errors = len(latest) - ok
        condition_truncated = sum(
            row.get("termination_reason") == "max_tokens"
            for row in latest.values()
            if row.get("status") == "ok"
        )
        if len(latest) != args.expected_questions:
            raise AssertionError(
                f"{condition}: {len(latest)}/{args.expected_questions} unique questions"
            )
        if condition_errors:
            raise AssertionError(f"{condition}: {condition_errors} failed questions")
        for qid, row in latest.items():
            if row.get("variant") != condition:
                raise AssertionError(f"{condition}/{qid}: incorrect variant field")
            if row.get("evaluation_format") != "open_ended":
                raise AssertionError(f"{condition}/{qid}: incorrect evaluation format")
            if "Options:" in row.get("prompt", ""):
                raise AssertionError(f"{condition}/{qid}: MCQ options leaked into prompt")
            if not row.get("image_sha256") or "raw_response" not in row:
                raise AssertionError(f"{condition}/{qid}: missing audit metadata")
        question_sets[condition] = set(latest)
        total += len(latest)
        errors += condition_errors
        truncated += condition_truncated
        print(
            f"[PASS] condition={condition} questions={len(latest)} "
            f"errors={condition_errors} max_token_stops={condition_truncated}"
        )

    baseline = question_sets["notext"]
    if any(qids != baseline for qids in question_sets.values()):
        raise AssertionError("Question IDs are not aligned across conditions")
    expected_total = args.expected_questions * len(CONDITIONS)
    if total != expected_total:
        raise AssertionError(f"Total records: {total}/{expected_total}")
    print(
        f"[COMPLETE] model={args.model_slug} records={total}/{expected_total} "
        f"errors={errors} max_token_stops={truncated} aligned_conditions=5/5"
    )


if __name__ == "__main__":
    main()
