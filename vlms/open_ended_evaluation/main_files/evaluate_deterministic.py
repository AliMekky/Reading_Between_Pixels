#!/usr/bin/env python3
"""Run deterministic open-ended answer classification and export unresolved cases."""

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path


CONDITIONS = (
    "notext",
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
)
REFERENCE_IDS = (
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
)
ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}
CONTRACTIONS = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hasn't": "has not", "haven't": "have not", "isn't": "is not",
    "it's": "it is", "shouldn't": "should not", "wasn't": "was not",
    "weren't": "were not", "won't": "will not", "wouldn't": "would not",
}
PUNCTUATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_answer(value):
    """Apply the preregistered VQA-style normalization without stemming."""
    text = "" if value is None else str(value).lower().strip()
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    tokens = text.translate(PUNCTUATION).split()
    tokens = [NUMBER_WORDS.get(token, token) for token in tokens if token not in ARTICLES]
    return " ".join(tokens)


def levenshtein_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def edit_similarity(left, right):
    denominator = max(len(left), len(right))
    return 1.0 if denominator == 0 else 1.0 - levenshtein_distance(left, right) / denominator


def classify(record, threshold):
    response = normalize_answer(record.get("raw_response"))
    references = {
        key: normalize_answer(record["reference_answers"].get(key))
        for key in REFERENCE_IDS
    }
    similarities = {
        key: round(edit_similarity(response, value), 6)
        for key, value in references.items()
    }
    duplicate_references = len(set(references.values())) != len(references)

    if record.get("status") != "ok" or not response:
        stage, category, matches = "invalid", "invalid", []
    else:
        matches = [key for key, value in references.items() if response == value]
        if len(matches) == 1:
            stage, category = "normalized_exact", matches[0]
        else:
            matches = [key for key, score in similarities.items() if score >= threshold]
            if len(matches) == 1:
                stage, category = "edit_similarity", matches[0]
            else:
                stage, category = "unresolved", None

    result = dict(record)
    result.update({
        "normalized_response": response,
        "normalized_references": references,
        "edit_similarities": similarities,
        "deterministic_stage": stage,
        "deterministic_category": category,
        "matching_reference_ids": matches,
        "needs_llm_judge": stage == "unresolved",
        "duplicate_normalized_references": duplicate_references,
        "edit_similarity_threshold": threshold,
    })
    return result


def judge_reference_orders(record, seed):
    reference_ids = list(REFERENCE_IDS)
    digest = hashlib.sha256(f"{seed}|{record['record_id']}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(reference_ids)
    return reference_ids, list(reversed(reference_ids))


def read_latest(path):
    latest = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["question_id"])] = row
    return latest


def completed_model_slugs(input_dir):
    suffix = "_summary.json"
    return sorted(
        path.name[:-len(suffix)]
        for path in input_dir.glob("*_summary.json")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_questions", type=int, default=474)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_slugs", nargs="*")
    args = parser.parse_args()

    model_slugs = args.model_slugs or completed_model_slugs(args.input_dir)
    if not model_slugs:
        raise RuntimeError("No completed model summaries found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summary = []

    for model_slug in model_slugs:
        model_dir = args.output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        unresolved = []
        baseline_ids = None

        for condition in CONDITIONS:
            source = args.input_dir / f"{model_slug}_{condition}.jsonl"
            rows = read_latest(source)
            if len(rows) != args.expected_questions:
                raise AssertionError(f"{model_slug}/{condition}: {len(rows)} questions")
            if baseline_ids is None:
                baseline_ids = set(rows)
            elif set(rows) != baseline_ids:
                raise AssertionError(f"{model_slug}/{condition}: question IDs are misaligned")

            destination = model_dir / f"{condition}.jsonl"
            with destination.open("w") as handle:
                for qid in sorted(rows):
                    result = classify(rows[qid], args.threshold)
                    counts[(condition, result["deterministic_stage"])] += 1
                    counts[("all", result["deterministic_stage"])] += 1
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result["needs_llm_judge"]:
                        order_1, order_2 = judge_reference_orders(result, args.seed)
                        unresolved.append({
                            "judge_case_id": "oej-" + hashlib.sha256(
                                result["record_id"].encode()
                            ).hexdigest()[:24],
                            "record_id": result["record_id"],
                            "question_id": result["question_id"],
                            "model_id": result["model_id"],
                            "variant": result["variant"],
                            "question": result["question"],
                            "raw_response": result["raw_response"],
                            "references": result["reference_answers"],
                            "judge_1_reference_order": order_1,
                            "judge_2_reference_order": order_2,
                        })

        unresolved_path = model_dir / "unresolved_for_judges.jsonl"
        with unresolved_path.open("w") as handle:
            for row in unresolved:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        total = args.expected_questions * len(CONDITIONS)
        summary = {
            "model_slug": model_slug,
            "total_records": total,
            "normalized_exact": counts[("all", "normalized_exact")],
            "edit_similarity": counts[("all", "edit_similarity")],
            "invalid": counts[("all", "invalid")],
            "unresolved": counts[("all", "unresolved")],
            "threshold": args.threshold,
            "condition_counts": {
                condition: {
                    stage: counts[(condition, stage)]
                    for stage in ("normalized_exact", "edit_similarity", "invalid", "unresolved")
                }
                for condition in CONDITIONS
            },
        }
        if sum(summary[key] for key in ("normalized_exact", "edit_similarity", "invalid", "unresolved")) != total:
            raise AssertionError(f"{model_slug}: stage counts do not sum to {total}")
        with (model_dir / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        all_summary.append(summary)
        print(
            f"[COMPLETE] model={model_slug} total={total} "
            f"exact={summary['normalized_exact']} edit={summary['edit_similarity']} "
            f"unresolved={summary['unresolved']} invalid={summary['invalid']}"
        )

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(all_summary, handle, indent=2)
    print(f"[PASS] completed_models={len(all_summary)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
