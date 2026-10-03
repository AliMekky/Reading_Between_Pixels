#!/usr/bin/env python3
"""Merge the two semantic judges into the intervention classifications."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "vlms/open_ended_evaluation/main_files"))

from merge_judge_results import load_gemini, load_manifest, load_openai  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic_dir", type=Path, required=True)
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--result_suffix", default="")
    args = parser.parse_args()

    manifest = load_manifest(args.batch_dir / "judge_manifest.jsonl")
    openai, openai_errors = load_openai(
        args.batch_dir / f"openai_results{args.result_suffix}.jsonl", manifest
    )
    gemini, gemini_errors = load_gemini(
        args.batch_dir / f"gemini_results{args.result_suffix}.jsonl", manifest
    )
    expected = set(manifest)
    validation = {
        "expected_cases": len(expected), "openai_valid": len(openai), "gemini_valid": len(gemini),
        "openai_missing": sorted(expected - set(openai)), "gemini_missing": sorted(expected - set(gemini)),
        "openai_errors": openai_errors, "gemini_errors": gemini_errors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "judge_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if any((validation["openai_missing"], validation["gemini_missing"], openai_errors, gemini_errors)):
        raise RuntimeError("judge outputs are incomplete or invalid")

    counts, records = Counter(), []
    with (args.deterministic_dir / "records.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["final_category"] is None:
                case_id = row["judge_case_id"]
                judge_1, judge_2 = openai[case_id], gemini[case_id]
                agreement = judge_1["mapped_category"] == judge_2["mapped_category"]
                row.update({
                    "openai_judgment": judge_1, "gemini_judgment": judge_2,
                    "judge_agreement": agreement,
                    "final_category": judge_1["mapped_category"] if agreement else "ambiguous",
                    "final_stage": "llm_agreement" if agreement else "llm_disagreement",
                    "classification_source": "two_semantic_judges",
                })
            counts[("stage", row["final_stage"])] += 1
            counts[("category", row["final_category"])] += 1
            records.append(row)
    if len(records) != 12200 or any(row["final_category"] is None for row in records):
        raise AssertionError("final record accounting failed")

    with (args.output_dir / "records_final.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "status": "complete", "records": len(records),
        "final_stage_counts": {key: value for (kind, key), value in counts.items() if kind == "stage"},
        "final_category_counts": {key: value for (kind, key), value in counts.items() if kind == "category"},
        "judge_agreement_cases": sum(openai[key]["mapped_category"] == gemini[key]["mapped_category"] for key in expected),
        "judge_disagreement_cases": sum(openai[key]["mapped_category"] != gemini[key]["mapped_category"] for key in expected),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[PASS] {summary}")


if __name__ == "__main__":
    main()
