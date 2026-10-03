#!/usr/bin/env python3
"""Read-only progress, integrity, and response audit for the relevance pilot."""
import argparse
import json
from collections import Counter
from pathlib import Path


def inspect(root, details=False):
    config_path = root / "configuration.json"
    if not config_path.exists():
        print("No configuration written yet")
        return
    config = json.loads(config_path.read_text())
    records = [json.loads(p.read_text()) for p in sorted((root / "samples").glob("*.json"))]
    results = [v for r in records for s in r["states"].values() for v in s.values()]
    complete = [r for r in records if r.get("status") == "complete"]
    counts = Counter(r["question_id"] for r in complete)
    summary = {
        "completed_generations": len(results),
        "expected_generations": len(config["selected_samples"]) * len(config["variants"]) * 18,
        "completed_question_overlay_pairs": len(complete),
        "questions_with_all_overlays_complete": sum(n == len(config["variants"]) for n in counts.values()),
        "no_op_checks_passed": sum(v.get("no_op_passed", False) for v in results),
        "prefix_checks_passed": sum(v.get("prefix_equivalence_passed", False) for v in results),
        "max_blocked_probability": max((v["integrity"]["max_blocked_probability"] for v in results), default=None),
        "max_attention_row_sum_error": max((v["integrity"]["max_attention_row_sum_error"] for v in results), default=None),
        "truncated_generations": sum(v["termination_reason"] == "max_tokens" for v in results),
    }
    completion = root / "completion.json"
    if completion.exists():
        summary["completion"] = json.loads(completion.read_text())
    print(json.dumps(summary, indent=2))
    if details:
        by_qid = {}
        for r in records:
            q = by_qid.setdefault(r["question_id"], {"question": r["question"],
                "references": r["references"], "variants": {}})
            q["variants"][r["variant"]] = {}
            for state, arms in r["states"].items():
                baseline = arms["baseline"]["raw_response"] if "baseline" in arms else None
                q["variants"][r["variant"]][state] = {"baseline": baseline,
                    "changes": {arm: v["raw_response"] for arm, v in arms.items()
                                if v["raw_response"] != baseline}}
        for qid, data in by_qid.items():
            print(json.dumps({"question_id": qid, **data}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    inspect(args.output, args.details)
