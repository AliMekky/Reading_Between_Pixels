#!/usr/bin/env python3
"""Merge independently run grounded and ungrounded layer-sweep reports."""

import argparse
import json
from collections import Counter
from pathlib import Path


def record_key(record):
    return (
        str(record["question_id"]), str(record["variant"]), int(record["layer"]),
        str(record["direction"]), str(record["region"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = []
    for path in args.inputs:
        with Path(path).open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") != "success":
            raise ValueError("Incomplete input report: {}".format(path))
        reports.append(report)
    expected_variants = [
        "misleading_groundable", "misleading_ungroundable", "irrelevant_word",
    ]
    actual_variants = [report["configuration"]["variants"] for report in reports]
    if actual_variants != [[variant] for variant in expected_variants]:
        raise ValueError("Input reports do not have the expected separate variants")
    reference = reports[0]
    for key in (
        "model_id", "dataset_revision", "selected_question_ids", "windows",
        "directions", "regions", "streams", "seed", "shuffle_options", "dtype",
    ):
        if any(reference["configuration"][key] != report["configuration"][key] for report in reports[1:]):
            raise ValueError("Configuration mismatch for {}".format(key))

    records = [record for report in reports for record in report["records"]]
    keys = {record_key(record) for record in records}
    if len(keys) != len(records):
        raise ValueError("Merged records contain duplicates")
    expected = sum(report["completion"]["expected_interventions"] for report in reports)
    if len(records) != expected:
        raise ValueError("Merged record count {}, expected {}".format(len(records), expected))

    configuration = dict(reference["configuration"])
    configuration["variants"] = expected_variants
    configuration["merged_from_separate_variant_jobs"] = True
    samples = {}
    for report in reports:
        for qid, sample in report["samples"].items():
            if qid not in samples:
                samples[qid] = dict(sample)
                samples[qid]["conditions"] = {}
            samples[qid]["conditions"].update(sample["conditions"])
    outcome_counts = Counter(record["outcome"] for record in records)
    merged = {
        "status": "success",
        "milestone": "shared_subset_complete_single_layer_sweep",
        "configuration": configuration,
        "selection": reference["selection"],
        "samples": samples,
        "records": records,
        "completion": {
            "expected_interventions": expected,
            "saved_interventions": len(records),
            "remaining_interventions": 0,
            "completed_question_ids": configuration["selected_question_ids"],
            "outcome_counts": dict(sorted(outcome_counts.items())),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    temporary.replace(output)
    print("[COMPLETE] merged_records={}/{} output={}".format(len(records), expected, output), flush=True)


if __name__ == "__main__":
    main()
