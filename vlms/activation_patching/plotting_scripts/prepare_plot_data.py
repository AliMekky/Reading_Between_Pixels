#!/usr/bin/env python3
"""Convert the three validated raw reports into compact plotting tables."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from plot_style import VARIANTS


DIRECTIONS = ("restoration", "insertion")
RAW_REGIONS = (
    "text_region", "correct_object_region", "grounded_object_region",
    "matched_random_region_1", "matched_random_region_2",
    "matched_random_region_3", "all_image_tokens",
)
SUMMARY_REGIONS = (
    "text_region", "matched_random_mean", "all_image_tokens",
    "correct_object_region", "grounded_object_region",
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values, rng, draws):
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        stop = min(start + 500, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return values.mean(), *np.quantile(means, (0.025, 0.975))


def wilcoxon_p(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 if np.all(values == 0) else wilcoxon(values, zero_method="zsplit").pvalue


def bh_adjust(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def wilson(successes, total):
    if not total:
        return np.nan, np.nan
    z, rate = 1.959964, successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    variants = list(VARIANTS)
    data = {}
    clean = {}
    predictions = {}
    qids = None
    correct_letters = {}

    for expected_variant, path in zip(variants, args.inputs):
        print("[LOAD] {}".format(path), flush=True)
        with Path(path).open(encoding="utf-8") as handle:
            report = json.load(handle)
        config = report["configuration"]
        if report.get("status") != "success" or config["variants"] != [expected_variant]:
            raise ValueError("Invalid or mismatched report: {}".format(path))
        current_qids = [str(value) for value in config["selected_question_ids"]]
        if qids is not None and current_qids != qids:
            raise ValueError("Question IDs differ across conditions")
        qids = current_qids
        correct_letters.update({str(qid): sample["correct_letter"] for qid, sample in report["samples"].items()})
        expected = int(report["completion"]["expected_interventions"])
        if len(report["records"]) != expected or report["completion"]["remaining_interventions"] != 0:
            raise ValueError("Incomplete report: {}".format(path))
        for record in report["records"]:
            key = (str(record["question_id"]), expected_variant, record["direction"], int(record["layer"]), record["region"])
            data[key] = float(record["effect"])
            clean[key] = record.get("clean_object_control")
            if record["region"] == "text_region":
                predictions[key[:-1]] = {
                    "recipient": record["recipient_prediction"],
                    "patched": record["patched_prediction"],
                    "target": record["target_option_letter"],
                }
        del report
        gc.collect()
        print("[VALID] condition={} records={}".format(expected_variant, expected), flush=True)

    rng = np.random.default_rng(args.seed)
    summary = []
    values_by_group = {}
    for variant in variants:
        for direction in DIRECTIONS:
            for layer in range(32):
                prefix = (variant, direction, layer)
                for region in SUMMARY_REGIONS:
                    values = []
                    for qid in qids:
                        if region == "matched_random_mean":
                            value = np.mean([data[(qid,) + prefix + ("matched_random_region_{}".format(index),)] for index in (1, 2, 3)])
                        else:
                            key = (qid,) + prefix + (region,)
                            if region.endswith("object_region") and (key not in clean or clean[key] is not True):
                                continue
                            value = data[key]
                        values.append(value)
                    values_by_group[prefix + (region,)] = np.asarray(values)
                    mean, low, high = mean_ci(values, rng, args.bootstrap_draws)
                    summary.append({
                        "variant": variant, "direction": direction, "layer": layer,
                        "region": region, "n": len(values), "mean_effect": mean,
                        "ci_95_low": low, "ci_95_high": high,
                    })

    comparison_names = (
        "ungrounded_minus_grounded", "grounded_minus_irrelevant",
        "ungrounded_minus_irrelevant", "grounded_text_minus_random",
        "ungrounded_text_minus_random", "irrelevant_text_minus_random",
    )
    comparisons = []
    for direction in DIRECTIONS:
        for layer in range(32):
            get = lambda variant, region: values_by_group[(variant, direction, layer, region)]
            arrays = {
                "ungrounded_minus_grounded": get(variants[1], "text_region") - get(variants[0], "text_region"),
                "grounded_minus_irrelevant": get(variants[0], "text_region") - get(variants[2], "text_region"),
                "ungrounded_minus_irrelevant": get(variants[1], "text_region") - get(variants[2], "text_region"),
                "grounded_text_minus_random": get(variants[0], "text_region") - get(variants[0], "matched_random_mean"),
                "ungrounded_text_minus_random": get(variants[1], "text_region") - get(variants[1], "matched_random_mean"),
                "irrelevant_text_minus_random": get(variants[2], "text_region") - get(variants[2], "matched_random_mean"),
            }
            row = {"direction": direction, "layer": layer, "n": len(qids)}
            for name, values in arrays.items():
                mean, low, high = mean_ci(values, rng, args.bootstrap_draws)
                row.update({name + "_mean": mean, name + "_ci_95_low": low,
                            name + "_ci_95_high": high, name + "_p_value": wilcoxon_p(values)})
            comparisons.append(row)
    for direction in DIRECTIONS:
        rows = [row for row in comparisons if row["direction"] == direction]
        for name in comparison_names:
            for row, q_value in zip(rows, bh_adjust([row[name + "_p_value"] for row in rows])):
                row[name + "_fdr_q_value"] = q_value

    prediction_rows = []
    for variant in variants:
        for direction in DIRECTIONS:
            for layer in range(32):
                rows = [(qid, predictions[(qid, variant, direction, layer)]) for qid in qids]
                if direction == "restoration":
                    eligible = [(qid, row) for qid, row in rows if row["recipient"] == row["target"]]
                    desired = lambda qid, row: row["patched"] == correct_letters[qid]
                    transition = "target_to_correct"
                else:
                    eligible = [(qid, row) for qid, row in rows if row["recipient"] == correct_letters[qid]]
                    desired = lambda qid, row: row["patched"] == row["target"]
                    transition = "correct_to_target"
                desired_n = sum(desired(qid, row) for qid, row in eligible)
                other_n = sum(row["patched"] != row["recipient"] and not desired(qid, row) for qid, row in eligible)
                unchanged_n = sum(row["patched"] == row["recipient"] for _, row in eligible)
                low, high = wilson(desired_n, len(eligible))
                prediction_rows.append({
                    "variant": variant, "direction": direction, "layer": layer,
                    "conditional_transition": transition, "eligible_n": len(eligible),
                    "desired_transition_count": desired_n,
                    "conditional_transition_rate": desired_n / len(eligible) if eligible else "",
                    "ci_95_low": low, "ci_95_high": high,
                    "other_flip_count": other_n, "unchanged_count": unchanged_n,
                })

    output = Path(args.output_dir)
    write_csv(output / "layerwise_summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "prediction_transitions.csv", prediction_rows)
    metadata = {"status": "success", "sample_count": len(qids), "layers": 32,
                "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
                "raw_interventions": len(data)}
    (output / "plot_data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[COMPLETE] samples={} raw_interventions={} summary_rows={} comparison_rows={} prediction_rows={}".format(
        len(qids), len(data), len(summary), len(comparisons), len(prediction_rows)), flush=True)


if __name__ == "__main__":
    main()
