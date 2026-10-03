#!/usr/bin/env python3
"""Convert validated raw reports into compact plotting tables."""

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
    parser.add_argument("--inputs", nargs="+", required=True)
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
    validation_rows = []

    if len(args.inputs) != len(variants):
        raise ValueError("Expected {} reports, received {}".format(len(variants), len(args.inputs)))
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
        condition_rows = [sample["conditions"][expected_variant] for sample in report["samples"].values()]
        maximum = lambda values: max(values, default=0.0)
        validation = {
            "variant": expected_variant, "questions": len(current_qids),
            "expected_interventions": expected, "saved_interventions": len(report["records"]),
            "remaining_interventions": int(report["completion"]["remaining_interventions"]),
            "max_no_text_noop_difference": maximum(float(row["noop"]["no_text_max_logit_difference"]) for row in condition_rows),
            "max_overlay_noop_difference": maximum(float(row["noop"]["overlay_max_logit_difference"]) for row in condition_rows),
            "max_cleaned_outside_mismatch_pixels": maximum(int(row["cleaned_overlay_validation"]["cleaned_outside_no_text_mismatch_pixels"]) for row in condition_rows),
            "max_cleaned_inside_mismatch_pixels": maximum(int(row["cleaned_overlay_validation"]["cleaned_inside_original_mismatch_pixels"]) for row in condition_rows),
            "nonfinite_effects": 0, "max_patched_donor_difference": 0.0,
            "max_unpatched_direct_change": 0.0,
        }
        for record in report["records"]:
            key = (str(record["question_id"]), expected_variant, record["direction"], int(record["layer"]), record["region"])
            data[key] = float(record["effect"])
            clean[key] = record.get("clean_object_control")
            if not np.isfinite(float(record["effect"])):
                validation["nonfinite_effects"] += 1
            for layer_integrity in record["integrity"]["by_layer"].values():
                validation["max_patched_donor_difference"] = max(
                    validation["max_patched_donor_difference"],
                    float(layer_integrity["patched_donor_max_abs_difference_after"]),
                )
                validation["max_unpatched_direct_change"] = max(
                    validation["max_unpatched_direct_change"],
                    float(layer_integrity["unpatched_positions_max_direct_change"]),
                )
            if record["region"] == "text_region":
                predictions[key[:-1]] = {
                    "recipient": record["recipient_prediction"],
                    "patched": record["patched_prediction"],
                    "target": record["target_option_letter"],
                }
        validation_rows.append(validation)
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

    short = {
        "misleading_groundable": "grounded",
        "misleading_ungroundable": "ungrounded",
        "irrelevant_word": "irrelevant",
        "correct_answer": "correct",
    }
    pairs = [
        ("misleading_ungroundable", "misleading_groundable"),
        ("misleading_groundable", "irrelevant_word"),
        ("misleading_ungroundable", "irrelevant_word"),
        ("correct_answer", "misleading_groundable"),
        ("correct_answer", "misleading_ungroundable"),
        ("correct_answer", "irrelevant_word"),
    ]
    comparison_names = [short[a] + "_minus_" + short[b] for a, b in pairs]
    comparison_names += [short[variant] + "_text_minus_random" for variant in variants]
    comparisons = []
    for direction in DIRECTIONS:
        for layer in range(32):
            get = lambda variant, region: values_by_group[(variant, direction, layer, region)]
            arrays = {
                short[a] + "_minus_" + short[b]: get(a, "text_region") - get(b, "text_region")
                for a, b in pairs
            }
            arrays.update({
                short[variant] + "_text_minus_random":
                get(variant, "text_region") - get(variant, "matched_random_mean")
                for variant in variants
            })
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
                if variant == "correct_answer" and direction == "restoration":
                    eligible = [(qid, row) for qid, row in rows if row["recipient"] == correct_letters[qid]]
                    desired = lambda qid, row: row["patched"] != correct_letters[qid]
                    transition = "correct_to_noncorrect"
                elif variant == "correct_answer":
                    eligible = [(qid, row) for qid, row in rows if row["recipient"] != correct_letters[qid]]
                    desired = lambda qid, row: row["patched"] == correct_letters[qid]
                    transition = "noncorrect_to_correct"
                elif direction == "restoration":
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

    early_summary, early_values = [], {}
    for variant in variants:
        for direction in DIRECTIONS:
            for region in SUMMARY_REGIONS:
                values = []
                for qid in qids:
                    if region == "matched_random_mean":
                        per_layer = [np.mean([
                            data[(qid, variant, direction, layer, "matched_random_region_{}".format(index))]
                            for index in (1, 2, 3)]) for layer in range(7)]
                    else:
                        keys = [(qid, variant, direction, layer, region) for layer in range(7)]
                        if region.endswith("object_region") and any(clean.get(key) is not True for key in keys):
                            continue
                        per_layer = [data[key] for key in keys]
                    values.append(np.mean(per_layer))
                values = np.asarray(values)
                early_values[(variant, direction, region)] = values
                mean, low, high = mean_ci(values, rng, args.bootstrap_draws)
                early_summary.append({
                    "variant": variant, "direction": direction, "layers": "0-6",
                    "region": region, "n": len(values), "mean_effect": mean,
                    "ci_95_low": low, "ci_95_high": high,
                })

    early_comparisons = []
    for direction in DIRECTIONS:
        arrays = {
            short[a] + "_minus_" + short[b]:
            early_values[(a, direction, "text_region")] - early_values[(b, direction, "text_region")]
            for a, b in pairs
        }
        arrays.update({
            short[variant] + "_text_minus_random":
            early_values[(variant, direction, "text_region")] - early_values[(variant, direction, "matched_random_mean")]
            for variant in variants
        })
        p_values = [wilcoxon_p(values) for values in arrays.values()]
        for (name, values), p_value in zip(arrays.items(), p_values):
            mean, low, high = mean_ci(values, rng, args.bootstrap_draws)
            early_comparisons.append({
                "direction": direction, "comparison": name, "n": len(values),
                "mean_difference": mean, "ci_95_low": low, "ci_95_high": high,
                "wilcoxon_p_value": p_value,
                "bonferroni_p_value": min(1.0, p_value * len(arrays)),
            })

    output = Path(args.output_dir)
    write_csv(output / "layerwise_summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "prediction_transitions.csv", prediction_rows)
    write_csv(output / "early_window_summary.csv", early_summary)
    write_csv(output / "early_window_comparisons.csv", early_comparisons)
    write_csv(output / "validation_summary.csv", validation_rows)
    validation_passed = all(
        row["remaining_interventions"] == 0 and row["nonfinite_effects"] == 0 and
        row["max_no_text_noop_difference"] <= 1e-3 and
        row["max_overlay_noop_difference"] <= 1e-3 and
        row["max_cleaned_outside_mismatch_pixels"] == 0 and
        row["max_cleaned_inside_mismatch_pixels"] == 0 and
        row["max_patched_donor_difference"] <= 1e-3 and
        row["max_unpatched_direct_change"] <= 1e-3
        for row in validation_rows
    )
    metadata = {"status": "success", "variants": variants,
                "sample_count_per_condition": len(qids), "layers": 32,
                "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
                "raw_interventions": len(data), "validation_passed": validation_passed}
    (output / "plot_data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[COMPLETE] samples={} raw_interventions={} summary_rows={} comparison_rows={} prediction_rows={}".format(
        len(qids), len(data), len(summary), len(comparisons), len(prediction_rows)), flush=True)


if __name__ == "__main__":
    main()
