#!/usr/bin/env python3
"""Audit and aggregate the full-305 Qwen3-VL-8B all-layer experiment."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "outputs/step6_8b_full_all_layers"
OUTPUT = HERE.parent / "outputs/step6_8b_analysis"
FULL_SELECTION = HERE.parents[1] / "activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json"
DISCOVERY_SELECTION = HERE / "step4_discovery_selection_seed42.json"
VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")
DIRECTIONS = ("restoration", "insertion")
REGIONS = ("text_region", "matched_random_region_1", "matched_random_region_2", "matched_random_region_3", "all_image_tokens")
LAYERS = tuple(range(36))
BOOTSTRAP_DRAWS = 10_000


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nested_values(value, key):
    found = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                found.append(float(current_value))
            found.extend(nested_values(current_value, key))
    elif isinstance(value, list):
        for current_value in value:
            found.extend(nested_values(current_value, key))
    return found


def bootstrap_columns(values, seed):
    values = np.asarray(values, dtype=np.float64)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return values.mean(axis=0), np.quantile(means, .025, axis=0), np.quantile(means, .975, axis=0)


def holm(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    full = json.loads(FULL_SELECTION.read_text())
    qids = [str(row["question_id"]) for row in full["selected_samples"]]
    discovery = {str(row["question_id"]) for row in json.loads(DISCOVERY_SELECTION.read_text())["selected_samples"]}
    if len(qids) != 305 or len(set(qids)) != 305 or len(discovery) != 40 or not discovery.issubset(qids):
        raise AssertionError("invalid full/discovery manifests")
    qid_index = {qid: index for index, qid in enumerate(qids)}
    shape = (len(VARIANTS), len(DIRECTIONS), len(LAYERS), len(REGIONS), len(qids))
    effects = np.full(shape, np.nan, dtype=np.float64)
    bundle = np.full((len(VARIANTS), len(DIRECTIONS), len(REGIONS), len(qids)), np.nan)
    recipient_correct = np.zeros(shape, dtype=bool)
    patched_correct = np.zeros(shape, dtype=bool)
    noop_max = patch_error = unpatched_change = 0.0
    sample_count = record_count = 0
    expected_keys = {
        ("single_layer_resid_pre", layer, region, direction)
        for layer in LAYERS for region in REGIONS for direction in DIRECTIONS
    } | {
        ("visual_source_bundle", None, region, direction)
        for region in REGIONS for direction in DIRECTIONS
    }

    for variant_index, variant in enumerate(VARIANTS):
        paths = sorted((INPUT / variant).glob("shard_*/samples/*.json"))
        if len(paths) != 305:
            raise AssertionError(f"{variant}: found {len(paths)} files, expected 305")
        observed = set()
        for number, path in enumerate(paths, 1):
            sample = json.loads(path.read_text())
            qid = str(sample["question_id"])
            if qid in observed or qid not in qid_index or sample.get("status") != "complete":
                raise AssertionError(f"invalid or duplicate sample: {path}")
            if sample.get("model_id") != "Qwen/Qwen3-VL-8B-Instruct" or sample.get("layers") != list(LAYERS):
                raise AssertionError(f"wrong model/layers: {path}")
            if len(sample.get("records", [])) != 370 or sample.get("expected_records") != 370:
                raise AssertionError(f"wrong record count: {path}")
            text_count = len(sample["regions"]["text_region"]["token_indices"])
            if any(len(sample["regions"][region]["token_indices"]) != text_count for region in REGIONS[1:4]):
                raise AssertionError(f"unmatched random control: {path}")
            noop_max = max(noop_max, *(float(value) for roles in sample["noop"].values() for value in roles.values()))
            found = set()
            qi = qid_index[qid]
            for record in sample["records"]:
                key = (record["intervention_type"], record["layer"], record["region"], record["direction"])
                if key in found or not math.isfinite(float(record["oriented_effect"])):
                    raise AssertionError(f"duplicate/non-finite record: {path} {key}")
                found.add(key)
                patch_error = max(patch_error, *nested_values(record["integrity"], "patched_to_donor_max_difference"))
                unpatched_change = max(unpatched_change, *nested_values(record["integrity"], "unpatched_direct_max_change"))
                di, ri = DIRECTIONS.index(record["direction"]), REGIONS.index(record["region"])
                if record["intervention_type"] == "single_layer_resid_pre":
                    li = int(record["layer"])
                    effects[variant_index, di, li, ri, qi] = float(record["oriented_effect"])
                    recipient_correct[variant_index, di, li, ri, qi] = record["recipient_preference"] == "correct"
                    patched_correct[variant_index, di, li, ri, qi] = record["patched_preference"] == "correct"
                else:
                    bundle[variant_index, di, ri, qi] = float(record["oriented_effect"])
            if found != expected_keys:
                raise AssertionError(f"incomplete intervention coverage: {path}")
            observed.add(qid)
            sample_count += 1
            record_count += len(sample["records"])
            if number == 1 or number % 50 == 0 or number == len(paths):
                print(f"[AUDIT] variant={variant} samples={number}/305", flush=True)
        if observed != set(qids):
            raise AssertionError(f"{variant}: IDs differ from shared 305")
    if np.isnan(effects).any() or np.isnan(bundle).any():
        raise AssertionError("effect tensor has missing cells")
    if noop_max > 1e-3 or patch_error > 1e-6 or unpatched_change > 1e-6:
        raise AssertionError("numerical integrity tolerance failed")

    heldout_mask = np.asarray([qid not in discovery for qid in qids])
    question_rows, summaries = [], []
    for vi, variant in enumerate(VARIANTS):
        for di, direction in enumerate(DIRECTIONS):
            raw_p = []
            pending = []
            for layer in LAYERS:
                text = effects[vi, di, layer, 0]
                random_mean = effects[vi, di, layer, 1:4].mean(axis=0)
                contrast = text - random_mean
                all_image = effects[vi, di, layer, 4]
                for qi, qid in enumerate(qids):
                    question_rows.append({"question_id": qid, "used_for_discovery": qid in discovery,
                                          "variant": variant, "direction": direction, "layer": layer,
                                          "text_effect": text[qi], "random_mean_effect": random_mean[qi],
                                          "text_minus_random": contrast[qi], "all_image_effect": all_image[qi]})
                try:
                    p_value = float(wilcoxon(contrast, zero_method="zsplit", alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                raw_p.append(p_value)
                pending.append((layer, text, random_mean, contrast, all_image))
            adjusted = holm(raw_p)
            for (layer, text, random_mean, contrast, all_image), p_value, p_holm in zip(pending, raw_p, adjusted):
                for scope_index, (scope, mask) in enumerate((("all_305", np.ones(305, dtype=bool)),
                                                            ("heldout_265", heldout_mask))):
                    columns = np.column_stack((text[mask], random_mean[mask], contrast[mask], all_image[mask]))
                    means, lows, highs = bootstrap_columns(columns, 42 + vi * 10_000 + di * 1_000 + layer * 10 + scope_index)
                    row = {"scope": scope, "variant": variant, "direction": direction, "layer": layer,
                           "n_questions": int(mask.sum()), "text_minus_random_p_raw_all305": p_value,
                           "text_minus_random_p_holm_all305": float(p_holm)}
                    for metric, mean, low, high in zip(("text_effect", "random_mean_effect", "text_minus_random", "all_image_effect"), means, lows, highs):
                        row[f"{metric}_mean"] = float(mean)
                        row[f"{metric}_ci95_low"] = float(low)
                        row[f"{metric}_ci95_high"] = float(high)
                    summaries.append(row)

    bundle_rows = []
    for vi, variant in enumerate(VARIANTS):
        for di, direction in enumerate(DIRECTIONS):
            text = bundle[vi, di, 0]
            random_mean = bundle[vi, di, 1:4].mean(axis=0)
            contrast = text - random_mean
            all_image = bundle[vi, di, 4]
            for scope_index, (scope, mask) in enumerate((("all_305", np.ones(305, dtype=bool)),
                                                        ("heldout_265", heldout_mask))):
                columns = np.column_stack((text[mask], random_mean[mask], contrast[mask], all_image[mask]))
                means, lows, highs = bootstrap_columns(columns, 50_042 + vi * 100 + di * 10 + scope_index)
                row = {"scope": scope, "variant": variant, "direction": direction, "n_questions": int(mask.sum())}
                for metric, mean, low, high in zip(("text_effect", "random_mean_effect", "text_minus_random", "all_image_effect"), means, lows, highs):
                    row[f"{metric}_mean"] = float(mean)
                    row[f"{metric}_ci95_low"] = float(low)
                    row[f"{metric}_ci95_high"] = float(high)
                bundle_rows.append(row)

    peak_rows = []
    for scope in ("all_305", "heldout_265"):
        for variant in VARIANTS:
            for direction in DIRECTIONS:
                cells = [row for row in summaries if row["scope"] == scope and row["variant"] == variant and row["direction"] == direction]
                peak = max(cells, key=lambda row: row["text_minus_random_mean"])
                peak_rows.append({key: peak[key] for key in ("scope", "variant", "direction", "layer", "n_questions",
                                                             "text_effect_mean", "random_mean_effect_mean", "text_minus_random_mean",
                                                             "text_minus_random_ci95_low", "text_minus_random_ci95_high",
                                                             "all_image_effect_mean", "text_minus_random_p_holm_all305")})

    audit = {"status": "pass", "model_id": "Qwen/Qwen3-VL-8B-Instruct",
             "question_condition_files": sample_count, "questions_per_condition": 305,
             "discovery_questions": 40, "heldout_questions": int(heldout_mask.sum()),
             "records": record_count, "expected_records": 451_400,
             "maximum_noop_logit_difference": noop_max,
             "maximum_patch_to_donor_difference": patch_error,
             "maximum_direct_unpatched_change": unpatched_change,
             "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "question_id"}
    if record_count != audit["expected_records"]:
        raise AssertionError("global record accounting failed")
    save_csv(OUTPUT / "question_level_layer_contrasts.csv", question_rows)
    save_csv(OUTPUT / "layer_summary.csv", summaries)
    save_csv(OUTPUT / "source_bundle_summary.csv", bundle_rows)
    save_csv(OUTPUT / "peak_layer_summary.csv", peak_rows)
    save_json(OUTPUT / "validation_audit.json", audit)
    print(f"[PASS] files={sample_count} records={record_count} heldout={heldout_mask.sum()} output={OUTPUT}")


if __name__ == "__main__":
    main()
