#!/usr/bin/env python3
"""Audit and aggregate the full Qwen3-VL-8B generation attention experiment."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = HERE.parent / "outputs/full"
OUTPUT = HERE.parent / "outputs/analysis"
SELECTION = ROOT / "vlms/activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json"
VARIANTS = ("correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word")
WINDOWS = tuple(f"layers_{start:02d}_{start + 5:02d}" for start in range(0, 36, 6))
BOOTSTRAPS = 10_000


def save_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def bootstrap(values, seed):
    values = np.asarray(values, dtype=np.float64)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def pvalue(values):
    try:
        return float(wilcoxon(values, zero_method="zsplit", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm(values):
    values = np.asarray(values); order = np.argsort(values); result = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index]); result[index] = min(running, 1.0)
    return result


def main():
    manifest = json.loads(SELECTION.read_text())
    qids = [str(x["question_id"]) for x in manifest["selected_samples"]]
    if len(qids) != 305 or len(set(qids)) != 305: raise AssertionError("invalid shared manifest")
    expected_ids = set(qids); question_rows = []; interaction_rows = []
    files = records = unavailable = 0; maxima = {"cache": 0., "noop": 0., "blocked": 0., "row": 0.}

    for vi, variant in enumerate(VARIANTS):
        paths = sorted((INPUT / variant).glob("shard_*/samples/*.json"))
        observed = set()
        if len(paths) != 305: raise AssertionError(f"{variant}: files={len(paths)}")
        for number, path in enumerate(paths, 1):
            sample = json.loads(path.read_text()); qid = str(sample["question_id"])
            if sample.get("status") != "complete" or qid in observed or qid not in expected_ids:
                raise AssertionError(f"invalid sample {path}")
            if sample.get("model_id") != "Qwen/Qwen3-VL-8B-Instruct" or tuple(sample["windows"]) != WINDOWS:
                raise AssertionError(f"wrong configuration {path}")
            validation = sample["validation"]
            if validation["saved_records"] != validation["expected_records"] or len(sample["records"]) != validation["saved_records"]:
                raise AssertionError(f"record mismatch {path}")
            for short, key in (("cache", "max_cached_logit_difference"), ("noop", "max_noop_logit_difference"),
                               ("blocked", "max_blocked_probability"), ("row", "max_attention_row_sum_error")):
                maxima[short] = max(maxima[short], float(validation[key]))
            records += len(sample["records"]); unavailable += int(validation["structurally_unavailable_records"])
            derived = {(x["window"], x["path"]): float(x["oriented_effect"])
                       for x in sample["derived_path_effects"]}
            if len(derived) != len(sample["derived_path_effects"]): raise AssertionError(f"duplicate derived key {path}")
            for (window, route), effect in derived.items():
                question_rows.append({"question_id": qid, "variant": variant, "window": window,
                                      "path": route, "oriented_effect": effect})
            for window in WINDOWS:
                for destination in ("Q", "A"):
                    random_keys = [(window, f"R{i}_to_{destination}") for i in (1, 2, 3)]
                    if all(key in derived for key in random_keys):
                        random = float(np.mean([derived[key] for key in random_keys]))
                        question_rows.append({"question_id": qid, "variant": variant, "window": window,
                                              "path": f"R_to_{destination}", "oriented_effect": random})
                        text = derived[(window, f"T_to_{destination}")]
                        question_rows.append({"question_id": qid, "variant": variant, "window": window,
                                              "path": f"T_minus_R_to_{destination}",
                                              "oriented_effect": text - random})
            for row in sample["factorial_interactions"]:
                interaction_rows.append({"question_id": qid, "variant": variant, "window": row["window"],
                                         "object": row["object"], "destination": row["destination_group"],
                                         "overlay_specific_interaction": row["overlay_specific_interaction"]})
            observed.add(qid); files += 1
            if number % 50 == 0 or number == 305: print(f"[AUDIT] {variant} {number}/305", flush=True)
        if observed != expected_ids: raise AssertionError(f"{variant}: question IDs differ")
    if maxima["cache"] > 1e-3 or maxima["noop"] > 1e-3 or maxima["blocked"] != 0 or maxima["row"] > 2e-3:
        raise AssertionError(f"validation tolerance failed {maxima}")

    grouped = {}
    for row in question_rows:
        key = (row["variant"], row["window"], row["path"])
        grouped.setdefault(key, []).append(float(row["oriented_effect"]))
    summaries = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        mean, low, high = bootstrap(values, 42 + index)
        summaries.append({"variant": key[0], "window": key[1], "path": key[2], "n_questions": len(values),
                          "mean_oriented_effect": mean, "ci95_low": low, "ci95_high": high,
                          "p_raw": pvalue(values)})
    for variant in VARIANTS:
        cells = [x for x in summaries if x["variant"] == variant]
        adjusted = holm([x["p_raw"] for x in cells])
        for row, value in zip(cells, adjusted): row["p_holm_within_condition"] = float(value)

    interaction_grouped = {}
    for row in interaction_rows:
        key = (row["variant"], row["window"], row["object"], row["destination"])
        interaction_grouped.setdefault(key, []).append(float(row["overlay_specific_interaction"]))
    interaction_summaries = []
    for index, (key, values) in enumerate(sorted(interaction_grouped.items())):
        mean, low, high = bootstrap(values, 100_042 + index)
        interaction_summaries.append({"variant": key[0], "window": key[1], "object": key[2],
                                      "destination": key[3], "n_questions": len(values),
                                      "mean_overlay_specific_interaction": mean,
                                      "ci95_low": low, "ci95_high": high, "p_raw": pvalue(values)})
    for variant in VARIANTS:
        cells = [x for x in interaction_summaries if x["variant"] == variant]
        adjusted = holm([x["p_raw"] for x in cells])
        for row, value in zip(cells, adjusted): row["p_holm_within_condition"] = float(value)

    question_lookup = {(x["question_id"], x["variant"], x["window"], x["path"]):
                       float(x["oriented_effect"]) for x in question_rows}
    groundedness = []
    for destination in ("Q", "A"):
        for wi, window in enumerate(WINDOWS):
            path = f"T_minus_R_to_{destination}"
            values = [question_lookup[(qid, "misleading_ungroundable", window, path)]
                      - question_lookup[(qid, "misleading_groundable", window, path)] for qid in qids]
            mean, low, high = bootstrap(values, 200_042 + (destination == "A") * 100 + wi)
            groundedness.append({"window": window, "destination": destination, "n_questions": len(values),
                                 "ungrounded_minus_grounded_mean": mean, "ci95_low": low,
                                 "ci95_high": high, "p_raw": pvalue(values)})
    adjusted = holm([x["p_raw"] for x in groundedness])
    for row, value in zip(groundedness, adjusted): row["p_holm_across_12_tests"] = float(value)

    audit = {"status": "pass", "model_id": "Qwen/Qwen3-VL-8B-Instruct", "question_condition_files": files,
             "questions_per_condition": 305, "saved_records": records,
             "structurally_unavailable_records": unavailable, "validation_failures": 0,
             "maximum_validation_values": maxima, "bootstrap_draws": BOOTSTRAPS,
             "bootstrap_unit": "question_id", "multiple_testing": "Holm within each condition"}
    save_csv(OUTPUT / "question_level_path_effects.csv", question_rows)
    save_csv(OUTPUT / "path_summary.csv", summaries)
    save_csv(OUTPUT / "primary_text_path_summary.csv",
             [row for row in summaries if row["path"].startswith("T_minus_R_to_")])
    save_csv(OUTPUT / "question_level_interactions.csv", interaction_rows)
    save_csv(OUTPUT / "interaction_summary.csv", interaction_summaries)
    save_csv(OUTPUT / "groundedness_contrast.csv", groundedness)
    (OUTPUT / "validation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[PASS] files={files} records={records} unavailable={unavailable} output={OUTPUT}")


if __name__ == "__main__":
    main()
