#!/usr/bin/env python3
"""Join behavior, activation patching, attention blocking, and free-generation effects."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


VARIANTS = ("misleading_groundable", "misleading_ungroundable")
RANDOM_REGIONS = tuple(f"matched_random_region_{index}" for index in (1, 2, 3))


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_question(path):
    return {str(row["question_id"]): row for row in read_jsonl(path)}


def sample_paths(root, variant):
    return sorted(path for shard in (0, 1) for path in (root / variant / f"shard_{shard}" / "samples").glob("*.json"))


def activation_metric(sample, direction, layers=range(5, 13)):
    records = {(row["layer"], row["region"]): row["oriented_effect"] for row in sample["records"]
               if row["intervention_type"] == "single_layer_resid_pre" and row["direction"] == direction}
    values = []
    for layer in layers:
        text = records[(layer, "text_region")]
        random = np.mean([records[(layer, region)] for region in RANDOM_REGIONS])
        values.append(text - random)
    return float(np.mean(values))


def path_metric(sample, window, source, destination):
    rows = {(row["window"], row["path"]): row["oriented_effect"] for row in sample["derived_path_effects"]}
    text = rows[(window, f"{source}_to_{destination}")]
    if source != "T":
        return float(text)
    random = np.mean([rows[(window, f"R{index}_to_{destination}")] for index in (1, 2, 3)])
    return float(text - random)


def free_measure(lookup, variant, qid, state, region, name):
    baseline = lookup[(variant, qid, state, "baseline")]
    blocked = lookup[(variant, qid, state, region)]
    before, after = baseline["final_category"], blocked["final_category"]
    if name == "changed":
        return float(baseline["output_token_ids"] != blocked["output_token_ids"])
    if name == "target_removed":
        return float(before == variant and after != variant)
    if name == "recovered":
        return float(before != "correct_answer" and after == "correct_answer")
    raise KeyError(name)


def adjusted_free_metric(lookup, variant, qid, name):
    def specific(state):
        text = free_measure(lookup, variant, qid, state, "text_region", name)
        random = np.mean([free_measure(lookup, variant, qid, state, region, name) for region in RANDOM_REGIONS])
        return text - random
    return float(specific("overlay") - specific("no_text"))


def interval(values, rng, draws):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def difference_interval(left, right, rng, draws):
    left, right = np.asarray(left), np.asarray(right)
    estimates = (left[rng.integers(0, len(left), size=(draws, len(left)))].mean(1)
                 - right[rng.integers(0, len(right), size=(draws, len(right)))].mean(1))
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def permutation_p(left, right, rng, draws):
    left, right = np.asarray(left), np.asarray(right)
    observed = abs(left.mean() - right.mean())
    pooled = np.concatenate([left, right]); size = len(left); exceed = 0
    for _ in range(draws):
        permuted = rng.permutation(pooled)
        exceed += abs(permuted[:size].mean() - permuted[size:].mean()) >= observed
    return (exceed + 1) / (draws + 1)


def holm(values):
    adjusted = [0.0] * len(values); running = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def partial_spearman(left, right, controls):
    """Rank correlation after linearly residualizing ranked control variables."""
    x, y = rankdata(left), rankdata(right)
    design = np.column_stack([np.ones(len(x)), *[rankdata(value) for value in controls]])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    result = pearsonr(x_residual, y_residual)
    return float(result.statistic), float(result.pvalue)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_dir", type=Path, required=True)
    parser.add_argument("--attention_dir", type=Path, required=True)
    parser.add_argument("--behavior_dir", type=Path, required=True)
    parser.add_argument("--free_records", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    behavior = {condition: by_question(args.behavior_dir / f"{condition}.jsonl")
                for condition in ("notext", *VARIANTS)}
    free = {}
    for row in read_jsonl(args.free_records):
        free[(row["variant"], str(row["question_id"]), row["image_state"], row["intervention"])] = row
    joined = []
    for variant in VARIANTS:
        activation = {path.stem: json.loads(path.read_text()) for path in sample_paths(args.activation_dir, variant)}
        attention = {path.stem: json.loads(path.read_text()) for path in sample_paths(args.attention_dir, variant)}
        if len(activation) != 305 or set(activation) != set(attention):
            raise AssertionError(f"unaligned causal samples for {variant}")
        for qid in sorted(activation):
            before = behavior["notext"][qid]["final_category"]
            overlay = behavior[variant][qid]["final_category"]
            group = ("fooled" if before == "correct_answer" and overlay == variant else
                     "robust" if before == "correct_answer" and overlay == "correct_answer" else
                     "baseline_correct_other" if before == "correct_answer" else "baseline_not_correct")
            a, t = activation[qid], attention[qid]
            joined.append({
                "variant": variant, "question_id": qid, "behavior_group": group,
                "no_text_category": before, "overlay_category": overlay,
                "baseline_margin_no_text": a["baseline_margins"]["no_text"],
                "baseline_margin_overlay": a["baseline_margins"]["overlay"],
                "overlay_margin_shift": a["baseline_margins"]["overlay"] - a["baseline_margins"]["no_text"],
                "text_token_count": len(t["groups"]["T"]),
                "activation_restoration_L5_12": activation_metric(a, "restoration"),
                "activation_insertion_L5_12": activation_metric(a, "insertion"),
                "attention_TA_L30_35": path_metric(t, "layers_30_35", "T", "A"),
                "attention_TQ_L12_17": path_metric(t, "layers_12_17", "T", "Q"),
                "attention_QA_L30_35": path_metric(t, "layers_30_35", "Q", "A"),
                "free_changed_adjusted": adjusted_free_metric(free, variant, qid, "changed"),
                "free_target_removed_adjusted": adjusted_free_metric(free, variant, qid, "target_removed"),
                "free_recovered_adjusted": adjusted_free_metric(free, variant, qid, "recovered"),
            })

    metrics = [key for key in joined[0] if key not in {
        "variant", "question_id", "behavior_group", "no_text_category", "overlay_category"
    }]
    rng = np.random.default_rng(args.seed); summaries, contrasts = [], []
    for variant in VARIANTS:
        current = [row for row in joined if row["variant"] == variant]
        condition_contrasts = []
        for group in ("fooled", "robust"):
            rows = [row for row in current if row["behavior_group"] == group]
            for metric in metrics:
                values = [float(row[metric]) for row in rows]
                low, high = interval(values, rng, args.resamples)
                summaries.append({"variant": variant, "group": group, "metric": metric, "n": len(rows),
                                  "mean": float(np.mean(values)), "ci_low": low, "ci_high": high})
        fooled = [row for row in current if row["behavior_group"] == "fooled"]
        robust = [row for row in current if row["behavior_group"] == "robust"]
        for metric in metrics:
            left = [float(row[metric]) for row in fooled]; right = [float(row[metric]) for row in robust]
            low, high = difference_interval(left, right, rng, args.resamples)
            condition_contrasts.append({
                "variant": variant, "contrast": "fooled_minus_robust", "metric": metric,
                "n_fooled": len(left), "n_robust": len(right), "difference": float(np.mean(left)-np.mean(right)),
                "ci_low": low, "ci_high": high, "p_raw": permutation_p(left, right, rng, args.resamples),
            })
        adjusted = holm([row["p_raw"] for row in condition_contrasts])
        for row, value in zip(condition_contrasts, adjusted): row["p_holm"] = value
        contrasts.extend(condition_contrasts)

    correlations = []
    pairs = (("activation_restoration_L5_12", "attention_TA_L30_35"),
             ("attention_TA_L30_35", "free_target_removed_adjusted"),
             ("activation_restoration_L5_12", "free_target_removed_adjusted"))
    for variant in VARIANTS:
        for subset in ("all", "baseline_correct"):
            rows = [row for row in joined if row["variant"] == variant and
                    (subset == "all" or row["no_text_category"] == "correct_answer")]
            for left, right in pairs:
                result = spearmanr([row[left] for row in rows], [row[right] for row in rows])
                partial_rho, partial_p = partial_spearman(
                    [row[left] for row in rows], [row[right] for row in rows],
                    [[row["text_token_count"] for row in rows]],
                )
                correlations.append({"variant": variant, "subset": subset, "x": left, "y": right,
                                     "n": len(rows), "spearman_rho": float(result.statistic),
                                     "p_raw": float(result.pvalue),
                                     "partial_rho_controlling_text_tokens": partial_rho,
                                     "partial_p": partial_p})

    write_csv(args.output_dir / "joined_sample_metrics.csv", joined)
    write_csv(args.output_dir / "group_summary.csv", summaries)
    write_csv(args.output_dir / "fooled_minus_robust.csv", contrasts)
    write_csv(args.output_dir / "correlations.csv", correlations)
    group_counts = {variant: {group: sum(row["variant"] == variant and row["behavior_group"] == group for row in joined)
                              for group in ("fooled", "robust", "baseline_correct_other", "baseline_not_correct")}
                    for variant in VARIANTS}
    summary = {"status": "complete", "joined_rows": len(joined), "group_counts": group_counts,
               "activation_window": [5, 12], "attention_answer_window": [30, 35],
               "attention_question_window": [12, 17], "bootstrap_resamples": args.resamples,
               "permutation_draws": args.resamples, "seed": args.seed}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[PASS] {summary}")


if __name__ == "__main__":
    main()
