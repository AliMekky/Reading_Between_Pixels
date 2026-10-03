#!/usr/bin/env python3
"""Audit pilot semantics and analyze existing Q->A teacher-forced margins."""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PILOT = HERE / "outputs/pilot_12_cached"
OUT = HERE / "outputs/relevance_score_followup"
VARIANTS = ("misleading_groundable", "misleading_ungroundable", "irrelevant_word")
WINDOWS = tuple(f"layers_{s:02d}_{s+5:02d}" for s in range(0, 36, 6))
sys.path.insert(0, str(BASE.parent / "open_ended_evaluation/main_files"))
from evaluate_deterministic import normalize_answer


def save_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pilot_semantics():
    records = [json.loads(p.read_text()) for p in sorted((PILOT / "samples").glob("*.json"))]
    assert len(records) == 48 and all(r["status"] == "complete" for r in records)
    qids = {r["question_id"] for r in records}
    prior = defaultdict(set)
    paths = list((BASE.parent / "open_ended_evaluation/outputs/final_classification_six_models/Qwen__Qwen3-VL-8B-Instruct").glob("*.jsonl"))
    paths += [BASE / "free_generation_intervention/outputs/evaluation/final/records_final.jsonl"]
    for path in paths:
        for row in read_jsonl(path):
            if str(row["question_id"]) in qids:
                prior[(str(row["question_id"]), normalize_answer(row["raw_response"]))].add(row["final_category"])
    # Explicit assistant semantic review, not a new independent two-judge evaluation.
    reviews = {
        ("10576809", "food"): ("other", "Food names contents, not bowls/container/plates/whole; no target adoption."),
        ("13539366", "tennis ball"): ("correct_answer", "Tennis ball is a more specific instance of the correct reference ball; not racket/bird/message."),
    }
    labels, audit, unique = {}, [], {}
    for r in records:
        refs = {k: normalize_answer(v) for k, v in r["references"].items()}
        for state, arms in r["states"].items():
            for arm, generation in arms.items():
                text = generation["raw_response"]
                key = (r["question_id"], normalize_answer(text))
                matches = [k for k, v in refs.items() if v == key[1]]
                inherited = prior.get(key, set())
                reason = ""
                if len(matches) == 1:
                    category, source = matches[0], "normalized_exact"
                elif len(inherited) == 1 and None not in inherited:
                    category, source = next(iter(inherited)), "inherited_existing_evaluation"
                elif len(inherited) > 1:
                    category, source = "ambiguous", "conflicting_existing_evaluations"
                    reason = "Existing categories: " + ", ".join(sorted(str(x) for x in inherited))
                elif key in reviews:
                    category, reason = reviews[key]
                    source = "assistant_semantic_review"
                else:
                    raise AssertionError(f"Unreviewed response: {key}")
                labels[(r["question_id"], r["variant"], state, arm)] = category
                row = {"question_id": r["question_id"], "variant": r["variant"], "state": state,
                    "arm": arm, "question": r["question"], "response": text, "category": category,
                    "label_source": source, "review_note": reason,
                    "overlay_reference": r["references"][r["variant"]]}
                audit.append(row)
                unique[key] = {k: v for k, v in row.items() if k not in ("variant", "state", "arm", "overlay_reference")}
    save_csv("pilot_semantic_audit.csv", audit)
    save_csv("pilot_unique_responses.csv", unique.values())
    count_rows, effects = [], []
    for variant in (*VARIANTS, "correct_answer"):
        selected = [r for r in records if r["variant"] == variant]
        arms = list(selected[0]["states"]["overlay"])
        def adoption(state, arm):
            return np.mean([labels[(r["question_id"], variant, state, arm)] == variant for r in selected])
        def effect(arm):
            return adoption("overlay", arm) - adoption("overlay", "baseline") - adoption("no_text", arm) + adoption("no_text", "baseline")
        for state in ("overlay", "no_text"):
            for arm in arms:
                categories = [labels[(r["question_id"], variant, state, arm)] for r in selected]
                count_rows.append({"variant": variant, "state": state, "arm": arm, "n": len(selected),
                    "adoption_count": sum(x == variant for x in categories),
                    "correct_count": sum(x == "correct_answer" for x in categories),
                    "other_count": sum(x == "other" for x in categories),
                    "ambiguous_count": sum(x == "ambiguous" for x in categories)})
        for window in ("middle", "late"):
            q = f"Q_{window}"
            effects.append({"variant": variant, "window": window,
                "q_adoption_DID_pp": 100 * effect(q),
                "q_effect_T_blocked_pp": 100 * (effect(q + "+T") - effect("T")),
                "text_specific_attenuation_pp": 100 * (effect(q + "+R") - effect("R") - effect(q + "+T") + effect("T"))})
    save_csv("pilot_semantic_counts.csv", count_rows)
    save_csv("pilot_semantic_contrasts.csv", effects)
    return records, labels, {"n_records": len(audit), "n_unique_responses": len(unique),
        "label_sources": dict(Counter(r["label_source"] for r in audit)),
        "categories": dict(Counter(r["category"] for r in audit)), "contrasts": effects}


def bootstrap_mean(values):
    x = np.asarray(values, dtype=float)
    idx = np.random.default_rng(271828).integers(0, len(x), size=(5000, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def score_metrics(score):
    assert abs(sum(score["token_logprobs"]) - score["sum_logprob"]) < 1e-6
    assert abs(score["mean_logprob"] * score["token_count"] - score["sum_logprob"]) < 1e-6
    return {"mean": score["mean_logprob"], "sum": score["sum_logprob"],
        "first": score["token_logprobs"][0]}


def analyze_scores(pilot_ids, labels):
    rows, pilot_rows, audit = [], [], {"files": 0, "max_derived_error": 0., "max_baseline_reproduction_error": 0.}
    for variant in VARIANTS:
        paths = sorted((BASE / "attention_intervention/outputs/full" / variant).glob("shard_*/samples/*.json"))
        assert len(paths) == 305
        seen = set()
        for path in paths:
            sample = json.loads(path.read_text())
            qid = str(sample["question_id"])
            assert qid not in seen and sample["status"] == "complete" and sample["variant"] == variant
            seen.add(qid)
            audit["files"] += 1
            lookup = {(x["image_state"], x["window"], x["path"]): x for x in sample["records"]}
            derived = {(x["window"], x["path"]): x["difference_in_differences"] for x in sample["derived_path_effects"]}
            baseline_path = OUT / "pilot_baselines" / f"{qid}_{variant}.json"
            baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
            if baseline:
                assert baseline["status"] == "complete" and baseline["model_revision"] == sample["model_revision"]
                for state, b in baseline["states"].items():
                    error = abs(b["margin"] - sample["baseline_margins"][state])
                    audit["max_baseline_reproduction_error"] = max(audit["max_baseline_reproduction_error"], error)
                    assert error <= 1e-3
            for window in WINDOWS:
                overlay = lookup[("overlay", window, "Q_to_A")]
                no_text = lookup[("no_text", window, "Q_to_A")]
                gain = -overlay["margin_change"]
                did = gain + no_text["margin_change"]
                error = abs(did + derived[(window, "Q_to_A")])
                audit["max_derived_error"] = max(audit["max_derived_error"], error)
                assert error < 1e-9
                score = overlay["scores"]["comparison"]
                row = {"question_id": qid, "variant": variant, "window": window,
                    "pilot": qid in pilot_ids, "correct_answer": sample["answers"]["correct"],
                    "displayed_answer": sample["answers"]["comparison"],
                    "baseline_correct_minus_displayed": sample["baseline_margins"]["overlay"],
                    "blocked_correct_minus_displayed": overlay["blocked_margin"],
                    "raw_relative_displayed_gain": gain, "no_text_relative_displayed_gain": -no_text["margin_change"],
                    "DID_relative_displayed_gain": did,
                    "blocked_displayed_mean_logprob": score["mean_logprob"],
                    "blocked_displayed_sum_logprob": score["sum_logprob"],
                    "blocked_displayed_first_logprob": score["token_logprobs"][0],
                    "blocked_displayed_first_probability": math.exp(score["token_logprobs"][0]),
                    "no_text_blocked_displayed_first_probability": math.exp(no_text["scores"]["comparison"]["token_logprobs"][0]),
                    "blocked_sum_correct_minus_displayed": overlay["scores"]["correct"]["sum_logprob"] - score["sum_logprob"],
                    "blocked_first_correct_minus_displayed": overlay["scores"]["correct"]["token_logprobs"][0] - score["token_logprobs"][0],
                    "shared_correct_first_token": overlay["scores"]["correct"]["token_ids"][0] == score["token_ids"][0],
                    "displayed_token_count": score["token_count"]}
                # These compare Q blocking to visual random blocks; they are NOT
                # raw changes from an unblocked absolute baseline or a matched Q control.
                for role in ("comparison", "correct"):
                    for metric in ("mean", "sum", "first"):
                        diffs = {}
                        for state in ("overlay", "no_text"):
                            qscore = score_metrics(lookup[(state, window, "Q_to_A")]["scores"][role])[metric]
                            random = np.mean([score_metrics(lookup[(state, window, f"R{i}_to_A")]["scores"][role])[metric] for i in (1, 2, 3)])
                            diffs[state] = float(qscore - random)
                        row[f"Q_minus_visual_random_{role}_{metric}"] = diffs["overlay"]
                        row[f"DID_Q_minus_visual_random_{role}_{metric}"] = diffs["overlay"] - diffs["no_text"]
                rows.append(row)
                if baseline and window in ("layers_18_23", "layers_30_35"):
                    arm = "Q_middle" if window == "layers_18_23" else "Q_late"
                    p = dict(row)
                    for state in ("overlay", "no_text"):
                        for role in ("comparison", "correct"):
                            before = baseline["states"][state]["scores"][role]
                            after = lookup[(state, window, "Q_to_A")]["scores"][role]
                            assert before["token_ids"] == after["token_ids"]
                            for metric in ("mean", "sum", "first"):
                                b, a = score_metrics(before)[metric], score_metrics(after)[metric]
                                p[f"{state}_{role}_{metric}_before"] = b
                                p[f"{state}_{role}_{metric}_after"] = a
                                p[f"{state}_{role}_{metric}_gain"] = a - b
                        p[f"{state}_baseline_category"] = labels[(qid, variant, state, "baseline")]
                        p[f"{state}_blocked_category"] = labels[(qid, variant, state, arm)]
                        component_relative_gain = p[f"{state}_comparison_mean_gain"] - p[f"{state}_correct_mean_gain"]
                        assert abs(component_relative_gain + lookup[(state, window, "Q_to_A")]["margin_change"]) <= 1e-3
                    pilot_rows.append(p)
        assert len(seen) == 305
        print(f"[SCORES] {variant}: 305 paired questions audited", flush=True)
    save_csv("question_level_scores.csv", rows)
    save_csv("pilot_absolute_scores.csv", pilot_rows)
    summaries = []
    for subset in ("all_305", "pilot_12"):
        for variant in VARIANTS:
            for window in WINDOWS:
                group = [r for r in rows if r["variant"] == variant and r["window"] == window and (subset == "all_305" or r["pilot"])]
                result = {"subset": subset, "variant": variant, "window": window, "n": len(group)}
                for key in ("baseline_correct_minus_displayed", "blocked_correct_minus_displayed",
                            "raw_relative_displayed_gain", "DID_relative_displayed_gain",
                            "blocked_displayed_mean_logprob", "blocked_displayed_first_logprob",
                            "Q_minus_visual_random_comparison_mean", "DID_Q_minus_visual_random_comparison_mean"):
                    values = [r[key] for r in group]
                    mean, low, high = bootstrap_mean(values)
                    result.update({key + "_mean": mean, key + "_ci_low": low, key + "_ci_high": high,
                                   key + "_median": float(np.median(values))})
                result.update({
                    "raw_relative_gain_positive_count": sum(r["raw_relative_displayed_gain"] > 0 for r in group),
                    "still_below_correct_count": sum(r["blocked_correct_minus_displayed"] > 0 for r in group),
                    "still_below_correct_sum_count": sum(r["blocked_sum_correct_minus_displayed"] > 0 for r in group),
                    "crossed_correct_score_count": sum(r["baseline_correct_minus_displayed"] > 0 and r["blocked_correct_minus_displayed"] <= 0 for r in group),
                    "blocked_first_probability_median": float(np.median([r["blocked_displayed_first_probability"] for r in group]))})
                summaries.append(result)
    save_csv("score_summary.csv", summaries)
    paired = []
    for window in WINDOWS:
        lookup = {(r["question_id"], r["variant"]): r for r in rows if r["window"] == window}
        for other in VARIANTS[:2]:
            values = [r["DID_relative_displayed_gain"] - lookup[(r["question_id"], other)]["DID_relative_displayed_gain"] for r in rows if r["variant"] == "irrelevant_word" and r["window"] == window]
            mean, low, high = bootstrap_mean(values)
            paired.append({"window": window, "contrast": f"irrelevant_minus_{other}", "n": len(values),
                "DID_relative_gain_difference": mean, "descriptive_ci_low": low, "descriptive_ci_high": high})
    save_csv("paired_condition_contrasts.csv", paired)
    generated_baselines = {}
    for r in read_jsonl(BASE / "free_generation_intervention/outputs/evaluation/final/records_final.jsonl"):
        if r["variant"] == "irrelevant_word" and r["image_state"] == "overlay" and r["intervention"] == "baseline":
            generated_baselines[str(r["question_id"])] = r
    assert len(generated_baselines) == 305
    high_support = []
    for r in rows:
        if r["variant"] != "irrelevant_word" or r["window"] != "layers_18_23" or r["blocked_displayed_first_probability"] <= .5:
            continue
        b = generated_baselines[r["question_id"]]
        high_support.append({"question_id": r["question_id"], "displayed_answer": r["displayed_answer"],
            "displayed_token_count": r["displayed_token_count"],
            "blocked_first_token_probability": r["blocked_displayed_first_probability"],
            "no_text_blocked_first_token_probability": r["no_text_blocked_displayed_first_probability"],
            "baseline_generated_answer": b["raw_response"], "baseline_category": b["final_category"],
            "already_adopted_at_baseline": b["final_category"] == "irrelevant_word", "in_pilot": r["pilot"]})
    save_csv("irrelevant_high_first_token_support.csv", high_support)
    audit["irrelevant_high_first_token_cases"] = len(high_support)
    audit["irrelevant_high_first_token_not_previously_adopted"] = sum(not r["already_adopted_at_baseline"] for r in high_support)
    audit["irrelevant_high_first_token_not_previously_adopted_in_pilot"] = sum(not r["already_adopted_at_baseline"] and r["in_pilot"] for r in high_support)
    audit["pilot_absolute_rows"] = len(pilot_rows)
    return rows, pilot_rows, summaries, audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, labels, semantic = pilot_semantics()
    rows, pilot_rows, summaries, audit = analyze_scores({r["question_id"] for r in records}, labels)
    (OUT / "analysis_audit.json").write_text(json.dumps({"semantic": semantic, "scores": audit,
        "confidence_intervals": "Exploratory percentile bootstrap of paired question units, 5000 draws, seed 271828; no confirmatory significance claims or multiplicity-adjusted inference."}, indent=2) + "\n")
    for r in summaries:
        if r["window"] == "layers_18_23":
            print(json.dumps({k: r[k] for k in ("subset", "variant", "n", "baseline_correct_minus_displayed_mean",
                "blocked_correct_minus_displayed_mean", "DID_relative_displayed_gain_mean", "still_below_correct_count", "blocked_first_probability_median")}))
    print(f"[DONE] {OUT}; pilot absolute rows={len(pilot_rows)}/72")


if __name__ == "__main__":
    main()
