#!/usr/bin/env python3
"""Analyze the fixed paired input pilot; refuse incomplete results."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "open_ended_evaluation/main_files"))
from evaluate_deterministic import normalize_answer as norm
OUT = HERE / "outputs/input_comparison_12"
KEYS = ["correct_answer", "misleading_groundable", "misleading_ungroundable", "irrelevant_word"]
STATES = ["question_only", "gray_image", "clean_image", *KEYS]
LABELS = ["Correct", "Grounded misleading", "Ungrounded misleading", "Irrelevant"]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    completion = json.loads((OUT / "completion.json").read_text())
    assert completion["status"] == "complete"
    config = json.loads((OUT / "configuration.json").read_text())
    qids = [x["question_id"] for x in config["selected_samples"]]
    records = {(r["question_id"], r["state"]): r for p in sorted((OUT / "samples").glob("*.json"))
               for r in [json.loads(p.read_text())]}
    assert set(records) == {(q, s) for q in qids for s in STATES}
    prior = {}
    with (HERE / "outputs/relevance_score_followup/pilot_unique_responses.csv").open() as handle:
        for row in csv.DictReader(handle):
            prior[(row["question_id"], norm(row["response"]))] = row
    # Optional explicit review, kept separate from model outputs and exact matching.
    reviews_path = OUT / "semantic_review.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
    flat, generations, checks = [], [], []
    for q in qids:
        reference = records[q, "gray_image"]
        for s in STATES:
            r = records[q, s]
            assert r["status"] == "complete"
            if s != "question_only":
                assert r["prompt_token_ids"] == reference["prompt_token_ids"]
                assert r["image_grid_thw"] == reference["image_grid_thw"]
            else:
                assert r["image_token_count"] == 0
            if "cache_validation" in r:
                checks.append(r["cache_validation"])
            for k in KEYS:
                v = r["scores"][k]
                assert v["token_ids"] == records[q, "question_only"]["scores"][k]["token_ids"]
                flat.append(dict(question_id=q, state=s, candidate=k, answer=v["answer"],
                    token_count=v["token_count"], mean_logprob=v["mean_logprob"], sum_logprob=v["sum_logprob"],
                    first_logprob=v["token_logprobs"][0], first_probability=math.exp(v["token_logprobs"][0])))
            response = r["generation"]["raw_response"]
            matches = [k for k, answer in r["references"].items() if norm(answer) == norm(response)]
            old = prior.get((q, norm(response)))
            review = reviews.get(q + "/" + norm(response))
            if len(matches) == 1:
                category, source = matches[0], "normalized_exact"
            elif review:
                category, source = review["category"], "explicit_assistant_review: " + review["reason"]
            elif old:
                category, source = old["category"], "prior_audit: " + old["label_source"]
            else:
                category, source = "unreviewed", "requires_review"
            generations.append(dict(question_id=q, state=s, question=r["question"], response=response,
                category=category, label_source=source, references=json.dumps(r["references"])))
    assert len(flat) == 336 and len(generations) == 84 and len(checks) == 6
    assert all(c["generation_tokens_identical"] and c["max_token_logprob_error"] <= .001 for c in checks)
    write_csv("candidate_scores.csv", flat)
    write_csv("generation_audit.csv", generations)

    def values(state, key, metric="mean_logprob"):
        return np.array([records[q, key if state == "own_overlay" else state]["scores"][key][metric] for q in qids])

    summaries, paired = [], []
    for metric in ["mean_logprob", "sum_logprob"]:
        for s in [*STATES, "own_overlay"]:
            for k in KEYS:
                x = values(s, k, metric)
                summaries.append(dict(metric=metric, state=s, candidate=k, mean=float(x.mean()), median=float(np.median(x)), n=len(x)))
        for s in STATES[:3]:
            gap = (values(s, KEYS[1], metric) + values(s, KEYS[2], metric))/2 - values(s, KEYS[3], metric)
            for q, v in zip(qids, gap):
                paired.append(dict(question_id=q, metric=metric, contrast="relevant_misleading_minus_irrelevant", state=s, candidate="paired", value=float(v)))
        for k in KEYS:
            for target, baseline in [("gray_image", "question_only"), ("clean_image", "question_only"),
                                     ("clean_image", "gray_image"), ("own_overlay", "clean_image")]:
                for q, v in zip(qids, values(target, k, metric)-values(baseline, k, metric)):
                    paired.append(dict(question_id=q, metric=metric, contrast=target+"_minus_"+baseline,
                                       state=target, candidate=k, value=float(v)))
    write_csv("score_summary.csv", summaries)
    write_csv("paired_contrasts.csv", paired)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = ["#2878b5", "#e69532", "#8b61b0", "#c94b49"]
    display = ["question_only", "gray_image", "clean_image", "own_overlay"]
    for k, label, color in zip(KEYS, LABELS, colors):
        axes[0].plot(range(4), [values(s,k).mean() for s in display], "o-", label=label, color=color)
    axes[0].set_xticks(range(4), ["Question only", "Gray image", "Clean scene", "Own overlay"])
    axes[0].set_ylabel("Mean candidate log probability per token (nats)")
    axes[0].set_title("Candidate support across inputs (12 questions)")
    axes[0].legend(fontsize=8)
    gaps = np.array([(values(s, KEYS[1])+values(s, KEYS[2]))/2-values(s,KEYS[3]) for s in STATES[:3]])
    for i in range(12):
        axes[1].plot(range(3), gaps[:,i], "o-", color="#999999", alpha=.45, linewidth=.8)
    axes[1].plot(range(3), gaps.mean(axis=1), "o-", color="#222222", linewidth=2.5, label="Mean; gray lines = questions")
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set_xticks(range(3), ["Question only", "Gray image", "Clean scene"])
    axes[1].set_ylabel("Relevant misleading minus irrelevant score (nats/token)")
    axes[1].set_title("Does the relevance gap precede scene evidence?")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / ("input_comparison." + suffix), dpi=180)
    plt.close(fig)
    lines = ["# Input comparison: fixed 12-question exploratory pilot", "",
        "Same 12 questions and four candidate answers in all inputs. The four input families expand to seven states: question only, gray image, clean scene, and each of four overlays. All 84 generations and 336 full-candidate scores completed.", "",
        "The pilot supports a preexisting question-conditioned disadvantage for irrelevant candidates: the mean relevance gap is positive with both question-only and gray-image inputs, and in 10/12 questions under each control. The clean scene widens the average gap. Adding irrelevant text raises its own score relative to the clean image but produces no irrelevant-answer adoption in these 12 questions. This is consistent with irrelevant answers starting too far behind; it does not isolate the source of that disadvantage or establish a Q→A mechanism.", "",
        "The controls are behaviorally different: gray images elicit absence statements or abstentions in 8/12 questions, versus 0/12 for question-only inputs. Retain both controls rather than treating gray inputs as a neutral substitute for no image.", "",
        "## Candidate support", "", "Scores below are mean log probability per answer token, averaged over questions (nats; higher is better). They are not answer probabilities. Each own-overlay entry scores that candidate on its corresponding overlay, so that column combines four image states.", "",
        "| Candidate | Question only | Gray image | Clean scene | Own overlay | Overlay − clean |",
        "|---|---:|---:|---:|---:|---:|"]
    for k, label in zip(KEYS, LABELS):
        means = [values(s,k).mean() for s in display]
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in [*means, means[-1]-means[-2]]) + " |")
    lines += ["", "## Within-question relevance gap", "",
        "The gap is the average of grounded and ungrounded misleading candidate scores minus the irrelevant candidate score. Positive values mean irrelevant answers start behind relevant misleading answers.", "",
        "| Input | Mean gap | Median gap | Positive gaps / 12 | Sum-logprob gap (sensitivity) |",
        "|---|---:|---:|---:|---:|"]
    for s, label, gap in zip(STATES[:3], ["Question only", "Gray image", "Clean scene"], gaps):
        total = (values(s,KEYS[1],"sum_logprob")+values(s,KEYS[2],"sum_logprob"))/2-values(s,KEYS[3],"sum_logprob")
        lines.append(f"| {label} | {gap.mean():.3f} | {np.median(gap):.3f} | {int((gap>0).sum())}/12 | {total.mean():.3f} |")
    lines += ["", "## Generated answers (secondary)", "",
        "Accuracy refers to the original scene's answer even when the scene is absent; it is a diagnostic of guessing, not visual correctness on a gray image. Unreviewed and ambiguous outputs are reported separately rather than silently counted as errors.", "",
        "| Input | Correct / 12 | Irrelevant adoption / 12 | Absence or abstention / 12 | Unreviewed or ambiguous / 12 |",
        "|---|---:|---:|---:|---:|"]
    for s in STATES:
        rows = [r for r in generations if r["state"] == s]
        counts = [sum(r["category"] == c for r in rows) for c in ["correct_answer", "irrelevant_word", "abstention"]]
        counts[2] += sum(r["category"] == "no_object" for r in rows)
        unresolved = sum(r["category"] in ["unreviewed", "ambiguous"] for r in rows)
        lines.append(f"| {s} | {counts[0]} | {counts[1]} | {counts[2]} | {unresolved} |")
    lines += ["", "## Scope and validation", "",
        "Question-only scores measure question-conditioned answer preferences, which combine lexical frequency, learned associations, and plausibility. Gray-image scores retain visual tokens but are not a pure language prior or a guaranteed in-distribution baseline. Agreement between these controls strengthens a descriptive interpretation; disagreement should be retained.", "",
        "No attention intervention is run here. This comparison can locate a preexisting score gap and describe scene/overlay changes; it cannot establish that language priors cause all taxonomy differences or that Q→A specifically implements those priors. Subtracting the same baseline from blocked and unblocked scores does not change their difference. The earlier intervention used an image-specific instruction; this pilot recomputes all states with one modality-neutral instruction, so absolute scores should not be spliced across runs.", "",
        "Twelve paired questions are exploratory. Candidate scores are teacher-forced, do not include EOS, and do not compete against the full set of generated answers. Token-length sensitivity is exported using summed log probabilities; first-token probabilities and all per-question results are also exported. No significance or causal fraction-explained claim is made.", "",
        f"Validation passed: 84 input records; 336 candidate scores; identical candidate tokenization across states; identical visual prompt IDs and grids across six images per question; six cached-versus-native generation checks; maximum cached token log-probability error {max(c['max_token_logprob_error'] for c in checks):.6g}. Overlay pixel checks are saved with each record.", "",
        "![Input comparison](input_comparison.png)", "",
        "Artifacts: [candidate scores](candidate_scores.csv), [paired contrasts](paired_contrasts.csv), [score summary](score_summary.csv), [generation audit](generation_audit.csv), [PDF figure](input_comparison.pdf)."]
    (OUT / "report.md").write_text("\n".join(lines)+"\n")
    print("PASS: 84 inputs, 336 scores, six cache validations; report written.")
    for row in generations:
        if row["category"] == "unreviewed":
            print(json.dumps(row))


if __name__ == "__main__":
    main()
