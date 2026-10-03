#!/usr/bin/env python3
"""Conservative exact-match pilot summary; not final semantic evaluation."""
import argparse
import csv
import json
import re
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.casefold())).strip()


def summarize(root):
    config = json.loads((root / "configuration.json").read_text())
    rows, contrasts, responses = [], [], []
    for variant in config["variants"]:
        records = [json.loads((root / "samples" / f"{e['question_id']}_{variant}.json").read_text())
                   for e in config["selected_samples"]]
        assert all(r.get("status") == "complete" for r in records)
        arms = list(records[0]["states"]["overlay"])
        def match(record, state, arm, reference):
            return int(normalize(record["states"][state][arm]["raw_response"]) == normalize(reference))
        def adoption(state, arm):
            return sum(match(r, state, arm, r["references"][variant]) for r in records) / len(records)
        def effect(arm):
            return ((adoption("overlay", arm) - adoption("overlay", "baseline"))
                    - (adoption("no_text", arm) - adoption("no_text", "baseline")))
        for state in ("no_text", "overlay"):
            for arm in arms:
                correct = sum(match(r, state, arm, r["references"]["correct_answer"]) for r in records)
                unmatched = sum(not any(match(r, state, arm, ref) for ref in r["references"].values()) for r in records)
                rows.append({"variant": variant, "state": state, "arm": arm, "n": len(records),
                    "overlay_exact_count": round(adoption(state, arm) * len(records)),
                    "correct_exact_count": correct, "unmatched_count": unmatched,
                    "truncated_count": sum(r["states"][state][arm]["termination_reason"] == "max_tokens" for r in records),
                    "answer_changed_count": sum(normalize(r["states"][state][arm]["raw_response"]) != normalize(r["states"][state]["baseline"]["raw_response"]) for r in records),
                    "new_overlay_adoption_count": sum(match(r, state, arm, r["references"][variant]) and not match(r, state, "baseline", r["references"][variant]) for r in records),
                    "lost_overlay_adoption_count": sum(not match(r, state, arm, r["references"][variant]) and match(r, state, "baseline", r["references"][variant]) for r in records)})
                for r in records:
                    result = r["states"][state][arm]
                    responses.append({"question_id": r["question_id"], "variant": variant,
                        "state": state, "arm": arm, "question": r["question"],
                        "overlay_reference": r["references"][variant],
                        "correct_reference": r["references"]["correct_answer"],
                        "response": result["raw_response"],
                        "baseline_response": r["states"][state]["baseline"]["raw_response"],
                        "termination_reason": result["termination_reason"]})
        for window in config["q_windows"]:
            q = f"Q_{window}"
            contrasts.append({"variant": variant, "q_window": window,
                "q_adoption_DID_pp": 100 * effect(q),
                "q_effect_without_T_pp": 100 * (effect(q + "+T") - effect("T")),
                "q_effect_without_R_pp": 100 * (effect(q + "+R") - effect("R")),
                "uncontrolled_attenuation_pp": 100 * (effect(q) - (effect(q + "+T") - effect("T"))),
                "text_specific_attenuation_pp": 100 * ((effect(q + "+R") - effect("R"))
                                                      - (effect(q + "+T") - effect("T")))})
    condition_contrasts = []
    for window in config["q_windows"]:
        by_variant = {c["variant"]: c for c in contrasts if c["q_window"] == window}
        for other in ("misleading_groundable", "misleading_ungroundable"):
            condition_contrasts.append({"q_window": window, "comparison": f"irrelevant_word_minus_{other}",
                "q_adoption_DID_difference_pp": by_variant["irrelevant_word"]["q_adoption_DID_pp"] - by_variant[other]["q_adoption_DID_pp"],
                "attenuation_difference_pp": by_variant["irrelevant_word"]["text_specific_attenuation_pp"] - by_variant[other]["text_specific_attenuation_pp"]})
    for name, data in (("exact_counts", rows), ("exploratory_contrasts", contrasts),
                       ("condition_contrasts", condition_contrasts), ("responses", responses)):
        with (root / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(contrasts, indent=2))
    print("Exploratory exact matching only. Unmatched is not incorrect. No significance or semantic-gating claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
