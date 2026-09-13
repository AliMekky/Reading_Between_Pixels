#!/usr/bin/env python3
"""Create compact paper-facing tables from saved classifications and statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics_dir", type=Path, required=True)
    parser.add_argument("--classification_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.statistics_dir / "metrics.csv")
    transitions = pd.read_csv(args.statistics_dir / "transition_summary.csv")

    selected = metrics[
        metrics.metric.isin(["accuracy", "accuracy_change", "adjusted_target_following"])
        & (metrics.condition != "ungrounded_minus_grounded")
    ].copy()
    selected.to_csv(args.output_dir / "main_results_table.csv", index=False)
    metrics[metrics.condition == "ungrounded_minus_grounded"].to_csv(
        args.output_dir / "grounded_ungrounded_contrasts.csv", index=False
    )
    wanted = {
        ("correct_answer", "helpful_targeted_flip"),
        ("misleading_groundable", "harmful_targeted_flip"),
        ("misleading_ungroundable", "harmful_targeted_flip"),
        ("irrelevant_word", "irrelevant_adoption"),
    }
    transitions[[pair in wanted for pair in zip(transitions.condition, transitions.transition)]].to_csv(
        args.output_dir / "targeted_transitions_table.csv", index=False
    )

    summaries = json.loads((args.classification_dir / "summary.json").read_text())
    stage, category = Counter(), Counter()
    for summary in summaries:
        stage.update(summary["stage_counts"]); category.update(summary["category_counts"])
    pd.DataFrame([{"stage": key, "count": value, "rate": value / sum(stage.values())}
                  for key, value in stage.items()]).to_csv(args.output_dir / "classification_stages.csv", index=False)
    pd.DataFrame([{"category": key, "count": value, "rate": value / sum(category.values())}
                  for key, value in category.items()]).to_csv(args.output_dir / "classification_categories.csv", index=False)
    print(f"[PASS] tables={args.output_dir}")


if __name__ == "__main__": main()
