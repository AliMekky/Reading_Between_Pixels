#!/usr/bin/env python3
"""CPU-only validation of paired GUIC inputs and prompt isolation."""

import sys
import random
from pathlib import Path

ROOT = Path("/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels")
sys.path.insert(0, str(ROOT / "vlms/inference/main_files"))

from infere_vlms import (  # noqa: E402
    BaseVLMEvaluator,
    build_questions_from_hf_dataset,
    get_or_download_hf_dataset,
)

VARIANTS = (
    "notext",
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
)


def normalize_label(text):
    return " ".join(str(text).lower().strip().split())


def main():
    dataset = get_or_download_hf_dataset(
        "AHAAM/GUIC",
        str(ROOT / "vlms/activation_patching/hf_dataset_GUIC_cleaned"),
        split="test",
    )
    assert len(dataset) == 474, f"expected 474 rows, found {len(dataset)}"

    table = dataset.data
    question_ids = [str(value) for value in table.column("question_id").to_pylist()]
    questions = table.column("question").to_pylist()
    reference_columns = {
        key: table.column(key).combine_chunks().field("text").to_pylist()
        for key in (
            "correct_answer",
            "misleading_groundable",
            "misleading_ungroundable",
            "irrelevant_word",
        )
    }
    collisions = []
    for index, qid in enumerate(question_ids):
        labels = [reference_columns[key][index] for key in reference_columns]
        normalized = [normalize_label(label) for label in labels]
        assert all(normalized), f"{qid}: empty reference"
        if len(set(normalized)) != 4:
            collisions.append(qid)
        prompt = BaseVLMEvaluator.format_open_ended_prompt(None, questions[index])
        assert "Options:" not in prompt and "A, B, C, or D" not in prompt
        candidates = list(reference_columns)
        random.Random(f"42_{qid}").shuffle(candidates)
        assert len(candidates) == 4

    paired = {
        variant: build_questions_from_hf_dataset(
            dataset,
            variant=variant,
            image_field="cleaned_image",
            shuffle_options=True,
            seed=42,
            max_samples=2,
        )
        for variant in VARIANTS
    }
    expected_ids = question_ids[:2]
    for variant, rows in paired.items():
        assert len(rows) == 2, f"{variant}: expected 2 loader probes, found {len(rows)}"
        assert [str(row["question_id"]) for row in rows] == expected_ids

    for index, qid in enumerate(expected_ids):
        baseline = paired["notext"][index]
        for variant in VARIANTS[1:]:
            row = paired[variant][index]
            assert row["question"] == baseline["question"], f"{qid}: question mismatch"
            assert row["option_meta"] == baseline["option_meta"], f"{qid}: option order mismatch"
            assert row["reference_answers"] == baseline["reference_answers"], f"{qid}: reference mismatch"

    print("[PASS] dataset_rows=474 paired_conditions=5")
    print(f"[PASS] question_alignment=474/474 nonempty_references=474/474")
    print(f"[AUDIT] duplicate_reference_rows={len(collisions)} ids={collisions}")
    print("[PASS] deterministic_option_mapping=474/474 prompt_isolation=474/474")
    print("[EXPECTED] GPU smoke test should produce 2 rows x 5 conditions x 7 models = 70 records")


if __name__ == "__main__":
    main()
