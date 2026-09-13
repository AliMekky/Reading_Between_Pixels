#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/open_ended_evaluation
cd "$EXPERIMENT_DIR"

python main_files/evaluate_deterministic.py \
  --input_dir outputs/full \
  --output_dir outputs/deterministic_evaluation \
  --expected_questions 474 \
  --threshold 0.90 \
  --seed 42
