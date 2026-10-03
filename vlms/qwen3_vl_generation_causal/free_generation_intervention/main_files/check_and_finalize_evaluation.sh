#!/bin/bash
set -euo pipefail

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
set -a
source /l/users/ali.mekky/.secrets/open_ended_eval.env
set +a

ROOT=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/free_generation_intervention
OPEN_EVAL=$ROOT/../../open_ended_evaluation/main_files
BATCH=$ROOT/outputs/evaluation/judge_batches
RETRY=$ROOT/outputs/evaluation/judge_batches_retry_1

python "$OPEN_EVAL/check_judge_batches.py" --batch_dir "$BATCH"
python "$OPEN_EVAL/check_judge_batches.py" --batch_dir "$RETRY"
test -f "$BATCH/openai_results.jsonl"
test -f "$BATCH/gemini_results.jsonl"
test -f "$RETRY/openai_results.jsonl"
test -f "$RETRY/gemini_results.jsonl"

python "$OPEN_EVAL/combine_judge_retries.py" \
  --batch_dir "$BATCH" --retry_dir "$RETRY"

python "$ROOT/main_files/merge_intervention_judges.py" \
  --deterministic_dir "$ROOT/outputs/evaluation/deterministic" \
  --batch_dir "$BATCH" \
  --output_dir "$ROOT/outputs/evaluation/final" \
  --result_suffix _combined

python "$ROOT/main_files/compute_intervention_statistics.py" \
  --records "$ROOT/outputs/evaluation/final/records_final.jsonl" \
  --output_dir "$ROOT/outputs/evaluation/statistics"
