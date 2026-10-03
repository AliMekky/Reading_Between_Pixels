#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/open_ended_evaluation
cd "$EXPERIMENT_DIR"
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
set -a
source /l/users/ali.mekky/.secrets/open_ended_eval.env
set +a

python main_files/check_judge_batches.py --batch_dir outputs/judge_batches_six_models
python main_files/check_judge_batches.py --batch_dir outputs/judge_batches_six_models_retry_1

test -f outputs/judge_batches_six_models/openai_results.jsonl
test -f outputs/judge_batches_six_models/gemini_results.jsonl
test -f outputs/judge_batches_six_models_retry_1/openai_results.jsonl
test -f outputs/judge_batches_six_models_retry_1/gemini_results.jsonl

python main_files/combine_judge_retries.py \
  --batch_dir outputs/judge_batches_six_models \
  --retry_dir outputs/judge_batches_six_models_retry_1

python main_files/merge_judge_results.py \
  --evaluation_dir outputs/deterministic_evaluation \
  --batch_dir outputs/judge_batches_six_models \
  --output_dir outputs/final_classification_six_models \
  --result_suffix _combined

python main_files/compute_open_ended_statistics.py \
  --classified_dir outputs/final_classification_six_models \
  --output_dir outputs/statistics_six_models \
  --resamples 10000 \
  --seed 42
