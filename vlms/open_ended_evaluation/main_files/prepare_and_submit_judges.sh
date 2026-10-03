#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/open_ended_evaluation
SECRETS=/l/users/ali.mekky/.secrets/open_ended_eval.env
cd "$EXPERIMENT_DIR"

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
set -a
source "$SECRETS"
set +a

python main_files/prepare_judge_batches.py \
  --evaluation_dir outputs/deterministic_evaluation \
  --output_dir outputs/judge_batches_six_models \
  --openai_model gpt-5.6-luna \
  --gemini_model gemini-3.5-flash

python main_files/submit_judge_batches.py \
  --batch_dir outputs/judge_batches_six_models
