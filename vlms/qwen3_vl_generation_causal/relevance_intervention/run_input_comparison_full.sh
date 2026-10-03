#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-50,gpu-51,gpu-54
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH --array=0-3%4
#SBATCH --job-name=qwen3_input_full
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
python -u /l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/relevance_intervention/run_input_comparison.py --full --shards 4 --shard "$SLURM_ARRAY_TASK_ID"
