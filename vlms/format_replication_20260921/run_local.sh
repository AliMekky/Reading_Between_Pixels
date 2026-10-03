#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-50,gpu-51,gpu-54
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=mcq_replication
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%A_%a.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%A_%a.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
cd /l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921
for model_index in "$SLURM_ARRAY_TASK_ID" "$((SLURM_ARRAY_TASK_ID + 2))" "$((SLURM_ARRAY_TASK_ID + 4))"; do
  python -u run_generation.py --model-index "$model_index" --smoke
  python -u run_generation.py --model-index "$model_index"
done
