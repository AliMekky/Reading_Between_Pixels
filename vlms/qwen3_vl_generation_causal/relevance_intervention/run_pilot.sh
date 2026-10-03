#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-50,gpu-51,gpu-54
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --job-name=qwen3_relevance_pilot
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%j.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%j.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
PILOT=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/relevance_intervention
nvidia-smi -L
python -u "$PILOT/run_pilot.py" --samples 12 --output "$PILOT/outputs/pilot_12"
python -u "$PILOT/summarize_pilot.py" "$PILOT/outputs/pilot_12"
