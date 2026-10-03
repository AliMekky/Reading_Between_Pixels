#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH --job-name=qwen3_relevance_pilot
#SBATCH --output=vlms/qwen3_vl_generation_causal/logs/%x_%j.out
#SBATCH --error=vlms/qwen3_vl_generation_causal/logs/%x_%j.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
PILOT=${REPO_ROOT}/vlms/qwen3_vl_generation_causal/relevance_intervention
nvidia-smi -L
python -u "$PILOT/run_pilot.py" --samples 12 --output "$PILOT/outputs/pilot_12"
python -u "$PILOT/summarize_pilot.py" "$PILOT/outputs/pilot_12"
