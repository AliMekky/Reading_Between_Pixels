#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --job-name=qwen3_input_comparison
#SBATCH --output=vlms/qwen3_vl_generation_causal/logs/%x_%j.out
#SBATCH --error=vlms/qwen3_vl_generation_causal/logs/%x_%j.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
python -u ${REPO_ROOT}/vlms/qwen3_vl_generation_causal/relevance_intervention/run_input_comparison.py
