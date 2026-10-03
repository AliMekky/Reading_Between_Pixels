#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH --array=0-3%4
#SBATCH --job-name=qwen3_input_full
#SBATCH --output=vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.out
#SBATCH --error=vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
python -u ${REPO_ROOT}/vlms/qwen3_vl_generation_causal/relevance_intervention/run_input_comparison.py --full --shards 4 --shard "$SLURM_ARRAY_TASK_ID"
