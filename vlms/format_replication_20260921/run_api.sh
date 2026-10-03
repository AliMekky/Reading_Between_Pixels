#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen32b_api_formats
#SBATCH --output=vlms/format_replication_20260921/logs/%x_%j.out
#SBATCH --error=vlms/format_replication_20260921/logs/%x_%j.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
cd ${REPO_ROOT}/vlms/format_replication_20260921
python -u run_generation.py --api --smoke
python -u run_generation.py --api
