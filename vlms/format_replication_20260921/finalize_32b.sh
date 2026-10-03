#!/bin/bash
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=eval32b_finalize
#SBATCH --output=vlms/format_replication_20260921/logs/%x_%j.out
#SBATCH --error=vlms/format_replication_20260921/logs/%x_%j.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
set -a
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"
source "${API_ENV_FILE:?set API_ENV_FILE to your API-key env file}"
set +a
cd ${REPO_ROOT}/vlms/format_replication_20260921
python -u finalize_32b.py
