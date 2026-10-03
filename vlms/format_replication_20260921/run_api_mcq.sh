#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen32b_api_mcq
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%j.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%j.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
cd /l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921
python -u run_generation.py --api --only-format mcq
