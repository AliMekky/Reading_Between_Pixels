#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --job-name=eval32b_finalize
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%j.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921/logs/%x_%j.err
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
set -a
source /l/users/ali.mekky/.secrets/open_ended_eval.env
set +a
cd /l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/format_replication_20260921
python -u finalize_32b.py
