#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-50,gpu-51
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=qwen3_2b_attn_resume
#SBATCH --output=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.out
#SBATCH --error=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.err

set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
ROOT=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal
SHARD_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
echo "[CONFIG] resume Qwen3-VL-2B irrelevant_word shard=$SHARD_ID/2; original shared-305 selection and seed"
echo "[EXPECTED] skip compatible completed questions; final totals 153/152 questions per shard; zero validation failures"
echo "[EXPECTED] no-op/cache<=1e-3; blocked probability=0; attention-row error<=2e-3; saved records=expected records"
nvidia-smi -L
python -u "$ROOT/attention_intervention/main_files/run_attention_full.py" \
  --model_id Qwen/Qwen3-VL-2B-Instruct --variant irrelevant_word \
  --selection "$ROOT/../activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json" \
  --output_dir "$ROOT/attention_intervention/outputs/full_2b/irrelevant_word/shard_$SHARD_ID" \
  --shard_id "$SHARD_ID" --num_shards 2 --seed 271828
echo "[PASS] remaining 2B attention shard complete"
