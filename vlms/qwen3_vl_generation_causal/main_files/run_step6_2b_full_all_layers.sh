#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-05,gpu-50,gpu-51
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=qwen3_2b_full_layers
#SBATCH --output=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.out
#SBATCH --error=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.err

set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

ROOT=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/qwen3_vl_generation_causal
SELECTION=$ROOT/../activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json
VARIANTS=(correct_answer misleading_groundable misleading_ungroundable irrelevant_word)
SHARD_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
mkdir -p "$ROOT/logs"

echo "[CUDA] host=$(hostname) visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
echo "[LAUNCH] step=6 model=Qwen/Qwen3-VL-2B-Instruct questions=305 variants=${VARIANTS[*]} layers=0-27 shard=$SHARD_ID/2"
echo "[EXPECTED] 290 records/question/variant; 353800 total across both shards; resumable checkpoints; four final [PASS] lines/task"

cd "$ROOT/main_files"
for VARIANT in "${VARIANTS[@]}"; do
  OUTPUT=$ROOT/outputs/step6_2b_full_all_layers/$VARIANT/shard_$SHARD_ID
  mkdir -p "$OUTPUT"
  echo "[VARIANT] start=$VARIANT shard=$SHARD_ID/2"
  python -u run_step4_discovery.py \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --step 6_2b_full_all_layers \
    --variant "$VARIANT" \
    --selection "$SELECTION" \
    --expected_questions 305 \
    --output_dir "$OUTPUT" \
    --shard_id "$SHARD_ID" \
    --num_shards 2
  echo "[VARIANT] complete=$VARIANT shard=$SHARD_ID/2"
done
