#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=qwen3_free_gen_full
#SBATCH --output=vlms/qwen3_vl_generation_causal/free_generation_intervention/logs/%x_%A_%a.out
#SBATCH --error=vlms/qwen3_vl_generation_causal/free_generation_intervention/logs/%x_%A_%a.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"

set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

ROOT=${REPO_ROOT}/vlms/qwen3_vl_generation_causal/free_generation_intervention
SELECTION=$ROOT/../../activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json
VARIANTS=(correct_answer misleading_groundable misleading_ungroundable irrelevant_word)
SHARD_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
mkdir -p "$ROOT/logs"

echo "[CUDA] host=$(hostname) visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
echo "[LAUNCH] variants=${VARIANTS[*]} shard=$SHARD_ID/2 samples=305 layers=30-35"
for VARIANT in "${VARIANTS[@]}"; do
  OUTPUT=$ROOT/outputs/full/$VARIANT/shard_$SHARD_ID
  mkdir -p "$OUTPUT"
  echo "[VARIANT] start=$VARIANT shard=$SHARD_ID/2"
  python -u "$ROOT/main_files/run_free_generation_full.py" \
    --variant "$VARIANT" --selection "$SELECTION" --output_dir "$OUTPUT" \
    --shard_id "$SHARD_ID" --num_shards 2
  echo "[VARIANT] complete=$VARIANT shard=$SHARD_ID/2"
done
