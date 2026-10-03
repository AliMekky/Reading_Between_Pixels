#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=qwen3_2b_replication
#SBATCH --output=vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.out
#SBATCH --error=vlms/qwen3_vl_generation_causal/logs/%x_%A_%a.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"

set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image
ROOT=${REPO_ROOT}/vlms/qwen3_vl_generation_causal
SELECTION=$ROOT/../activation_patching/main_files/activation_patch_confirmation_selection_shared_305.json
SHARD_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODEL=Qwen/Qwen3-VL-2B-Instruct
VARIANTS=(correct_answer misleading_groundable misleading_ungroundable irrelevant_word)
echo "[CONFIG] model=$MODEL shard=$SHARD_ID/2 questions=305 conditions=${VARIANTS[*]}"
echo "[WINDOWS] attention=0-4,5-8,9-13,14-18,19-22,23-27 (zero-based relative-depth bins, not discovered causal windows)"
echo "[EXPECTED] activation: 353800 records total; attention: all eligible paths in six windows on both paired images"
echo "[EXPECTED] each attention gate must PASS before full work; checkpointed samples survive time limits"
nvidia-smi -L

# Validate all conditions using the actual full runner on one question per shard.
for VARIANT in "${VARIANTS[@]}"; do
  python -u "$ROOT/attention_intervention/main_files/run_attention_full.py" \
    --model_id "$MODEL" --variant "$VARIANT" --selection "$SELECTION" \
    --output_dir "$ROOT/attention_intervention/outputs/gate_2b/$VARIANT/shard_$SHARD_ID" \
    --shard_id "$SHARD_ID" --num_shards 2 --max_samples 1
done
echo "[PASS] 2B attention gates complete for all four conditions"

bash "$ROOT/main_files/run_step6_2b_full_all_layers.sh"
echo "[PASS] 2B all-layer activation-patching shard complete"

for VARIANT in "${VARIANTS[@]}"; do
  python -u "$ROOT/attention_intervention/main_files/run_attention_full.py" \
    --model_id "$MODEL" --variant "$VARIANT" --selection "$SELECTION" \
    --output_dir "$ROOT/attention_intervention/outputs/full_2b/$VARIANT/shard_$SHARD_ID" \
    --shard_id "$SHARD_ID" --num_shards 2
done
echo "[PASS] 2B replication shard complete: activation patching and attention intervention"
