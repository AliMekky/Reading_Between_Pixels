#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR=/l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/open_ended_evaluation
cd "$EXPERIMENT_DIR"
mkdir -p logs outputs/full

ARRAY_RANGE="${1:-0-1}"
if [[ ! "$ARRAY_RANGE" =~ ^(0-1|2-3|3-4|4-5|6)$ ]]; then
    echo "Usage: $0 [0-1|2-3|3-4|4-5|6]" >&2
    exit 2
fi

echo "[SUBMIT] model_indices=$ARRAY_RANGE"
if [[ "$ARRAY_RANGE" == "6" ]]; then
    echo "[SUBMIT] Qwen3-VL-32B requires two 40-GB GPUs for FP16 weights"
    sbatch --array=6 --gres=gpu:2 main_files/run_open_ended_full.sh
else
    sbatch --array="$ARRAY_RANGE" main_files/run_open_ended_full.sh
fi
