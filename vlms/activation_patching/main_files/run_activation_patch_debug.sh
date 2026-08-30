#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
#SBATCH --job-name=activation_patch_debug
#SBATCH --output=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching/logs/%x_%j.out
#SBATCH --error=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching/logs/%x_%j.err

set -euo pipefail

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

# SLURM executes a spool copy of this file, so BASH_SOURCE[0] does not reliably
# identify the repository. Use the real project path, with an override for an
# intentionally relocated checkout.
ACTIVATION_PATCH_ROOT="${ACTIVATION_PATCH_ROOT:-/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching}"
MAIN_DIR="${ACTIVATION_PATCH_ROOT}/main_files"
LOG_DIR="${ACTIVATION_PATCH_ROOT}/logs"
OUTPUT_DIR="${ACTIVATION_PATCH_ROOT}/debug_outputs"

if [[ ! -f "${MAIN_DIR}/activation_patch_llava_next_debug.py" ]]; then
    echo "[FAIL] Experiment script not found: ${MAIN_DIR}/activation_patch_llava_next_debug.py" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
test -w "$LOG_DIR" || { echo "[FAIL] Log directory is not writable: ${LOG_DIR}" >&2; exit 1; }
test -w "$OUTPUT_DIR" || { echo "[FAIL] Output directory is not writable: ${OUTPUT_DIR}" >&2; exit 1; }
cd "$MAIN_DIR"

QUESTION_ID="${QUESTION_ID:-14412508}"
VARIANT="${VARIANT:-misleading_groundable}"
LAYER="${LAYER:-15}"

echo "[LAUNCH] question_id=${QUESTION_ID} variant=${VARIANT} layer=${LAYER}"
echo "[LAUNCH] dataset_revision=27b45899d1154ef1f08ce5c40d45d2468e4ea3e2 overlay_image_field=cleaned_image"
echo "[LAUNCH] main_dir=${MAIN_DIR}"
echo "[LAUNCH] log_dir=${LOG_DIR}"
echo "[LAUNCH] output_dir=${OUTPUT_DIR}"
echo "[EXPECTED] The job must end with '[COMPLETE] All milestone validation checks passed'."

python -u activation_patch_llava_next_debug.py \
    --question_id "$QUESTION_ID" \
    --variant "$VARIANT" \
    --layer "$LAYER" \
    --dataset_revision 27b45899d1154ef1f08ce5c40d45d2468e4ea3e2 \
    --hf_cache_dir "${ACTIVATION_PATCH_ROOT}/hf_dataset_GUIC_cleaned" \
    --dataset_validation_file "${ACTIVATION_PATCH_ROOT}/hf_dataset_GUIC_cleaned/remote_validation.json" \
    --overlay_image_field cleaned_image \
    --shuffle_options \
    --streams base,mosaic \
    --out_dir "$OUTPUT_DIR"
