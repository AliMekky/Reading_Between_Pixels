#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 08:00:00
#SBATCH --job-name=activation_patch_sweep
#SBATCH --output=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching/logs/%x_%j.out
#SBATCH --error=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching/logs/%x_%j.err

set -euo pipefail

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

ROOT=/nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/activation_patching
MAIN_DIR="$ROOT/main_files"
OUTPUT_DIR="$ROOT/layer_sweep_outputs"
QUESTION_ID="${QUESTION_ID:-14412508}"
REPORT="$OUTPUT_DIR/$QUESTION_ID/text_random_all_layers.json"
PLOT="$OUTPUT_DIR/$QUESTION_ID/text_random_all_layers.png"

mkdir -p "$ROOT/logs" "$OUTPUT_DIR"
test -w "$ROOT/logs" || { echo "[FAIL] Log directory is not writable" >&2; exit 1; }
test -w "$OUTPUT_DIR" || { echo "[FAIL] Output directory is not writable" >&2; exit 1; }
cd "$MAIN_DIR"

echo "[LAUNCH] question_id=$QUESTION_ID expected_interventions=256"
echo "[LAUNCH] variants=grounded,ungrounded directions=restoration,insertion regions=text,matched_random layers=all"
echo "[LAUNCH] dataset_revision=27b45899d1154ef1f08ce5c40d45d2468e4ea3e2 overlay_image_field=cleaned_image"
echo "[EXPECTED] Text and random controls must have identical base/mosaic token counts."
echo "[EXPECTED] The run must end with all 256 interventions validated."

python -u activation_patch_layer_sweep.py \
    --question_id "$QUESTION_ID" \
    --dataset_revision 27b45899d1154ef1f08ce5c40d45d2468e4ea3e2 \
    --hf_cache_dir "$ROOT/hf_dataset_GUIC_cleaned" \
    --dataset_validation_file "$ROOT/hf_dataset_GUIC_cleaned/remote_validation.json" \
    --shuffle_options \
    --streams base,mosaic \
    --out_dir "$OUTPUT_DIR"

python -u plot_activation_patch_layer_sweep.py \
    --input "$REPORT" \
    --output "$PLOT"

echo "[COMPLETE] Layer sweep and preliminary plot completed successfully."

