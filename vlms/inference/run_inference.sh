#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --job-name=analyze_text
#SBATCH --output=jobs_logs/%x_%j.out
#SBATCH --error=jobs_logs/%x_%j.err

mkdir -p jobs_logs

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/inference

MODELS=(
    "qwen-vl"
    "llava-next"
    "llava"
    "internvl"
)

HF_DATASET="AHAAM/CIM"
HF_CACHE_DIR="./hf_cache/AHAAM__CIM"
OUTPUT_BASE="./results"

mkdir -p "$HF_CACHE_DIR"
mkdir -p "$OUTPUT_BASE"

echo "Starting inference runs at $(date)"
echo "================================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "Processing model: $model"
    echo "----------------------------------------"

    OUTPUT_FILE="$OUTPUT_BASE/${model}_results.json"

    echo "Command: python infere_vlms.py --model_type $model --hf_dataset $HF_DATASET --hf_cache_dir $HF_CACHE_DIR --output $OUTPUT_FILE"

    START_TIME=$(date +%s)

    python -u infere_vlms.py \
        --model_type "$model" \
        --hf_dataset "$HF_DATASET" \
        --hf_cache_dir "$HF_CACHE_DIR" \
        --output "$OUTPUT_FILE" \

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Success! Duration: ${DURATION}s"
        echo "Outputs saved with prefix: $OUTPUT_FILE"
    else
        echo "❌ Failed with exit code: $EXIT_CODE"
    fi
done

echo ""
echo "================================================"
echo "All inference runs completed at $(date)"
echo "Results saved in: $OUTPUT_BASE"