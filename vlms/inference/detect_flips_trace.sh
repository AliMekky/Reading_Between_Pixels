#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-53
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --job-name=flip_detection
#SBATCH --output=jobs_logs/%x_%j.out
#SBATCH --error=jobs_logs/%x_%j.err

mkdir -p jobs_logs

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/inference/

nvidia-smi

model="llava-next"
output_dir="./layerwise_flip_cache"


echo "Starting integrated gradients runs at $(date)"
echo "================================================"

echo ""
echo "Processing model: $model"
echo "----------------------------------------"

START_TIME=$(date +%s)

for variant in notext correct_answer misleading_groundable misleading_ungroundable irrelevant_word; do
    python -u detect_flip_layer.py \
        --model_id llava-hf/llava-v1.6-mistral-7b-hf \
        --hf_dataset AHAAM/GUIC \
        --hf_cache_dir ./hf_cache_GUIC \
        --split test \
        --variant "$variant" \
        --qid_file ./no_overlap_question_ids.txt \
        --out_dir "${output_dir}/${variant}" \
        --shuffle_options \
        --seed 42 \
        --device cuda \
        --save_full_trace
done


wait

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Success! Duration: ${DURATION}s"
    echo "Outputs saved with prefix: $OUTPUT_FILE"
else
    echo "❌ Failed with exit code: $EXIT_CODE"
fi


echo ""
echo "================================================"
echo "Occlusion finished at $(date)"
echo "Results saved in: $output_dir"