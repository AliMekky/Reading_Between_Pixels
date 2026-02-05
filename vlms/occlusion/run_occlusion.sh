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

cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/occlusion/


model="llava-next"
output_dir="./${model}_occlusions_results"


echo "Starting activations extraction runs at $(date)"
echo "================================================"

echo ""
echo "Processing model: $model"
echo "----------------------------------------"

START_TIME=$(date +%s)

python -u updated_occlusions.py \
    --output_dir "$output_dir" \
    --model_type "$model" \

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