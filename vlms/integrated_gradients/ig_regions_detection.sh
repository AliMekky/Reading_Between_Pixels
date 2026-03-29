#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --job-name=ig_regions_detection
#SBATCH --output=jobs_logs/%x_%j.out
#SBATCH --error=jobs_logs/%x_%j.err

mkdir -p jobs_logs

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/integrated_gradients/

nvidia-smi

model="llava-next"
# output_dir="./${model}_ig_token_outputs_correct_answer_token"
# start=400
# end=500


echo "Starting integrated gradients regions detection runs at $(date)"
echo "================================================"

echo ""
echo "Processing model: $model"
echo "----------------------------------------"

START_TIME=$(date +%s)

python -u run_ig_for_all.py \
  --region_script ./ig_regions.py \
  --ig_root /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/vlms/integrated_gradients/llava-next_ig_token_outputs\
  --out_dir ./one_question_three_regions_mask_based_strict_sign \
  --ids_file ../inference/no_overlap_question_ids.txt \
  --skip_existing

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