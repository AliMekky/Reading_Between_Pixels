#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --job-name=images_segmentation
#SBATCH --output=jobs_logs/%x_%j.out
#SBATCH --error=jobs_logs/%x_%j.err

source /apps/local/anaconda3/conda_init.sh
conda activate textdiffuser2

echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "CUDA devices:"
nvidia-smi

cd /l/users/ali.mekky/reading_between_pixels/Reading_Between_Pixels/scenetap/
python -u save_som_images.py \
  --start 1500 \
  --end 1593 &

# python -u save_som_images.py \
#   --start 1300 \
#   --end 1400 &

# python -u save_som_images.py \
#   --start 1400 \
#   --end 1500 &

  wait
