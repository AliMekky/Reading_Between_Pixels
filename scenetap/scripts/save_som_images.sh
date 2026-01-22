#!/bin/bash
#SBATCH --account=cscc-users
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --job-name=images_segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /apps/local/anaconda3/conda_init.sh
conda activate textdiffuser2

echo "Job started on $(date)"
echo "Node: $(hostname)"
echo "CUDA devices:"
nvidia-smi

cd /nfs-stor/ali.mekky/reading_between_pixels/Reading_Between_Pixels/scenetap/
python -u save_som_images.py \
  --seed 42 \
  --dataset typo_base_color \
  --slider 3 \
  --filter 12 \
  --image-folder ./data/images \
  --question-file ./data/questions.json \
  --log_dir ./som_images
