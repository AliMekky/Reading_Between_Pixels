#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH --array=0-6%2
#SBATCH --job-name=open_ended_full
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
: "${REPO_ROOT:?set REPO_ROOT to the repository root}"

set -euo pipefail

source /apps/local/anaconda3/conda_init.sh
conda activate text_in_image

MODEL_TYPES=(llava llava-next qwen-vl internvl qwen3-vl qwen3-vl qwen3-vl)
MODEL_IDS=(
  llava-hf/llava-1.5-7b-hf
  llava-hf/llava-v1.6-mistral-7b-hf
  Qwen/Qwen2.5-VL-7B-Instruct
  OpenGVLab/InternVL3_5-8B
  Qwen/Qwen3-VL-2B-Instruct
  Qwen/Qwen3-VL-8B-Instruct
  Qwen/Qwen3-VL-32B-Instruct
)

ROOT=${REPO_ROOT}
EXPERIMENT_DIR="$ROOT/vlms/open_ended_evaluation"
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODEL_TYPE=${MODEL_TYPES[$INDEX]}
MODEL_ID=${MODEL_IDS[$INDEX]}
MODEL_SLUG=${MODEL_ID//\//__}

cd "$EXPERIMENT_DIR"
mkdir -p outputs/full logs

echo "[CONFIG] task=$INDEX model_type=$MODEL_TYPE model_id=$MODEL_ID"
echo "[CONFIG] questions=474 conditions=5 expected_records=2370"
echo "[EXPECTED] each condition must finish 474/474 with zero missing records"
echo "[NODE] hostname=$(hostname) slurm_nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import sys
import torch

available = torch.cuda.is_available()
count = torch.cuda.device_count()
print(f"[GPU VALIDATION] torch_cuda_available={available} visible_device_count={count}")
if not available or count < 1:
    print("[FAIL] Slurm allocation exposes no CUDA device to PyTorch", file=sys.stderr)
    raise SystemExit(1)
print(f"[PASS] cuda_device_0={torch.cuda.get_device_name(0)}")
PY

python -u "$ROOT/vlms/inference/main_files/infere_vlms.py" \
  --model_type "$MODEL_TYPE" \
  --model_id "$MODEL_ID" \
  --hf_dataset anonymous/GUIC \
  --hf_cache_dir "$ROOT/vlms/activation_patching/hf_dataset_GUIC_cleaned" \
  --output "outputs/full/${MODEL_SLUG}.jsonl" \
  --evaluation_format open_ended \
  --image_field cleaned_image \
  --max_tokens 32 \
  --max_retries 3 \
  --seed 42 \
  --device cuda

python main_files/validate_generation_outputs.py \
  --input_dir outputs/full \
  --model_slug "$MODEL_SLUG" \
  --expected_questions 474

echo "[PASS] model=$MODEL_ID completed and validated 2370 records"
