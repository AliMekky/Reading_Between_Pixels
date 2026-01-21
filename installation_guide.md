# Complete SceneTAP Local Setup Guide

This guide covers the complete setup for running SceneTAP on your local images, including all dependencies (SoM and TextDiffuser-2).

## Overview

SceneTAP requires three main components:
1. **SoM (Set-of-Mark)** - For scene segmentation and marking
2. **TextDiffuser-2** - For text rendering and integration
3. **SceneTAP** - The main adversarial attack framework

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- PyTorch with CUDA support
- Git

---

## Part 1: Set Up SoM (Set-of-Mark)

### Step 1.1: Clone and Install SoM

```bash
# Clone the repository
git clone https://github.com/microsoft/SoM.git
cd SoM

# Create conda environment
conda create -n som python=3.8
conda activate som

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 1.2: Install Segmentation Packages

```bash
# Install SEEM
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package

# Install SAM (Segment Anything)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install Semantic-SAM
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package

# Build Deformable Convolution for Semantic-SAM
cd ops && bash make.sh && cd ..

# Common error fix (if needed)
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
```

### Step 1.3: Download Pretrained Models

```bash
# Download checkpoints
sh download_ckpt.sh
```

### Step 1.4: Test SoM Installation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=YOUR_API_KEY

# Run demo
python demo_som.py
```

---

## Part 2: Set Up TextDiffuser-2

### Step 2.1: Clone Repository

```bash
# Navigate to parent directory
cd ..

# Clone UniLM repository
git clone https://github.com/microsoft/unilm.git
cd unilm/textdiffuser-2
```

### Step 2.2: Create Environment and Install Dependencies

```bash
# Create new environment
conda create -n textdiffuser2 python=3.8
conda activate textdiffuser2

# Install requirements
pip install -r requirements.txt

# Install PyTorch, torchvision, and xformers (adjust for your CUDA version)
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
```

### Step 2.3: Install Flash Attention (Optional, for training)

```bash
# Only if you plan to train the layout planner
pip install flash-attn --no-build-isolation
```

### Step 2.4: Install Modified Diffusers (for text inpainting)

```bash
# Install modified diffusers package
pip install git+https://github.com/JingyeChen/diffusers_td2.git
```

### Step 2.5: Fix Attention Processor Error (if needed)

If you encounter `RuntimeError: expected scalar type float Float but found Half`:

```bash
# Download the fixed file from the repo and replace
# Copy assets/attention_processor.py to your installed diffusers library
# Usually at: ~/.conda/envs/textdiffuser2/lib/python3.8/site-packages/diffusers/models/
```

### Step 2.6: Add Font File

```bash
# Create font directory
mkdir -p assets/font

# Copy Arial.ttf to assets/font/
# Download Arial.ttf and place it in assets/font/
```

### Step 2.7: Download Pretrained Models

Download the following from HuggingFace:

1. **Layout Planner**: https://huggingface.co/JingyeChen22/textdiffuser2_layout_planner
2. **Full Fine-tuned Model**: https://huggingface.co/JingyeChen22/textdiffuser2-full-ft
3. **LoRA Fine-tuned Model**: https://huggingface.co/JingyeChen22/textdiffuser2-lora-ft

```bash
# Using git-lfs to download models
git lfs install
git clone https://huggingface.co/JingyeChen22/textdiffuser2_layout_planner
git clone https://huggingface.co/JingyeChen22/textdiffuser2-full-ft
git clone https://huggingface.co/JingyeChen22/textdiffuser2-lora-ft
```

---

## Part 3: Set Up SceneTAP

### Step 3.1: Clone SceneTAP Repository

```bash
# Navigate to parent directory
cd ../..

# Clone SceneTAP
git clone https://github.com/tsingqguo/scenetap.git
cd scenetap
```

### Step 3.2: Create Environment

```bash
# Create environment (or reuse one of the above)
conda create -n scenetap python=3.8
conda activate scenetap

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow numpy opencv-python tqdm
pip install openai  # for GPT-4 API access
```

### Step 3.3: Configure API Keys

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here
```

---

## Part 4: Run SceneTAP on Local Images

### Step 4.1: Prepare Your Data

Create a directory structure:

```
scenetap/
├── data/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── questions.json
```

### Step 4.2: Create Questions File

Create `data/questions.json`:

```json
[
  {
    "question_id": 1,
    "image": "image1.jpg",
    "text": "What is shown in this image?",
    "category": "general",
    "answer": "expected_answer"
  },
  {
    "question_id": 2,
    "image": "image2.jpg",
    "text": "Describe the main objects",
    "category": "description",
    "answer": "expected_answer"
  }
]
```

### Step 4.3: Generate SoM Images

```bash
python save_som_images.py \
  --seed 42 \
  --dataset typo_base_color \
  --slider 3 \
  --filter 12 \
  --image-folder ./data/images \
  --question-file ./data/questions.json \
  --log_dir ./som_images
```

**Parameters explained:**
- `--seed`: Random seed for reproducibility
- `--dataset`: Dataset type identifier
- `--slider`: Granularity level for segmentation (1-5, higher = more segments)
- `--filter`: Minimum segment size threshold
- `--image-folder`: Path to your images
- `--question-file`: Path to questions JSON
- `--log_dir`: Output directory for SoM results

### Step 4.4: Run SceneTAP Attack

```bash
python chatgpt_test.py \
  --model gpt-4o \
  --dataset custom \
  --attack attack_plan_som_avoid_target_give_answer_ablation_resize_combine \
  --slider 3 \
  --filter 12 \
  --question-file ./data/questions.json \
  --image-folder ./data/images \
  --log_dir ./logs
```

**Parameters explained:**
- `--model`: Vision-language model to attack (gpt-4o, gpt-4-vision, etc.)
- `--dataset`: Dataset identifier
- `--attack`: Attack strategy name
- `--log_dir`: Directory for output logs and results

or you can use a loop to rerun the script whenever there is a error.

```bash
until python -u chatgpt_test.py --dataset typo_base_color --attack TextAnalysis --slider 3 --filter 12 --question-file ./data/questions.json --image-folder ./data/images --log_dir ./logs ; do
  code=$?
  echo "your_script.py crashed with exit code $code. Restarting in 2s..." >&2
  sleep 2
done
```

### Step 4.5: Check Results

Results will be saved in:
- `./som_images/` - Segmented images with marks
- `./logs/` - Attack results and generated adversarial images

---

## Part 5: Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or image resolution
# Or use gradient checkpointing
```

**2. Missing Dependencies**
```bash
pip install opencv-python pillow matplotlib scipy scikit-image
```

**3. SoM Segmentation Fails**
```bash
# Try different slider values (1-5)
# Reduce image resolution
# Check if CUDA is available: python -c "import torch; print(torch.cuda.is_available())"
```

**4. TextDiffuser Errors**
```bash
# Ensure font file exists in assets/font/
# Check model checkpoints are downloaded
# Verify diffusers version matches requirements
```

**5. OpenAI API Errors**
```bash
# Verify API key is set correctly
export OPENAI_API_KEY=your_key_here
# Check API quota and rate limits
```

---

## Part 6: Simplified Local Inference Script

For simpler local testing without the full attack pipeline:

```python
# simple_inference.py
import torch
from PIL import Image
import numpy as np

def process_single_image(image_path):
    """
    Process a single image through SoM segmentation
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # TODO: Add your SoM processing code here
    # This would use the SoM models to generate segmentation masks
    
    return image

if __name__ == "__main__":
    image_path = "./data/images/test.jpg"
    result = process_single_image(image_path)
    result.save("./output/result.jpg")
```

---

## Part 7: Environment Management

### Quick Environment Setup Script

Save as `setup_all.sh`:

```bash
#!/bin/bash

# Setup SoM
conda create -n som python=3.8 -y
conda activate som
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package

# Setup TextDiffuser-2
conda create -n textdiffuser2 python=3.8 -y
conda activate textdiffuser2
pip install -r requirements.txt
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118

# Setup SceneTAP
conda create -n scenetap python=3.8 -y
conda activate scenetap
pip install torch torchvision transformers pillow numpy opencv-python tqdm openai

echo "All environments created successfully!"
```

---

## Resources

- **SoM Paper**: https://arxiv.org/pdf/2310.11441.pdf
- **TextDiffuser-2 Paper**: https://arxiv.org/pdf/2311.16465.pdf
- **SceneTAP Paper**: https://arxiv.org/pdf/2412.00114.pdf
- **SoM Demo**: https://huggingface.co/spaces/Roboflow/SoM
- **TextDiffuser-2 Demo**: https://huggingface.co/spaces/JingyeChen22/TextDiffuser-2

---

## Notes

- GPU with 16GB+ VRAM recommended for optimal performance
- Processing time depends on image resolution and model complexity
- For production use, consider batch processing and caching
- API costs apply when using OpenAI's GPT-4V/GPT-4o

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{cao2024scenetap,
  title={SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments},
  author={Cao, Yue and Xing, Yun and Zhang, Jie and Lin, Di and Zhang, Tianwei and Tsang, Ivor and Liu, Yang and Guo, Qing},
  journal={arXiv preprint arXiv:2412.00114},
  year={2024}
}
```
