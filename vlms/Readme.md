# VLM MCQ Evaluator

A Python tool that tests vision-language AI models on multiple-choice questions about images.

## What Does This Do?

This script:
1. Takes images and multiple-choice questions about those images
2. Feeds them to AI vision models (like LLaVA, Qwen-VL, etc.)
3. Extracts the model's answer (A, B, C, or D)
4. Calculates accuracy and saves results to JSON

**Example:**
- Image: A photo of a cat
- Question: "What animal is in this image?"
- Options: A) Dog, B) Cat, C) Bird, D) Fish
- Model predicts: B
- Result: ✓ Correct!

## Installation

```bash
# Basic requirements
pip install torch transformers pillow tqdm

# For Qwen-VL models (optional)
pip install qwen-vl-utils torchvision

# For InternVL models (optional)
pip install torchvision einops timm
```

## How to Use

### Step 1: Prepare Your Data

Create a folder with your images:
```
my_images/
├── cat.jpg
├── dog.jpg
└── bird.jpg
```

Create a `questions.json` file:
```json
[
  {
    "image": "cat.jpg",
    "question": "What animal is shown?",
    "options": {
      "A": "Dog",
      "B": "Cat",
      "C": "Bird",
      "D": "Fish"
    },
    "answer": "B"
  }
]
```

### Step 2: Run the Script

**Simplest command:**
```bash
python infer_vlms.py \
  --image_folder ./my_images/ \
  --questions questions.json
```

This uses the default LLaVA model and saves results to `results.json`.

**Try different models:**
```bash
# Use Qwen-VL
python infer_vlms.py --model_type qwen-vl --image_folder ./my_images/ --questions questions.json

# Use InternVL
python infer_vlms.py --model_type internvl --image_folder ./my_images/ --questions questions.json
```

**See what models are available:**
```bash
python infer_vlms.py --list_models
```

### Step 3: Check Results

The script prints accuracy to console:
```
Correct: 45/50
Accuracy: 90.00%
```

And saves detailed results to `results.json`:
```json
[
  {
    "image": "cat.jpg",
    "question": "What animal is shown?",
    "predicted_answer": "B",
    "correct_answer": "B",
    "is_correct": true,
    "full_response": "User: What animal is shown?\nAssistant: B"
  }
]
```

## Common Options

| Option | What It Does | Example |
|--------|--------------|---------|
| `--model_type` | Choose which AI model to use | `--model_type qwen-vl` |
| `--output` | Where to save results | `--output my_results.json` |
| `--batch_size` | Process multiple images at once (faster) | `--batch_size 8` |
| `--device` | Use GPU or CPU | `--device cpu` |
| `--max_tokens` | Maximum length of answer | `--max_tokens 100` |

## Supported Models

| Model | Command | Notes |
|-------|---------|-------|
| **LLaVA 1.5** | `--model_type llava` | Default, works out of the box |
| **Qwen-VL** | `--model_type qwen-vl` | Requires: `pip install qwen-vl-utils torchvision` |
| **LLaVA-NeXT** | `--model_type llava-next` | Requires: `pip install -U transformers` |
| **InternVL** | `--model_type internvl` | Requires: `pip install torchvision einops timm` |

## Quick Examples

**Create a sample questions file:**
```bash
python infer_vlms.py --create_sample my_questions.json
```

**Process with larger batch size (faster):**
```bash
python infer_vlms.py \
  --image_folder ./images/ \
  --questions questions.json \
  --batch_size 16
```

**Use CPU instead of GPU:**
```bash
python infer_vlms.py \
  --image_folder ./images/ \
  --questions questions.json \
  --device cpu
```

**Use a specific model variant:**
```bash
python infer_vlms.py \
  --model_type llava \
  --model_id llava-hf/llava-1.5-13b-hf \
  --image_folder ./images/ \
  --questions questions.json
```

**Save results to custom file:**
```bash
python infer_vlms.py \
  --image_folder ./images/ \
  --questions questions.json \
  --output my_experiment_results.json
```

## Complete Command Reference

```bash
python infer_vlms.py [OPTIONS]

Options:
  --model_type TYPE        Model type: llava, qwen-vl, llava-next, internvl
  --model_id ID           Specific HuggingFace model ID
  --image_folder PATH     Path to folder with images (required)
  --questions PATH        Path to questions JSON file (required)
  --output PATH           Output JSON file (default: results.json)
  --batch_size N          Images per batch (default: 4)
  --max_tokens N          Max tokens per answer (default: 50)
  --device DEVICE         Device: cuda, cpu, or auto (default: auto)
  --list_models           List all available models
  --create_sample FILE    Create sample questions file
```

