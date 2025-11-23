# Image Filtration Pipeline

Filters images for typography overlay using 3 criteria: answer length, existing text (OCR), and writable surfaces (segmentation).

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python filtration_pipeline.py
```

## Input

**questions.json:**
```json
[
  {
    "image": "img001.jpg",
    "question": "What color is the car?",
    "A": "Red",
    "B": "Blue",
    "C": "Green",
    "D": "Yellow"
  }
]
```

**Directory structure:**
```
project/
├── filtration_pipeline.py
├── questions.json
└── images/
    └── img001.jpg
```

## Output

Results saved to `filtered_results/`:
- `all_results.json` - All images with metrics
- `passed_images.json` - Passed images only
- `failed_images.json` - Failed images only
- `summary.json` - Statistics

## Configuration

Edit thresholds in `filtration_pipeline.py`:
```python
self.max_answer_words = 2           # Max words per answer
self.max_text_chars = 50            # Max existing text characters
self.max_non_writable_ratio = 0.70  # Max non-writable surface (sky, people, grass)
```