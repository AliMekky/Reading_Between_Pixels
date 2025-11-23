import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0

# test_installation.py
import torch
import easyocr
import cv2
from transformers import SegformerImageProcessor

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name(0))
print("✓ OpenCV:", cv2.__version__)
print("✓ EasyOCR installed")
print("✓ Transformers installed")

# Test EasyOCR initialization
print("\nTesting EasyOCR initialization...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
print("✓ EasyOCR ready")