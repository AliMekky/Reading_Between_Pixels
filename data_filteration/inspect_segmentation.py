from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

# Load model
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model.eval()

# Load your image
image = Image.open("../scenetap/data/images/image_4.jpg")

# Run segmentation
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=image.size[::-1],
    mode="bilinear"
)
segmentation = logits.argmax(dim=1)[0].cpu().numpy()

# Get unique classes in THIS image
unique_classes = np.unique(segmentation)

print(f"Classes found in this image:")
print("="*60)
id2label = model.config.id2label
for class_id in unique_classes:
    class_name = id2label[class_id]
    pixel_count = (segmentation == class_id).sum()
    percentage = pixel_count / segmentation.size * 100
    print(f"{class_id:3d}: {class_name:20s} - {percentage:5.2f}%")
