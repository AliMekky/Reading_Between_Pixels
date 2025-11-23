from transformers import SegformerForSemanticSegmentation
import json

# Load model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512"
)

# Get class mapping
id2label = model.config.id2label

# Save to JSON
with open('ade20k_classes.json', 'w') as f:
    json.dump(id2label, f, indent=2)

print("✓ Saved to ade20k_classes.json")

# Also save as readable text
with open('ade20k_classes.txt', 'w') as f:
    for class_id, class_name in sorted(id2label.items()):
        f.write(f"{class_id:3d}: {class_name}\n")

print("✓ Saved to ade20k_classes.txt")