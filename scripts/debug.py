import sys
import os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from facenet_pytorch import MTCNN
from src.mtcnn.simplified_mtcnn import detect_face_simplified
from PIL import Image
import torch
import json
import numpy as np

device = torch.device('cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Load baseline
with open('baseline/wider_face_baseline.json') as f:
    baseline = json.load(f)

# Get first image
first_key = list(baseline.keys())[0]
print(f"Image key: {first_key}")
print(f"Baseline saved: {baseline[first_key]}")

# Run original MTCNN right now
img_path = os.path.join('data', 'widerface', 'WIDER_val', 'images', first_key)
print(f"\nImage path: {img_path}")
print(f"Image exists: {os.path.exists(img_path)}")

img = Image.open(img_path).convert('RGB')
print(f"Image size: {img.size}")

# Original
boxes_orig, probs_orig = mtcnn.detect(img)
print(f"\nOriginal MTCNN now:")
print(f"  Boxes: {boxes_orig}")
print(f"  Probs: {probs_orig}")

# Simplified
with torch.no_grad():
    batch_boxes, _ = detect_face_simplified(
        img,
        mtcnn.min_face_size,
        mtcnn.pnet,
        mtcnn.rnet,
        mtcnn.onet,
        mtcnn.thresholds,
        mtcnn.factor,
        device
    )

print(f"\nSimplified MTCNN now:")
print(f"  Boxes: {batch_boxes[0]}")