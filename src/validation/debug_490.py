"""
Debug specific image: 0_Parade_Parade_0_490.jpg
This image has 53 faces and a sorting mismatch at box index 46.
"""

import sys
import os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

key = '0--Parade\\0_Parade_Parade_0_490.jpg'
orig_boxes = np.array(baseline[key]['boxes'], dtype=np.float64)

print(f"Image: {key}")
print(f"Total faces in baseline: {len(orig_boxes)}")

# Run simplified
img_path = os.path.join('data', 'widerface', 'WIDER_val', 'images', key)
img = Image.open(img_path).convert('RGB')
print(f"Image size: {img.size}")

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

simp_boxes = np.array(batch_boxes[0][:, :4], dtype=np.float64)
print(f"Total faces in simplified: {len(simp_boxes)}")

# Sort both
orig_rounded = np.round(orig_boxes, 1)
simp_rounded = np.round(simp_boxes, 1)

orig_idx = np.lexsort((orig_rounded[:, 1], orig_rounded[:, 0]))
simp_idx = np.lexsort((simp_rounded[:, 1], simp_rounded[:, 0]))

orig_sorted = orig_boxes[orig_idx]
simp_sorted = simp_boxes[simp_idx]

# Print all boxes side by side
print(f"\n{'Idx':<5} {'Orig x1':>10} {'Orig y1':>10} {'Simp x1':>10} {'Simp y1':>10} {'Diff':>10}")
print("-" * 55)
for i in range(len(orig_sorted)):
    diff = abs(orig_sorted[i][0] - simp_sorted[i][0]) + abs(orig_sorted[i][1] - simp_sorted[i][1])
    marker = " <-- MISMATCH" if diff > 2.0 else ""
    print(
        f"{i:<5} "
        f"{orig_sorted[i][0]:>10.2f} "
        f"{orig_sorted[i][1]:>10.2f} "
        f"{simp_sorted[i][0]:>10.2f} "
        f"{simp_sorted[i][1]:>10.2f} "
        f"{diff:>10.2f}"
        f"{marker}"
    )