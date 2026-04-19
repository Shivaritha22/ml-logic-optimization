"""
Run original MTCNN on WIDER FACE validation set and save baseline outputs.
These outputs are the ground truth we compare against after simplification.
Usage: python scripts/save_baseline.py
"""

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from facenet_pytorch import MTCNN
from PIL import Image
import torch
import json
import numpy as np


def save_baseline(max_images=None):
    print("Saving WIDER FACE baseline outputs")
    print("-" * 60)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    wider_val_dir = os.path.join(
        _PROJECT_ROOT, 'data', 'widerface', 'WIDER_val', 'images'
    )

    if not os.path.exists(wider_val_dir):
        print(f"ERROR: Directory not found: {wider_val_dir}")
        sys.exit(1)

    # Collect all image paths
    all_images = []
    for event_folder in sorted(os.listdir(wider_val_dir)):
        event_path = os.path.join(wider_val_dir, event_folder)
        if not os.path.isdir(event_path):
            continue
        for img_file in sorted(os.listdir(event_path)):
            if img_file.lower().endswith('.jpg'):
                all_images.append(os.path.join(event_folder, img_file))

    if max_images:
        all_images = all_images[:max_images]

    print(f"Total images found: {len(all_images)}")
    print(f"Device: {device}")
    print("-" * 60)

    baseline = {}
    failed = 0

    for i, rel_path in enumerate(all_images):
        if i % 100 == 0:
            print(f"Processing {i+1}/{len(all_images)} ...")

        img_path = os.path.join(wider_val_dir, rel_path)

        try:
            img = Image.open(img_path).convert('RGB')
            boxes, probs = mtcnn.detect(img)

            if boxes is not None:
                baseline[rel_path] = {
                    'boxes': boxes.tolist(),
                    'probs': probs.tolist(),
                    'num_faces': len(boxes)
                }
            else:
                baseline[rel_path] = {
                    'boxes': [],
                    'probs': [],
                    'num_faces': 0
                }

        except Exception as e:
            print(f"ERROR on {rel_path}: {e}")
            failed += 1
            continue

    # Save baseline
    os.makedirs(os.path.join(_PROJECT_ROOT, 'baseline'), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, 'baseline', 'wider_face_baseline.json')

    with open(out_path, 'w') as f:
        json.dump(baseline, f)

    print("-" * 60)
    print(f"Done")
    print(f"Images processed: {len(baseline)}")
    print(f"Failed: {failed}")
    print(f"Saved to: {out_path}")

    # Quick summary
    total_faces = sum(v['num_faces'] for v in baseline.values())
    images_with_faces = sum(1 for v in baseline.values() if v['num_faces'] > 0)
    print(f"Total faces detected: {total_faces}")
    print(f"Images with at least 1 face: {images_with_faces}/{len(baseline)}")


if __name__ == "__main__":
    save_baseline()