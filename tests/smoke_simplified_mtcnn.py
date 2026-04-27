"""
Test that simplified pipeline produces identical output to original.
"""

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from facenet_pytorch import MTCNN
from src.mtcnn.simplified_mtcnn import detect_face_simplified
from PIL import Image
import torch
import numpy as np


def test_simplified_matches_original():
    print("TEST: Simplified pipeline vs original")
    print("-" * 60)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    celeba_dir = os.path.join(
        _PROJECT_ROOT, 'data', 'celeba',
        'img_align_celeba', 'img_align_celeba'
    )

    all_images = sorted(os.listdir(celeba_dir))[:10]

    matches = 0
    mismatches = 0

    for img_file in all_images:
        img_path = os.path.join(celeba_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        # Run original
        boxes_orig, probs_orig = mtcnn.detect(img)

        # Run simplified
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
        boxes_simp = batch_boxes[0]

        # Compare
        orig_count = len(boxes_orig) if boxes_orig is not None else 0
        simp_count = len(boxes_simp) if boxes_simp is not None else 0

        if orig_count != simp_count:
            print(f"{img_file}: MISMATCH - orig={orig_count} simp={simp_count}")
            mismatches += 1
            continue

        if orig_count == 0:
            print(f"{img_file}: both detected 0 faces - MATCH")
            matches += 1
            continue

        diff = np.abs(
            np.array(boxes_orig) - np.array(boxes_simp[:, :4])
        )

        if diff.max() < 0.001:
            print(f"{img_file}: {orig_count} face(s) - MATCH (diff={diff.max():.6f})")
            matches += 1
        else:
            print(f"{img_file}: MISMATCH (diff={diff.max():.4f})")
            mismatches += 1

    print("-" * 60)
    print(f"Matches:    {matches}/{len(all_images)}")
    print(f"Mismatches: {mismatches}/{len(all_images)}")

    if mismatches == 0:
        print("TEST PASSED - simplified pipeline is equivalent")
    else:
        print("TEST FAILED - simplified pipeline differs from original")

    return mismatches == 0


if __name__ == "__main__":
    success = test_simplified_matches_original()
    sys.exit(0 if success else 1)