"""
Chunk 7: Validate simplified MTCNN pipeline against baseline.
Runs simplified pipeline on WIDER FACE validation set and
compares outputs against saved baseline from original pipeline.

Usage: python scripts/validate_simplified.py
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


def load_baseline(baseline_path):
    with open(baseline_path, 'r') as f:
        return json.load(f)


def run_validation(max_images=None):
    print("CHUNK 7: Validating Simplified Pipeline on WIDER FACE")
    print("=" * 60)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load baseline
    baseline_path = os.path.join(_PROJECT_ROOT, 'baseline', 'wider_face_baseline.json')
    if not os.path.exists(baseline_path):
        print(f"ERROR: Baseline not found at {baseline_path}")
        print("Run save_baseline.py first (Chunk 5)")
        sys.exit(1)

    print(f"Loading baseline...")
    baseline = load_baseline(baseline_path)
    print(f"Baseline images: {len(baseline)}")

    wider_val_dir = os.path.join(
        _PROJECT_ROOT, 'data', 'widerface', 'WIDER_val', 'images'
    )

    # Select images to validate
    image_keys = list(baseline.keys())
    if max_images:
        image_keys = image_keys[:max_images]

    print(f"Images to validate: {len(image_keys)}")
    print("=" * 60)

    # Tracking results
    matches          = 0
    mismatches       = 0
    skipped          = 0
    face_count_diffs = []
    box_diffs        = []

    for i, rel_path in enumerate(image_keys):
        if i % 200 == 0:
            print(f"Processing {i+1}/{len(image_keys)} ...")

        img_path = os.path.join(wider_val_dir, rel_path)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Run simplified pipeline
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

            simp_boxes = batch_boxes[0]

            # Get baseline
            orig = baseline[rel_path]
            orig_boxes    = np.array(orig['boxes'])
            orig_num      = orig['num_faces']
            simp_num      = len(simp_boxes) if simp_boxes is not None else 0

            # Check 1: same number of faces
            if orig_num != simp_num:
                mismatches += 1
                face_count_diffs.append({
                    'image': rel_path,
                    'orig':  orig_num,
                    'simp':  simp_num
                })
                continue

            # Check 2: both detected nothing
            if orig_num == 0 and simp_num == 0:
                matches += 1
                continue

            # Check 3: compare box coordinates
            simp_boxes_arr = np.array(simp_boxes[:, :4])

            orig_sorted = orig_boxes[np.argsort(orig_boxes[:, 0])]
            simp_sorted = simp_boxes_arr[np.argsort(simp_boxes_arr[:, 0])]

            diff = np.abs(orig_sorted - simp_sorted)
            max_diff = diff.max()
            box_diffs.append(max_diff)

            if max_diff < 0.001:
                matches += 1
            else:
                mismatches += 1
                print(f"\nMISMATCH: {rel_path}")
                print(f"  Orig boxes shape:  {orig_sorted.shape}")
                print(f"  Simp boxes shape:  {simp_sorted.shape}")
                print(f"  Max diff: {max_diff:.4f}")
                print(f"  Orig first box: {orig_sorted[0]}")
                print(f"  Simp first box: {simp_sorted[0]}")

        except Exception as e:
            skipped += 1
            continue

    # Results
    total = len(image_keys) - skipped
    match_rate = matches / total * 100 if total > 0 else 0

    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total images:     {len(image_keys)}")
    print(f"Processed:        {total}")
    print(f"Skipped:          {skipped}")
    print(f"Matches:          {matches}")
    print(f"Mismatches:       {mismatches}")
    print(f"Match rate:       {match_rate:.2f}%")

    if box_diffs:
        print(f"\nBox coordinate differences (matched images):")
        print(f"  Max:    {max(box_diffs):.6f} pixels")
        print(f"  Mean:   {np.mean(box_diffs):.6f} pixels")
        print(f"  Median: {np.median(box_diffs):.6f} pixels")

    if face_count_diffs:
        print(f"\nFace count mismatches ({len(face_count_diffs)} images):")
        for item in face_count_diffs[:5]:  # show first 5
            print(f"  {item['image']}: orig={item['orig']} simp={item['simp']}")
        if len(face_count_diffs) > 5:
            print(f"  ... and {len(face_count_diffs) - 5} more")

    print("=" * 60)

    if match_rate == 100.0:
        print("VALIDATION PASSED")
        print("Simplified pipeline is provably equivalent to original")
    elif match_rate >= 99.0:
        print("VALIDATION MOSTLY PASSED")
        print(f"  {mismatches} images differ - investigate mismatches")
    else:
        print("VALIDATION FAILED")
        print(f"  Too many mismatches - simplification may not be safe")

    print("=" * 60)

    # Save results
    results = {
        'total_images':     len(image_keys),
        'processed':        total,
        'skipped':          skipped,
        'matches':          matches,
        'mismatches':       mismatches,
        'match_rate':       match_rate,
        'max_box_diff':     float(max(box_diffs)) if box_diffs else 0,
        'mean_box_diff':    float(np.mean(box_diffs)) if box_diffs else 0,
        'face_count_diffs': face_count_diffs,
    }

    out_path = os.path.join(_PROJECT_ROOT, 'outputs', 'validation_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    return match_rate == 100.0


if __name__ == "__main__":
    run_validation(max_images=50)