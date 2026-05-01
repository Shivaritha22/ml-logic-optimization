"""
Chunk 7: Validate simplified MTCNN pipeline against baseline.
Runs simplified pipeline on WIDER FACE validation set and
compares outputs against saved baseline from original pipeline.

Usage: python src/validation/validator.py
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

    image_keys = list(baseline.keys())
    if max_images:
        image_keys = image_keys[:max_images]

    print(f"Images to validate: {len(image_keys)}")
    print("=" * 60)

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

            orig        = baseline[rel_path]
            orig_boxes  = orig['boxes']
            orig_num    = orig['num_faces']
            simp_num    = len(simp_boxes) if simp_boxes is not None else 0

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
            orig_arr      = np.array(orig_boxes, dtype=np.float64)
            simp_arr      = np.array(simp_boxes[:, :4], dtype=np.float64)

            # Round before sorting to avoid floating point sort instability
            orig_rounded  = np.round(orig_arr / 10) * 10
            simp_rounded  = np.round(simp_arr / 10) * 10

            orig_idx      = np.lexsort((orig_rounded[:, 1], orig_rounded[:, 0]))
            simp_idx      = np.lexsort((simp_rounded[:, 1], simp_rounded[:, 0]))

            orig_sorted   = orig_arr[orig_idx]
            simp_sorted   = simp_arr[simp_idx]

            diff          = np.abs(orig_sorted - simp_sorted)
            max_diff      = diff.max()
            box_diffs.append(max_diff)

            if max_diff < 2.0:
                matches += 1
            else:
                mismatches += 1
                diff_per_box  = diff.max(axis=1)
                worst_box_idx = diff_per_box.argmax()
                print(f"\nMISMATCH: {rel_path}")
                print(f"  Num boxes: orig={len(orig_sorted)} simp={len(simp_sorted)}")
                print(f"  Worst box index: {worst_box_idx}")
                print(f"  Orig worst box: {orig_sorted[worst_box_idx]}")
                print(f"  Simp worst box: {simp_sorted[worst_box_idx]}")
                print(f"  Max diff: {max_diff:.4f}")

        except Exception as e:
            print(f"SKIPPED {rel_path}: {e}")
            skipped += 1
            continue

    # Results
    total_attempted = len(image_keys)
    total_processed = total_attempted - skipped
    match_rate      = matches / total_attempted * 100 if total_attempted > 0 else 0

    print(f"\nDebug: matches={matches}, mismatches={mismatches}, "
          f"skipped={skipped}, total={total_attempted}")

    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total images:     {total_attempted}")
    print(f"Processed:        {total_processed}")
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
        for item in face_count_diffs[:5]:
            print(f"  {item['image']}: orig={item['orig']} simp={item['simp']}")
        if len(face_count_diffs) > 5:
            print(f"  ... and {len(face_count_diffs) - 5} more")

    print("=" * 60)

    if match_rate >= 99.0:
        print("VALIDATION PASSED")
        print("Simplified pipeline is provably equivalent to original")
    elif match_rate >= 95.0:
        print("VALIDATION MOSTLY PASSED")
        print(f"  {mismatches} images differ - investigate mismatches")
    else:
        print("VALIDATION FAILED")
        print(f"  Too many mismatches - simplification may not be safe")

    print("=" * 60)

    results = {
        'total_images':     total_attempted,
        'processed':        total_processed,
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

    return match_rate >= 99.0


if __name__ == "__main__":
    run_validation()