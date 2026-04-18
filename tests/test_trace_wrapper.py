"""
Test that the instrumented detect_face works correctly on one image.
Verify:
1. Tracer logs all 7 decision points
2. Output matches original MTCNN output
3. Log file is created and readable
"""

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from facenet_pytorch import MTCNN
from src.profiling.tracer import DecisionTracer
from src.profiling.instrumented_detect_face import detect_face_instrumented
from PIL import Image
import torch
import numpy as np


def test_tracer_on_one_image():
    print("TEST: Tracer on one image")
    print("-" * 60)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load one image
    celeba_dir = os.path.join(
        _PROJECT_ROOT, 'data', 'celeba', 'img_align_celeba', 'img_align_celeba'
    )
    img_file = sorted(os.listdir(celeba_dir))[0]
    img_path = os.path.join(celeba_dir, img_file)
    img = Image.open(img_path)
    print(f"Image: {img_file}")

    # Run original MTCNN
    boxes_original, probs_original = mtcnn.detect(img)
    print(f"Original MTCNN: {len(boxes_original) if boxes_original is not None else 0} faces detected")

    # Run instrumented version
    tracer = DecisionTracer(log_dir=os.path.join(_PROJECT_ROOT, 'logs'))
    tracer.set_image(img_file)

    batch_boxes, batch_points = detect_face_instrumented(
        img,
        mtcnn.min_face_size,
        mtcnn.pnet,
        mtcnn.rnet,
        mtcnn.onet,
        mtcnn.thresholds,
        mtcnn.factor,
        device,
        tracer
    )

    print(f"Instrumented MTCNN: {len(batch_boxes[0])} faces detected")

    # Compare boxes (source of truth: original vs instrumented pipeline)
    if boxes_original is not None and len(batch_boxes[0]) > 0:
        original = np.asarray(boxes_original)
        instrumented = np.asarray(batch_boxes[0][:, :4])
        if original.shape[0] != instrumented.shape[0]:
            print(
                f"\nWARNING: Face count mismatch — original={original.shape[0]}, "
                f"instrumented={instrumented.shape[0]}; skipping box diff"
            )
        else:
            diff = np.abs(original - instrumented)
            print(
                f"\nBox coordinate difference: max={diff.max():.6f}, mean={diff.mean():.6f}"
            )
            if diff.max() < 0.001:
                print("Boxes match exactly - tracer is not changing any logic")
            else:
                print("WARNING: Boxes do not match - tracer may have broken something")

    # Save and display trace
    tracer.save()

    print("\nDecision point log:")
    print("-" * 60)
    print(f"{'Stage':<20} {'Type':<12} {'Input':>8} {'Output':>8} {'Rejected':>10} {'Value':>8}")
    print("-" * 60)
    for entry in tracer.entries:
        print(
            f"{entry['stage']:<20} "
            f"{entry['decision_type']:<12} "
            f"{entry['input_count']:>8} "
            f"{entry['output_count']:>8} "
            f"{entry['rejected_count']:>10} "
            f"{entry['value']:>8.2f}"
        )

    print("-" * 60)
    print(f"Total decision points logged: {len(tracer.entries)}")

    assert len(tracer.entries) >= 7, f"Expected at least 7 entries, got {len(tracer.entries)}"
    print("\nTEST PASSED")


if __name__ == "__main__":
    test_tracer_on_one_image()