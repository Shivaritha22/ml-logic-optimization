"""
Smoke test to verify MTCNN works on a given dataset.
Usage:
    python tests/smoke_test.py celeba
    python tests/smoke_test.py wider
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facenet_pytorch import MTCNN
from PIL import Image
import torch


def collect_images(dataset):
    """
    Returns a list of image paths based on dataset type.
    Supports: celeba, wider
    """
    if dataset == 'celeba':
        base_dir = os.path.join('data', 'celeba', 'img_align_celeba', 'img_align_celeba')
        if not os.path.exists(base_dir):
            print(f"ERROR: Directory not found: {base_dir}")
            return []
        all_images = sorted(os.listdir(base_dir))
        return [os.path.join(base_dir, f) for f in all_images[:5]]

    elif dataset == 'wider':
        base_dir = os.path.join('data', 'widerface', 'WIDER_val', 'images')
        if not os.path.exists(base_dir):
            print(f"ERROR: Directory not found: {base_dir}")
            return []
        images = []
        for event_folder in sorted(os.listdir(base_dir)):
            event_path = os.path.join(base_dir, event_folder)
            if not os.path.isdir(event_path):
                continue
            for img_file in sorted(os.listdir(event_path)):
                if img_file.lower().endswith('.jpg'):
                    images.append(os.path.join(event_path, img_file))
                    break
            if len(images) == 5:
                break
        return images

    else:
        print(f"ERROR: Unknown dataset '{dataset}'. Use 'celeba' or 'wider'")
        return []


def run_smoke_test(dataset):
    print(f"SMOKE TEST: MTCNN on {dataset.upper()}")
    print("-" * 60)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=True, device=device)
    print("MTCNN initialized")

    test_images = collect_images(dataset)
    if not test_images:
        return False

    print(f"Testing on {len(test_images)} images")
    print("-" * 60)

    success_count = 0
    for img_path in test_images:
        img = Image.open(img_path).convert('RGB')
        boxes, probs = mtcnn.detect(img)

        img_name = os.path.basename(img_path)
        if boxes is not None:
            print(f"{img_name}: Detected {len(boxes)} face(s), confidence: {probs[0]:.4f}")
            success_count += 1
        else:
            print(f"{img_name}: No faces detected")

    print("-" * 60)
    print(f"Results: {success_count}/{len(test_images)} successful")

    return success_count >= 3


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/smoke_test.py [celeba|wider]")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    success = run_smoke_test(dataset)

    if success:
        print("SMOKE TEST PASSED")
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED")
        sys.exit(1)