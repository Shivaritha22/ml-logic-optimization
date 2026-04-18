"""
Smoke test to verify MTCNN works on CelebA images.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from facenet_pytorch import MTCNN
from PIL import Image
import torch


def test_mtcnn_basic():
    """Test basic MTCNN functionality"""
    print("SMOKE TEST: MTCNN Basic Functionality")
    print("-" * 60)
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    mtcnn = MTCNN(keep_all=True, device=device)
    print("MTCNN initialized")
    
    celeba_dir = os.path.join('data', 'celeba', 'img_align_celeba', 'img_align_celeba')
    
    if not os.path.exists(celeba_dir):
        print(f"ERROR: Directory not found: {celeba_dir}")
        return False
    
    all_images = sorted(os.listdir(celeba_dir))
    test_images = all_images[:5]
    
    print(f"Testing on {len(test_images)} images")
    print("-" * 60)
    
    success_count = 0
    for img_file in test_images:
        img_path = os.path.join(celeba_dir, img_file)
        img = Image.open(img_path)
        boxes, probs = mtcnn.detect(img)
        
        if boxes is not None:
            print(f"{img_file}: Detected {len(boxes)} face(s), confidence: {probs[0]:.4f}")
            success_count += 1
        else:
            print(f"{img_file}: No faces detected")
    
    print("-" * 60)
    print(f"Results: {success_count}/{len(test_images)} successful")
    
    return success_count >= 3


if __name__ == "__main__":
    success = test_mtcnn_basic()
    if success:
        print("SMOKE TEST PASSED")
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED")
        sys.exit(1)