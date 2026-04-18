"""
Run instrumented MTCNN on CelebA images and collect decision point traces.
Usage: python scripts/run_profiling.py
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


def run_profiling(num_images=1000):
    print("MTCNN Profiling on CelebA")
    print("-" * 60)

    device = torch.device('cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    celeba_dir = os.path.join(
        _PROJECT_ROOT, 'data', 'celeba', 'img_align_celeba', 'img_align_celeba'
    )

    all_images = sorted(os.listdir(celeba_dir))
    selected_images = all_images[:num_images]

    print(f"Total images available: {len(all_images)}")
    print(f"Images to profile:      {len(selected_images)}")
    print(f"Device:                 {device}")
    print("-" * 60)

    tracer = DecisionTracer(log_dir=os.path.join(_PROJECT_ROOT, 'logs'))

    failed = 0
    for i, img_file in enumerate(selected_images):

        # Progress update every 100 images
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(selected_images)} ...")

        img_path = os.path.join(celeba_dir, img_file)

        try:
            img = Image.open(img_path).convert('RGB')
            tracer.set_image(img_file)

            detect_face_instrumented(
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

        except Exception as e:
            print(f"ERROR on {img_file}: {e}")
            failed += 1
            continue

    tracer.save()

    print("-" * 60)
    print(f"Profiling complete")
    print(f"Images processed: {len(selected_images) - failed}/{len(selected_images)}")
    print(f"Failed:           {failed}")
    print(f"Total log entries:{len(tracer.entries)}")
    print(f"Log file:         {tracer.log_path}")


if __name__ == "__main__":
    run_profiling(num_images=1000)