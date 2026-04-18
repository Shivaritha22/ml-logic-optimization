"""
Verify all dependencies are installed correctly.
"""

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"ERROR PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"ERROR torchvision: {e}")
        return False
    
    try:
        from facenet_pytorch import MTCNN
        print(f"facenet-pytorch imported successfully")
    except ImportError as e:
        print(f"ERROR facenet-pytorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"numpy {np.__version__}")
    except ImportError as e:
        print(f"ERROR numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"pandas {pd.__version__}")
    except ImportError as e:
        print(f"ERROR pandas: {e}")
        return False
    
    try:
        import matplotlib
        print(f"matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"ERROR matplotlib: {e}")
        return False
    
    try:
        import cv2
        print(f"opencv {cv2.__version__}")
    except ImportError as e:
        print(f"ERROR opencv: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"Pillow imported successfully")
    except ImportError as e:
        print(f"ERROR Pillow: {e}")
        return False
    
    print("\nAll imports successful")
    return True

if __name__ == "__main__":
    import sys
    success = test_imports()
    sys.exit(0 if success else 1)