# util/model_utils.py
"""
Utility functions for model and transform handling.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(img_size: int):
    """Get augmentation pipeline for inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])