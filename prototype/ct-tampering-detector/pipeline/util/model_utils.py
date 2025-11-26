# util/model_utils.py
"""
Utility functions for model and transform handling.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def get_transform(img_size: int):
    """Get augmentation pipeline for inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

def load_model(checkpoint_path, model, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel wrapping
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model