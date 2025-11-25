# pipeline/data_loaders/inference_dataset.py

"""
Custom Dataset for batched inference on CT slices.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    def __init__(self, slice_data: List[Dict[str, Any]], preprocess_fn: Callable[[np.ndarray], torch.Tensor]):
        """
        Args:
            slice_data: List of {"fname": str, "data": np.ndarray (H, W)}.
            preprocess_fn: Function to preprocess raw -> tensor (e.g., windowing + transform).
        """
        self.slice_data = slice_data
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.slice_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.slice_data[idx]
        raw = item["data"].astype(np.float32)  # Ensure float32
        img_tensor = self.preprocess_fn(raw)
        H_orig, W_orig = raw.shape
        return {
            "images": img_tensor,  # (C, H, W)
            "fnames": item["fname"],
            "orig_shapes": (H_orig, W_orig),
            "raw": raw  # Optional: keep for post-processing if needed
        }

class MultiChannelCTDataset(Dataset):
    def __init__(self, 
                 slice_data: List[Dict[str, Any]], 
                 transform: Callable = None,
                 img_size: int = 384):
        """
        Args:
            slice_data: List of {"fname": str, "channels": dict with CT, ROI, FFT}
            transform: Optional torchvision transform
            img_size: Target image size
        """
        self.slice_data = slice_data
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self) -> int:
        return len(self.slice_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.slice_data[idx]
        channels = item["channels"]
        fname = item["fname"]
        
        # Stack channels in correct order: [CT, ROI, FFT]
        multi_channel = np.stack([
            channels['CT'],
            channels['ROI'], 
            channels['FFT']
        ], axis=-1)  # Shape: (H, W, 3)
        
        # Convert to tensor and normalize
        if self.transform:
            multi_channel = self.transform(multi_channel)
        else:
            # Default normalization
            multi_channel = torch.from_numpy(multi_channel).permute(2, 0, 1).float()
            multi_channel = (multi_channel - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return {
            "images": multi_channel,  # (3, H, W)
            "fnames": fname,
            "orig_shapes": item.get("original_shape", (self.img_size, self.img_size))
        }

class EvalTransforms:
    """Evaluation transforms matching training preprocessing"""
    def __init__(self, img_size=384):
        self.img_size = img_size
        
    def __call__(self, x_np: np.ndarray) -> torch.Tensor:
        import torchvision.transforms as transforms
        
        transform_chain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform_chain(x_np)