# dataloaders/inference_dataset.py
"""
Custom Dataset for batched inference on CT slices.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Callable
import numpy as np

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