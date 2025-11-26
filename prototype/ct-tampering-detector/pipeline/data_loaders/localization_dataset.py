# pipeline/data_loaders/localization_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LocalizationDataset(Dataset):
    def __init__(self, slice_data: List[Dict[str, Any]], preprocess_fn=None):
        """
        Args:
            slice_data: List of {"fname": str, "data": np.ndarray (H, W)}
            preprocess_fn: Function to preprocess raw CT slice
        """
        self.slice_data = slice_data
        self.preprocess_fn = preprocess_fn
        
    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):
        item = self.slice_data[idx]
        raw_data = item["data"]
        fname = item["fname"]
        
        # Preprocess if function provided
        if self.preprocess_fn:
            processed_data = self.preprocess_fn(raw_data)
        else:
            processed_data = raw_data
            
        return {
            "raw_data": raw_data,  # Keep original for resizing
            "processed_data": processed_data,
            "fname": fname,
            "original_shape": raw_data.shape
        }