# pipeline/data_loaders/inference_dataset.py

"""
Custom Dataset for batched inference on CT slices.
"""

import logging
from typing import List, Dict, Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    """Simple dataset for single-channel CT slice inference."""
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

class StackedChannelsToRGB:
    """Utility to convert multi-channel numpy arrays into a RGB PIL.Image."""
    def __call__(self, arr: np.ndarray) -> Image.Image:
        # Ensure input is numpy
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        arr = arr.astype(np.float32)

        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError("Input must be HxWx1 or HxWx3")
        # Normalize per channel to [0,1] then scale to 0-255
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        if arr.shape[2] == 3:
            for c in range(3):
                ch = arr[:, :, c]
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
                rgb[:, :, c] = (ch * 255).astype(np.uint8)
        else:
            ch = arr[:, :, 0]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
            single = (ch * 255).astype(np.uint8)
            rgb = np.stack([single] * 3, axis=-1)
        return Image.fromarray(rgb)

class MultiChannelCTDataset(Dataset):
    """Dataset for multi-channel CT slices represented as dict-channels (CT, ROI, FFT)."""
    def __init__(self,
                 slice_data: List[Dict[str, Any]],
                 transform: Optional[Callable] = None,
                 img_size: int = 384):
        """
        Args:
            slice_data: List of {"fname": str, "channels": dict with 'CT','ROI','FFT'}
            transform: Optional callable/transform. If None, returns tensor of shape (3, H, W).
            img_size: Target image size recorded as original shape fallback.
        """
        self.slice_data = slice_data
        self.transform = transform
        self.img_size = img_size
        self.to_rgb = StackedChannelsToRGB()

    def __len__(self) -> int:
        return len(self.slice_data)

    # In MultiChannelCTDataset.__getitem__ (replace the stacking part)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.slice_data[idx]
        channels = item["channels"]
        fname = item["fname"]

        # For DenseNet: Use pre-stacked identical 3ch
        if "multi_channel" in item:
            multi_channel = item["multi_channel"]

        # Case 2: Only CT provided → auto-create CT, BLACK, FFT
        elif "CT" in channels and (
            "ROI" not in channels or "FFT" not in channels
        ):
            ct = channels["CT"].astype(np.float32)

            # ROI = black
            roi = np.zeros_like(ct, dtype=np.float32)

            # FFT magnitude
            fft = np.fft.fftshift(np.fft.fft2(ct))
            fft = np.abs(fft).astype(np.float32)
            fft = (fft - fft.min()) / (fft.max() - fft.min() + 1e-8)

            multi_channel = np.stack([ct, roi, fft], axis=-1)

        # Case 3: Old behavior (already complete)
        else:
            multi_channel = np.stack([
                channels['CT'],
                channels['ROI'],
                channels['FFT']
            ], axis=-1)

        # Rest unchanged: to_rgb(PIL) if needed, then transform(PIL)
        img_for_transform = multi_channel
        if self.transform:

            transformed = self.transform(img_for_transform)
        else:
            transformed = torch.from_numpy(multi_channel).permute(2, 0, 1).float()

        return {
            "images": transformed,  # (3, H, W)
            "fnames": fname,
            "orig_shapes": item.get("original_shape", (self.img_size, self.img_size))
        }

class EvalTransforms:
    """Evaluation transforms for 3-channel models (default 384x384)."""
    def __init__(self, img_size=384, mean=None, std=None):
        self.img_size = img_size
        self.mean = mean or [0.5, 0.5, 0.5]
        self.std = std or [0.5, 0.5, 0.5]

    def __call__(self, x_np: np.ndarray) -> torch.Tensor:
        transform_chain = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return transform_chain(x_np)

class InjRemEvalTransforms:
    """Evaluation transforms for injected/removed model (288x288) with per-channel stats."""
    def __init__(self, img_size=288, channel_stats=None):
        self.img_size = img_size
        self.channel_stats = channel_stats or {
            0: {'mean': 0.5, 'std': 0.5},
            1: {'mean': 0.5, 'std': 0.5},
            2: {'mean': 0.5, 'std': 0.5}
        }
        self.to_rgb = StackedChannelsToRGB()
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def __call__(self, x_np: np.ndarray) -> torch.Tensor:
        rgb_img = self.to_rgb(x_np)
        tensor_img = self.base_transform(rgb_img)
        # Normalize per channel using stored stats
        for c in range(3):
            if c in self.channel_stats:
                stats = self.channel_stats[c]
                tensor_img[c] = (tensor_img[c] - stats['mean']) / (stats['std'] + 1e-6)
        return tensor_img