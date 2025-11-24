# localizePipe/Injected.py
"""
Injected Tampering Localization Module.

This class handles the localization of injected regions in tampered CT slices
using a pre-trained U-Net++ segmentation model. It processes a list of slice data
dicts, filters for affected filenames, performs inference (with TTA), resizes
predictions back to original resolution, extracts bounding boxes from binary
masks, and returns serializable reports including heatmaps as base64-encoded
PNG strings for JSON compatibility.
"""

import os
import base64
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from skimage import measure  # For connected components if needed; install if missing
from tqdm import tqdm  # For progress in loops

from segmentation_models_pytorch import UnetPlusPlus
from pipeline.dataloaders.inference_dataset import InferenceDataset  
from pipeline.util.model_utils import load_model, get_transform 
from logs.logger import get_logger  
from config.configs import config
logger = get_logger(__name__)

# ------------------ CONFIG (Hardcoded; move to config/ if needed) ------------------
IMG_SIZE = 320
RADIUS_PX = 48  # Not used in inference, but for reference
TOLERANCE_PX = 48  # Not used in inference
TTA_FLIPS = ['horizontal']
BEST_CHECKPOINT =  config.BEST_CHECKPOINT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4  # For batched inference; adjust based on GPU memory

class Injected:
    def __init__(self, slice_data: List[Dict[str, Any]], num_slices: int):
        """
        Initialize the Injected localizer.

        Args:
            slice_data: List of dicts, each with {"fname": str, "data": np.ndarray (H, W)}.
            num_slices: Total number of slices (for logging).
        """
        self.slice_data = slice_data
        self.num_slices = num_slices
        self.model = None
        self.transform = None
        self._load_model()
        logger.info(f"Initialized Injected localizer for {num_slices} slices")

    def _load_model(self):
        """Load the pre-trained U-Net++ model."""
        ckpt_path = Path(BEST_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        cfg = ckpt.get('config', {'encoder': 'resnet34'})  # Default encoder if missing

        self.model = UnetPlusPlus(
            encoder_name=cfg.get('encoder', 'resnet34'),
            encoder_weights=None,
            decoder_attention_type='scse',
            classes=1,
            activation=None
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = get_transform(IMG_SIZE)  # Albumentations transform
        logger.info("Injected model loaded successfully")

    def _preprocess_slice(self, raw: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single raw slice (H, W) to tensor (C, IMG_SIZE, IMG_SIZE).

        Applies windowing, CLAHE, channel stacking, resize, and normalization.
        """
        H_orig, W_orig = raw.shape

        # Windowing + CLAHE for 3 channels
        ch1 = self._window_image(raw, -600, 1500, True)
        ch1 = self._apply_clahe(ch1)
        ch2 = self._window_image(raw, 40, 400, True)
        ch3 = self._window_image(raw, 400, 1800, True)
        img = np.stack([ch1, ch2, ch3], axis=-1)  # H_orig x W_orig x 3

        # Transform
        aug = self.transform(image=img)
        return aug['image'].to(DEVICE)  # C x H x W

    @staticmethod
    def _window_image(img: np.ndarray, wc: int, ww: int, to_uint8: bool = True) -> np.ndarray:
        """HU windowing."""
        img_min = wc - ww // 2
        img_max = wc + ww // 2
        w = np.clip(img, img_min, img_max)
        w = (w - img_min) / (img_max - img_min + 1e-6)
        return (w * 255).astype(np.uint8) if to_uint8 else w

    @staticmethod
    def _apply_clahe(img: np.ndarray, clip: float = 2.0, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(img)

    def _tta_predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Test-Time Augmentation prediction.

        Args:
            img_tensor: (C, H, W) tensor on device.

        Returns:
            Mean prediction tensor (1, H_out, W_out).
        """
        preds = []
        # Base prediction
        out = torch.sigmoid(self.model(img_tensor.unsqueeze(0)))
        preds.append(out.squeeze(0))

        # Flips
        for flip in TTA_FLIPS:
            if flip == 'horizontal':
                flipped = torch.flip(img_tensor, dims=[2])
                p = torch.sigmoid(self.model(flipped.unsqueeze(0))).squeeze(0)
                p = torch.flip(p, dims=[2])
            else:  # vertical, if added
                flipped = torch.flip(img_tensor, dims=[1])
                p = torch.sigmoid(self.model(flipped.unsqueeze(0))).squeeze(0)
                p = torch.flip(p, dims=[1])
            preds.append(p)

        stacked = torch.stack(preds, dim=0)
        return torch.mean(stacked, dim=0)  # (1, H, W)

    def _extract_bbox(self, mask_bool: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Extract quadrilateral bounding box from binary mask.

        Assumes single connected component; returns None if empty.
        Box format: [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] where (x1,y1) is top-left.
        """
        if not np.any(mask_bool):
            return None

        # Find bounding box (min/max rows/cols)
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add small padding if needed (e.g., 5px)
        pad = 5
        rmin = max(0, rmin - pad)
        rmax = min(mask_bool.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(mask_bool.shape[1], cmax + pad)

        return [
            (cmin, rmin),  # top-left
            (cmax, rmin),  # top-right
            (cmax, rmax),  # bottom-right
            (cmin, rmax)   # bottom-left
        ]

    def _prob_to_base64_png(self, prob_map: np.ndarray) -> str:
        """
        Encode probability heatmap as base64 PNG string for serialization.
        """
        # Normalize to 0-255 uint8, apply jet colormap for visualization
        prob_norm = (prob_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(colored)
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_results(self, affected_fnames: List[str]) -> List[Dict[str, Any]]:
        """
        Run localization inference on affected slices.

        Args:
            affected_fnames: List of filenames to localize.

        Returns:
            List of dicts: [{"fname": str, "coords": List[Tuple[int,int]] or None, "heatmap": str (base64 PNG)}]
        """
        if not affected_fnames:
            logger.warning("No affected filenames provided for localization")
            return []

        # Filter slice_data for affected_fnames
        affected_data = [s for s in self.slice_data if s["fname"] in affected_fnames]
        if len(affected_data) != len(affected_fnames):
            logger.warning(f"Mismatch: Found {len(affected_data)} / {len(affected_fnames)} affected slices")

        # Use custom Dataset and DataLoader for batched inference
        dataset = InferenceDataset(affected_data, self._preprocess_slice)  # Preprocess in collate if needed
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # num_workers=0 for simplicity

        reports = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Localizing injected regions"):
                # batch: list of (img_tensor, fname, orig_shape, raw_data) or similar; adjust based on Dataset
                # For simplicity, assume loop over singles if batching complex; implement batch predict if needed
                for idx in range(len(batch['images'])):  # Pseudo-batch handling
                    img_tensor = batch['images'][idx].to(DEVICE)
                    fname = batch['fnames'][idx]
                    H_orig, W_orig = batch['orig_shapes'][idx]

                    # Predict
                    prob_t = self._tta_predict(img_tensor)
                    prob_np = prob_t.detach().cpu().numpy().squeeze()  # (H_model, W_model)

                    # Binary mask at original res
                    pred_bin_model = (prob_np > 0.5).astype(np.uint8)
                    pred_bin_orig = cv2.resize(pred_bin_model, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
                    pred_bool = pred_bin_orig.astype(bool)

                    # Extract bbox
                    coords = self._extract_bbox(pred_bool)

                    # Heatmap as base64 PNG
                    prob_orig = cv2.resize(prob_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                    heatmap_b64 = self._prob_to_base64_png(prob_orig)

                    reports.append({
                        "fname": fname,
                        "coords": coords,  # List of 4 tuples or None
                        "heatmap": heatmap_b64  # base64 str
                    })

        logger.info(f"Generated {len(reports)} localization reports")
        return reports