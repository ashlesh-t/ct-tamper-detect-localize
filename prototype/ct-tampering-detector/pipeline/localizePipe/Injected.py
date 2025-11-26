# pipeline/localizePipe/Injected.py
import os
import base64
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from segmentation_models_pytorch import UnetPlusPlus
from pipeline.data_loaders.localization_dataset import LocalizationDataset
from pipeline.util.model_utils import load_model, get_transform
from logs.logger import get_logger
from config.configs import config

logger = get_logger(__name__)

class Injected:
    def __init__(self, slice_data: List[Dict[str, Any]], num_slices: int):
        """
        Initialize the Injected localizer.

        Args:
            slice_data: List of dicts, each with {"fname": str, "data": np.ndarray (H, W)}
            num_slices: Total number of slices
        """
        self.slice_data = slice_data
        self.num_slices = num_slices
        self.model = None
        self.transform = get_transform(config.LOCALIZATION_IMG_SIZE)
        self.device = torch.device(config.DEVICE)
        self._load_model()
        logger.info(f"Initialized Injected localizer for {num_slices} slices")

    def _load_model(self):
        """Load the pre-trained U-Net++ model for injection localization."""
        ckpt_path = Path(config.BEST_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
        
        # Initialize model
        self.model = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1,
            activation=None
        )
        
        # Load weights
        self.model = load_model(ckpt_path, self.model, self.device)
        logger.info("Injected localization model loaded successfully")

    def _preprocess_slice(self, raw: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single raw CT slice for localization model.
        
        Args:
            raw: Original CT slice (H, W)
            
        Returns:
            Processed tensor (3, H, W)
        """
        H_orig, W_orig = raw.shape

        # Create 3 channels with different windowing
        ch1 = self._window_image(raw, -600, 1500, True)  # Lung window
        ch1 = self._apply_clahe(ch1)
        ch2 = self._window_image(raw, 40, 400, True)     # Soft tissue window
        ch2 = self._apply_clahe(ch2)
        ch3 = self._window_image(raw, 400, 1800, True)   # Bone window
        ch3 = self._apply_clahe(ch3)
        
        # Stack channels
        img = np.stack([ch1, ch2, ch3], axis=-1)  # H_orig x W_orig x 3

        # Apply transforms
        augmented = self.transform(image=img)
        return augmented['image']  # (3, H, W)

    @staticmethod
    def _window_image(img: np.ndarray, wc: int, ww: int, to_uint8: bool = True) -> np.ndarray:
        """Apply HU windowing."""
        img_min = wc - ww // 2
        img_max = wc + ww // 2
        windowed = np.clip(img, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min + 1e-6)
        return (windowed * 255).astype(np.uint8) if to_uint8 else windowed

    @staticmethod
    def _apply_clahe(img: np.ndarray, clip: float = 2.0, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE to single channel image."""
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(img)

    def _tta_predict(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Test-Time Augmentation prediction.
        
        Args:
            img_tensor: (C, H, W) tensor
            
        Returns:
            Mean prediction probability map (H, W)
        """
        preds = []
        
        # Base prediction
        with torch.no_grad():
            out = self.model(img_tensor.unsqueeze(0))
            pred = torch.sigmoid(out).squeeze(0).squeeze(0)  # (H, W)
            preds.append(pred)
            
            # Horizontal flip
            flipped_h = torch.flip(img_tensor, dims=[2])
            out_h = self.model(flipped_h.unsqueeze(0))
            pred_h = torch.sigmoid(out_h).squeeze(0).squeeze(0)
            pred_h = torch.flip(pred_h, dims=[1])
            preds.append(pred_h)
            
            # Vertical flip
            flipped_v = torch.flip(img_tensor, dims=[1])
            out_v = self.model(flipped_v.unsqueeze(0))
            pred_v = torch.sigmoid(out_v).squeeze(0).squeeze(0)
            pred_v = torch.flip(pred_v, dims=[0])
            preds.append(pred_v)
        
        # Average predictions
        stacked = torch.stack(preds, dim=0)
        return torch.mean(stacked, dim=0)  # (H, W)

    def _extract_bbox(self, mask_bool: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Extract bounding box coordinates from binary mask.
        
        Args:
            mask_bool: Binary mask (H, W)
            
        Returns:
            List of 4 points representing bounding box: [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
            Returns None if no mask detected
        """
        if not np.any(mask_bool):
            return None
            
        # Find contours
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(mask_bool.shape[1] - x, w + 2 * pad)
        h = min(mask_bool.shape[0] - y, h + 2 * pad)
        
        # Return 4 points
        return [
            (x, y),           # top-left
            (x + w, y),       # top-right
            (x + w, y + h),   # bottom-right
            (x, y + h)        # bottom-left
        ]

    def _create_heatmap_b64(self, prob_map: np.ndarray, original_shape: Tuple[int, int]) -> str:
        """
        Create base64 encoded heatmap from probability map.
        
        Args:
            prob_map: Probability map (H_model, W_model)
            original_shape: Target shape for resizing (H_orig, W_orig)
            
        Returns:
            base64 encoded PNG string
        """
        # Resize to original dimensions
        H_orig, W_orig = original_shape
        prob_resized = cv2.resize(prob_map, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        
        # Convert to heatmap
        heatmap = (prob_resized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to base64
        pil_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_results(self, affected_fnames: List[str]) -> List[Dict[str, Any]]:
        """
        Run localization on affected slices.
        
        Args:
            affected_fnames: List of filenames to process
            
        Returns:
            List of localization results with coordinates and heatmaps
        """
        if not affected_fnames:
            logger.warning("No affected filenames provided for localization")
            return []

        # Filter for affected slices
        affected_slices = [s for s in self.slice_data if s["fname"] in affected_fnames]
        logger.info(f"Processing {len(affected_slices)} slices for injection localization")

        results = []
        
        for slice_data in tqdm(affected_slices, desc="Localizing injections"):
            try:
                fname = slice_data["fname"]
                raw_data = slice_data["data"]
                original_shape = raw_data.shape
                
                # Preprocess
                processed_tensor = self._preprocess_slice(raw_data)
                processed_tensor = processed_tensor.to(self.device)
                
                # Predict with TTA
                prob_map = self._tta_predict(processed_tensor)
                prob_np = prob_map.cpu().numpy()  # (H_model, W_model)
                
                # Create binary mask
                binary_mask = (prob_np > 0.5).astype(np.uint8)
                
                # Resize binary mask to original dimensions
                binary_mask_orig = cv2.resize(
                    binary_mask, 
                    (original_shape[1], original_shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Extract bounding box
                coords = self._extract_bbox(binary_mask_orig.astype(bool))
                
                # Create heatmap
                heatmap_b64 = self._create_heatmap_b64(prob_np, original_shape)
                
                results.append({
                    "fname": fname,
                    "coords": coords,  # List of 4 points or None
                    "heatmap": heatmap_b64,  # base64 encoded PNG
                    "prob_max": float(np.max(prob_np)),  # Maximum probability for confidence
                    "mask_area": int(np.sum(binary_mask_orig))  # Pixel area of detected region
                })
                
            except Exception as e:
                logger.error(f"Error processing {slice_data.get('fname', 'unknown')}: {e}")
                # Return empty result for this slice
                results.append({
                    "fname": slice_data.get("fname", "unknown"),
                    "coords": None,
                    "heatmap": "",
                    "prob_max": 0.0,
                    "mask_area": 0
                })

        logger.info(f"Completed localization for {len(results)} slices")
        return results