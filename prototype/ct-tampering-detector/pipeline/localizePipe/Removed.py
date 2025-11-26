# pipeline/localizePipe/Removed.py
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
import pandas as pd

from pipeline.architectures.removal_localization_arch import MultiChannelUNet
from pipeline.util.forensic_filters import (
    generate_ela_map, generate_noise_residual, 
    generate_fft_energy_map, apply_lung_window
)
from pipeline.data_loaders.localization_dataset import LocalizationDataset
from pipeline.util.model_utils import load_model
from logs.logger import get_logger
from pipeline.config.configs import Config
config = Config()

logger = get_logger(__name__)

class Removed:
    def __init__(self, slice_data: List[Dict[str, Any]], num_slices: int):
        """
        Initialize the Removal localizer.
        
        Args:
            slice_data: List of dicts with {"fname": str, "data": np.ndarray (H, W)}
            num_slices: Total number of slices
        """
        self.slice_data = slice_data
        self.num_slices = num_slices
        self.model = None
        self.device = torch.device(config.DEVICE)
        self._load_model()
        logger.info(f"Initialized Removal localizer for {num_slices} slices")

    def _load_model(self):
        """Load the pre-trained MultiChannelUNet for removal localization."""
        # Try to load best dice model first, then best loss model
        dice_path = Path(config.REMOVAL_BEST_DICE_MODEL)
        
        model_path = None
        if dice_path.exists():
            model_path = dice_path
            logger.info("Loading removal localization model: best_dice_model.pth")
        else:
            raise FileNotFoundError(
                f"No removal localization model found at {config.REMOVAL_LOCALIZATION_DIR}"
            )
        
        # Initialize model
        self.model = MultiChannelUNet(n_channels=4, n_classes=1)
        self.model = load_model(model_path, self.model, self.device)
        logger.info("Removal localization model loaded successfully")

    def _preprocess_slice(self, raw: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single raw CT slice for removal localization.
        Creates 4 forensic channels as in your training.
        
        Args:
            raw: Original CT slice (H, W)
            
        Returns:
            Processed tensor (4, 512, 512)
        """
        # Apply lung window if needed
        if raw.min() < -500 or raw.max() > 500:
            image = apply_lung_window(raw)
        else:
            image = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        
        image = image.astype(np.float32)
        
        # Resize to 512x512 as in training
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Generate forensic channels
        ch_ela = generate_ela_map(image)
        ch_noise = generate_noise_residual(image)
        ch_fft = generate_fft_energy_map(image)
        
        # Stack all channels: [original, ela, noise, fft]
        combined = np.stack([image, ch_ela, ch_noise, ch_fft], axis=0)
        
        return torch.from_numpy(combined).float()

    def _predict_single_slice(self, processed_tensor: torch.Tensor) -> np.ndarray:
        """
        Predict removal probability map for a single slice.
        
        Args:
            processed_tensor: (4, 512, 512) tensor
            
        Returns:
            Probability map (512, 512)
        """
        with torch.no_grad():
            processed_tensor = processed_tensor.to(self.device).unsqueeze(0)  # Add batch dimension
            logits = self.model(processed_tensor)
            prob_map = torch.sigmoid(logits)
            return prob_map.squeeze().squeeze().cpu().numpy()  # Remove batch and channel dims

    def _extract_removal_bbox(self, prob_map: np.ndarray, threshold: float = 0.5) -> Optional[List[Tuple[int, int]]]:
        """
        Extract bounding box from removal probability map.
        Removal typically shows as larger regions, so we use different parameters.
        
        Args:
            prob_map: Probability map (512, 512)
            threshold: Binary threshold
            
        Returns:
            List of 4 points representing bounding box
        """
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        if not np.any(binary_mask):
            return None
        
        # Find contours - removal might have multiple regions
        contours, _ = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get the largest contour (main removal region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle with padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # For removal, use larger padding since regions are typically bigger
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(prob_map.shape[1] - x, w + 2 * pad)
        h = min(prob_map.shape[0] - y, h + 2 * pad)
        
        # Return 4 points
        return [
            (x, y),           # top-left
            (x + w, y),       # top-right
            (x + w, y + h),   # bottom-right
            (x, y + h)        # bottom-left
        ]

    def _create_removal_heatmap_b64(self, prob_map: np.ndarray) -> str:
        """
        Create base64 encoded heatmap for removal localization.
        
        Args:
            prob_map: Probability map (512, 512)
            
        Returns:
            base64 encoded PNG string
        """
        # Convert to heatmap
        heatmap = (prob_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to base64
        pil_img = Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _calculate_removal_metrics(self, prob_map: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate metrics for removal detection.
        
        Args:
            prob_map: Probability map
            threshold: Binary threshold
            
        Returns:
            Dictionary with detection metrics
        """
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        return {
            'max_probability': float(np.max(prob_map)),
            'mean_probability': float(np.mean(prob_map)),
            'detected_area': int(np.sum(binary_mask)),
            'detection_confidence': float(np.max(prob_map))  # Use max probability as confidence
        }

    def get_results(self, affected_fnames: List[str]) -> List[Dict[str, Any]]:
        """
        Run removal localization on affected slices.
        
        Args:
            affected_fnames: List of filenames to process
            
        Returns:
            List of removal localization results
        """
        if not affected_fnames:
            logger.warning("No affected filenames provided for removal localization")
            return []

        # Filter for affected slices
        affected_slices = [s for s in self.slice_data if s["fname"] in affected_fnames]
        logger.info(f"Processing {len(affected_slices)} slices for removal localization")

        results = []
        
        for slice_data in tqdm(affected_slices, desc="Localizing removals"):
            try:
                fname = slice_data["fname"]
                raw_data = slice_data["data"]
                
                # Preprocess (creates 4 forensic channels)
                processed_tensor = self._preprocess_slice(raw_data)
                
                # Predict
                prob_map = self._predict_single_slice(processed_tensor)
                
                # Extract bounding box
                coords = self._extract_removal_bbox(prob_map)
                
                # Create heatmap
                heatmap_b64 = self._create_removal_heatmap_b64(prob_map)
                
                # Calculate metrics
                metrics = self._calculate_removal_metrics(prob_map)
                
                results.append({
                    "fname": fname,
                    "coords": coords,  # Bounding box coordinates
                    "heatmap": heatmap_b64,  # base64 encoded heatmap
                    "prob_max": metrics['max_probability'],
                    "prob_mean": metrics['mean_probability'],
                    "mask_area": metrics['detected_area'],
                    "detection_confidence": metrics['detection_confidence'],
                    "localization_type": "removal"
                })
                
            except Exception as e:
                logger.error(f"Error processing {slice_data.get('fname', 'unknown')} for removal: {e}")
                # Return empty result for this slice
                results.append({
                    "fname": slice_data.get("fname", "unknown"),
                    "coords": None,
                    "heatmap": "",
                    "prob_max": 0.0,
                    "prob_mean": 0.0,
                    "mask_area": 0,
                    "detection_confidence": 0.0,
                    "localization_type": "removal"
                })

        logger.info(f"Completed removal localization for {len(results)} slices")
        return results