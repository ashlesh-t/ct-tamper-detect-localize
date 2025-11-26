import numpy as np
import cv2

# pipeline/preProces/preProcess.py
import numpy as np
import cv2
from typing import List, Dict, Any
import logging
from scipy import fft

logger = logging.getLogger(__name__)


def global_preprocess(img, target_img_size=256, 
                      do_clahe=True, do_gamma=True, do_sharpen=True):
    """
    Apply all global preprocessing steps to a single-channel CT slice.
    Input: 2D np.array
    Output: 2D np.array (float32), resized and enhanced
    """

    # 1. Min-Max Normalize
    img = img.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # 2. CLAHE
    if do_clahe:
        img_u8 = (img_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
    else:
        img_clahe = img_norm

    # 3. Gamma correction
    if do_gamma:
        gamma = 0.8
        img_gamma = np.power(img_clahe, gamma)
        img_gamma = (img_gamma - img_gamma.min()) / (img_gamma.max() - img_gamma.min() + 1e-6)
    else:
        img_gamma = img_clahe

    # 4. Remove black borders
    thresh = (img_gamma > 0.05).astype(np.uint8)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_crop = img_gamma[y:y+h, x:x+w]
    else:
        img_crop = img_gamma

    # 5. Resize to target size
    img_resized = cv2.resize(img_crop, (target_img_size, target_img_size), interpolation=cv2.INTER_LINEAR)

    # 6. Sharpen
    if do_sharpen:
        sharpen_kernel = np.array([[0, -1,  0],
                                   [-1, 5, -1],
                                   [0, -1,  0]], dtype=np.float32)
        img_sharp = cv2.filter2D(img_resized, -1, sharpen_kernel)
        img_final = np.clip(img_sharp, 0, 1)
    else:
        img_final = img_resized

    return img_final.astype(np.float32)


def preprocess(sorted_file_list):
    """
    sorted_file_list: list of dicts with keys:
        - "fname": filename
        - "data": path to .npy file
    Returns: list of dicts with processed arrays
    """

    all_samples = []

    for data in sorted_file_list:
        fpath = data.get("data")
        fname = data.get("fname")

        if fpath is None or fname is None:
            continue

        try:
            img = np.load(fpath)

            # ensure it's 2D
            if img.ndim > 2:
                img = img.squeeze()
            
            pre_processed_img = global_preprocess(img)

            all_samples.append({
                "fname": fname,
                "data": pre_processed_img
            })

        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    return all_samples


class CTMultiChannelPreprocessor:
    def __init__(self, target_size=384):
        self.target_size = target_size
        
    def global_preprocess_ct(self, img: np.ndarray) -> np.ndarray:
        """Preprocess CT channel (same as your original function)"""
        # 1. Min-Max Normalize
        img = img.astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # 2. CLAHE
        img_u8 = (img_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0

        # 3. Gamma correction
        gamma = 0.8
        img_gamma = np.power(img_clahe, gamma)
        img_gamma = (img_gamma - img_gamma.min()) / (img_gamma.max() - img_gamma.min() + 1e-6)

        # 4. Remove black borders
        thresh = (img_gamma > 0.05).astype(np.uint8)
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_crop = img_gamma[y:y+h, x:x+w]
        else:
            img_crop = img_gamma

        # 5. Resize to target size
        img_resized = cv2.resize(img_crop, (self.target_size, self.target_size), 
                               interpolation=cv2.INTER_LINEAR)

        # 6. Sharpen
        sharpen_kernel = np.array([[0, -1,  0],
                                 [-1, 5, -1],
                                 [0, -1,  0]], dtype=np.float32)
        img_sharp = cv2.filter2D(img_resized, -1, sharpen_kernel)
        img_final = np.clip(img_sharp, 0, 1)

        return img_final.astype(np.float32)
    
    def extract_roi_channel(self, ct_channel: np.ndarray) -> np.ndarray:
        """Extract ROI channel using adaptive thresholding"""
        # Use Otsu's thresholding to find ROI
        ct_normalized = (ct_channel * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(ct_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original CT
        roi_channel = ct_channel * (binary_mask / 255.0)
        
        return roi_channel
    
    def compute_fft_channel(self, ct_channel: np.ndarray) -> np.ndarray:
        """Compute FFT channel with magnitude spectrum"""
        # Compute 2D FFT
        fft_transform = fft.fft2(ct_channel)
        fft_shifted = fft.fftshift(fft_transform)
        
        # Compute magnitude spectrum and log scale
        magnitude_spectrum = np.abs(fft_shifted)
        log_spectrum = np.log1p(magnitude_spectrum)
        
        # Normalize
        fft_normalized = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min() + 1e-6)
        
        return fft_normalized.astype(np.float32)
    
    def preprocess_single_slice(self, ct_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess single slice to generate all channels"""
        # Ensure 2D
        if ct_data.ndim > 2:
            ct_data = ct_data.squeeze()
        
        # Process CT channel
        ct_processed = self.global_preprocess_ct(ct_data)
        
        # Generate ROI channel
        roi_channel = self.extract_roi_channel(ct_processed)
        
        # Generate FFT channel
        fft_channel = self.compute_fft_channel(ct_processed)
        
        return {
            'CT': ct_processed,
            'ROI': roi_channel,
            'FFT': fft_channel
        }

# Add this helper at the top (after imports)
def ensure_3ch(arr, img_size=384):
    if arr is None: 
        return np.zeros((img_size, img_size, 3), dtype=np.float32)
    if arr.ndim == 2: 
        return np.stack([arr] * 3, axis=-1)  # Identical stack like working code
    if arr.shape[-1] >= 3: 
        return arr[:, :, :3]
    ch = arr.shape[-1]
    pad = np.zeros((arr.shape[0], arr.shape[1], 3 - ch), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)

# Update the preprocess overload (replace the existing def preprocess at bottom)
def preprocess(sorted_file_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocessing aligned with working standalone: Stack identical CT to 3ch.
    (ROI/FFT optional for multi-stream; disabled for now to match DenseNet).
    """
    # Reuse your global_preprocess_ct for CT (CLAHE/gamma/etc.)
    def process_ct_slice(ct_data: np.ndarray, target_size=384) -> np.ndarray:
        # Your existing global_preprocess_ct code here (copy from class)
        img = ct_data.astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img_u8 = (img_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
        gamma = 0.8
        img_gamma = np.power(img_clahe, gamma)
        img_gamma = (img_gamma - img_gamma.min()) / (img_gamma.max() - img_gamma.min() + 1e-6)
        thresh = (img_gamma > 0.05).astype(np.uint8)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_crop = img_gamma[y:y+h, x:x+w]
        else:
            img_crop = img_gamma
        img_resized = cv2.resize(img_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img_sharp = cv2.filter2D(img_resized, -1, sharpen_kernel)
        img_final = np.clip(img_sharp, 0, 1)
        return img_final.astype(np.float32)

    processed_samples = []
    for data_item in sorted_file_list:
        fname = data_item.get("fname")
        ct_data = data_item.get("data")
        if ct_data is None or fname is None:
            logger.warning(f"Skipping invalid data item: {fname}")
            continue
        try:
            # Process CT only
            ct_processed = process_ct_slice(ct_data, target_size=384)
            # Stack identical to 3ch (match working code)
            multi_channel = ensure_3ch(ct_processed, img_size=384)
            # For compatibility (keep "channels" key, but single-derived)
            channels = {'CT': ct_processed, 'ROI': ct_processed, 'FFT': ct_processed}  # Identical for now
            processed_samples.append({
                "fname": fname,
                "channels": channels,
                "original_shape": ct_data.shape,
                "multi_channel": multi_channel  # New: Direct 3ch np for DenseNet
            })
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            continue
    logger.info(f"Successfully processed {len(processed_samples)} slices")
    return processed_samples