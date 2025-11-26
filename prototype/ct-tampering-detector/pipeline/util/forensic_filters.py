# pipeline/util/forensic_filters.py
import numpy as np
import cv2

def generate_ela_map(image_norm, quality=90):
    """Generate Error Level Analysis map"""
    img_uint8 = (image_norm * 255).astype(np.uint8)
    _, encoded = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, 0)
    ela = np.abs(img_uint8.astype(np.float32) - decoded.astype(np.float32))
    return np.clip(ela / 255.0 * 20.0, 0, 1)

def generate_noise_residual(image_norm):
    """Generate noise residual map"""
    img_uint8 = (image_norm * 255).astype(np.uint8)
    denoised = cv2.medianBlur(img_uint8, 3)
    residual = np.abs(img_uint8.astype(np.float32) - denoised.astype(np.float32))
    return np.clip(residual / 255.0 * 5.0, 0, 1)

def generate_fft_energy_map(image_norm, block_size=32):
    """Generate FFT energy map"""
    gy, gx = np.gradient(image_norm)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    energy_map = cv2.boxFilter(gradient_magnitude, -1, (block_size, block_size))
    return np.clip(energy_map * 5.0, 0, 1)

def apply_lung_window(image, level=-600, width=1500):
    """Apply lung window to CT image"""
    lower = level - (width / 2)
    upper = level + (width / 2)
    image = np.clip(image, lower, upper)
    return (image - lower) / (upper - lower)