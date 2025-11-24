# utils/processing.py
import cv2
import numpy as np

def apply_histeq(img):
    if img.ndim == 3:
        return np.stack([cv2.equalizeHist(img[i].astype(np.uint8)) for i in range(img.shape[0])])
    else:
        return cv2.equalizeHist(img.astype(np.uint8))