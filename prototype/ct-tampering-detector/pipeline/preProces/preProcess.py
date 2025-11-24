import numpy as np
import cv2

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
