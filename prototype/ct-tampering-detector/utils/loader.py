# utils/loader.py
import streamlit as st
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image
import os
from pathlib import Path
import re

def natural_sort_key(s):
    """Sort filenames like humans: slice_1.npy before slice_10.npy"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def load_medical_image(input_data) -> np.ndarray:
    """
    Handles:
    - Single file: .dcm, .nii/.nii.gz, .npy (1D/2D/3D), .jpg/.png
    - Multiple files (folder): DICOMs OR .npy slices
    """
    if input_data is None:
        return None

    # Case 1: Folder upload (list of UploadedFile)
    if isinstance(input_data, list) and len(input_data) > 1:
        files = sorted(input_data, key=lambda x: natural_sort_key(x.name))

        # Try loading as stack of .npy slices first
        if any(f.name.lower().endswith('.npy') for f in files):
            arrays = []
            for f in files:
                if not f.name.lower().endswith('.npy'):
                    st.warning(f"Skipping non-npy: {f.name}")
                    continue
                arr = np.load(f)
                if arr.ndim == 2:
                    arrays.append(arr)
                else:
                    st.warning(f"Skipping {f.name}: expected 2D slice, got shape {arr.shape}")
            if arrays:
                volume = np.stack(arrays, axis=0)
                st.success(f"Loaded {len(arrays)} .npy slices → {volume.shape}")
                return volume.astype(np.float32)

        # Otherwise assume DICOM folder
        arrays = []
        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                arrays.append(ds.pixel_array.astype(np.float32))
            except:
                st.warning(f"Could not read {f.name} as DICOM")
        if arrays:
            volume = np.stack(arrays, axis=0)
            st.success(f"Loaded {len(arrays)} DICOM slices → {volume.shape}")
            return volume

    # Case 2: Single file
    file = input_data if not isinstance(input_data, list) else input_data[0]
    name = file.name.lower()
    file.seek(0)

    try:
        if name.endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(file)
            img = ds.pixel_array.astype(np.float32)
            return np.expand_dims(img, 0)

        elif name.endswith(('.nii', '.nii.gz')):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            nii = nib.load(tmp_path)
            vol = nii.get_fdata().astype(np.float32)
            os.unlink(tmp_path)
            return vol

        elif name.endswith('.npy'):
            arr = np.load(file)
            if arr.ndim == 2:
                return np.expand_dims(arr, 0)           # (1, H, W)
            elif arr.ndim == 3:
                return arr.astype(np.float32)           # (D, H, W)
            elif arr.ndim == 1:
                size = int(np.sqrt(len(arr)))
                arr = arr.reshape(size, size)
                return np.expand_dims(arr, 0)
            else:
                st.error(f"Unsupported .npy shape: {arr.shape}")
                return None

        elif name.endswith(('.jpg', '.jpeg', '.png')):
            img = np.array(Image.open(file).convert('L'))
            return np.expand_dims(img, 0).astype(np.float32)

    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")

    return None