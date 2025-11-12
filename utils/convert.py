import os
import numpy as np
import pydicom
from pathlib import Path

# Input and output dirs
input_dir = Path("/mnt/mydata/main")
output_dir = Path("/mnt/mydata/main2")
output_dir.mkdir(parents=True, exist_ok=True)

def dcm_to_hu(dcm_path):
    """Convert a single DICOM slice to Hounsfield Units (HU) numpy array."""
    ds = pydicom.dcmread(dcm_path)

    # Get pixel array
    image = ds.pixel_array.astype(np.int16)

    # Handle missing values
    image[image == -2000] = 0

    # Get rescale slope/intercept (for HU conversion)
    intercept = getattr(ds, "RescaleIntercept", 0)
    slope = getattr(ds, "RescaleSlope", 1)

    if slope != 1:
        image = image.astype(np.float64) * slope
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return image

print(f"Converting DICOMs from {input_dir} → {output_dir}...")

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".dcm"):
            dcm_path = Path(root) / file

            # Get UUID folder name relative to input_dir
            uuid_folder = dcm_path.parent.relative_to(input_dir)

            # Make the same folder in main2
            target_folder = output_dir / uuid_folder
            target_folder.mkdir(parents=True, exist_ok=True)

            # Save as .npy in the same structure
            npy_path = target_folder / (dcm_path.stem + ".npy")

            try:
                hu_image = dcm_to_hu(dcm_path)
                np.save(npy_path, hu_image)
                print(f"Saved {npy_path}")
            except Exception as e:
                print(f"Error converting {dcm_path}: {e}")

print("\n✅ Done! All .dcm converted to .npy in main2/")
