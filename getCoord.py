import nibabel as nib
import numpy as np
import random
import glob
import os

# Path to the folder containing all lobe segmentations
lobe_mask_dir = "normal003_mask"

# Map for easier display
lobe_files = {
    "upper_lobe_left": os.path.join(lobe_mask_dir, "lung_upper_lobe_left.nii.gz"),
    "upper_lobe_right": os.path.join(lobe_mask_dir, "lung_upper_lobe_right.nii.gz"),
    "lower_lobe_left": os.path.join(lobe_mask_dir, "lung_lower_lobe_left.nii.gz"),
    "lower_lobe_right": os.path.join(lobe_mask_dir, "lung_lower_lobe_right.nii.gz"),
}

def sample_coords_from_lobe(mask_path, lobe_name, z_range=(30, 110), N=1):
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.uint8)
    mask = np.transpose(mask, (2, 1, 0))  # shape: (z, y, x)
    
    coords = np.argwhere(mask == 1)
    
    # Filter by z-range (optional)
    coords = coords[(coords[:, 0] >= z_range[0]) & (coords[:, 0] <= z_range[1])]
    
    if len(coords) < N:
        print(f"Warning: Not enough voxels in {lobe_name} to sample {N} points.")
        return []
    
    sampled = coords[np.random.choice(len(coords), N, replace=False)]
    return [(int(x), int(y), int(z), lobe_name) for z, y, x in sampled]

# Sample 1 coordinate from each lobe (adjust N per lobe if needed)
final_coords = []
for lobe_name, path in lobe_files.items():
    final_coords += sample_coords_from_lobe(path, lobe_name, z_range=(50, 150), N=1)

# Print results
print("Sampled tumor injection coordinates (x, y, z, lobe):")
for coord in final_coords:
    print(coord)



'''
pip install TotalSegmentator
totalsegmentator -i your_input_dicom_folder -o output_folder --fast
# This eill get the plausible tumour location coord
'''