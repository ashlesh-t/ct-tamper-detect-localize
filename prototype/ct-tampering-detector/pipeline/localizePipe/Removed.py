# localizePipe/Removed.py
"""
Removed Tampering Localization Module.

Similar to Injected, but for removed regions. Update checkpoint/windowing as needed.
"""

# Copy Injected.py content, rename class to Removed, update logger messages, checkpoint path (e.g., "models/CT_Removal_*.pth")
# For now, assume same model; customize preprocessing if FFT/magnitude needed for removal detection.

class Removed():  # Inherit if similar
    def __init__(self, slice_data, num_slices):
        super().__init__(slice_data, num_slices)
        # Override checkpoint if different
        # self._load_model()  # Calls parent's, but set different BEST_CHECKPOINT
        print("Started Removed Localization")