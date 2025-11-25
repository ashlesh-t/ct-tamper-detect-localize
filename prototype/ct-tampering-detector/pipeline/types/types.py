# pipeline/types/types.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class CTMultiChannelData:
    """Container for multi-channel CT data"""
    fname: str
    ct_channel: np.ndarray  # Original CT slice
    roi_channel: Optional[np.ndarray] = None  # ROI channel
    fft_channel: Optional[np.ndarray] = None  # FFT channel
    metadata: Optional[Dict[str, Any]] = None
    
class Types:
    type1 = "REAL"
    type2 = "FAKE"
    type3 = "INJECTED"
    type4 = "REMOVED"