"""
Advanced Multi-Head Forensic Network for CT Scan Tampering Detection
With Self-Supervised Pretraining and Hard Negative Mining
Enhanced Features:
1. Self-supervised pretraining on TB (real) data
2. Pyramid sliding with multi-frequency band analysis
3. Coarse-fine structure relationship learning
4. Hard negative memory bank
5. Chunk-wise auxiliary anatomical consistency
"""
import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pywt
import math
# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    PREPROCESSED_PATH = "/kaggle/input/ct-removal-processed"
    PREPROCESSED_PATH_TB =  ""
    TB_PATH = "/kaggle/input/true-benign"
    REMOVAL_PATH = os.path.join(PREPROCESSED_PATH, "2") # CT_Removal
   
    OUTPUT_DIR = "/kaggle/working/ct_forensic_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.3
    RANDOM_SEED = 42
   
    # Training hyperparameters - PRETRAINING (IMPROVED)
    PRETRAIN_EPOCHS = 60  # Increased from 15
    PRETRAIN_BATCH_SIZE = 64  # Increased from 16
    PRETRAIN_LR = 1e-3  # Adjusted from 5e-4
    PRETRAIN_PATIENCE = 25
   
    # Training hyperparameters - FINETUNING
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
   
    # Model architecture
    IMG_SIZE = 256
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 512
    MEMORY_BANK_SIZE = 2000
    HARD_NEGATIVE_RATIO = 0.3 # 30% hard negatives
    NUM_PYRAMID_LEVELS = 3
    PYRAMID_SLIDE_STRIDE = 64 # For pyramid sliding
    WAVELET = 'db4'
   
    # Chunk-wise auxiliary
    CHUNK_SIZE = 64
    NUM_CHUNKS_PER_IMAGE = 16
   
    # Bayesian uncertainty
    NUM_MC_SAMPLES = 10
    DROPOUT_RATE = 0.3
   
    # Loss weights
    WEIGHT_DICE = 1.0
    WEIGHT_FOCAL = 1.0
    WEIGHT_SMOOTHNESS = 0.1
    WEIGHT_CONTRASTIVE = 0.5
    WEIGHT_ANATOMICAL_AUX = 0.3
   
    # Contrastive learning (IMPROVED)
    TEMPERATURE = 0.1  # Increased from 0.07 for better stability
   
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
   
    # Checkpointing
    SAVE_EVERY = 5
    PATIENCE = 20

# ============================================================================
# AUGMENTATION FUNCTIONS FOR PRETRAINING
# ============================================================================
class PretrainAugmentations:
    """Advanced augmentations for self-supervised pretraining"""
    
    @staticmethod
    def random_horizontal_flip(img):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        return img
    
    @staticmethod
    def random_rotation(img, max_angle=25):  # Increased from 15 to 25
        if random.random() > 0.5:
            angle = random.uniform(-max_angle, max_angle)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img
    
    @staticmethod
    def random_brightness_contrast(img):
        if random.random() > 0.5:
            # Brightness - wider range
            brightness = random.uniform(0.6, 1.4)  # Changed from (0.8, 1.2)
            img = img * brightness
            img = np.clip(img, 0, 1)
            
            # Contrast - wider range
            contrast = random.uniform(0.6, 1.4)  # Changed from (0.8, 1.2)
            mean = np.mean(img)
            img = (img - mean) * contrast + mean
            img = np.clip(img, 0, 1)
        return img
    
    @staticmethod
    def random_gaussian_noise(img, noise_std=0.04):  # Increased from 0.02
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)  # Random noise level
            noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
            img = img + noise
            img = np.clip(img, 0, 1)
        return img
    
    @staticmethod
    def random_color_shift(img):
        """Simulate different CT windowing levels"""
        if random.random() > 0.5:
            shift = random.uniform(-0.1, 0.1)
            img = img + shift
            img = np.clip(img, 0, 1)
        return img
    
    @staticmethod
    def apply_augmentations(img):
        """Apply all augmentations to create one view"""
        img = img.copy()
        img = PretrainAugmentations.random_horizontal_flip(img)
        img = PretrainAugmentations.random_rotation(img)
        img = PretrainAugmentations.random_brightness_contrast(img)
        img = PretrainAugmentations.random_gaussian_noise(img)
        img = PretrainAugmentations.random_color_shift(img)  # Added new augmentation
        return img
    
    @staticmethod
    def create_two_views(img):
        """Create two different augmented views of the same image"""
        view1 = PretrainAugmentations.apply_augmentations(img)
        view2 = PretrainAugmentations.apply_augmentations(img)
        return view1, view2

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def iou_score(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)
def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()
   
    dice = dice_coefficient(pred_binary, target_binary)
    iou = iou_score(pred_binary, target_binary)
   
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().float()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().float()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().float()
   
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
   
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }
# ============================================================================
# DATASET CLASSES
# ============================================================================
class PretrainDataset(Dataset):
    """Dataset for self-supervised pretraining on real (TB) data with two-view augmentation"""
   
    def __init__(self, tb_samples, fb_samples=None):
        self.tb_samples = tb_samples # Real data
        # Balance by taking min(len(tb), len(fb))
        min_size = min(len(tb_samples), len(fb_samples)) if fb_samples else len(tb_samples)
        self.fb_samples = random.sample(fb_samples, min_size)  # NO REPEAT
       
    def __len__(self):
        return len(self.tb_samples) + len(self.fb_samples)
   
    def __getitem__(self, idx):
        if idx < len(self.tb_samples):
            # Real sample
            data = np.load(self.tb_samples[idx]['path'])
            label = 0 # Real
        else:
            # Fake sample
            fb_idx = idx - len(self.tb_samples)
            data = np.load(self.fb_samples[fb_idx]['path'])
            label = 1 # Fake
       
        # Extract CT channel only for pretraining
        ct_img = data[:, :, 0]
        
        # Create two augmented views for contrastive learning
        view1 = PretrainAugmentations.apply_augmentations(ct_img)
        view2 = PretrainAugmentations.apply_augmentations(ct_img)
        
        # Convert to tensors
        view1_tensor = torch.from_numpy(view1).unsqueeze(0).float()
        view2_tensor = torch.from_numpy(view2).unsqueeze(0).float()
        
        # Stack to 3-channel by repeating (simulating the 3-channel input)
        view1_tensor = view1_tensor.repeat(3, 1, 1)
        view2_tensor = view2_tensor.repeat(3, 1, 1)
        
        return view1_tensor, view2_tensor, label
class CTForensicDataset(Dataset):
    """Dataset for CT forensic analysis with ROI masks"""
   
    def __init__(self, samples, csv_path=None, augment=False):
        self.samples = samples
        self.augment = augment
       
        # Load CSV for coordinate lookup
        self.roi_lookup = {}
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['path'] = df['path'].astype(str)
            df['cur_slice'] = df['cur_slice'].astype(str).apply(
                lambda x: str(int(float(x))) if str(x).replace('.','',1).isdigit() else str(x)
            )
            for _, row in df.iterrows():
                key = f"{row['path']}_{row['cur_slice']}"
                self.roi_lookup[key] = (int(row['x']), int(row['y']))
   
    def __len__(self):
        return len(self.samples)
   
    def create_roi_mask(self, img_shape, x, y, radius=32):
        mask = np.zeros(img_shape, dtype=np.float32)
        if x is None or y is None:
            return mask
       
        h, w = img_shape
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = np.exp(-(dist**2) / (2 * radius**2))
        return mask
   
    def augment_sample(self, img, mask):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
       
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
       
        return img, mask
   
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample['path'])
       
        ct_img = data[:, :, 0]
        roi_channel = data[:, :, 1]
        fft_channel = data[:, :, 2]
       
        patient_id = sample['patient_id']
        slice_name = os.path.splitext(os.path.basename(sample['path']))[0]
        lookup_key = f"{patient_id}_{slice_name}"
       
        x, y = self.roi_lookup.get(lookup_key, (None, None))
        gt_mask = self.create_roi_mask(ct_img.shape, x, y, radius=32)
       
        if self.augment:
            ct_img, gt_mask = self.augment_sample(ct_img, gt_mask)
       
        img_tensor = torch.from_numpy(np.stack([ct_img, roi_channel, fft_channel], axis=0)).float()
        mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).float()
       
        return img_tensor, mask_tensor, sample['patient_id']
def load_and_split_data(config):
    """Load data and create splits"""
    print("Loading CT Removal data...")
   
    removal_path = config.REMOVAL_PATH
    csv_path = os.path.join(removal_path, "data_v2.csv")  # Fixed to v2
   
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    print(f"Using CSV: {csv_path}")
   
    # Debug CSV
    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} rows")
    print("Sample rows:\n", df.head())
   
    all_files = glob(os.path.join(removal_path, "*", "*.npy"))
    print(f"Found {len(all_files)} total removal slices")
   
    patient_files = defaultdict(list)
    for fpath in all_files:
        patient_id = os.path.basename(os.path.dirname(fpath))
        patient_files[patient_id].append({
            'path': fpath,
            'patient_id': patient_id
        })
   
    print(f"Found {len(patient_files)} removal patients")
   
    # Load TB data for pretraining
    tb_path = config.TB_PATH
    tb_files = glob(os.path.join(tb_path, "*", "*.npy"))
    tb_samples = [{'path': f, 'patient_id': os.path.basename(os.path.dirname(f))} for f in tb_files]
    print(f"Found {len(tb_samples)} TB (real) samples for pretraining")
   
    # Patient-level split for removal data
    patient_ids = list(patient_files.keys())
    train_patients, val_patients = train_test_split(
        patient_ids,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED
    )
   
    train_samples = []
    val_samples = []
   
    for pid in train_patients:
        train_samples.extend(patient_files[pid])
    for pid in val_patients:
        val_samples.extend(patient_files[pid])
   
    print(f"Train: {len(train_samples)} slices from {len(train_patients)} patients")
    print(f"Val: {len(val_samples)} slices from {len(val_patients)} patients")
   
    split_info = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'tb_samples': len(tb_samples),
        'random_seed': config.RANDOM_SEED,
        'timestamp': datetime.now().isoformat()
    }
   
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, 'data_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
   
    return train_samples, val_samples, tb_samples, csv_path
# ============================================================================
# PYRAMID SLIDING WINDOW MODULE
# ============================================================================
class PyramidSlidingWindow(nn.Module):
    """Multi-scale analysis with sliding window and cross-scale consistency"""
   
    def __init__(self, num_levels=3, wavelet='db4', stride=64):
        super().__init__()
        self.num_levels = num_levels
        self.wavelet = wavelet
        self.stride = stride
       
        # Per-level feature extractors
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            ) for _ in range(num_levels)
        ])
       
        # Cross-scale fusion
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv2d(num_levels * 64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
   
    def extract_sliding_windows(self, img, window_size=64, stride=64):
        """Extract sliding windows from image"""
        b, c, h, w = img.shape
        windows = []
        positions = []
       
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = img[:, :, y:y+window_size, x:x+window_size]
                windows.append(window)
                positions.append((y, x))
       
        if len(windows) == 0:
            windows = [img]
            positions = [(0, 0)]
           
        return torch.cat(windows, dim=0), positions
   
    def aggregate_windows(self, window_feats, positions, orig_shape, window_size=64):
        """Aggregate window features back to full image with mean pooling"""
        b, c, h, w = orig_shape
        agg_map = torch.zeros((b, window_feats.shape[1], h, w), device=window_feats.device)
        count_map = torch.zeros_like(agg_map)
       
        feat_idx = 0
        for y, x in positions:
            agg_map[:, :, y:y+window_size, x:x+window_size] += window_feats[feat_idx:feat_idx+1]
            count_map[:, :, y:y+window_size, x:x+window_size] += 1
            feat_idx += 1
       
        agg_map = agg_map / (count_map + 1e-6)
        return agg_map
   
    def wavelet_pyramid(self, x):
        """Create wavelet pyramid"""
        batch_size = x.shape[0]
        device = x.device
       
        pyramid_levels = []
       
        for b in range(batch_size):
            img_np = x[b, 0].cpu().numpy()
           
            level_features = []
            current = img_np
           
            for level in range(self.num_levels):
                coeffs = pywt.dwt2(current, self.wavelet)
                cA, (cH, cV, cD) = coeffs
               
                hf_energy = np.sqrt(cH**2 + cV**2 + cD**2)
                hf_resized = cv2.resize(hf_energy, (256, 256), interpolation=cv2.INTER_LINEAR)
                level_features.append(hf_resized)
               
                current = cA
           
            pyramid_levels.append(np.stack(level_features, axis=0))
       
        pyramid_tensor = torch.from_numpy(np.stack(pyramid_levels, axis=0)).float().to(device)
        return pyramid_tensor
   
    def forward(self, x):
        """
        x: (batch, 3, H, W)
        Returns: Cross-scale consistency map
        """
        # Get wavelet pyramid
        pyramid = self.wavelet_pyramid(x) # (batch, num_levels, H, W)
       
        # Process each level with sliding windows
        level_features = []
        orig_shape = pyramid.shape
        for i in range(self.num_levels):
            level = pyramid[:, i:i+1, :, :]
            windows, positions = self.extract_sliding_windows(level, 64, self.stride)
            window_feats = self.level_encoders[i](windows)
            agg_feats = self.aggregate_windows(window_feats, positions, orig_shape, 64)
            level_features.append(agg_feats)
       
        # Concatenate and fuse
        combined = torch.cat(level_features, dim=1)
        output = self.cross_scale_fusion(combined)
       
        return output
# ============================================================================
# COARSE-FINE STRUCTURE MODULE
# ============================================================================
class CoarseFineStructure(nn.Module):
    """Learn coarse-fine structural relationships with phase fusion"""
   
    def __init__(self):
        super().__init__()
       
        # Coarse path (downsampled)
        self.coarse_path = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
       
        # Fine path (full resolution)
        self.fine_path = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
       
        # Phase analyzer
        self.phase_analyzer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
       
        # Fusion with phase
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64 + 32, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
   
    def get_phase_info(self, x):
        """Extract phase information from FFT"""
        img = x[:, 0:1, :, :]
        fft = torch.fft.fft2(img)
        fft_shift = torch.fft.fftshift(fft)
        phase = torch.angle(fft_shift)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        return phase_norm
   
    def forward(self, x):
        """
        x: (batch, 3, H, W)
        """
        # Coarse features
        coarse = self.coarse_path(x)
        coarse_up = F.interpolate(coarse, size=(256, 256), mode='bilinear', align_corners=True)
       
        # Fine features
        fine = self.fine_path(x)
       
        # Phase features
        phase = self.get_phase_info(x)
        phase_features = self.phase_analyzer(phase)
       
        # Fuse all
        combined = torch.cat([coarse_up, fine, phase_features], dim=1)
        output = self.fusion(combined)
       
        return output
# ============================================================================
# HARD NEGATIVE MEMORY BANK
# ============================================================================
class HardNegativeMemoryBank(nn.Module):
    """Memory bank with hard negative mining"""
   
    def __init__(self, feature_dim=512, bank_size=2000, hard_ratio=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.hard_ratio = hard_ratio
        self.hard_size = int(bank_size * hard_ratio)
       
        # Memory banks
        self.register_buffer('memory_real', torch.randn(bank_size - self.hard_size, feature_dim))
        self.register_buffer('memory_hard_neg', torch.randn(self.hard_size, feature_dim))
        self.register_buffer('hard_neg_scores', torch.zeros(self.hard_size))
        self.register_buffer('ptr_real', torch.zeros(1, dtype=torch.long))
        self.register_buffer('ptr_hard', torch.zeros(1, dtype=torch.long))
       
        # Feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
       
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
   
    @torch.no_grad()
    def update_memory(self, features, is_hard_negative=False, difficulty_scores=None):
        """Update memory bank with optional hard negative mining"""
        batch_size = features.shape[0]
       
        if is_hard_negative and difficulty_scores is not None:
            # Update hard negatives
            for i in range(batch_size):
                if difficulty_scores[i] > self.hard_neg_scores.min():
                    min_idx = self.hard_neg_scores.argmin()
                    self.memory_hard_neg[min_idx] = features[i]
                    self.hard_neg_scores[min_idx] = difficulty_scores[i]
        else:
            # Update regular memory
            ptr = int(self.ptr_real)
            if ptr + batch_size <= self.memory_real.shape[0]:
                self.memory_real[ptr:ptr + batch_size] = features
                ptr += batch_size
            else:
                remaining = self.memory_real.shape[0] - ptr
                self.memory_real[ptr:] = features[:remaining]
                self.memory_real[:batch_size - remaining] = features[remaining:]
                ptr = batch_size - remaining
            self.ptr_real[0] = ptr % self.memory_real.shape[0]
   
    def forward(self, x, return_features=False):
        """
        x: (batch, 3, H, W)
        """
        # Extract features
        features = self.encoder(x)
        features = features.squeeze(-1).squeeze(-1)
        features_norm = F.normalize(features, dim=1)
       
        # Combine memory banks
        memory_combined = torch.cat([self.memory_real, self.memory_hard_neg], dim=0)
        memory_norm = F.normalize(memory_combined, dim=1)
       
        # Compute similarity
        similarity = torch.mm(features_norm, memory_norm.t())
        max_sim, _ = similarity.max(dim=1, keepdim=True)
       
        # Anomaly score
        anomaly_score = 1 - max_sim
        anomaly_map = anomaly_score.view(-1, 1, 1, 1).expand(-1, 1, 256, 256)
       
        if return_features:
            return anomaly_map, features_norm
        return anomaly_map
# ============================================================================
# CHUNK-WISE ANATOMICAL AUXILIARY
# ============================================================================
class ChunkWiseAnatomicalAuxiliary(nn.Module):
    """Chunk-wise auxiliary task for anatomical consistency"""
   
    def __init__(self, chunk_size=64, num_chunks=16, hidden_dim=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
       
        # Chunk encoder
        self.chunk_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU()
        )
       
        # Auxiliary classifier (real vs fake chunk)
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # Binary: real/fake
        )
       
        # Global context aggregator
        self.global_aggregator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 1, 1)
        )
   
    def extract_random_chunks(self, x, num_chunks):
        """Extract random chunks from image"""
        b, c, h, w = x.shape
        chunks = []
        positions = []
    
        for _ in range(num_chunks * b):  # b images × num_chunks
            img_idx = _ // num_chunks
            y = random.randint(0, h - self.chunk_size)
            x_pos = random.randint(0, w - self.chunk_size)
            chunk = x[img_idx:img_idx+1, :, y:y+self.chunk_size, x_pos:x_pos+self.chunk_size]
            chunks.append(chunk)
            positions.append((y, x_pos))
    
        if len(chunks) == 0:
            chunks = [x[:, :, :self.chunk_size, :self.chunk_size]]
            positions = [(0, 0)]
    
        return torch.cat(chunks, dim=0), positions
       
    def forward(self, x, masks=None, compute_aux_loss=False):
        global_map = self.global_aggregator(x)
        global_map = F.interpolate(global_map, size=(256, 256), mode='bilinear', align_corners=True)
       
        aux_loss = None
        chunks = None
        positions = None
        chunk_features = None
    
        if compute_aux_loss and self.training and masks is not None:
            # Extract chunks from all images in batch
            chunks, positions = self.extract_random_chunks(x, self.num_chunks)
            chunk_features = self.chunk_encoder(chunks)
           
            # === Compute labels using masks ===
            chunk_labels = []
            batch_size = masks.shape[0]
            total_chunks = batch_size * self.num_chunks
    
            for i in range(len(positions)):
                batch_idx = i // self.num_chunks
                if batch_idx >= batch_size:
                    continue
                y, x_pos = positions[i]
                mask_patch = masks[batch_idx:batch_idx+1, :, y:y+self.chunk_size, x_pos:x_pos+self.chunk_size]
                overlap = (mask_patch > 0.5).float().mean().item()
                label = 1 if overlap > 0.5 else 0
                chunk_labels.append(label)
           
            if len(chunk_labels) == chunk_features.shape[0]:
                chunk_labels = torch.tensor(chunk_labels, device=x.device, dtype=torch.long)
                aux_preds = self.aux_classifier(chunk_features)
                aux_loss = F.cross_entropy(aux_preds, chunk_labels)
    
        return global_map, aux_loss, chunks, positions, chunk_features
# ============================================================================
# MAIN FORENSIC NETWORK WITH SELF-SUPERVISED PRETRAINING
# ============================================================================
class MultiHeadForensicNetwork(nn.Module):
    """Enhanced forensic network with all 5 advanced features"""
   
    def __init__(self, config):
        super().__init__()
        self.config = config
       
        # Feature 1: Self-supervised encoder (will be pretrained)
        self.self_supervised_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )
       
        # Projection head for contrastive pretraining
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),  # Added extra layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)   # Final embedding
        )
       
        # Feature 2: Pyramid sliding
        self.pyramid_head = PyramidSlidingWindow(
            num_levels=config.NUM_PYRAMID_LEVELS,
            wavelet=config.WAVELET,
            stride=config.PYRAMID_SLIDE_STRIDE
        )
       
        # Feature 3: Coarse-fine structure
        self.coarse_fine_head = CoarseFineStructure()
       
        # Feature 4: Hard negative memory bank
        self.memory_head = HardNegativeMemoryBank(
            feature_dim=config.EMBEDDING_DIM,
            bank_size=config.MEMORY_BANK_SIZE,
            hard_ratio=config.HARD_NEGATIVE_RATIO
        )
       
        # Feature 5: Chunk-wise anatomical auxiliary
        self.anatomy_head = ChunkWiseAnatomicalAuxiliary(
            chunk_size=config.CHUNK_SIZE,
            num_chunks=config.NUM_CHUNKS_PER_IMAGE
        )
       
        # Fusion network (5 heads)
        self.fusion = nn.Sequential(
            nn.Conv2d(5, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(config.DROPOUT_RATE),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(config.DROPOUT_RATE),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
       
        # Decoder from self-supervised features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )
   
    def forward_encoder(self, x):
        """Forward through self-supervised encoder"""
        features = self.self_supervised_encoder(x)
        embeddings = self.projection_head(features)
        return embeddings, features
   
    def forward(self, x, masks=None, compute_aux_loss=False):
        _, ss_features = self.forward_encoder(x)
        ss_map = self.decoder(ss_features)
        ss_map = F.interpolate(ss_map, size=(256, 256), mode='bilinear', align_corners=True)
    
        pyramid_out = self.pyramid_head(x)
        coarse_fine_out = self.coarse_fine_head(x)
        memory_out, _ = self.memory_head(x, return_features=True)
        anatomy_out, aux_loss, chunks, positions, chunk_features = self.anatomy_head(
            x, masks=masks, compute_aux_loss=compute_aux_loss
        )
    
        all_heads = torch.cat([ss_map, pyramid_out, coarse_fine_out, memory_out, anatomy_out], dim=1)
        output = self.fusion(all_heads)
    
        return output, aux_loss, chunks, positions, chunk_features
       
        def predict_with_uncertainty(self, x, num_samples=10):
            """Monte Carlo Dropout for uncertainty"""
            self.train()
           
            predictions = []
            for _ in range(num_samples):
                with torch.no_grad():
                    pred, _, _, _, _ = self.forward(x)
                    predictions.append(pred)
           
            predictions = torch.stack(predictions, dim=0)
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)
           
            self.eval()
            return mean_pred, uncertainty
# ============================================================================
# IMPROVED CONTRASTIVE LOSS FOR PRETRAINING
# ============================================================================
# ============================================================================
# IMPROVED CONTRASTIVE LOSS FOR PRETRAINING
# ============================================================================
class ImprovedContrastiveLoss(nn.Module):
    """Improved SimCLR-style contrastive loss with two-view learning"""
   
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
   
    def forward(self, embeddings1, embeddings2, labels):
        """
        embeddings1: (batch, embedding_dim) - first augmented view
        embeddings2: (batch, embedding_dim) - second augmented view  
        labels: (batch,) - 0 for real, 1 for fake
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute similarity between two views (diagonal should be high)
        sim_matrix = torch.mm(embeddings1, embeddings2.t()) / self.temperature
        
        # Positive pairs are on the diagonal (same image, different augmentations)
        positive_sim = torch.diag(sim_matrix)
        
        # Compute loss using cross entropy style
        labels = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
# ============================================================================
# WARMUP SCHEDULER
# ============================================================================
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create scheduler with warmup followed by cosine decay"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# ============================================================================
# COMBINED LOSS
# ============================================================================
class CombinedLoss(nn.Module):
    """Combined loss for forensic localization"""
   
    def __init__(self, config):
        super().__init__()
        self.config = config
   
    def dice_loss(self, logits, target):
        pred = torch.sigmoid(logits)
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
   
    def focal_loss(self, logits, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
   
    def smoothness_loss(self, logits):
        pred = torch.sigmoid(logits)
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        return dx.mean() + dy.mean()
   
    def forward(self, logits, target, aux_loss=None):
        dice = self.dice_loss(logits, target)
        focal = self.focal_loss(logits, target)
        smooth = self.smoothness_loss(logits)
       
        total_loss = (
            self.config.WEIGHT_DICE * dice +
            self.config.WEIGHT_FOCAL * focal +
            self.config.WEIGHT_SMOOTHNESS * smooth
        )
       
        if aux_loss is not None:
            total_loss += self.config.WEIGHT_ANATOMICAL_AUX * aux_loss
       
        return total_loss, {
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'smooth_loss': smooth.item(),
            'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
            'total_loss': total_loss.item()
        }
# ============================================================================
# IMPROVED PRETRAINING FUNCTIONS
# ============================================================================
def pretrain_self_supervised(model, tb_samples, fb_samples, config):
    """Improved self-supervised pretraining with two-view contrastive learning"""
    print("\n" + "="*80)
    print("PHASE 1: Improved Self-Supervised Pretraining")
    print("="*80)
    print("Features: Two-view augmentation + SimCLR loss + Cosine annealing")
   
    # Create pretraining dataset
    num_fake = min(len(tb_samples), len(fb_samples))
    fb_balanced = random.sample(fb_samples, num_fake)
    pretrain_dataset = PretrainDataset(tb_samples[:num_fake], fb_balanced)
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
   
    # Optimizer for pretraining - only train encoder and projection head
    pretrain_params = list(model.self_supervised_encoder.parameters()) + list(model.projection_head.parameters())
    optimizer = optim.AdamW(pretrain_params, lr=config.PRETRAIN_LR)
    
    # Cosine annealing scheduler
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=config.PRETRAIN_EPOCHS)
    
    contrastive_criterion = ImprovedContrastiveLoss(temperature=config.TEMPERATURE)
    scaler = torch.amp.GradScaler('cuda')
   
    # Pretraining history with type
    pretrain_history = []
   
    # Resume logic
    start_epoch = 0
    best_pretrain_loss = float('inf')
    patience_counter = 0
    pretrain_checkpoint_path = os.path.join(config.OUTPUT_DIR, 'pretrain_checkpoint.pth')
    if os.path.exists(pretrain_checkpoint_path):
        print(f"Loading pretrain checkpoint from {pretrain_checkpoint_path}")
        checkpoint = torch.load(pretrain_checkpoint_path,weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_pretrain_loss = checkpoint['best_loss']
        pretrain_history = checkpoint['history']
        print(f"Resuming pretraining from epoch {start_epoch}, best loss: {best_pretrain_loss:.4f}")
   
    model.train()
   
    for epoch in range(start_epoch, config.PRETRAIN_EPOCHS):
        total_loss = 0
        pbar = tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}/{config.PRETRAIN_EPOCHS}")
       
        for view1, view2, labels in pbar:
            view1 = view1.to(config.DEVICE)
            view2 = view2.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
           
            optimizer.zero_grad()
           
            with torch.amp.autocast('cuda'):
                # Get embeddings for both views
                embeddings1, _ = model.forward_encoder(view1)
                embeddings2, _ = model.forward_encoder(view2)
                
                # Compute improved contrastive loss
                loss = contrastive_criterion(embeddings1, embeddings2, labels)
           
            scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pretrain_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
           
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
       
        # Update learning rate
        scheduler.step()
        
        avg_loss = total_loss / len(pretrain_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Pretrain Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
       
        # Store history with type
        pretrain_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': current_lr,
            'type': 'pretraining'
        })
       
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_pretrain_loss,
            'history': pretrain_history
        }
        torch.save(checkpoint, pretrain_checkpoint_path)
       
        # Save best
        if avg_loss < best_pretrain_loss:
            best_pretrain_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'pretrain_best_model.pth'))
            print(f" *** New best pretrain model! Loss: {best_pretrain_loss:.4f} ***")
        else:
            patience_counter += 1
       
        if patience_counter >= config.PRETRAIN_PATIENCE:
            print(f"\nPretrain early stopping at epoch {epoch+1}")
            break
   
    # Save history
    history_path = os.path.join(config.OUTPUT_DIR, 'pretrain_history.json')
    with open(history_path, 'w') as f:
        json.dump(pretrain_history, f, indent=2)
    print(f"Pretrain history saved to: {history_path}")
   
    # Plot (simple loss plot)
    plt.figure()
    epochs = [h['epoch'] for h in pretrain_history]
    losses = [h['loss'] for h in pretrain_history]
    plt.plot(epochs, losses, 'b-')
    plt.title('Improved Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'pretrain_history.png'))
    plt.close()
   
    print("\nImproved pretraining complete! Encoder learned robust real vs fake representations.")
   
    # Save final pretrained weights
    pretrain_path = os.path.join(config.OUTPUT_DIR, 'pretrained_encoder.pth')
    torch.save(model.state_dict(), pretrain_path)
    print(f"Pretrained model saved to: {pretrain_path}")
# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0
    all_metrics = defaultdict(list)

    pbar = tqdm(dataloader, desc="Training")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs, aux_loss, _, _, _ = model(images, masks=masks, compute_aux_loss=True)
            loss, loss_dict = criterion(outputs, masks, aux_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Hard negative update
        with torch.no_grad():
            _, features_norm = model.memory_head(images, return_features=True)
            memory = torch.cat([model.memory_head.memory_real, model.memory_head.memory_hard_neg], dim=0)
            memory_norm = F.normalize(memory, dim=1)
            sim = torch.mm(features_norm, memory_norm.t())
            difficulty = 1 - sim.max(dim=1)[0]
            model.memory_head.update_memory(features_norm, is_hard_negative=True, difficulty_scores=difficulty)

        metrics = calculate_metrics(outputs.detach().cpu(), masks.cpu())
        total_loss += loss.item()
        for k, v in metrics.items():
            all_metrics[k].append(v)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
            'aux': f"{aux_loss.item() if aux_loss is not None else 0:.3f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics

def validate_epoch(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_metrics = defaultdict(list)
   
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
           
            outputs, _, _, _, _ = model(images, compute_aux_loss=False)
            loss, _ = criterion(outputs, masks)
           
            metrics = calculate_metrics(outputs.cpu(), masks.cpu())
           
            total_loss += loss.item()
            for k, v in metrics.items():
                all_metrics[k].append(v)
           
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
   
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
   
    return avg_loss, avg_metrics
# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_predictions(model, dataset, device, config, num_samples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
   
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
   
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, patient_id = dataset[idx]
           
            img_batch = img.unsqueeze(0).to(device)
            pred_raw, _, _, _, _ = model(img_batch)
            pred = torch.sigmoid(pred_raw).squeeze(0).cpu().numpy()[0]
           
            _, uncertainty = model.predict_with_uncertainty(img_batch, num_samples=5)
            uncertainty = uncertainty.squeeze(0).cpu().numpy()[0]
           
            ct_img = img[0].numpy()
            gt_mask = mask[0].numpy()
           
            pred_binary = (pred > 0.5).astype(np.uint8)
            if pred_binary.sum() > 0:
                y_coords, x_coords = np.where(pred_binary > 0)
                center_y, center_x = int(y_coords.mean()), int(x_coords.mean())
            else:
                center_y, center_x = np.unravel_index(pred.argmax(), pred.shape)
           
            axes[i, 0].imshow(ct_img, cmap='gray')
            axes[i, 0].set_title('CT Image')
            axes[i, 0].axis('off')
           
            axes[i, 1].imshow(ct_img, cmap='gray')
            axes[i, 1].imshow(gt_mask, alpha=0.5, cmap='Reds')
            axes[i, 1].set_title('GT Mask Overlay')
            axes[i, 1].axis('off')
           
            axes[i, 2].imshow(ct_img, cmap='gray')
            axes[i, 2].imshow(pred, alpha=0.6, cmap='hot')
            circle = plt.Circle((center_x, center_y), 32, color='cyan', fill=False, linewidth=2)
            axes[i, 2].add_patch(circle)
            axes[i, 2].plot(center_x, center_y, 'c*', markersize=15)
            axes[i, 2].set_title(f'Pred Heatmap ({center_x}, {center_y})')
            axes[i, 2].axis('off')
           
            axes[i, 3].imshow(uncertainty, cmap='viridis')
            axes[i, 3].set_title('Uncertainty')
            axes[i, 3].axis('off')
   
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions saved to: {save_path}")
def plot_training_history(history, config, is_pretrain=False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = [h['epoch'] for h in history]
   
    if is_pretrain:
        losses = [h['loss'] for h in history]
        axes[0, 0].plot(epochs, losses, 'b-')
        axes[0, 0].set_title('Pretraining Loss')
        axes[0, 0].grid(True)
        save_path = os.path.join(config.OUTPUT_DIR, 'pretrain_history.png')
    else:
        train_loss = [h['train_loss'] for h in history]
        val_loss = [h['val_loss'] for h in history]
        train_dice = [h['train_dice'] for h in history]
        val_dice = [h['val_dice'] for h in history]
        # ... (add others similarly)
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train')
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        # Plot dice, iou, etc. as before
        save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"History saved to: {save_path}")
# ============================================================================
# MAIN
# ============================================================================
def main():
    config = Config()
    set_seed(config.RANDOM_SEED)
   
    print("="*80)
    print("Advanced CT Forensic Localization")
    print("="*80)
    print(f"Device: {config.DEVICE}")
   
    # Load data
    train_samples, val_samples, tb_samples, csv_path = load_and_split_data(config)
   
    # Initialize model
    print("\nInitializing model...")
    model = MultiHeadForensicNetwork(config).to(config.DEVICE)
   
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
   
    # PHASE 1: Self-supervised pretraining
    # === BALANCE REAL AND FAKE FOR PRETRAINING ===
    num_per_class = min(len(tb_samples), len(train_samples))
    tb_balanced = random.sample(tb_samples, num_per_class)
    fb_balanced = random.sample(train_samples, num_per_class)
    
    print(f"Pretraining on {num_per_class} real + {num_per_class} fake samples (balanced, no duplication)")
    
    pretrain_self_supervised(model, tb_balanced, fb_balanced, config)
   
    # PHASE 2: Supervised fine-tuning
    print("\n" + "="*80)
    print("PHASE 2: Supervised Fine-Tuning")
    print("="*80)
   
    train_dataset = CTForensicDataset(train_samples, csv_path, augment=True)
    val_dataset = CTForensicDataset(val_samples, csv_path, augment=False)
   
    # Sanity check masks
    print("\nSanity check: First 5 train mask sums")
    for i in range(5):
        _, mask, _ = train_dataset[i]
        print(f"Mask {i} sum: {mask.sum().item():.2f}")
   
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
   
    criterion = CombinedLoss(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=10, verbose=True)
    scaler = torch.amp.GradScaler('cuda')
   
    history = []
   
    start_epoch = 0
    best_val_dice = 0
    patience_counter = 0
   
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path,weights_only = False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint['best_val_dice']
        history = checkpoint['history']
        print(f"Resuming from epoch {start_epoch}, best dice: {best_val_dice:.4f}")
   
    print("\nStarting fine-tuning...")
   
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 80)
       
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE, config
        )
       
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, config.DEVICE, config
        )
       
        scheduler.step(val_metrics['dice'])
        current_lr = optimizer.param_groups[0]['lr']
       
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'lr': current_lr,
            'type': 'training'
        }
        history.append(history_entry)
       
        print(f"\nEpoch {epoch+1} Summary:")
        print(f" Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}")
        print(f" Val - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}")
        print(f" LR: {current_lr:.2e}")
       
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
       
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f" *** New best model! Dice: {best_val_dice:.4f} ***")
        else:
            patience_counter += 1
       
        if (epoch + 1) % config.SAVE_EVERY == 0:
            periodic_path = os.path.join(config.OUTPUT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), periodic_path)
       
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
   
    history_path = os.path.join(config.OUTPUT_DIR, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to: {history_path}")
   
    plot_training_history(history, config, is_pretrain=False)
   
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
   
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path,weights_only = False))
   
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.DEVICE, config)
   
    print(f"\nFinal Validation Metrics:")
    print(f" Loss: {val_loss:.4f}")
    print(f" Dice: {val_metrics['dice']:.4f}")
    print(f" IoU: {val_metrics['iou']:.4f}")
    print(f" Precision: {val_metrics['precision']:.4f}")
    print(f" Recall: {val_metrics['recall']:.4f}")
   
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_dataset, config.DEVICE, config, num_samples=5)
   
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best model: {best_model_path}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
if __name__ == "__main__":
    main()