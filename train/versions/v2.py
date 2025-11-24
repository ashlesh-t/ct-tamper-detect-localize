"""
Robust Baseline Model for CT Tampering Localization - V5 (Dual-Curriculum)

Key Improvements:
- V5: Implements a Dual-Curriculum Learning strategy.
- 1. Channel Dropout Schedule (Epoch 0-20): Gradually increases
     the hint dropout rate from 0% to 75%.
- 2. Hint Noise Schedule (Epoch 25-35): After the model is stable,
     gradually increases the hint noise from 0.15 to 0.30.
- This forces the model to first learn WITH the hint, then learn
  WITHOUT it, and finally to learn to DISTRUST a noisy hint.
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
# Use torch.amp imports for autocast/scaler
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths
    REMOVAL_PATH = "/kaggle/input/ct-removal-processed/2"
    CSV_PATH = "/kaggle/input/ct-removal-processed/2/data_v2.csv"
    OUTPUT_DIR = "/kaggle/working/baseline_robust_v5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Shape configuration (matching visualization code)
    ORIGINAL_SHAPE = (512, 512)  # CSV coordinates are in this space
    TARGET_SHAPE = (256, 256)    # Actual .npy file shape
    
    # Data
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    MASK_RADIUS = 32  # For both input heatmap and target mask
    MIN_MASK_SUM = 100  # More strict filtering
    
    # --- Robust Training ---
    # 1. Channel Dropout Schedule (Epochs 0-20)
    CHANNEL_DROPOUT_RATE_START = 0.0  # Start with 100% hints
    CHANNEL_DROPOUT_RATE_END = 0.75   # Gradually increase to 75%
    CHANNEL_DROPOUT_SCHEDULE_EPOCHS = 20 # Linearly increase over 20 epochs
    
    # 2. Hint Augmentation
    HINT_FADE_MIN = 0.4
    HINT_FADE_MAX = 0.9
    
    # 3. Hint Noise Schedule (Epochs 25-35)
    HINT_NOISE_STD_START = 0.15
    HINT_NOISE_STD_END = 0.30   # Make it much noisier later
    HINT_NOISE_SCHEDULE_START_EPOCH = 25
    HINT_NOISE_SCHEDULE_DURATION_EPOCHS = 10
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 60
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    
    # Model
    IMG_SIZE = 256
    IN_CHANNELS = 2  # CT + Heatmap (ALWAYS 2 channels now)
    BASE_CHANNELS = 64
    
    # Loss weights
    WEIGHT_DICE = 1.0
    WEIGHT_FOCAL = 1.0
    WEIGHT_MSE = 0.0  # *** DISABLE MSE *** - It confuses the model
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    
    # Early stopping
    PATIENCE = 15
    SAVE_EVERY = 5
    MIN_EPOCHS = 10  # Don't early stop before this

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

def create_gaussian_heatmap(shape, x, y, radius=32):
    """
    Create Gaussian heatmap - SAME as visualization code
    """
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return heatmap
    
    x, y = int(x), int(y)
    
    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return heatmap
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    heatmap = np.exp(-(dist**2) / (2 * radius**2)).astype(np.float32) 
    
    return heatmap

def build_coordinate_lookup(csv_path, config):
    """Build and scale coordinate lookup - SAME as visualization code"""
    print("\n--- Building and Scaling Coordinate Lookup Table ---")
    
    df = pd.read_csv(csv_path)
    lookup = {}
    
    # Calculate scaling factors
    x_scale = config.TARGET_SHAPE[1] / config.ORIGINAL_SHAPE[1]
    y_scale = config.TARGET_SHAPE[0] / config.ORIGINAL_SHAPE[0]
    print(f"✓ Scaling coordinates from {config.ORIGINAL_SHAPE} to {config.TARGET_SHAPE}")
    print(f"✓ X-scale: {x_scale:.4f}, Y-scale: {y_scale:.4f}")
    
    # Normalize columns
    df['path'] = df['path'].astype(str)
    df['cur_slice'] = df['cur_slice'].astype(str).apply(
        lambda x: str(int(float(x))) if str(x).replace('.','',1).isdigit() else str(x)
    )
    
    for _, row in df.iterrows():
        key = f"{row['path']}_{row['cur_slice']}"
        
        # Scale coordinates to match target image size
        scaled_x = int(row['x'] * x_scale)
        scaled_y = int(row['y'] * y_scale)
        
        lookup[key] = (scaled_x, scaled_y)
    
    print(f"✓ Built lookup table with {len(lookup)} entries")
    return lookup

def dice_coefficient(pred, target, smooth=1e-6):
    """Dice coefficient for evaluation"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    """IoU score for evaluation"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_localization_error(pred, target, threshold=0.5):
    """Calculate center-of-mass distance"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    if pred_binary.sum() > 0 and target_binary.sum() > 0:
        # Get coordinates
        pred_coords = (pred_binary > 0).nonzero(as_tuple=False).float()
        target_coords = (target_binary > 0).nonzero(as_tuple=False).float()
        
        pred_center = pred_coords.mean(dim=0)
        target_center = target_coords.mean(dim=0)
        
        return torch.norm(pred_center - target_center).item()
    else:
        return 999.0

def calculate_metrics(pred_logits, target, threshold=0.5):
    """Calculate comprehensive metrics"""
    pred = torch.sigmoid(pred_logits)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    dice = dice_coefficient(pred_binary, target_binary)
    iou = iou_score(pred_binary, target_binary)
    
    tp = ((pred_binary == 1) & (target_binary == 1)).sum().float()
    fp = ((pred_binary == 1) & (target_binary == 0)).sum().float()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum().float()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    loc_error = calculate_localization_error(pred.squeeze(), target.squeeze(), threshold)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'loc_error': loc_error
    }

# ============================================================================
# DATASET
# ============================================================================
class CTTamperingDataset(Dataset):
    """
    Dataset for CT tampering localization.
    ALWAYS outputs a 2-channel tensor:
    - Train: [CT, Noisy_Hint] or [CT, Zeros]
    - Val:   [CT, Zeros]
    """
    
    def __init__(self, samples, lookup, config, augment=False):
        self.samples = samples
        self.lookup = lookup
        self.config = config
        self.augment = augment
        
        # --- Add scheduler variables ---
        self.current_dropout_rate = config.CHANNEL_DROPOUT_RATE_START
        self.current_hint_noise_std = config.HINT_NOISE_STD_START
        
        # Pre-filter valid samples
        self.valid_samples = self._filter_valid_samples()
        print(f"  Valid samples: {len(self.valid_samples)}/{len(samples)}")
    
    def _filter_valid_samples(self):
        """Filter samples with valid coordinates and masks"""
        valid = []
        
        for sample in tqdm(self.samples, desc="  Filtering samples"):
            patient_id = sample['patient_id']
            slice_name = os.path.splitext(os.path.basename(sample['path']))[0]
            key = f"{patient_id}_{slice_name}"
            
            coords = self.lookup.get(key, (None, None))
            if coords[0] is not None and coords[1] is not None:
                # Test if mask is valid
                test_mask = create_gaussian_heatmap(
                    self.config.TARGET_SHAPE, 
                    coords[0], 
                    coords[1], 
                    self.config.MASK_RADIUS
                )
                
                if test_mask.sum() > self.config.MIN_MASK_SUM:
                    valid.append(sample)
        
        return valid
    
    def __len__(self):
        return len(self.valid_samples)
    
    def _augment(self, ct_img, heatmap, target_mask):
        """
        Apply augmentation.
        CRITICAL: Geometric transforms (flip, rotate) are applied to all.
        Noise/fade transforms are applied ONLY to the ct_img and heatmap (hint).
        The target_mask remains clean.
        """
        
        # --- 1. Geometric Augmentations (Applied to all) ---
        # Horizontal flip
        if random.random() > 0.5:
            ct_img = np.fliplr(ct_img).copy()
            heatmap = np.fliplr(heatmap).copy()
            target_mask = np.fliplr(target_mask).copy()
        
        # Vertical flip
        if random.random() > 0.5:
            ct_img = np.flipud(ct_img).copy()
            heatmap = np.flipud(heatmap).copy()
            target_mask = np.flipud(target_mask).copy()
        
        # Small rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = ct_img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            
            ct_img = cv2.warpAffine(ct_img, M, (w, h), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT)
            heatmap = cv2.warpAffine(heatmap, M, (w, h), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REFLECT)
            target_mask = cv2.warpAffine(target_mask, M, (w, h), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_REFLECT)
        
        # --- 2. CT-Only Augmentations ---
        # Brightness adjustment (only on CT)
        if random.random() > 0.5:
            brightness = random.uniform(0.85, 1.15)
            ct_img = np.clip(ct_img * brightness, 0, 1)
        
        # Gaussian noise (only on CT)
        if random.random() > 0.3:
            noise = np.random.normal(0, 0.02, ct_img.shape).astype(np.float32)
            ct_img = np.clip(ct_img + noise, 0, 1)
            
        # --- 3. HINT-Only Augmentations (The important part!) ---
        # This breaks the "copy" behavior
        
        # Fade the hint
        if random.random() > 0.5:
            fade = random.uniform(self.config.HINT_FADE_MIN, self.config.HINT_FADE_MAX)
            heatmap = heatmap * fade
            
        # Add noise to the hint (using the scheduled noise level)
        if random.random() > 0.5:
            # *** USE SCHEDULED NOISE ***
            noise = np.random.normal(0, self.current_hint_noise_std, heatmap.shape).astype(np.float32)
            heatmap = np.clip(heatmap + noise, 0, 1)
        
        return ct_img, heatmap, target_mask
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load CT data
        data = np.load(sample['path'])
        ct_img = data[:, :, 0].astype(np.float32)
        
        # Get scaled coordinates
        patient_id = sample['patient_id']
        slice_name = os.path.splitext(os.path.basename(sample['path']))[0]
        key = f"{patient_id}_{slice_name}"
        x, y = self.lookup.get(key, (None, None))
        
        # --- Generate separate Hint and Target ---
        # 1. This is the "perfect" ground truth mask.
        target_mask = create_gaussian_heatmap(
            ct_img.shape, x, y, self.config.MASK_RADIUS
        )
        
        # 2. This is the "imperfect" input hint.
        input_heatmap = create_gaussian_heatmap(
            ct_img.shape, x, y, self.config.MASK_RADIUS
        )
        
        # Augmentation
        if self.augment:
            ct_img, input_heatmap, target_mask = self._augment(
                ct_img, input_heatmap, target_mask
            )
        
        # === START OF CRITICAL UPDATE V2 ===
        if self.augment:
            # Training Mode: [CT, Noisy_Hint] or [CT, Zeros]
            
            # *** Channel Dropout via ZEROED HINT ***
            # *** Use SCHEDULED dropout rate ***
            if random.random() < self.current_dropout_rate:
                # Drop the hint by replacing it with zeros
                input_heatmap = np.zeros_like(ct_img, dtype=np.float32)
            
            # Stack [CT, (Noisy_Heatmap OR Zero_Heatmap)]
            img = np.stack([ct_img, input_heatmap], axis=0).astype(np.float32)
                
        else:
            # Validation Mode: [CT, Zeros]
            # We simulate the 1-channel input by providing a zeroed-out
            # hint channel. This forces the model to use its CT-only
            # learned pathway, and keeps tensor shapes consistent.
            zero_heatmap = np.zeros_like(ct_img, dtype=np.float32)
            img = np.stack([ct_img, zero_heatmap], axis=0).astype(np.float32)
        # === END OF CRITICAL UPDATE V2 ===
        
        # To tensors
        img_tensor = torch.from_numpy(img)
        target_tensor = torch.from_numpy(target_mask).unsqueeze(0)
        
        return img_tensor, target_tensor, patient_id

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class DoubleConv(nn.Module):
    """Double convolution block with batch norm and dropout"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ImprovedUNet(nn.Module):
    """
    Improved U-Net with:
    - Attention gates
    - Residual connections
    - Deep supervision
    - *** UNIFIED 2-CHANNEL INPUT ***
    """
    
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        
        # Encoder
        # This is the single input block. It ALWAYS takes 2 channels.
        # (CT + Hint) or (CT + Zeros)
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.1)
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, dropout=0.3)
        
        # Attention gates
        self.att4 = AttentionBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.att3 = AttentionBlock(base_channels * 4, base_channels * 4, base_channels * 2)
        self.att2 = AttentionBlock(base_channels * 2, base_channels * 2, base_channels)
        self.att1 = AttentionBlock(base_channels, base_channels, base_channels // 2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout=0.2)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout=0.2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout=0.1)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout=0.1)
        
        # Output heads
        self.out_final = nn.Conv2d(base_channels, 1, 1)
        
        # Deep supervision outputs
        self.out_deep4 = nn.Conv2d(base_channels * 8, 1, 1)
        self.out_deep3 = nn.Conv2d(base_channels * 4, 1, 1)
        self.out_deep2 = nn.Conv2d(base_channels * 2, 1, 1)
    
    def forward(self, x):
        # Encoder
        # The input x is ALWAYS (B, 2, H, W)
        e1 = self.enc1(x)
        
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        # Outputs
        out_final = self.out_final(d1)
        
        # Deep supervision (only during training)
        if self.training:
            out_deep4 = F.interpolate(self.out_deep4(d4), scale_factor=8, mode='bilinear', align_corners=False)
            out_deep3 = F.interpolate(self.out_deep3(d3), scale_factor=4, mode='bilinear', align_corners=False)
            out_deep2 = F.interpolate(self.out_deep2(d2), scale_factor=2, mode='bilinear', align_corners=False)
            return out_final, out_deep4, out_deep3, out_deep2
        
        return out_final

# ============================================================================
# LOSS FUNCTION
# ============================================================================
class CombinedLoss(nn.Module):
    """Combined loss: Dice + Focal (MSE is disabled)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def dice_loss(self, logits, target):
        pred = torch.sigmoid(logits)
        smooth = 1e-6
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def focal_loss(self, logits, target, alpha=0.25, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def mse_loss(self, logits, target):
        """MSE on sigmoid outputs (soft heatmap matching)"""
        pred = torch.sigmoid(logits)
        return F.mse_loss(pred, target)
    
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            # Deep supervision
            out_final, out_deep4, out_deep3, out_deep2 = outputs
            
            loss_final = (
                self.config.WEIGHT_DICE * self.dice_loss(out_final, target) +
                self.config.WEIGHT_FOCAL * self.focal_loss(out_final, target) +
                self.config.WEIGHT_MSE * self.mse_loss(out_final, target)
            )
            
            # Use a simpler loss for deep supervision
            loss_deep4 = self.dice_loss(out_deep4, target)
            loss_deep3 = self.dice_loss(out_deep3, target)
            loss_deep2 = self.dice_loss(out_deep2, target)
            
            total_loss = loss_final + 0.3 * (loss_deep4 + loss_deep3 + loss_deep2)
            
            return total_loss, {
                'total': total_loss.item(),
                'final': loss_final.item(),
                'deep': (loss_deep4 + loss_deep3 + loss_deep2).item() / 3
            }
        else:
            # Inference
            dice = self.dice_loss(outputs, target)
            focal = self.focal_loss(outputs, target)
            mse = self.mse_loss(outputs, target)
            
            total_loss = (
                self.config.WEIGHT_DICE * dice +
                self.config.WEIGHT_FOCAL * focal +
                self.config.WEIGHT_MSE * mse
            )
            
            return total_loss, {
                'total': total_loss.item(),
                'dice': dice.item(),
                'focal': focal.item(),
                'mse': mse.item()
            }

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(config):
    """Load and split data by patient"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Build coordinate lookup
    lookup = build_coordinate_lookup(config.CSV_PATH, config)
    
    # Get all files
    all_files = glob(os.path.join(config.REMOVAL_PATH, "*", "*.npy"))
    print(f"\nFound {len(all_files)} total files")
    
    # Group by patient
    patient_files = defaultdict(list)
    for fpath in all_files:
        patient_id = os.path.basename(os.path.dirname(fpath))
        patient_files[patient_id].append({
            'path': fpath,
            'patient_id': patient_id
        })
    
    print(f"From {len(patient_files)} patients")
    
    # Patient-level split
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
    
    print(f"\nTrain: {len(train_samples)} slices from {len(train_patients)} patients")
    print(f"Val: {len(val_samples)} slices from {len(val_patients)} patients")
    
    return train_samples, val_samples, lookup

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_metrics = defaultdict(list)
    
    pbar = tqdm(loader, desc="Training")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Use torch.amp.autocast
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics (use final output if deep supervision)
        out_for_metric = outputs[0] if isinstance(outputs, tuple) else outputs
        metrics = calculate_metrics(out_for_metric.detach().cpu(), masks.cpu())
        
        total_loss += loss.item()
        for k, v in metrics.items():
            all_metrics[k].append(v)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}"
        })
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Run model (will be [CT, Zeros])
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss, _ = criterion(outputs, masks)
            
            metrics = calculate_metrics(outputs.cpu(), masks.cpu())
            total_loss += loss.item()
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics

# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_predictions(model, dataset, device, output_dir, num_samples=15):
    """Visualize model predictions"""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, patient_id = dataset[idx]
            
            img_batch = img.unsqueeze(0).to(device)
            
            with autocast(device_type='cuda'):
                pred = model(img_batch)
                
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            # img is (2, H, W) for validation: [CT, Zeros]
            ct_img = img[0].numpy()
            heatmap_input = img[1].numpy() # This will be all zeros
            gt_mask = mask[0].numpy()
            
            heatmap_title = 'Input Heatmap (Zeros)'

            # Find centers
            if pred_prob.max() > 0.3:
                pred_center = np.unravel_index(pred_prob.argmax(), pred_prob.shape)
                pred_center = (pred_center[1], pred_center[0])
            else:
                pred_center = (128, 128)
            
            gt_center = np.unravel_index(gt_mask.argmax(), gt_mask.shape)
            gt_center = (gt_center[1], gt_center[0])
            
            # Calculate distance
            dist = np.sqrt((pred_center[0]-gt_center[0])**2 + (pred_center[1]-gt_center[1])**2)
            
            # Plot
            axes[i, 0].imshow(ct_img, cmap='gray')
            axes[i, 0].set_title(f'CT (Input)\n{patient_id[:25]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap_input, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title(heatmap_title)
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(ct_img, cmap='gray')
            axes[i, 2].imshow(gt_mask, alpha=0.5, cmap='Greens')
            axes[i, 2].plot(gt_center[0], gt_center[1], 'g*', markersize=15)
            axes[i, 2].set_title('Ground Truth (Target)')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(pred_prob, cmap='hot')
            axes[i, 3].set_title(f'Prediction\nMax: {pred_prob.max():.3f}')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(ct_img, cmap='gray')
            axes[i, 4].imshow(pred_prob, alpha=0.6, cmap='Reds')
            axes[i, 4].plot(pred_center[0], pred_center[1], 'r*', markersize=15, label='Pred')
            axes[i, 4].plot(gt_center[0], gt_center[1], 'g*', markersize=15, label='GT')
            axes[i, 4].set_title(f'Overlay\nError: {dist:.1f}px')
            axes[i, 4].legend(loc='upper right')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_training_history(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = [h['epoch'] for h in history]
    
    # Loss
    axes[0, 0].plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs, [h['train_dice'] for h in history], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, [h['val_dice'] for h in history], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_title('Dice Score', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 2].plot(epochs, [h['train_iou'] for h in history], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, [h['val_iou'] for h in history], 'r-', label='Val', linewidth=2)
    axes[0, 2].set_title('IoU Score', fontsize=14)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # F1
    axes[1, 0].plot(epochs, [h['train_f1'] for h in history], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, [h['val_f1'] for h in history], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_title('F1 Score', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Localization error
    axes[1, 1].plot(epochs, [h['train_loc'] for h in history], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, [h['val_loc'] for h in history], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_title('Localization Error (px)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 2].plot(epochs, [h['lr'] for h in history], 'g-', linewidth=2)
    axes[1, 2].set_title('Learning Rate', fontsize=14)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")
    
class CTInferenceDataset(Dataset):
    """
    Dataset for testing WITHOUT CSV (CT only).
    Produces 2-channel input: [CT, Zeros]
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]['path']
        data = np.load(path)
        ct_img = data[:, :, 0].astype(np.float32)
        
        # Create a zero heatmap to match the 2-channel input
        zero_heatmap = np.zeros_like(ct_img, dtype=np.float32)
        img = np.stack([ct_img, zero_heatmap], axis=0).astype(np.float32)
        
        return torch.from_numpy(img), self.samples[idx]['patient_id']

# ============================================================================
# MAIN
# ============================================================================
def main():
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    print("="*80)
    print("ROBUST CT TAMPERING LOCALIZATION - V5 (Dual-Curriculum)")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Input Channels: {config.IN_CHANNELS} (CT + Heatmap/Zeros)")
    print(f"Output: {config.OUTPUT_DIR}\n")
    
    # Load data
    train_samples, val_samples, lookup = load_data(config)
    
    # Create datasets
    print("\n" + "="*80)
    print("CREATING DATASETS")
    print("="*80)
    train_dataset = CTTamperingDataset(train_samples, lookup, config, augment=True)
    val_dataset = CTTamperingDataset(val_samples, lookup, config, augment=False)
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n⚠️ ERROR: No valid samples!")
        return
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    model = ImprovedUNet(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    criterion = CombinedLoss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6
    )
    scaler = GradScaler()
    
    # Training loop
    history = []
    best_val_dice = 0
    patience_counter = 0
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-"*80)
        
        # --- CURRICULUM LEARNING: Update Dropout Rate ---
        if epoch < config.CHANNEL_DROPOUT_SCHEDULE_EPOCHS:
            # Linearly increase from START to END
            progress = epoch / config.CHANNEL_DROPOUT_SCHEDULE_EPOCHS
            new_rate = (config.CHANNEL_DROPOUT_RATE_START + 
                        (config.CHANNEL_DROPOUT_RATE_END - config.CHANNEL_DROPOUT_RATE_START) * progress)
        else:
            # Stay at the final rate
            new_rate = config.CHANNEL_DROPOUT_RATE_END
            
        train_dataset.current_dropout_rate = new_rate
        
        # --- CURRICULUM LEARNING: Update Noise Std ---
        if epoch < config.HINT_NOISE_SCHEDULE_START_EPOCH:
            new_noise_std = config.HINT_NOISE_STD_START
        else:
            # Linearly increase from START to END
            progress = (epoch - config.HINT_NOISE_SCHEDULE_START_EPOCH) / config.HINT_NOISE_SCHEDULE_DURATION_EPOCHS
            progress = min(progress, 1.0) # Cap at 1.0
            new_noise_std = (config.HINT_NOISE_STD_START +
                             (config.HINT_NOISE_STD_END - config.HINT_NOISE_STD_START) * progress)
        
        train_dataset.current_hint_noise_std = new_noise_std
        
        # Log the curriculum parameters periodically
        if (epoch) % 5 == 0 or epoch == 0:
             print(f"  ✓ Set Channel Dropout Rate to {new_rate:.3f}")
             print(f"  ✓ Set Hint Noise Std to {new_noise_std:.3f}")
        # --- End Curriculum Update ---
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Scheduler
        scheduler.step(val_metrics['dice'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_iou': val_metrics['iou'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_loc': train_metrics['loc_error'],
            'val_loc': val_metrics['loc_error'],
            'lr': current_lr
        })
        
        # Print summary
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_metrics['dice']:.4f}, "
              f"IoU={train_metrics['iou']:.4f}, LocErr={train_metrics['loc_error']:.1f}px")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_metrics['dice']:.4f}, "
              f"IoU={val_metrics['iou']:.4f}, LocErr={val_metrics['loc_error']:.1f}px")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_val_dice,
                'config': vars(config)
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))
            print(f"  ✓ New best! Dice={best_val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE and epoch >= config.MIN_EPOCHS:
            print(f"\n⏹ Early stopping at epoch {epoch+1}")
            break
        
        # Periodic save
        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save(model.state_dict(), 
                       os.path.join(config.OUTPUT_DIR, f'model_epoch_{epoch+1}.pth'))
    
    # Save history
    with open(os.path.join(config.OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    try:
        checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
    except FileNotFoundError:
        print("⚠️ Warning: 'best_model.pth' not found. Using last model state.")
        checkpoint = {'epoch': 'N/A'} # Dummy checkpoint for print
        
    val_loss, val_metrics = validate_epoch(model, val_loader, criterion, config.DEVICE)
    
    print(f"\nBest Model (Epoch {checkpoint['epoch']}):")
    print(f"  Dice:      {val_metrics['dice']:.4f}")
    print(f"  IoU:       {val_metrics['iou']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  Loc Error: {val_metrics['loc_error']:.1f} pixels")
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history, config.OUTPUT_DIR)
    visualize_predictions(model, val_dataset, config.DEVICE, config.OUTPUT_DIR, num_samples=15)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Best Dice: {best_val_dice:.4f}")
    print(f"Output: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()