# =================================================================================
# OPTIMIZED UNet++ FINE-TUNING v9.2 - PROPER PREVIOUS MODEL LOADING
# Key improvements:
# 1. Properly loads both base model AND feature head from previous experiment
# 2. Builds upon existing trained weights (frozen base + trainable enhancements)
# 3. Progressive unfreezing strategy
# =================================================================================
!pip install -q segmentation_models_pytorch
!pip install -q torchinfo

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging import version
from collections import defaultdict
import gc
from torchinfo import summary

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =================================================================================
# 1. ENHANCED CONFIG - PROPER PREVIOUS MODEL LOADING
# =================================================================================
class CFG:
    # Paths
    BASE_PATH = "/kaggle/input/ct-injection"
    PREV_BASE_PATH = "/kaggle/input/base-model"
    INJECTION_DIR = "CT_Injection"
    INJECTION_CSV = "CT_Injection/data_v1.csv"
    BASE_PATH_output = "/kaggle/working"
    # Previous experiment details - THIS IS CRUCIAL
    prev_experiment = "unetpp_v8_softknn_injection_only_v2"
    prev_ckpt_path = os.path.join(PREV_BASE_PATH, prev_experiment, "best_model.pth")
    
    # Current experiment
    experiment_name = "CT_Injection_finetune_v3.0"
    ckpt_dir = os.path.join(BASE_PATH_output, "checkpoints", experiment_name)
    log_dir = os.path.join(BASE_PATH_output, "logs", experiment_name)
    split_dir = os.path.join(BASE_PATH_output, "data_splits", experiment_name)
    
    # Model - SAME AS PREVIOUS EXPERIMENT
    encoder = "resnet34"
    img_size = 320
    radius_px = 42
    batch_size = 8
    accum_steps = 8
    num_workers = 2

    # Resume control
    isResume = True

    # Optimizer - FOCUS ON NEW COMPONENTS
    encoder_lr = 5e-6   # Low for frozen encoder
    decoder_lr = 1e-4   # Moderate for decoder
    head_lr = 2e-4      # Higher for new enhancement layers
    weight_decay = 1e-4
    
    # Progressive unfreezing
    unfreeze_epoch = 15  # Later unfreezing to stabilize first
    unfreeze_layers = ['layer4']  # Start with last layer only
    
    # KNN - SAME AS BEFORE
    knn_support_per_batch = 256
    knn_beta = 3.0
    knn_feat_channels = 8
    knn_loss_weight = 0.03
    knn_enabled = True

    # Training
    total_epochs = 200
    patience = 40  # More patience for fine-tuning
    val_split = 0.30
    
    # Loss weights
    w_dice = 0.6
    w_focal = 0.3
    w_tversky = 0.1
    
    # Enhanced metrics
    save_samples = 10
    test_samples = 20

os.makedirs(CFG.ckpt_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)
os.makedirs(CFG.split_dir, exist_ok=True)

# =================================================================================
# ENHANCEMENT LAYERS (NEW COMPONENTS ON TOP OF EXISTING MODEL)
# =================================================================================
class CBAM(nn.Module):
    """NEW: Convolutional Block Attention Module - ADDED ON TOP"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x

class EnhancementBlock(nn.Module):
    """NEW: Enhancement block added on top of existing feature head"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cbam(x)  # NEW attention
        x = self.conv2(x)
        return x

# =================================================================================
# PROPER PREVIOUS MODEL LOADING FUNCTIONS
# =================================================================================
def load_previous_model_and_feathead(cfg, device):
    """PROPERLY load both model and feature head from previous experiment"""
    print(f"🔄 Loading previous model from: {cfg.prev_ckpt_path}")
    
    if not os.path.exists(cfg.prev_ckpt_path):
        print(f"❌ Previous checkpoint not found: {cfg.prev_ckpt_path}")
        return None, None
    
    try:
        # Load the previous checkpoint
        checkpoint = torch.load(cfg.prev_ckpt_path, map_location=device)
        print(f"✅ Loaded previous checkpoint")
        
        # Create model with same architecture as before
        model = smp.UnetPlusPlus(
            encoder_name=cfg.encoder,
            encoder_weights=None,
            decoder_attention_type='scse',
            classes=1,
            activation=None
        )
        
        # Create the original feature head (same as previous)
        class OriginalPixelFeatureHead(nn.Module):
            def __init__(self, in_ch=4, out_ch=cfg.knn_feat_channels):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, out_ch, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        feat_head = OriginalPixelFeatureHead()
        
        # Move to device
        model.to(device)
        feat_head.to(device)
        
        # Load weights - handle different checkpoint formats
        if is_state_dict(checkpoint):
            # Raw state_dict
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights (raw state_dict)")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded model weights from 'model_state_dict'")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("✅ Loaded model weights from 'model'")
        else:
            # Try direct loading
            try:
                model.load_state_dict(checkpoint)
                print("✅ Loaded model weights (direct)")
            except:
                print("❌ Could not load model weights")
                return None, None
        
        # Load feature head if available
        feat_head_loaded = False
        if 'feat_head' in checkpoint:
            try:
                feat_head.load_state_dict(checkpoint['feat_head'])
                print("✅ Loaded feature head weights from 'feat_head'")
                feat_head_loaded = True
            except Exception as e:
                print(f"⚠️ Could not load feature head: {e}")
        elif 'feat_head_state_dict' in checkpoint:
            try:
                feat_head.load_state_dict(checkpoint['feat_head_state_dict'])
                print("✅ Loaded feature head weights from 'feat_head_state_dict'")
                feat_head_loaded = True
            except Exception as e:
                print(f"⚠️ Could not load feature head: {e}")
        
        if not feat_head_loaded:
            print("ℹ️ Using original feature head architecture (no saved weights)")
        
        return model, feat_head
        
    except Exception as e:
        print(f"❌ Error loading previous model: {e}")
        return None, None

def create_enhanced_feature_head(original_feat_head, cfg):
    """Create enhanced feature head that uses original as base"""
    class EnhancedFeatureHead(nn.Module):
        def __init__(self, original_head, enhancement_channels=32):
            super().__init__()
            # Keep the original feature head FROZEN
            self.original_head = original_head
            for param in self.original_head.parameters():
                param.requires_grad = False
            
            # NEW: Add enhancement layers on top
            self.enhancement = EnhancementBlock(
                cfg.knn_feat_channels,  # Input from original head
                enhancement_channels   # Enhanced features
            )
            
            print(f"🆕 Added enhancement block on top of original feature head")
            print(f"   Original features: {cfg.knn_feat_channels} -> Enhanced: {enhancement_channels}")
            
        def forward(self, x):
            # Get features from original (frozen) head
            with torch.no_grad():
                original_features = self.original_head(x)
            
            # Enhance features with new trainable layers
            enhanced_features = self.enhancement(original_features)
            return enhanced_features
    
    return EnhancedFeatureHead(original_feat_head)

# =================================================================================
# EXISTING UTILITY FUNCTIONS (Keep mostly same)
# =================================================================================
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def is_state_dict(checkpoint):
    """Check if the checkpoint is a raw state_dict"""
    if not isinstance(checkpoint, dict):
        return False
    model_keys = [k for k in checkpoint.keys() if any(x in k for x in ['encoder.', 'decoder.', 'segmentation_head.'])]
    metadata_keys = [k for k in checkpoint.keys() if k in ['epoch', 'best_dice', 'model', 'optimizer', 'history']]
    return len(model_keys) > 0 and len(metadata_keys) == 0

# Keep all dataset, dataloader, transform functions exactly the same
# [Previous dataset, dataloader, transform code remains unchanged]
# =================================================================================
# Dataset & Transforms (UNCHANGED - same as before)
# =================================================================================
def window_image(img, window_center, window_width, to_uint8=True):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    normalized_img = (windowed_img - img_min) / (img_max - img_min + 1e-6)
    if to_uint8:
        return (normalized_img * 255).astype(np.uint8)
    return normalized_img

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def get_transforms(is_train: bool, img_size: int):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    if is_train:
        transforms = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=25, p=0.7),
            A.OneOf([
                A.ElasticTransform(alpha=15, sigma=6, alpha_affine=10, p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(transforms)

def draw_disk(mask, cx, cy, r):
    H, W = mask.shape
    y, x = np.ogrid[:H, :W]
    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    mask[dist2 <= r * r] = 1

class SegDataset(Dataset):
    def __init__(self, samples, img_size, is_train, radius_px):
        self.samples = samples
        self.radius_px = radius_px
        self.transforms = get_transforms(is_train, img_size)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_raw = np.load(s['fpath']).astype(np.float32)
        
        # Multi-window preprocessing
        ch1 = window_image(img_raw, -600, 1500, True)
        ch2 = window_image(img_raw, 40, 400, True)
        ch3 = window_image(img_raw, 400, 1800, True)
        ch1 = apply_clahe(ch1)
        img = np.stack([ch1, ch2, ch3], axis=-1)
        
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        for x, y in s.get('annotations', []):
            if x is not None and y is not None:
                cx, cy = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
                draw_disk(mask, cx, cy, self.radius_px)
        
        augmented = self.transforms(image=img, mask=mask)
        return augmented['image'].float(), augmented['mask'].unsqueeze(0).float(), s

class PatientGroupedBatchSampler(Sampler):
    def __init__(self, samples, patient_groups, batch_size, shuffle_patients=True):
        self.samples = samples
        self.patient_indices = {}
        for pid in patient_groups:
            self.patient_indices[pid] = [i for i, s in enumerate(samples) if s['patient_id'] == pid]
        self.pids = list(patient_groups.keys())
        self.batch_size = batch_size
        self.shuffle_patients = shuffle_patients

    def __iter__(self):
        pids = self.pids[:]
        if self.shuffle_patients:
            random.shuffle(pids)
        for pid in pids:
            indices = self.patient_indices[pid][:]
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and len(indices) >= self.batch_size:
                    batch += indices[:self.batch_size - len(batch)]
                yield batch

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size 
                   for indices in self.patient_indices.values())

def custom_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    samples = [b[2] for b in batch]
    return imgs, masks, samples

def build_tamper_samples(root_dir, csv_path, dataset_label):
    df = pd.read_csv(csv_path)
    groups = df.groupby(["path", "cur_slice"])
    samples = []
    for (patient_rel, slice_id), g in groups:
        fpath = os.path.join(root_dir, str(patient_rel), f"{int(slice_id)}.npy")
        if not os.path.exists(fpath): 
            continue
        ann = [(int(r["x"]), int(r["y"])) for _, r in g.iterrows() if pd.notna(r["x"])]
        samples.append({
            "fpath": fpath, 
            "patient_id": f"inj_{patient_rel}", 
            "annotations": ann, 
            "dataset": dataset_label
        })
    return samples

def get_dataloaders(cfg):
    inj_samples = build_tamper_samples(
        os.path.join(cfg.BASE_PATH, cfg.INJECTION_DIR), 
        os.path.join(cfg.BASE_PATH, cfg.INJECTION_CSV), 
        "inj"
    )
    print(f"Found {len(inj_samples)} injection slices.")

    # PATIENT-WISE SPLITTING (no data leakage)
    train_patients_path = os.path.join(cfg.split_dir, "train_patients.json")
    if os.path.exists(train_patients_path):
        with open(train_patients_path, 'r') as f:
            train_patients = json.load(f)
        with open(os.path.join(cfg.split_dir, "val_patients.json"), 'r') as f:
            val_patients = json.load(f)
        print("Loaded existing patient split.")
    else:
        patients = list(set(s['patient_id'] for s in inj_samples))
        train_patients, val_patients = train_test_split(
            patients, test_size=cfg.val_split, random_state=42
        )
        with open(train_patients_path, 'w') as f:
            json.dump(train_patients, f)
        with open(os.path.join(cfg.split_dir, "val_patients.json"), 'w') as f:
            json.dump(val_patients, f)
        print("Created new patient split.")

    train_samples = [s for s in inj_samples if s['patient_id'] in train_patients]
    val_samples = [s for s in inj_samples if s['patient_id'] in val_patients]

    train_patient_groups = defaultdict(list)
    for s in train_samples:
        train_patient_groups[s['patient_id']].append(s)

    train_ds = SegDataset(train_samples, cfg.img_size, True, cfg.radius_px)
    val_ds = SegDataset(val_samples, cfg.img_size, False, cfg.radius_px)

    train_batch_sampler = PatientGroupedBatchSampler(
        train_samples, train_patient_groups, cfg.batch_size
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_batch_sampler, 
        num_workers=cfg.num_workers, pin_memory=True, 
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, pin_memory=True, 
        collate_fn=custom_collate
    )

    print(f"Train: {len(train_samples)} samples ({len(train_patient_groups)} patients)")
    print(f"Val: {len(val_samples)} samples ({len(set(s['patient_id'] for s in val_samples))} patients)")
    
    return train_loader, val_loader, val_samples, train_samples

# =================================================================================
# LOSS AND METRICS (Keep same)
# =================================================================================
class EnhancedCombinedLoss(nn.Module):
    def __init__(self, w_dice=CFG.w_dice, w_focal=CFG.w_focal, w_tversky=CFG.w_tversky):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary', gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_tversky = w_tversky

    def forward(self, logits, targets):
        return (self.w_dice * self.dice(logits, targets) + 
                self.w_focal * self.focal(logits, targets) +
                self.w_tversky * self.tversky(logits, targets))

@torch.no_grad()
def compute_detailed_metrics(preds, targets):
    """Compute comprehensive metrics"""
    preds_flat = preds.view(preds.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)
    
    tp = (preds_flat & targets_flat).sum(dim=1).float()
    fp = (preds_flat & ~targets_flat).sum(dim=1).float()
    fn = (~preds_flat & targets_flat).sum(dim=1).float()
    tn = (~preds_flat & ~targets_flat).sum(dim=1).float()
    
    epsilon = 1e-7
    
    dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)
    
    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'specificity': specificity.mean().item(),
        'accuracy': accuracy.mean().item(),
    }

# =================================================================================
# KNN UTILITIES (Keep same)
# =================================================================================
class SoftKNN:
    def __init__(self, beta=CFG.knn_beta, device='cpu'):
        self.beta = beta
        self.device = device

    @torch.no_grad()
    def predict(self, support_feats, support_labels, query_feats, chunk_size=4096):
        support_feats = support_feats.to(self.device)
        support_labels = support_labels.to(self.device)
        query_feats = query_feats.to(self.device)

        Ns = support_feats.shape[0]
        Nq = query_feats.shape[0]
        
        if Ns == 0 or Nq == 0:
            return torch.zeros((Nq,), device=self.device)

        support_feats = F.normalize(support_feats, dim=1)
        query_feats = F.normalize(query_feats, dim=1)

        probs_chunks = []
        for start in range(0, Nq, chunk_size):
            end = min(Nq, start + chunk_size)
            q_chunk = query_feats[start:end]
            
            similarities = q_chunk @ support_feats.t()
            weights = F.softmax(self.beta * similarities, dim=1)
            
            probs_chunk = (weights * support_labels.unsqueeze(0)).sum(dim=1)
            probs_chunks.append(probs_chunk)

        return torch.cat(probs_chunks, dim=0)

def sample_support_pixels(features, masks, max_support=CFG.knn_support_per_batch):
    B, C, H, W = features.shape
    device = features.device
    
    feats_flat = features.permute(0,2,3,1).reshape(-1, C)
    masks_flat = masks.view(-1)
    
    pos_idx = (masks_flat > 0.5).nonzero(as_tuple=False).squeeze(1)
    neg_idx = (masks_flat <= 0.5).nonzero(as_tuple=False).squeeze(1)

    half = max_support // 2
    chosen_idx = []
    
    if pos_idx.numel() > 0:
        pos_sample = min(half, len(pos_idx))
        chosen_idx.append(pos_idx[torch.randperm(len(pos_idx))[:pos_sample]])
    
    if neg_idx.numel() > 0:
        neg_sample = min(max_support - len(chosen_idx[0]) if chosen_idx else half, len(neg_idx))
        chosen_idx.append(neg_idx[torch.randperm(len(neg_idx))[:neg_sample]])
    
    if not chosen_idx:
        return torch.empty((0, C), device=device), torch.empty((0,), device=device)
    
    chosen_idx = torch.cat(chosen_idx)
    return feats_flat[chosen_idx], masks_flat[chosen_idx].float()

# =================================================================================
# MODEL SETUP WITH PROPER FREEZING
# =================================================================================
def freeze_encoder_completely(model):
    """Freeze encoder completely - PRESERVE PREVIOUS TRAINING"""
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("🧊 Encoder completely frozen (preserving previous training)")

def freeze_decoder_completely(model):
    """Freeze decoder completely - PRESERVE PREVIOUS TRAINING"""
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.segmentation_head.parameters():
        param.requires_grad = False
    print("🧊 Decoder completely frozen (preserving previous training)")

def unfreeze_encoder_layers(model, layer_names):
    """Carefully unfreeze specific encoder layers"""
    for name, param in model.encoder.named_parameters():
        if any(layer in name for layer in layer_names):
            param.requires_grad = True
    print(f"🔓 Unfrozen encoder layers: {layer_names}")

def setup_optimizer(model, feat_head, cfg, epoch=0):
    """Setup optimizer focusing ONLY on new components initially"""
    
    # Initially: Only train the ENHANCEMENT layers
    if epoch < cfg.unfreeze_epoch:
        # Only enhancement layers in feature head are trainable
        enhancement_params = []
        for name, param in feat_head.named_parameters():
            if 'enhancement' in name:
                enhancement_params.append(param)
                print(f"🎯 Training enhancement parameter: {name}")
        
        param_groups = [
            {'params': enhancement_params, 'lr': cfg.head_lr, 'name': 'enhancement'}
        ]
        print("🔧 Training ONLY enhancement layers")
        
    else:
        # After unfreeze_epoch: Add some encoder layers
        encoder_params = []
        enhancement_params = []
        
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:  # Only unfrozen layers
                encoder_params.append(param)
        
        for name, param in feat_head.named_parameters():
            if 'enhancement' in name:
                enhancement_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': cfg.encoder_lr, 'name': 'encoder'},
            {'params': enhancement_params, 'lr': cfg.head_lr, 'name': 'enhancement'}
        ]
        print("🔧 Training encoder layers + enhancement layers")
    
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer

# =================================================================================
# VISUALIZATION AND TESTING
# =================================================================================
def visualize_predictions(model, dataloader, device, num_samples=5, save_dir=None):
    """Visualize model predictions"""
    model.eval()
    samples_shown = 0
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for imgs, masks, sample_info in dataloader:
            if samples_shown >= num_samples:
                break
                
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            logits = model(imgs)
            preds = torch.sigmoid(logits)
            binary_preds = (preds > 0.5).float()
            
            for i in range(min(imgs.shape[0], num_samples - samples_shown)):
                idx = samples_shown + i
                
                img_np = imgs[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()
                pred_np = preds[i, 0].cpu().numpy()
                binary_pred_np = binary_preds[i, 0].cpu().numpy()
                
                axes[idx, 0].imshow(img_np, cmap='gray')
                axes[idx, 0].set_title('Input Image')
                axes[idx, 0].axis('off')
                
                axes[idx, 1].imshow(mask_np, cmap='jet')
                axes[idx, 1].set_title('Ground Truth')
                axes[idx, 1].axis('off')
                
                axes[idx, 2].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
                axes[idx, 2].set_title('Prediction Probability')
                axes[idx, 2].axis('off')
                
                axes[idx, 3].imshow(binary_pred_np, cmap='jet')
                
                sample_metrics = compute_detailed_metrics(
                    binary_preds[i:i+1] > 0.5, 
                    masks[i:i+1] > 0.5
                )
                
                axes[idx, 3].set_title(f'Binary Pred\nDice: {sample_metrics["dice"]:.3f}')
                axes[idx, 3].axis('off')
                
            samples_shown += imgs.shape[0]
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()

def test_model(model, test_loader, device):
    """Comprehensive model testing"""
    model.eval()
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for imgs, masks, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).bool()
            targets = (masks > 0.5).bool()
            
            batch_metrics = compute_detailed_metrics(preds, targets)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
    
    final_metrics = {}
    for k, v in all_metrics.items():
        final_metrics[k] = np.mean(v)
    
    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)
    for metric, value in final_metrics.items():
        print(f"{metric.upper():<12}: {value:.4f}")
    print("="*60)
    
    return final_metrics

# =================================================================================
# MAIN TRAINING LOOP - PROPER FINE-TUNING
# =================================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, val_samples, train_samples = get_dataloaders(CFG)
    
    # PROPERLY LOAD PREVIOUS MODEL AND FEATURE HEAD
    model, original_feat_head = load_previous_model_and_feathead(CFG, device)
    
    if model is None:
        print("❌ Failed to load previous model. Exiting.")
        return None, None, None, None
    
    # CREATE ENHANCED FEATURE HEAD THAT USES ORIGINAL AS BASE
    feat_head = create_enhanced_feature_head(original_feat_head, CFG)
    feat_head.to(device)
    
    # FREEZE EVERYTHING EXCEPT NEW ENHANCEMENT LAYERS
    freeze_encoder_completely(model)
    freeze_decoder_completely(model)
    
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    print("✅ Base U-Net++: LOADED FROM PREVIOUS EXPERIMENT (FROZEN)")
    print("✅ Original Feature Head: LOADED FROM PREVIOUS EXPERIMENT (FROZEN)") 
    print("🆕 Enhancement Layers: NEW ADDITIONS (TRAINABLE)")
    print("="*80)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_feat_params = sum(p.numel() for p in feat_head.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params + trainable_feat_params:,}")
    print(f"   - Model: {trainable_params:,}")
    print(f"   - Feature head enhancements: {trainable_feat_params:,}")
    
    # Enhanced logging
    writer = SummaryWriter(log_dir=CFG.log_dir)
    
    # Track best metrics
    best_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    patience_counter = 0
    history = {
        'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [], 'train_dice': [], 'lr': []
    }
    
    start_epoch = 0
    
    print(f"\n{'='*80}")
    print(f"STARTING PROPER FINE-TUNING")
    print(f"Building upon: {CFG.prev_experiment}")
    print(f"Enhancement unfreezing at epoch: {CFG.unfreeze_epoch}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, CFG.total_epochs):
        # Progressive unfreezing
        if epoch == CFG.unfreeze_epoch:
            unfreeze_encoder_layers(model, CFG.unfreeze_layers)
        
        # Setup optimizer (focuses on trainable components)
        optimizer = setup_optimizer(model, feat_head, CFG, epoch)
        
        # Enhanced scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.total_epochs//4, eta_min=1e-7)
        
        # Loss and training utilities
        seg_loss_fn = EnhancedCombinedLoss()
        scaler = GradScaler()
        knn = SoftKNN(device=device) if CFG.knn_enabled else None
        
        # Training phase
        model.train()
        feat_head.train()
        
        epoch_loss = 0.0
        train_metrics_sum = defaultdict(float)
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.total_epochs}")
        
        for batch_idx, (imgs, masks, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast(dtype=torch.float16):
                # Forward pass through FROZEN base model
                with torch.no_grad():
                    logits = model(imgs)
                
                seg_loss = seg_loss_fn(logits, masks)
                
                # KNN loss with ENHANCED features
                knn_loss = torch.tensor(0.0, device=device)
                if CFG.knn_enabled and knn is not None:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                    feat_input = torch.cat([imgs, probs.detach()], dim=1)
                    pixel_feats = feat_head(feat_input)  # Uses enhanced features
                    
                    support_feats, support_labels = sample_support_pixels(
                        pixel_feats.detach(), masks
                    )
                    
                    if support_feats.shape[0] > 0:
                        B, C, H, W = pixel_feats.shape
                        query_feats = pixel_feats.permute(0,2,3,1).reshape(B, -1, C)
                        
                        knn_loss_batch = 0.0
                        for b in range(B):
                            soft_probs = knn.predict(
                                support_feats, support_labels, query_feats[b]
                            )
                            soft_probs = soft_probs.view(1, 1, H, W).clamp(1e-7, 1-1e-7)
                            target_b = masks[b:b+1]
                            
                            knn_logits = torch.logit(soft_probs.clamp(1e-7, 1-1e-7))
                            knn_loss_batch += F.binary_cross_entropy_with_logits(
                                knn_logits, target_b, reduction='mean'
                            )
                        knn_loss = knn_loss_batch / B
                
                total_loss = seg_loss + CFG.knn_loss_weight * knn_loss
                total_loss = total_loss / CFG.accum_steps
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % CFG.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(feat_head.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += total_loss.item() * CFG.accum_steps
            
            # Compute training metrics
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).bool()
                targets = (masks > 0.5).bool()
                batch_metrics = compute_detailed_metrics(preds, targets)
                train_metrics_sum['dice'] += batch_metrics['dice']
            
            pbar.set_postfix({
                'loss': f"{total_loss.item() * CFG.accum_steps:.4f}",
                'dice': f"{batch_metrics['dice']:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Validation phase
        model.eval()
        feat_head.eval()
        
        val_metrics_sum = defaultdict(float)
        
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).bool()
                targets = (masks > 0.5).bool()
                
                batch_metrics = compute_detailed_metrics(preds, targets)
                for k, v in batch_metrics.items():
                    if k not in ['tp', 'fp', 'fn', 'tn']:
                        val_metrics_sum[k] += v
        
        # Calculate averages
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_metrics_sum['dice'] / len(train_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)
        
        # Enhanced TensorBoard logging
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Dice/train", avg_train_dice, epoch)
        writer.add_scalar("Dice/val", val_metrics['dice'], epoch)
        writer.add_scalar("IoU/val", val_metrics['iou'], epoch)
        writer.add_scalar("F1/val", val_metrics['f1'], epoch)
        writer.add_scalar("Precision/val", val_metrics['precision'], epoch)
        writer.add_scalar("Recall/val", val_metrics['recall'], epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f}")
        print(f"Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Scheduler step
        scheduler.step()
        
        # Enhanced checkpointing
        current_dice = val_metrics['dice']
        current_f1 = val_metrics['f1']
        
        # Use composite score for best model determination
        composite_score = current_dice * 0.6 + current_f1 * 0.4
        best_composite = best_metrics['dice'] * 0.6 + best_metrics['f1'] * 0.4
        
        is_best = composite_score > best_composite
        
        if is_best:
            best_metrics.update(val_metrics)
            patience_counter = 0
            
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'feat_head': feat_head.state_dict(),
                'original_feat_head': original_feat_head.state_dict(),
                'best_dice': best_metrics['dice'],
                'best_f1': best_metrics['f1'],
                'best_iou': best_metrics['iou'],
                'history': history,
                'config': CFG.__dict__
            }
            
            torch.save(save_dict, os.path.join(CFG.ckpt_dir, "best_model.pth"))
            print(f"🎉 NEW BEST! Dice: {best_metrics['dice']:.4f}, F1: {best_metrics['f1']:.4f} → Model saved!")
            
            # Visualize some predictions on best model
            if epoch % 10 == 0:
                print("📊 Generating sample predictions...")
                visualize_predictions(model, val_loader, device, num_samples=5, save_dir=CFG.ckpt_dir)
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{CFG.patience}")
            
            if patience_counter >= CFG.patience:
                print(f"🏁 Early stopping at epoch {epoch}")
                break
        
        clear_memory()
    
    # Final model saving and testing
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE - RUNNING FINAL EVALUATION")
    print("="*80)
    
    # Load best model for final evaluation
    best_checkpoint_path = os.path.join(CFG.ckpt_dir, "best_model.pth")
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(best_checkpoint['model'])
        feat_head.load_state_dict(best_checkpoint['feat_head'])
        print("✅ Loaded best model for final evaluation")
    
    # Final comprehensive testing
    final_metrics = test_model(model, val_loader, device)
    
    # Generate final visualizations
    print("\n📊 Generating final sample predictions...")
    visualize_predictions(model, val_loader, device, num_samples=CFG.save_samples, save_dir=CFG.ckpt_dir)
    
    # Save final model
    final_save_path = os.path.join(CFG.ckpt_dir, "final_model.pth")
    torch.save({
        'model': model.state_dict(),
        'feat_head': feat_head.state_dict(),
        'original_feat_head': original_feat_head.state_dict(),
        'config': CFG.__dict__,
        'final_metrics': final_metrics,
        'history': history
    }, final_save_path)
    
    print(f"\n💾 Final model saved to: {final_save_path}")
    print(f"📈 TensorBoard logs: {CFG.log_dir}")
    print("To view TensorBoard: %tensorboard --logdir=" + CFG.log_dir)
    
    writer.close()
    
    return model, feat_head, history, final_metrics

if __name__ == "__main__":
    final_model, final_feat_head, training_history, final_metrics = train()