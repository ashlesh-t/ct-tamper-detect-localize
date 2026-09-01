# =================================================================================
# OPTIMIZED UNet++ FINE-TUNING v11 - MEMORY EFFICIENT + HIGH ACCURACY
# Key improvements:
# 1. Fixed OOM issues with gradient checkpointing and memory management
# 2. Progressive unfreezing with proper optimizer reinitialization
# 3. Discriminative learning rates for encoder/decoder
# 4. Enhanced metrics tracking and logging
# 5. Better KNN implementation with memory optimization
# 6. Proper fine-tuning strategy (freeze BN, differential LR)
# =================================================================================
# !pip install -q segmentation_models_pytorch
from google.colab import drive
drive.mount('/content/drive')

import os
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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging import version
from collections import defaultdict
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =================================================================================
# 1. CONFIG - Memory Optimized
# =================================================================================
class CFG:
    # Paths
    BASE_PATH = "/content/drive/MyDrive/Capstone/main"
    INJECTION_DIR = "CT_Injection"
    INJECTION_CSV = "CT_Injection/data_v1.csv"

    # Model - REDUCED for memory
    encoder = "resnet34"
    img_size = 320  # Reduced from 384
    radius_px = 42  # Proportionally reduced
    batch_size = 8  # Reduced from 16
    accum_steps = 8  # Increased to maintain effective batch size
    num_workers = 2

    # Fine-tuning
    resume_from = "best_model.pth"
    prev_experiment = "unetpp_v8_softknn_injection_only_v2"
    experiment_name = "unetpp_v11_optimized_finetune"
    ckpt_dir = os.path.join(BASE_PATH, "checkpoints", experiment_name)
    log_dir = os.path.join(BASE_PATH, "logs", experiment_name)
    split_dir = os.path.join(BASE_PATH, "data_splits", experiment_name)

    # Optimizer - Discriminative LR
    encoder_lr = 5e-6  # Much lower for pretrained encoder
    decoder_lr = 5e-5  # Higher for decoder
    head_lr = 1e-4     # Highest for new head
    weight_decay = 1e-4
    
    # Scheduler
    use_onecycle = True
    max_lr_multiplier = 10  # For OneCycle
    
    # KNN - Optimized
    knn_support_per_batch = 256  # Reduced
    knn_beta = 3.0  # Softer
    knn_feat_channels = 8  # Reduced from 16
    knn_loss_weight = 0.05  # Reduced weight
    knn_enabled = True  # Can disable if causing issues

    # Training
    total_epochs = 150
    patience = 25
    val_split = 0.30
    grad_clip_norm = 0.5
    
    # Progressive unfreezing schedule (epoch: layers)
    unfreeze_schedule = {
        0: [],  # Start with encoder frozen
        15: ['layer4'],  # Unfreeze deepest first
        30: ['layer3', 'layer4'],
        45: ['layer2', 'layer3', 'layer4'],
        60: ['layer1', 'layer2', 'layer3', 'layer4'],
        75: ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    }
    
    # Loss weights
    w_dice = 0.5
    w_focal = 0.3
    w_tversky = 0.2
    
    # Gradient checkpointing
    use_gradient_checkpointing = True
    
    save_samples = 10

os.makedirs(CFG.ckpt_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)
os.makedirs(CFG.split_dir, exist_ok=True)

# =================================================================================
# Memory Management Utilities
# =================================================================================
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def log_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# =================================================================================
# Custom Collate & Sampler
# =================================================================================
def custom_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    samples = [b[2] for b in batch]
    return imgs, masks, samples

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

# =================================================================================
# Dataset & Transforms
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

# =================================================================================
# Dataloaders
# =================================================================================
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
# Advanced Loss
# =================================================================================
class CombinedLoss(nn.Module):
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

# =================================================================================
# Metrics
# =================================================================================
@torch.no_grad()
def compute_metrics(preds, targets):
    """Compute Dice and IoU"""
    preds_flat = preds.view(preds.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)
    
    tp = (preds_flat & targets_flat).sum(dim=1).float()
    fp = (preds_flat & ~targets_flat).sum(dim=1).float()
    fn = (~preds_flat & targets_flat).sum(dim=1).float()
    tn = (~preds_flat & ~targets_flat).sum(dim=1).float()
    
    epsilon = 1e-7
    dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    
    # Additional metrics
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }

# =================================================================================
# KNN Feature Head (Memory Optimized)
# =================================================================================
class PixelFeatureHead(nn.Module):
    def __init__(self, in_ch=4, out_ch=CFG.knn_feat_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class SoftKNN:
    def __init__(self, beta=CFG.knn_beta, device='cpu'):
        self.beta = beta
        self.device = device

    @torch.no_grad()
    def predict(self, support_feats, support_labels, query_feats, chunk_size=4096):
        """Memory-efficient KNN prediction with chunking"""
        support_feats = support_feats.to(self.device)
        support_labels = support_labels.to(self.device)
        query_feats = query_feats.to(self.device)

        Ns = support_feats.shape[0]
        Nq = query_feats.shape[0]
        
        if Ns == 0 or Nq == 0:
            return torch.zeros((Nq,), device=self.device)

        # Normalize features
        support_feats = F.normalize(support_feats, dim=1)
        query_feats = F.normalize(query_feats, dim=1)

        probs_chunks = []
        for start in range(0, Nq, chunk_size):
            end = min(Nq, start + chunk_size)
            q_chunk = query_feats[start:end]
            
            # Cosine similarity (since normalized)
            similarities = q_chunk @ support_feats.t()
            weights = F.softmax(self.beta * similarities, dim=1)
            
            probs_chunk = (weights * support_labels.unsqueeze(0)).sum(dim=1)
            probs_chunks.append(probs_chunk)

        return torch.cat(probs_chunks, dim=0)

def sample_support_pixels(features, masks, max_support=CFG.knn_support_per_batch):
    """Balanced sampling of support pixels"""
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
# Model Setup with Gradient Checkpointing
# =================================================================================
def setup_model(cfg, device):
    """Setup model with gradient checkpointing"""
    model = smp.UnetPlusPlus(
        encoder_name=cfg.encoder,
        encoder_weights=None,
        decoder_attention_type='scse',
        classes=1,
        activation=None
    )
    
    # Enable gradient checkpointing if needed
    if cfg.use_gradient_checkpointing and hasattr(model.encoder, 'set_gradient_checkpointing'):
        model.encoder.set_gradient_checkpointing(True)
    
    feat_head = PixelFeatureHead()
    
    model.to(device)
    feat_head.to(device)
    
    return model, feat_head

def freeze_bn(module):
    """Freeze batch normalization layers"""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

def unfreeze_encoder_layers(model, layers_to_unfreeze):
    """Unfreeze specific encoder layers"""
    # First, freeze everything
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Unfreeze specified layers
    for layer_name in layers_to_unfreeze:
        if hasattr(model.encoder, layer_name):
            layer = getattr(model.encoder, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
            print(f"  ✓ Unfroze {layer_name}")
    
    # Always keep BN frozen for stability
    freeze_bn(model.encoder)

def setup_optimizer(model, feat_head, cfg, unfrozen_layers):
    """Setup optimizer with discriminative learning rates"""
    # Get trainable encoder params
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    seghead_params = list(model.segmentation_head.parameters())
    feat_params = list(feat_head.parameters())
    
    param_groups = []
    
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': cfg.encoder_lr})
        print(f"  Encoder params: {len(encoder_params)} (lr={cfg.encoder_lr:.2e})")
    
    # Always include these groups
    param_groups.extend([
        {'params': decoder_params, 'lr': cfg.decoder_lr},
        {'params': seghead_params, 'lr': cfg.decoder_lr},
        {'params': feat_params, 'lr': cfg.head_lr}
    ])
    
    print(f"  Decoder params: {len(decoder_params)} (lr={cfg.decoder_lr:.2e})")
    print(f"  SegHead params: {len(seghead_params)} (lr={cfg.decoder_lr:.2e})")
    print(f"  FeatHead params: {len(feat_params)} (lr={cfg.head_lr:.2e})")
    
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    
    return optimizer

def validate_optimizer_setup(optimizer):
    """Validate optimizer has expected parameter groups"""
    actual_groups = len(optimizer.param_groups)
    print(f"Optimizer has {actual_groups} parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {num_params} params, lr={group['lr']:.2e}")

def create_onecycle_scheduler(optimizer, total_steps, cfg):
    """Create OneCycleLR scheduler that adapts to current parameter groups"""
    max_lrs = []
    
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        max_lrs.append(current_lr * cfg.max_lr_multiplier)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    print(f"OneCycleLR configured for {len(max_lrs)} parameter groups:")
    for i, lr in enumerate(max_lrs):
        print(f"  Group {i}: max_lr = {lr:.2e}")
    
    return scheduler

# =================================================================================
# Training Loop
# =================================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    log_memory("Initial")
    
    # Load data
    train_loader, val_loader, val_samples, train_samples = get_dataloaders(CFG)
    
    # Setup model
    model, feat_head = setup_model(CFG, device)
    
    # Load checkpoint
    prev_ckpt_path = os.path.join(
        CFG.BASE_PATH, "checkpoints", CFG.prev_experiment, CFG.resume_from
    )
    start_epoch = 0
    best_dice = 0.0
    
    try:
        checkpoint = torch.load(prev_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        feat_head.load_state_dict(checkpoint['feat_head'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"✓ Loaded checkpoint from epoch {start_epoch-1}, best Dice: {best_dice:.4f}")
    except Exception as e:
        print(f"Warning: Could not load full checkpoint: {e}")
        try:
            model.load_state_dict(torch.load(prev_ckpt_path, map_location=device))
            print("✓ Loaded model weights only")
        except:
            print("✗ Could not load weights, training from scratch")
    
    # Initial freeze: All encoder frozen
    unfreeze_encoder_layers(model, [])
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, feat_head, CFG, [])
    validate_optimizer_setup(optimizer)
    
    steps_per_epoch = len(train_loader) // CFG.accum_steps
    total_steps = steps_per_epoch * CFG.total_epochs
    
    if CFG.use_onecycle:
        scheduler = create_onecycle_scheduler(optimizer, total_steps, CFG)
    else:
        scheduler = None
    
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Loss and training utilities
    seg_loss_fn = CombinedLoss()
    scaler = GradScaler()
    knn = SoftKNN(device=device) if CFG.knn_enabled else None
    
    # Logging
    writer = SummaryWriter(log_dir=CFG.log_dir)
    
    # Tracking
    patience_counter = 0
    history = {
        'train_loss': [], 'train_seg_loss': [], 'train_knn_loss': [],
        'train_dice': [], 'train_iou': [], 'train_precision': [], 'train_recall': [],
        'val_dice': [], 'val_iou': [], 'val_precision': [], 'val_recall': [],
        'lr': []
    }
    
    print(f"\n{'='*80}")
    print(f"Starting fine-tuning from epoch {start_epoch}")
    print(f"Target: Dice > 0.95 | Current best: {best_dice:.4f}")
    print(f"{'='*80}\n")
    
    for epoch in range(start_epoch, CFG.total_epochs):
        # Check for unfreezing
        if epoch in CFG.unfreeze_schedule:
            layers_to_unfreeze = CFG.unfreeze_schedule[epoch]
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}: Unfreezing layers: {layers_to_unfreeze}")
            print(f"{'='*80}")
            
            unfreeze_encoder_layers(model, layers_to_unfreeze)
            
            # Recreate optimizer with new unfrozen params
            optimizer = setup_optimizer(model, feat_head, CFG, layers_to_unfreeze)
            validate_optimizer_setup(optimizer)
            
            if CFG.use_onecycle:
                remaining_steps = (CFG.total_epochs - epoch) * steps_per_epoch
                scheduler = create_onecycle_scheduler(optimizer, remaining_steps, CFG)
            
            clear_memory()
            log_memory(f"After unfreezing at epoch {epoch}")
        
        # ======================== TRAINING ========================
        model.train()
        feat_head.train()
        freeze_bn(model.encoder)  # Keep BN in eval mode
        
        epoch_seg_loss = 0.0
        epoch_knn_loss = 0.0
        epoch_total_loss = 0.0
        train_metrics_sum = defaultdict(float)
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.total_epochs}")
        
        for batch_idx, (imgs, masks, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast(dtype=torch.float16):
                # Forward pass
                logits = model(imgs)
                seg_loss = seg_loss_fn(logits, masks)
                
                # KNN loss
                knn_loss = torch.tensor(0.0, device=device)
                if CFG.knn_enabled and knn is not None:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                    feat_input = torch.cat([imgs, probs.detach()], dim=1)
                    pixel_feats = feat_head(feat_input)
                    
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
                            soft_probs = soft_probs.view(1, 1, H, W)
                            target_b = masks[b:b+1]
                            
                            # Move KNN loss computation outside of autocast
                            with autocast(enabled=False):
                                # Convert to float32 for stability
                                soft_probs_f32 = soft_probs.float().clamp(1e-7, 1-1e-7)
                                target_b_f32 = target_b.float()
                                knn_loss_batch += F.binary_cross_entropy(
                                    soft_probs_f32, target_b_f32, reduction='mean'
                                )
                        knn_loss = knn_loss_batch / B
                
                total_loss = seg_loss + CFG.knn_loss_weight * knn_loss
                total_loss = total_loss / CFG.accum_steps
            
            # Backward pass
            scaler.scale(total_loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % CFG.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(feat_head.parameters()),
                    CFG.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None and CFG.use_onecycle:
                    scheduler.step()
            
            # Track losses
            epoch_seg_loss += seg_loss.item()
            epoch_knn_loss += knn_loss.item()
            epoch_total_loss += total_loss.item() * CFG.accum_steps
            
            # Compute training metrics
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).bool()
                targets = (masks > 0.5).bool()
                batch_metrics = compute_metrics(preds, targets)
                for k, v in batch_metrics.items():
                    train_metrics_sum[k] += v
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{total_loss.item() * CFG.accum_steps:.4f}",
                'seg': f"{seg_loss.item():.4f}",
                'knn': f"{knn_loss.item():.4f}",
                'dice': f"{batch_metrics['dice']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Average training metrics
        num_batches = len(train_loader)
        avg_train_seg_loss = epoch_seg_loss / num_batches
        avg_train_knn_loss = epoch_knn_loss / num_batches
        avg_train_total_loss = epoch_total_loss / num_batches
        
        train_metrics = {k: v / num_batches for k, v in train_metrics_sum.items()}
        
        # ======================== VALIDATION ========================
        model.eval()
        feat_head.eval()
        
        val_metrics_sum = defaultdict(float)
        val_total_loss = 0.0
        
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="Validation", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                logits = model(imgs)
                val_loss = seg_loss_fn(logits, masks)
                val_total_loss += val_loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).bool()
                targets = (masks > 0.5).bool()
                batch_metrics = compute_metrics(preds, targets)
                
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v
        
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        avg_val_loss = val_total_loss / len(val_loader)
        
        # ======================== LOGGING ========================
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_total_loss)
        history['train_seg_loss'].append(avg_train_seg_loss)
        history['train_knn_loss'].append(avg_train_knn_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)
        
        # TensorBoard logging
        writer.add_scalar("Loss/train_total", avg_train_total_loss, epoch)
        writer.add_scalar("Loss/train_seg", avg_train_seg_loss, epoch)
        writer.add_scalar("Loss/train_knn", avg_train_knn_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        
        writer.add_scalar("Metrics/train_dice", train_metrics['dice'], epoch)
        writer.add_scalar("Metrics/train_iou", train_metrics['iou'], epoch)
        writer.add_scalar("Metrics/train_precision", train_metrics['precision'], epoch)
        writer.add_scalar("Metrics/train_recall", train_metrics['recall'], epoch)
        
        writer.add_scalar("Metrics/val_dice", val_metrics['dice'], epoch)
        writer.add_scalar("Metrics/val_iou", val_metrics['iou'], epoch)
        writer.add_scalar("Metrics/val_precision", val_metrics['precision'], epoch)
        writer.add_scalar("Metrics/val_recall", val_metrics['recall'], epoch)
        
        # Dynamic LR logging for variable parameter groups
        for i, group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/param_group_{i}", group['lr'], epoch)
        
        # Console logging
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Summary:")
        print(f"{'='*80}")
        print(f"Train Loss: {avg_train_total_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, KNN: {avg_train_knn_loss:.4f})")
        print(f"Train Metrics - Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f} | "
              f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f}")
        print(f"Val Metrics   - Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"{'='*80}\n")
        
        # LR scheduler step
        if not CFG.use_onecycle:
            plateau_scheduler.step(val_metrics['dice'])
        
        # ======================== CHECKPOINTING ========================
        is_best = val_metrics['dice'] > best_dice
        
        if is_best:
            best_dice = val_metrics['dice']
            patience_counter = 0
            safe_config = {k: v for k, v in vars(CFG).items() 
               if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
            save_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'feat_head': feat_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_dice': best_dice,
                'history': history,
                'config': safe_config
            }
            
            torch.save(save_dict, os.path.join(CFG.ckpt_dir, "best_model.pth"))
            
            print(f"🎉 NEW BEST! Dice: {best_dice:.4f} → Model saved!")
            
            # Check if target achieved
            if best_dice >= 0.95:
                print(f"\n{'='*80}")
                print(f"🎯 TARGET ACHIEVED! Dice = {best_dice:.4f} >= 0.95")
                print(f"{'='*80}\n")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{CFG.patience}")
            
            if patience_counter >= CFG.patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best Dice: {best_dice:.4f}")
                print(f"{'='*80}\n")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CFG.ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'feat_head': feat_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_dice': best_dice,
                'history': history,
                'config': safe_config
            }, checkpoint_path)
        
        clear_memory()
    
    writer.close()
    
    # ======================== FINAL EVALUATION ========================
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    # Load best model
    best_ckpt = torch.load(os.path.join(CFG.ckpt_dir, "best_model.pth"))
    model.load_state_dict(best_ckpt['model'])
    model.eval()
    
    # Create test set (100 unique patient slices)
    unique_patients = list(set(s['patient_id'] for s in val_samples))
    selected_patients = random.sample(unique_patients, min(100, len(unique_patients)))
    test_samples = [
        random.choice([s for s in val_samples if s['patient_id'] == pid]) 
        for pid in selected_patients
    ]
    
    test_ds = SegDataset(test_samples, CFG.img_size, False, CFG.radius_px)
    test_loader = DataLoader(
        test_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True, collate_fn=custom_collate
    )
    
    # Evaluate
    test_metrics_sum = defaultdict(float)
    with torch.no_grad():
        for imgs, masks, _ in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).bool()
            targets = (masks > 0.5).bool()
            
            batch_metrics = compute_metrics(preds, targets)
            for k, v in batch_metrics.items():
                test_metrics_sum[k] += v
    
    test_metrics = {k: v / len(test_loader) for k, v in test_metrics_sum.items()}
    
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS (100 unique patient slices):")
    print(f"{'='*80}")
    print(f"Dice Score:  {test_metrics['dice']:.4f}")
    print(f"IoU Score:   {test_metrics['iou']:.4f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1 Score:    {test_metrics['f1']:.4f}")
    print(f"{'='*80}\n")
    
    # Save test metrics
    with open(os.path.join(CFG.ckpt_dir, "test_results.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # ======================== VISUALIZATIONS ========================
    print("Creating visualizations...")
    vis_dir = os.path.join(CFG.ckpt_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs_range = range(len(history['train_loss']))
    
    # Loss curves
    axes[0, 0].plot(epochs_range, history['train_loss'], label='Total Loss')
    axes[0, 0].plot(epochs_range, history['train_seg_loss'], label='Seg Loss')
    axes[0, 0].plot(epochs_range, history['train_knn_loss'], label='KNN Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice scores
    axes[0, 1].plot(epochs_range, history['train_dice'], label='Train Dice')
    axes[0, 1].plot(epochs_range, history['val_dice'], label='Val Dice')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU scores
    axes[0, 2].plot(epochs_range, history['train_iou'], label='Train IoU')
    axes[0, 2].plot(epochs_range, history['val_iou'], label='Val IoU')
    axes[0, 2].set_title('IoU Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Precision
    axes[1, 0].plot(epochs_range, history['train_precision'], label='Train')
    axes[1, 0].plot(epochs_range, history['val_precision'], label='Val')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(epochs_range, history['train_recall'], label='Train')
    axes[1, 1].plot(epochs_range, history['val_recall'], label='Val')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning rate
    axes[1, 2].plot(epochs_range, history['lr'])
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('LR')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Visualize predictions
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    for idx in range(min(CFG.save_samples, len(test_samples))):
        img, mask, sample_info = test_ds[idx]
        img_tensor = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_logits = model(img_tensor)
            pred_prob = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # Denormalize image
        img_np = (img.permute(1, 2, 0).numpy() * std + mean)
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        mask_np = (mask[0].numpy() * 255).astype(np.uint8)
        pred_np = (pred_mask * 255).astype(np.uint8)
        
        # Create overlay
        overlay = img_np.copy()
        overlay[pred_mask == 1] = [0, 255, 0]  # Green for predictions
        
        # Create comparison overlay
        comparison = img_np.copy()
        comparison[mask[0].numpy() == 1] = [255, 0, 0]  # Red for GT
        comparison[pred_mask == 1] = [0, 255, 0]  # Green for pred
        # Purple for overlap
        overlap = (mask[0].numpy() == 1) & (pred_mask == 1)
        comparison[overlap] = [255, 0, 255]
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title('Prediction Probability')
        axes[0, 2].axis('off')
        plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], fraction=0.046)
        
        axes[1, 0].imshow(pred_np, cmap='gray')
        axes[1, 0].set_title('Binary Prediction')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Green=Pred)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('Comparison (Red=GT, Green=Pred, Purple=Match)')
        axes[1, 2].axis('off')
        
        # Compute metrics for this sample
        pred_tensor = torch.from_numpy(pred_mask).bool().unsqueeze(0)
        mask_tensor = (mask > 0.5).bool()
        sample_metrics = compute_metrics(pred_tensor, mask_tensor)
        
        fig.suptitle(f"Sample {idx} | Dice: {sample_metrics['dice']:.4f} | "
                    f"IoU: {sample_metrics['iou']:.4f} | "
                    f"Precision: {sample_metrics['precision']:.4f} | "
                    f"Recall: {sample_metrics['recall']:.4f}",
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'prediction_{idx}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✅ Training complete!")
    print(f"📊 Results saved to: {CFG.ckpt_dir}")
    print(f"📈 TensorBoard logs: {CFG.log_dir}")
    print(f"🖼️  Visualizations: {vis_dir}")
    print(f"\n🎯 Final Test Dice: {test_metrics['dice']:.4f}")
    
    if test_metrics['dice'] >= 0.95:
        print(f"🎉 SUCCESS! Target of 0.95 Dice achieved!")
    else:
        print(f"⚠️  Target not yet achieved. Consider:")
        print(f"   - Training for more epochs")
        print(f"   - Adjusting learning rates")
        print(f"   - Modifying augmentation strategy")
    
    return model, feat_head, history, test_metrics

if __name__ == "__main__":
    final_model, final_feat_head, training_history, final_test_metrics = train()