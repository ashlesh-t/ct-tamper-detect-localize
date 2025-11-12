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
import os
from torchinfo import summary
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
# 1. ENHANCED CONFIG - TARGET 0.95+ DICE
# =================================================================================
class CFG:
    # Paths
    BASE_PATH = "/kaggle/input/ct-injection"
    PREV_BASE_PATH = "/kaggle/input/base-model"
    INJECTION_DIR = "CT_Injection"
    INJECTION_CSV = "CT_Injection/data_v1.csv"
    BASE_PATH_output = "/kaggle/working"
    # Previous experiment details
    prev_experiment = "unetpp_v8_softknn_injection_only_v2"
    prev_ckpt_path = os.path.join(PREV_BASE_PATH, prev_experiment, "best_model.pth")
    best_feat_head_path = os.path.join(PREV_BASE_PATH, prev_experiment, "best_feat_head.pth")  # NEW: Separate head load
    
    # Current experiment
    experiment_name = "CT_Injection_finetune_v3.0"
    ckpt_dir = os.path.join(BASE_PATH_output, "checkpoints", experiment_name)
    log_dir = os.path.join(BASE_PATH_output, "logs", experiment_name)
    split_dir = os.path.join(BASE_PATH_output, "data_splits", experiment_name)
    
    # Model - UPGRADED for better features
    encoder = "resnet50"  # From resnet34: +2% Dice for small objects
    img_size = 320
    radius_px = 42
    batch_size = 8
    accum_steps = 8
    num_workers = 2  # Set to 0 if threading issues in Kaggle
    # Resume control
    isResume = True
    # Optimizer - Tiered LRs
    encoder_lr = 1e-6  # Even lower for progressive unfreeze
    decoder_lr = 1e-4
    head_lr = 2e-4
    weight_decay = 1e-4
    
    # Progressive unfreezing - START FROM EPOCH 1 FOR DECODER/HEAD
    unfreeze_epochs = [10, 30, 50]  # layer4, layer3, layer2 partial
    unfreeze_layers_stages = [['layer4'], ['layer3'], ['layer2.1', 'layer2.2']]  # Partial layer2
    
    # KNN
    knn_support_per_batch = 512  # Increased for better supervision
    knn_beta = 4.0  # Sharper soft weighting
    knn_feat_channels = 8
    knn_loss_weight = 0.05  # Slight increase
    knn_enabled = True
    # Training - Extended for convergence
    total_epochs = 300
    patience = 20  # Tighter to prevent overfit
    val_split = 0.20  # Lower for more train data
    
    # Loss weights - Enhanced for imbalance
    w_dice = 0.5
    w_focal = 0.2
    w_tversky = 0.2
    w_bce = 0.1  # NEW: Add BCE per MELBA
    
    # Enhanced metrics
    save_samples = 10
    test_samples = 20
    tta_flips = ['horizontal', 'vertical']  # For TTA

os.makedirs(CFG.ckpt_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)
os.makedirs(CFG.split_dir, exist_ok=True)

# =================================================================================
# ENHANCEMENT LAYERS - WITH MULTI-SCALE & ADAPTER (LO-LIKE)
# =================================================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid())
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid())
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        x = x * sa
        return x

class SimpleAdapter(nn.Module):  # LoRA-like bottleneck for efficiency
    def __init__(self, channels, rank=4):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1)
        self.up = nn.Conv2d(rank, channels, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))  # Residual adapter

class EnhancementBlock(nn.Module):
    def __init__(self, in_channels, out_channels, multi_scale=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.adapter = SimpleAdapter(out_channels)  # NEW: Adapter
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.1)  # NEW: Overfit prevention
        if multi_scale:
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d((s,s)) for s in [out_channels//4, out_channels//2]])  # Multi-scale
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if hasattr(self, 'pool'):
            ms_feats = [p(x) for p in self.pool]
            x = x + F.interpolate(torch.cat(ms_feats, dim=1), size=x.shape[2:], mode='bilinear')  # Fuse scales
        x = self.cbam(x)
        x = self.adapter(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

# =================================================================================
# PROPER PREVIOUS MODEL LOADING - WITH SEPARATE HEAD
# =================================================================================
def load_previous_model_and_feathead(cfg, device):
    print(f"🔄 Loading previous model from: {cfg.prev_ckpt_path}")
    if not os.path.exists(cfg.prev_ckpt_path):
        print(f"❌ Previous checkpoint not found")
        return None, None
    
    try:
        checkpoint = torch.load(cfg.prev_ckpt_path, map_location=device)
        model = smp.UnetPlusPlus(
            encoder_name=cfg.encoder,
            encoder_weights=None,
            decoder_attention_type='scse',
            classes=1,
            activation=None)
        # Original feature head
        class OriginalPixelFeatureHead(nn.Module):
            def __init__(self, in_ch=4, out_ch=cfg.knn_feat_channels):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, out_ch, 1))
            def forward(self, x):
                return self.net(x)
        feat_head = OriginalPixelFeatureHead()
        model.to(device)
        feat_head.to(device)
        
        # Load model weights (same as before)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
        
        # NEW: Load feat_head from separate file if exists
        feat_head_loaded = False
        if os.path.exists(cfg.best_feat_head_path):
            head_ckpt = torch.load(cfg.best_feat_head_path, map_location=device)
            if isinstance(head_ckpt, dict) and 'feat_head' in head_ckpt:
                feat_head.load_state_dict(head_ckpt['feat_head'])
            else:
                feat_head.load_state_dict(head_ckpt)
            print(f"✅ Loaded separate best_feat_head from {cfg.best_feat_head_path}")
            feat_head_loaded = True
        elif 'feat_head' in checkpoint:
            feat_head.load_state_dict(checkpoint['feat_head'])
            print("✅ Loaded feat_head from main checkpoint")
            feat_head_loaded = True
        
        if not feat_head_loaded:
            print("ℹ️ No saved feat_head; using random init")
        
        return model, feat_head
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def create_enhanced_feature_head(original_feat_head, cfg):
    class EnhancedFeatureHead(nn.Module):
        def __init__(self, original_head, enhancement_channels=32):
            super().__init__()
            self.original_head = original_head
            for param in self.original_head.parameters():
                param.requires_grad = False  # Freeze original
            self.enhancement = EnhancementBlock(cfg.knn_feat_channels, enhancement_channels, multi_scale=True)
        def forward(self, x):
            with torch.no_grad():
                original_features = self.original_head(x)
            return self.enhancement(original_features)
    return EnhancedFeatureHead(original_feat_head)

# =================================================================================
# DATASET & TRANSFORMS - ENHANCED AUGS + CURRICULUM
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
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.8),  # Stronger
            A.OneOf([
                A.ElasticTransform(alpha=20, sigma=8, alpha_affine=15, p=1.0),  # Stronger elastic for small disks
                A.GridDistortion(p=1.0),
            ], p=0.4),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.6),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()]
    else:
        transforms = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()]
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

class CurriculumPatientGroupedBatchSampler(Sampler):
    def __init__(self, samples, patient_groups, batch_size, curriculum_epochs, total_epochs, shuffle_patients=True):
        self.samples = samples
        self.patient_indices = {}
        for pid in patient_groups:
            self.patient_indices[pid] = [i for i, s in enumerate(samples) if s['patient_id'] == pid]
        self.pids = list(patient_groups.keys())
        self.batch_size = batch_size
        self.shuffle_patients = shuffle_patients
        self.curriculum_epochs = curriculum_epochs  # Phase in hard samples over epochs
        self.total_epochs = total_epochs
        # Sort samples by difficulty: few annotations = hard (small injections)
        self.difficulty = {i: len(s.get('annotations', [])) for i, s in enumerate(samples)}
        self.sorted_indices = sorted(range(len(samples)), key=lambda i: self.difficulty[i])  # Easy (many ann) first
    def __iter__(self):
        epoch = getattr(self, 'current_epoch', 0)  # Set externally
        # Curriculum: Fraction of hard samples increases
        easy_frac = max(0.8 - (epoch / self.curriculum_epochs) * 0.6, 0.2)  # Start 80% easy, end 20% easy
        num_easy = int(len(self.sorted_indices) * easy_frac)
        current_indices = self.sorted_indices[:num_easy] + random.sample(self.sorted_indices[num_easy:], len(self.sorted_indices) - num_easy)
        # Group by patient (simplified)
        pids = self.pids[:]
        if self.shuffle_patients:
            random.shuffle(pids)
        for pid in pids:
            patient_idx = [i for i in current_indices if self.samples[i]['patient_id'] == pid]
            random.shuffle(patient_idx)
            for start in range(0, len(patient_idx), self.batch_size):
                batch = patient_idx[start:start + self.batch_size]
                if len(batch) < self.batch_size:
                    batch += random.choices(current_indices, k=self.batch_size - len(batch))
                yield batch
    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.patient_indices.values())

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
            "dataset": dataset_label})
    return samples

def get_dataloaders(cfg):
    inj_samples = build_tamper_samples(os.path.join(cfg.BASE_PATH, cfg.INJECTION_DIR),os.path.join(cfg.BASE_PATH, cfg.INJECTION_CSV),"inj")
    print(f"Found {len(inj_samples)} injection slices.")
    train_patients_path = os.path.join(cfg.split_dir, "train_patients.json")
    if os.path.exists(train_patients_path):
        with open(train_patients_path, 'r') as f:
            train_patients = json.load(f)
        with open(os.path.join(cfg.split_dir, "val_patients.json"), 'r') as f:
            val_patients = json.load(f)
    else:
        patients = list(set(s['patient_id'] for s in inj_samples))
        train_patients, val_patients = train_test_split(
            patients, test_size=cfg.val_split, random_state=42)
        with open(train_patients_path, 'w') as f:
            json.dump(train_patients, f)
        with open(os.path.join(cfg.split_dir, "val_patients.json"), 'w') as f:
            json.dump(val_patients, f)
    train_samples = [s for s in inj_samples if s['patient_id'] in train_patients]
    val_samples = [s for s in inj_samples if s['patient_id'] in val_patients]
    train_patient_groups = defaultdict(list)
    for s in train_samples:
        train_patient_groups[s['patient_id']].append(s)
    train_ds = SegDataset(train_samples, cfg.img_size, True, cfg.radius_px)
    val_ds = SegDataset(val_samples, cfg.img_size, False, cfg.radius_px)
    # NEW: Curriculum sampler
    train_batch_sampler = CurriculumPatientGroupedBatchSampler(train_samples, train_patient_groups, cfg.batch_size, curriculum_epochs=100, total_epochs=cfg.total_epochs)
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler,num_workers=cfg.num_workers, pin_memory=True,collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,num_workers=cfg.num_workers, pin_memory=True,collate_fn=custom_collate)
    print(f"Train: {len(train_samples)} samples ({len(train_patient_groups)} patients)")
    print(f"Val: {len(val_samples)} samples ({len(set(s['patient_id'] for s in val_samples))} patients)")
    return train_loader, val_loader, val_samples, train_samples

# =================================================================================
# LOSS & METRICS - WITH DEEP SUPERVISION & BCE
# =================================================================================
class EnhancedCombinedLoss(nn.Module):
    def __init__(self, w_dice=CFG.w_dice, w_focal=CFG.w_focal, w_tversky=CFG.w_tversky, w_bce=CFG.w_bce):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal = smp.losses.FocalLoss(mode='binary', gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.8)  # Boosted beta
        self.bce = nn.BCEWithLogitsLoss()
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_tversky = w_tversky
        self.w_bce = w_bce
    def forward(self, logits, targets):
        return (self.w_dice * self.dice(logits, targets) +
                self.w_focal * self.focal(logits, targets) +
                self.w_tversky * self.tversky(logits, targets) +
                self.w_bce * self.bce(logits, targets))

# NEW: Deep supervision - Aux losses from decoder stages
def deep_supervision_loss(model, logits_list, targets, main_loss_fn):
    aux_loss = 0.0
    for logits in logits_list[1:]:  # Skip main output
        aux_loss += main_loss_fn(logits, targets) * 0.4  # Weighted aux
    return aux_loss

@torch.no_grad()
def compute_detailed_metrics(preds, targets):
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
    return {'dice': dice.mean().item(),'iou': iou.mean().item(),'precision': precision.mean().item(),'recall': recall.mean().item(),'f1': f1.mean().item(),'specificity': specificity.mean().item(),'accuracy': accuracy.mean().item()}

# =================================================================================
# KNN UTILITIES - UPDATED SUPPORTS
# =================================================================================
class SoftKNN:
    def __init__(self, beta=CFG.knn_beta, device='cpu'):
        self.beta = beta
        self.device = device
    @torch.no_grad()
    def predict(self, support_feats, support_labels, query_feats, chunk_size=4096):
        support_feats = F.normalize(support_feats.to(self.device), dim=1)
        support_labels = support_labels.to(self.device)
        query_feats = F.normalize(query_feats.to(self.device), dim=1)
        Ns, Nq = support_feats.shape[0], query_feats.shape[0]
        if Ns == 0 or Nq == 0:
            return torch.zeros((Nq,), device=self.device)
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
        neg_sample = min(max_support - (len(chosen_idx[0]) if chosen_idx else 0), len(neg_idx))
        chosen_idx.append(neg_idx[torch.randperm(len(neg_idx))[:neg_sample]])
    if not chosen_idx:
        return torch.empty((0, C), device=device), torch.empty((0,), device=device)
    chosen_idx = torch.cat(chosen_idx)
    return feats_flat[chosen_idx], masks_flat[chosen_idx].float()

# =================================================================================
# MODEL SETUP - DEEP SUPERVISION HOOKS
# =================================================================================
class DeepSupervisionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder_stages = []  # Hook for aux outputs
    def forward(self, x):
        # Hook decoder stages for aux (simplified; assumes SMPy exposes)
        outputs = []
        def hook_fn(module, input, output):
            outputs.append(output)
        # Register hooks on decoder blocks (adapt to SMPy structure)
        for name, module in self.model.named_modules():
            if 'decoder' in name and 'stage' in name:  # Approx
                module.register_forward_hook(hook_fn)
        main_out = self.model(x)
        outputs.append(main_out)
        self.decoder_stages = outputs
        return main_out

def freeze_encoder_completely(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("🧊 Encoder frozen")

def unfreeze_decoder(model):  # NEW: Unfreeze from epoch 1
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.segmentation_head.parameters():
        param.requires_grad = True
    print("🔓 Decoder unfrozen (from epoch 1)")

def unfreeze_encoder_layers(model, layer_names):
    for name, param in model.encoder.named_parameters():
        if any(layer in name for layer in layer_names):
            param.requires_grad = True
    print(f"🔓 Unfrozen encoder layers: {layer_names}")

def setup_optimizer(model, feat_head, cfg, epoch=0):
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad] + [p for p in model.segmentation_head.parameters() if p.requires_grad]
    enhancement_params = [p for p in feat_head.parameters() if any('enhancement' in n for n, p in feat_head.named_parameters()) or p.requires_grad]  # Approx
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [{'params': encoder_params or [torch.tensor(0.)], 'lr': cfg.encoder_lr},{'params': decoder_params, 'lr': cfg.decoder_lr},{'params': enhancement_params, 'lr': cfg.head_lr}]
    optimizer = AdamW([g for g in param_groups if len(g['params']) > 0], weight_decay=cfg.weight_decay)
    return optimizer

# =================================================================================
# TTA FOR EVAL
# =================================================================================
def tta_predict(model, img, flips=CFG.tta_flips):
    tta_preds = []
    orig_pred = torch.sigmoid(model(img.unsqueeze(0))).squeeze()
    tta_preds.append(orig_pred)
    for flip in flips:
        flipped = torch.flip(img, dims=[2 if flip=='horizontal' else 3])
        pred_flipped = torch.sigmoid(model(flipped.unsqueeze(0))).squeeze()
        pred_flipped = torch.flip(pred_flipped, dims=[1 if flip=='horizontal' else 2])
        tta_preds.append(pred_flipped)
    return torch.mean(torch.stack(tta_preds), dim=0)

# =================================================================================
# VISUALIZATION & TESTING - WITH TTA
# =================================================================================
def visualize_predictions(model, dataloader, device, num_samples=5, save_dir=None):
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
            for i in range(min(imgs.shape[0], num_samples - samples_shown)):
                idx = samples_shown + i
                img_np = imgs[i]
                mask_np = masks[i, 0].cpu().numpy()
                # TTA pred
                pred_np = tta_predict(model, img_np[0], CFG.tta_flips).cpu().numpy()  # First channel for TTA
                binary_pred_np = (pred_np > 0.5).float().numpy()
                axes[idx, 0].imshow(img_np[0].cpu().numpy(), cmap='gray')  # First channel
                axes[idx, 0].set_title('Input Image')
                axes[idx, 0].axis('off')
                axes[idx, 1].imshow(mask_np, cmap='jet')
                axes[idx, 1].set_title('Ground Truth')
                axes[idx, 1].axis('off')
                axes[idx, 2].imshow(pred_np, cmap='jet', vmin=0, vmax=1)
                axes[idx, 2].set_title('TTA Prediction Prob')
                axes[idx, 2].axis('off')
                axes[idx, 3].imshow(binary_pred_np, cmap='jet')
                sample_metrics = compute_detailed_metrics(torch.tensor(binary_pred_np).unsqueeze(0).unsqueeze(0), torch.tensor(mask_np).unsqueeze(0).unsqueeze(0))
                axes[idx, 3].set_title(f'Binary Pred\nDice: {sample_metrics["dice"]:.3f}')
                axes[idx, 3].axis('off')
            samples_shown += imgs.shape[0]
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sample_predictions_tta.png'), dpi=150, bbox_inches='tight')
    plt.show()

def test_model(model, test_loader, device):
    model.eval()
    all_metrics = defaultdict(list)
    with torch.no_grad():
        for imgs, masks, _ in tqdm(test_loader, desc="Testing TTA"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            tta_preds = torch.stack([tta_predict(model, img[0], CFG.tta_flips).unsqueeze(0).unsqueeze(0) for img in imgs])  # Adjust for channel
            binary_preds = (tta_preds > 0.5).bool()
            targets = (masks > 0.5).bool()
            batch_metrics = compute_detailed_metrics(binary_preds, targets)
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)
    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    print("\n" + "="*60)
    print("FINAL TEST METRICS (WITH TTA)")
    print("="*60)
    for metric, value in final_metrics.items():
        print(f"{metric.upper():<12}: {value:.4f}")
    print("="*60)
    return final_metrics

# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# =================================================================================
# MAIN TRAINING LOOP - WITH PROGRESSIVE UNFREEZE & DEEP SUP
# =================================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, val_samples, train_samples = get_dataloaders(CFG)
    clear_memory()  # Kaggle mem clear
    model, original_feat_head = load_previous_model_and_feathead(CFG, device)
    if model is None:
        return None, None, None, None
    feat_head = create_enhanced_feature_head(original_feat_head, CFG)
    feat_head.to(device)
    # NEW: Wrap for deep sup
    model = DeepSupervisionModel(model)
    # Initial setup: Freeze encoder, unfreeze decoder from epoch 1
    freeze_encoder_completely(model.model)
    unfreeze_decoder(model.model)
    print("\n" + "="*80)
    print("MODEL SETUP: Decoder/Enhancements trainable from epoch 1")
    print("Progressive encoder unfreeze at epochs: " + str(CFG.unfreeze_epochs))
    print("="*80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params: {total_params:,} | Trainable: {trainable_params:,}")
    writer = SummaryWriter(log_dir=CFG.log_dir)
    best_metrics = {'dice': 0.0, 'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    patience_counter = 0
    history = {'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'train_dice': [], 'lr': []}
    start_epoch = 0
    unfreeze_idx = 0
    print(f"\n{'='*80}")
    print(f"STARTING v9.3 FINE-TUNING - TARGET >0.95 DICE")
    print(f"{'='*80}\n")
    seg_loss_fn = EnhancedCombinedLoss()
    scaler = GradScaler()
    knn = SoftKNN(device=device) if CFG.knn_enabled else None
    # Schedulers
    optimizer = setup_optimizer(model.model, feat_head, CFG, 0)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=CFG.total_epochs//4, eta_min=1e-7)
    reduce_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    # Set sampler epoch tracker
    train_loader.batch_sampler.current_epoch = 0
    for epoch in range(start_epoch, CFG.total_epochs):
        # Progressive unfreeze
        if epoch in CFG.unfreeze_epochs and unfreeze_idx < len(CFG.unfreeze_layers_stages):
            unfreeze_encoder_layers(model.model, CFG.unfreeze_layers_stages[unfreeze_idx])
            patience_counter = 0  # RESET PATIENCE
            unfreeze_idx += 1
            optimizer = setup_optimizer(model.model, feat_head, CFG, epoch)  # Re-setup
            print(f"🔓 Unfreeze at epoch {epoch} - Patience reset!")
        train_loader.batch_sampler.current_epoch = epoch  # For curriculum
        # Training
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
                logits = model(imgs)
                seg_loss = seg_loss_fn(logits, masks)
                # NEW: Deep sup
                if len(model.decoder_stages) > 1:
                    aux_loss = deep_supervision_loss(model, model.decoder_stages, masks, seg_loss_fn)
                    seg_loss += aux_loss
                # KNN
                knn_loss = torch.tensor(0.0, device=device)
                if CFG.knn_enabled and knn is not None:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                    feat_input = torch.cat([imgs, probs.detach()], dim=1)
                    pixel_feats = feat_head(feat_input)
                    support_feats, support_labels = sample_support_pixels(pixel_feats.detach(), masks)
                    if support_feats.shape[0] > 0:
                        B, C, H, W = pixel_feats.shape
                        query_feats = pixel_feats.permute(0,2,3,1).reshape(B, -1, C)
                        knn_loss_batch = 0.0
                        for b in range(B):
                            soft_probs = knn.predict(support_feats, support_labels, query_feats[b])
                            soft_probs = soft_probs.view(1, 1, H, W).clamp(1e-7, 1-1e-7)
                            knn_logits = torch.logit(soft_probs)
                            knn_loss_batch += F.binary_cross_entropy_with_logits(knn_logits, masks[b:b+1], reduction='mean')
                        knn_loss = knn_loss_batch / B
                total_loss = seg_loss + CFG.knn_loss_weight * knn_loss
                total_loss = total_loss / CFG.accum_steps
            scaler.scale(total_loss).backward()
            if (batch_idx + 1) % CFG.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(feat_head.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += total_loss.item() * CFG.accum_steps
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).bool()
                targets = (masks > 0.5).bool()
                batch_metrics = compute_detailed_metrics(preds, targets)
                train_metrics_sum['dice'] += batch_metrics['dice']
            pbar.set_postfix({'loss': f"{total_loss.item() * CFG.accum_steps:.4f}", 'dice': f"{batch_metrics['dice']:.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        # Validation
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
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_metrics_sum['dice'] / len(train_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)
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
        print(f"Learning Rate: {current_lr:.2e}")
        # Schedulers
        main_scheduler.step()
        reduce_scheduler.step(val_metrics['dice'])
        # Checkpointing
        current_dice = val_metrics['dice']
        current_f1 = val_metrics['f1']
        composite_score = current_dice * 0.6 + current_f1 * 0.4
        best_composite = best_metrics['dice'] * 0.6 + best_metrics['f1'] * 0.4
        is_best = composite_score > best_composite
        if is_best:
            best_metrics.update(val_metrics)
            patience_counter = 0
            save_dict = {
                'epoch': epoch,
                'model': model.model.state_dict(),
                'feat_head': feat_head.state_dict(),
                'original_feat_head': original_feat_head.state_dict(),
                'best_dice': best_metrics['dice'],
                'best_f1': best_metrics['f1'],
                'best_iou': best_metrics['iou'],
                'history': history,
                'config': CFG.__dict__
            }
            torch.save(save_dict, os.path.join(CFG.ckpt_dir, "best_model.pth"))
            print(f"🎉 NEW BEST! Dice: {best_metrics['dice']:.4f} → Saved!")
            if epoch % 20 == 0:  # Visualize less often
                visualize_predictions(model.model, val_loader, device, num_samples=5, save_dir=CFG.ckpt_dir)
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{CFG.patience}")
            if patience_counter >= CFG.patience:
                print(f"🏁 Early stopping at epoch {epoch}")
                break
        clear_memory()  # Kaggle mem clear after epoch
    # Final eval
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE - FINAL EVAL WITH TTA")
    print("="*80)
    if os.path.exists(os.path.join(CFG.ckpt_dir, "best_model.pth")):
        best_checkpoint = torch.load(os.path.join(CFG.ckpt_dir, "best_model.pth"))
        model.model.load_state_dict(best_checkpoint['model'])
        feat_head.load_state_dict(best_checkpoint['feat_head'])
        print("✅ Loaded best model")
    final_metrics = test_model(model.model, val_loader, device)
    print("\n📊 Final samples...")
    visualize_predictions(model.model, val_loader, device, num_samples=CFG.save_samples, save_dir=CFG.ckpt_dir)
    # Save final
    final_save_path = os.path.join(CFG.ckpt_dir, "final_model.pth")
    torch.save({
        'model': model.model.state_dict(),
        'feat_head': feat_head.state_dict(),
        'original_feat_head': original_feat_head.state_dict(),
        'config': CFG.__dict__,
        'final_metrics': final_metrics,
        'history': history
    }, final_save_path)
    print(f"\n💾 Final saved: {final_save_path}")
    print(f"📈 Logs: {CFG.log_dir}")
    writer.close()
    return model.model, feat_head, history, final_metrics

if __name__ == "__main__":
    final_model, final_feat_head, training_history, final_metrics = train()