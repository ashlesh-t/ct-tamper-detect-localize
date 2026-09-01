# binary_ct_training_enhanced_v5.py
import os
import random
import numpy as np
import json
from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from multiprocessing import cpu_count
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from numba import jit
import pandas as pd
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Speed: CUDA & Torch Optimizations
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Enhanced class weight computation
@jit(nopython=True)
def compute_sample_weights(labels, class_counts):
    inv_counts = 1.0 / class_counts.astype(np.float32)
    return inv_counts[labels]

# Enhanced Dataset with better caching
class FastPreprocessedCTDataset(Dataset):
    def __init__(self, samples, transform=None, img_size=224, mmap_mode='r', cache_size=256):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.mmap_mode = mmap_mode
        self._cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fpath, label, patient_id = sample["fpath"], sample["label"], sample["patient_id"]

        if fpath in self._cache:
            multi_channel = self._cache[fpath]
        else:
            try:
                multi_channel = np.load(fpath, mmap_mode=self.mmap_mode).astype(np.float32)
                if multi_channel.shape[-1] != 3:
                    if len(multi_channel.shape) == 2:
                        multi_channel = np.stack([multi_channel] * 3, axis=-1)
                    else:
                        multi_channel = multi_channel[:, :, :3]
                # Cache management
                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[fpath] = multi_channel
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                multi_channel = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel, torch.tensor(label, dtype=torch.long), patient_id

# Enhanced Transforms with more augmentation
DATASET_MEAN = [0.5, 0.5, 0.5]
DATASET_STD = [0.5, 0.5, 0.5]

class TrainTransforms:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3))], p=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

    def __call__(self, x_np):
        t = torch.from_numpy(x_np).permute(2,0,1).float()
        return self.augment(t)

class EvalTransforms:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

    def __call__(self, x_np):
        t = torch.from_numpy(x_np).permute(2,0,1).float()
        return self.transforms(t)

# Enhanced Activation
class HybridActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()

    def forward(self, x):
        return self.gelu(x) * 0.7 + self.selu(x) * 0.3

# Enhanced Model with attention
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            HybridActivation(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        return x * ca

class EnhancedCTModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, use_ml_head=True, use_attention=True):
        super().__init__()
        self.use_ml_head = use_ml_head
        self.use_attention = use_attention

        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.backbone = densenet.features
        num_backbone_features = 1024

        if use_attention:
            self.attention = AttentionModule(num_backbone_features)

        if use_ml_head:
            self.projection_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.BatchNorm1d(num_backbone_features), nn.Dropout(0.4),
                nn.Linear(num_backbone_features, 512), HybridActivation(),
                nn.BatchNorm1d(512), nn.Dropout(0.4),
                nn.Linear(512, 256), HybridActivation(),
                nn.BatchNorm1d(256), nn.Dropout(0.3),
            )
            classifier_input = 256
        else:
            self.projection_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            classifier_input = num_backbone_features

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128), HybridActivation(),
            nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        if self.use_attention:
            f = self.attention(f)
        p = self.projection_head(f)
        return self.classifier(p)

# Enhanced Loss with class balancing
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is None:
            self.alpha = torch.ones(inputs.size(1)).to(inputs.device)

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss

        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, focal_gamma=2.5, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = BalancedFocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, preds, targets):
        return self.alpha * self.focal_loss(preds, targets) + (1 - self.alpha) * self.ce_loss(preds, targets)

# Enhanced Optimizers & Schedulers
def get_optimizer_scheduler(model, phase, lr, weight_decay, max_steps, current_epoch=0, total_epochs=50, resume=False):
    if phase == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if not resume:
            print(f"Initializing OneCycleLR with max_steps={max_steps}")
            scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max_steps, pct_start=0.1, anneal_strategy='cos')
        else:
            scheduler = None  # Will be loaded from checkpoint
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        if not resume:
            print(f"Initializing CosineAnnealingLR with T_max={total_epochs}")
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=lr*0.01)
        else:
            scheduler = None

    return optimizer, scheduler

# Enhanced Balanced Sampling with Real Class Distribution
def create_balanced_dataset(samples, target_real_per_class=None, seed=42):
    """
    Create balanced dataset ensuring equal distribution from both real classes (0 and 1)
    and matching tampered classes (2 and 3)
    """
    np.random.seed(seed)

    # Separate by original labels
    real_class_0 = [s for s in samples if s['original_label'] == 0]
    real_class_1 = [s for s in samples if s['original_label'] == 1]
    tampered_class_2 = [s for s in samples if s['original_label'] == 2]
    tampered_class_3 = [s for s in samples if s['original_label'] == 3]

    print(f"Original distribution:")
    print(f"  Real Class 0: {len(real_class_0)}")
    print(f"  Real Class 1: {len(real_class_1)}")
    print(f"  Tampered Class 2: {len(tampered_class_2)}")
    print(f"  Tampered Class 3: {len(tampered_class_3)}")

    # Set default target if not provided
    if target_real_per_class is None:
        min_real_per_class = min(len(real_class_0), len(real_class_1))
        min_tampered_per_class = min(len(tampered_class_2), len(tampered_class_3))
        target_real_per_class = min(min_real_per_class, min_tampered_per_class)
        print(f"Auto-setting target_real_per_class to: {target_real_per_class}")

    # Sample from each class
    sampled_real_0 = random.sample(real_class_0, min(target_real_per_class, len(real_class_0)))
    sampled_real_1 = random.sample(real_class_1, min(target_real_per_class, len(real_class_1)))
    sampled_tampered_2 = random.sample(tampered_class_2, min(target_real_per_class, len(tampered_class_2)))
    sampled_tampered_3 = random.sample(tampered_class_3, min(target_real_per_class, len(tampered_class_3)))

    # Combine
    balanced_samples = sampled_real_0 + sampled_real_1 + sampled_tampered_2 + sampled_tampered_3
    random.shuffle(balanced_samples)

    print(f"Balanced dataset:")
    print(f"  Real Class 0: {len(sampled_real_0)}")
    print(f"  Real Class 1: {len(sampled_real_1)}")
    print(f"  Tampered Class 2: {len(sampled_tampered_2)}")
    print(f"  Tampered Class 3: {len(sampled_tampered_3)}")
    print(f"  Total: {len(balanced_samples)}")
    print(f"  Binary distribution - Real: {len(sampled_real_0) + len(sampled_real_1)}, "
          f"Tampered: {len(sampled_tampered_2) + len(sampled_tampered_3)}")

    return balanced_samples

# Enhanced Patient-wise Split with balanced sampling
def enhanced_patient_wise_split(all_samples, test_size=0.3, seed=42, split_dir=None,
                               target_real_per_class=None, create_test_set=False):
    """
    Enhanced split that maintains balanced class distribution in train/val/test
    """
    if split_dir and os.path.exists(os.path.join(split_dir, "enhanced_split_info.json")):
        print(f"Loading cached enhanced split from {split_dir}")
        with open(os.path.join(split_dir, "enhanced_split_info.json"), 'r') as f:
            info = json.load(f)

        # Reconstruct samples from saved info
        train_samples = info['train_samples']
        val_samples = info['val_samples']
        test_samples = info.get('test_samples', [])

        return train_samples, val_samples, test_samples, info

    print("Computing new enhanced patient-wise split with balanced sampling...")

    # Group by patient
    patient_to_samples = defaultdict(list)
    for s in tqdm(all_samples, desc="Mapping patients to samples"):
        patient_to_samples[s['patient_id']].append(s)

    # Get unique patients with their predominant class
    patient_info = []
    for pid, samples in patient_to_samples.items():
        labels = [s['label'] for s in samples]
        predominant_label = max(set(labels), key=labels.count)
        original_labels = [s['original_label'] for s in samples]
        predominant_original = max(set(original_labels), key=original_labels.count)
        patient_info.append({
            'patient_id': pid,
            'label': predominant_label,
            'original_label': predominant_original,
            'sample_count': len(samples)
        })

    # Convert to DataFrame for easier stratification
    patient_df = pd.DataFrame(patient_info)

    if create_test_set:
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            patient_df['patient_id'].values,
            test_size=test_size,
            random_state=seed,
            stratify=patient_df[['label', 'original_label']]
        )

        # Second split: train vs val
        train_val_df = patient_df[patient_df['patient_id'].isin(train_val_patients)]
        train_patients, val_patients = train_test_split(
            train_val_df['patient_id'].values,
            test_size=test_size/(1-test_size),  # Adjust test_size for second split
            random_state=seed,
            stratify=train_val_df[['label', 'original_label']]
        )
    else:
        # Simple train-val split
        train_patients, val_patients = train_test_split(
            patient_df['patient_id'].values,
            test_size=test_size,
            random_state=seed,
            stratify=patient_df[['label', 'original_label']]
        )
        test_patients = []

    # Collect samples
    train_samples = [s for pid in train_patients for s in patient_to_samples[pid]]
    val_samples = [s for pid in val_patients for s in patient_to_samples[pid]]
    test_samples = [s for pid in test_patients for s in patient_to_samples[pid]] if create_test_set else []

    # Apply balanced sampling to train set
    print("Applying balanced sampling to training set...")
    train_samples = create_balanced_dataset(train_samples, target_real_per_class, seed)

    # Also balance validation set if needed
    if len(val_samples) > 0:
        val_samples = create_balanced_dataset(val_samples,
                                            target_real_per_class=min(500, len(val_samples)//4),
                                            seed=seed)

    print(f"Final split:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples" if test_samples else "  Test: Using val as test")

    # Save split info
    split_info = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'split_params': {
            'test_size': test_size,
            'seed': seed,
            'target_real_per_class': target_real_per_class,
            'create_test_set': create_test_set
        }
    }

    if split_dir:
        print(f"Saving enhanced split to {split_dir}")
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "enhanced_split_info.json"), 'w') as f:
            json.dump(split_info, f, indent=2, default=str)

    return train_samples, val_samples, test_samples, split_info

# Enhanced Training Loop with better metrics tracking
def enhanced_train_model(model, train_loader, val_loader, device, ckpt_dir, hyperparams, resume=False):
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training phases
    adamw_epochs = 10
    sgd_early_epochs = 5
    sgd_epochs = hyperparams['total_epochs'] - adamw_epochs - sgd_early_epochs

    # Enhanced loss with class weights
    train_labels = [s['label'] for s in train_loader.dataset.samples]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], dtype=torch.float32).to(device)

    criterion = EnhancedCombinedLoss(alpha=0.7, focal_gamma=2.0, class_weights=class_weights)
    scaler = GradScaler()

    # Optimizers and schedulers
    steps_per_epoch = len(train_loader)
    adamw_steps = adamw_epochs * steps_per_epoch
    sgd_early_steps = sgd_early_epochs * steps_per_epoch

    # Initialize all optimizers and schedulers
    optimizer_adamw, scheduler_adamw = get_optimizer_scheduler(
        model, "adamw", hyperparams['adamw_lr'], hyperparams['weight_decay'],
        max_steps=adamw_steps, total_epochs=hyperparams['total_epochs'], resume=resume
    )
    optimizer_sgd_early, scheduler_sgd_early = get_optimizer_scheduler(
        model, "sgd", hyperparams['sgd_lr'], hyperparams['weight_decay'],
        max_steps=sgd_early_steps, total_epochs=hyperparams['total_epochs'], resume=resume
    )
    optimizer_sgd, scheduler_sgd = get_optimizer_scheduler(
        model, "sgd", hyperparams['sgd_lr'] * 0.1, hyperparams['weight_decay'],
        max_steps=hyperparams['total_epochs'], total_epochs=hyperparams['total_epochs'], resume=resume
    )

    # Enhanced history tracking
    history = defaultdict(list)
    val_probs_history = []

    # Checkpoint paths
    latest_ckpt = os.path.join(ckpt_dir, "enhanced_latest_model.pth")
    best_ckpt = os.path.join(ckpt_dir, "enhanced_best_model.pth")
    history_path = os.path.join(ckpt_dir, "enhanced_training_history.json")

    # Enhanced resume logic
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    patience = 12
    epochs_no_improve = 0
    current_phase = "adamw"  # Track current phase

    def verify_checkpoint(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
        print("Checkpoint contents:")
        for key in checkpoint.keys():
            if hasattr(checkpoint[key], 'shape' if torch.is_tensor(checkpoint[key]) else '__len__'):
                size = checkpoint[key].shape if torch.is_tensor(checkpoint[key]) else len(checkpoint[key])
                print(f"  {key}: {type(checkpoint[key])} with size/shape {size}")
            else:
                print(f"  {key}: {checkpoint[key]}")

        return checkpoint

    if resume and os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        checkpoint = verify_checkpoint(latest_ckpt)

        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        best_val_precision = checkpoint.get('best_val_precision', 0.0)
        history = checkpoint.get('history', defaultdict(list))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        current_phase = checkpoint.get('phase', 'adamw')

        # Load ALL optimizers and schedulers from checkpoint
        if 'optimizer_adamw_state_dict' in checkpoint:
            optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw_state_dict'])
            print("✓ Loaded AdamW optimizer state")
        if 'scheduler_adamw_state_dict' in checkpoint and scheduler_adamw is not None:
            scheduler_adamw.load_state_dict(checkpoint['scheduler_adamw_state_dict'])
            print("✓ Loaded AdamW scheduler state")
        if 'optimizer_sgd_early_state_dict' in checkpoint:
            optimizer_sgd_early.load_state_dict(checkpoint['optimizer_sgd_early_state_dict'])
            print("✓ Loaded SGD Early optimizer state")
        if 'scheduler_sgd_early_state_dict' in checkpoint and scheduler_sgd_early is not None:
            scheduler_sgd_early.load_state_dict(checkpoint['scheduler_sgd_early_state_dict'])
            print("✓ Loaded SGD Early scheduler state")
        if 'optimizer_sgd_state_dict' in checkpoint:
            optimizer_sgd.load_state_dict(checkpoint['optimizer_sgd_state_dict'])
            print("✓ Loaded SGD optimizer state")
        if 'scheduler_sgd_state_dict' in checkpoint and scheduler_sgd is not None:
            scheduler_sgd.load_state_dict(checkpoint['scheduler_sgd_state_dict'])
            print("✓ Loaded SGD scheduler state")

        print(f"✓ Resumed from epoch {start_epoch}, phase: {current_phase}, best_val_f1: {best_val_f1:.4f}")

    # Enhanced training loop
    for epoch in range(start_epoch, hyperparams['total_epochs']):
        # Select phase based on current epoch
        if epoch < adamw_epochs:
            phase = "adamw"
            optimizer = optimizer_adamw
            scheduler = scheduler_adamw
        elif epoch < adamw_epochs + sgd_early_epochs:
            phase = "sgd_early"
            optimizer = optimizer_sgd_early
            scheduler = scheduler_sgd_early
        else:
            phase = "sgd"
            optimizer = optimizer_sgd
            scheduler = scheduler_sgd

        # Training
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams['total_epochs']} [{phase.upper()}] Train")
        for imgs, labels, _ in train_pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']

        # Calculate training metrics
        train_metrics = calculate_detailed_metrics(all_labels, all_preds, epoch_loss / len(train_loader.dataset))

        # Validation
        val_metrics, val_probs = enhanced_validate_model(model, val_loader, criterion, device)

        # Store history
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        history['learning_rates'].append(current_lr)

        val_probs_history.append({
            'epoch': epoch + 1,
            'probs': val_probs['probs'],
            'labels': val_probs['labels'],
            'preds': val_probs['preds']
        })

        # Enhanced logging
        print(f"\nEpoch {epoch+1}/{hyperparams['total_epochs']} [{phase.upper()}] LR: {current_lr:.2e}")
        print(f"Train: Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.4f} "
              f"Prec={train_metrics['precision']:.4f} Rec={train_metrics['recall']:.4f} F1={train_metrics['f1']:.4f}")
        print(f"Val:   Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} "
              f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} "
              f"AUC={val_metrics['auc']:.4f}")

        # Enhanced early stopping based on both F1 and Precision
        current_val_f1 = val_metrics['f1']
        current_val_precision = val_metrics['precision']

        # Save best model (prioritizing F1 but also considering precision)
        improvement_threshold = 0.001

        if current_val_f1 > best_val_f1 + improvement_threshold or \
           (abs(current_val_f1 - best_val_f1) < improvement_threshold and current_val_precision > best_val_precision):

            best_val_f1 = max(current_val_f1, best_val_f1)
            best_val_precision = max(current_val_precision, best_val_precision)

            print(f"🎉 New best model! Val F1: {current_val_f1:.4f}, Precision: {current_val_precision:.4f}")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_f1': current_val_f1,
                'val_precision': current_val_precision,
                'val_auc': val_metrics['auc'],
                'history': dict(history),
                'hyperparams': hyperparams
            }, best_ckpt)

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (Best F1: {best_val_f1:.4f})")

        # Save latest checkpoint with ALL optimizer states
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_val_f1': best_val_f1,
            'best_val_precision': best_val_precision,
            'history': dict(history),
            'phase': phase,
            'epochs_no_improve': epochs_no_improve,
            'hyperparams': hyperparams
        }

        # Add all optimizer and scheduler states
        checkpoint_data['optimizer_adamw_state_dict'] = optimizer_adamw.state_dict()
        if scheduler_adamw is not None:
            checkpoint_data['scheduler_adamw_state_dict'] = scheduler_adamw.state_dict()

        checkpoint_data['optimizer_sgd_early_state_dict'] = optimizer_sgd_early.state_dict()
        if scheduler_sgd_early is not None:
            checkpoint_data['scheduler_sgd_early_state_dict'] = scheduler_sgd_early.state_dict()

        checkpoint_data['optimizer_sgd_state_dict'] = optimizer_sgd.state_dict()
        if scheduler_sgd is not None:
            checkpoint_data['scheduler_sgd_state_dict'] = scheduler_sgd.state_dict()

        torch.save(checkpoint_data, latest_ckpt)
        print(f"💾 Saved checkpoint for epoch {epoch+1}")

        # Save history with proper serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        save_json_serializable(dict(history), history_path)
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

    return dict(history), val_probs_history

def calculate_detailed_metrics(true_labels, predictions, loss):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }

def enhanced_validate_model(model, val_loader, criterion, device):
    """Enhanced validation with comprehensive metrics"""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.inference_mode(), autocast():
        for imgs, labels, _ in tqdm(val_loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    # Calculate metrics
    metrics = calculate_detailed_metrics(all_labels, all_preds, val_loss)

    # Add AUC
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = 0.5

    return metrics, {'probs': all_probs, 'labels': all_labels, 'preds': all_preds}

# Enhanced Evaluation with comprehensive analysis
def enhanced_evaluate_model(model, test_loader, device="cuda", ckpt_dir=None, threshold=0.5):
    """Comprehensive evaluation with detailed analysis"""
    model.eval()
    all_preds, all_labels, all_patients, all_probs = [], [], [], []
    sample_images = []

    with torch.inference_mode(), autocast():
        for batch_idx, (imgs, labels, patients) in enumerate(tqdm(test_loader, desc="Testing")):
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_patients.extend(patients)
            all_probs.extend(probs.cpu().numpy())

            # Collect sample images for visualization
            if len(sample_images) < 8 and batch_idx % 10 == 0:
                denorm_imgs = imgs * torch.tensor(DATASET_STD).view(3,1,1).to(device) + torch.tensor(DATASET_MEAN).view(3,1,1).to(device)
                denorm_imgs = torch.clamp(denorm_imgs, 0, 1).permute(0,2,3,1).cpu().numpy()
                sample_images.extend(denorm_imgs[:2])

    all_probs_tampered = [p[1] for p in all_probs]
    all_preds_bin = (np.array(all_probs_tampered) > threshold).astype(int)

    # Comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds_bin)
    precision = precision_score(all_labels, all_preds_bin, zero_division=0)
    recall = recall_score(all_labels, all_preds_bin, zero_division=0)
    f1 = f1_score(all_labels, all_preds_bin, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs_tampered)
    except:
        auc = 0.5

    # Additional metrics
    cm = confusion_matrix(all_labels, all_preds_bin)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    # Precision-Recall curve metrics
    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs_tampered)
    avg_precision = average_precision_score(all_labels, all_probs_tampered)

    report = classification_report(all_labels, all_preds_bin, output_dict=True, zero_division=0)

    print("\n" + "="*80)
    print("ENHANCED TEST EVALUATION")
    print("="*80)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"AUC-ROC:       {auc:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"Balanced Acc:  {balanced_accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds_bin, zero_division=0))

    # Enhanced visualization
    if ckpt_dir:
        create_comprehensive_plots(
            all_labels, all_probs_tampered, all_preds_bin, cm,
            sample_images, all_preds[:len(sample_images)] if sample_images else [],
            all_probs[:len(sample_images)] if sample_images else [], ckpt_dir
        )

    eval_results = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'roc_auc': auc, 'average_precision': avg_precision, 'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'probs': all_probs_tampered,
        'labels': all_labels,
        'preds': all_preds_bin.tolist(),
        'counts': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
    }

    if ckpt_dir:
        save_json_serializable(eval_results, os.path.join(ckpt_dir, "enhanced_eval_results.json"))

    return eval_results

def create_comprehensive_plots(true_labels, probs, preds, cm, sample_images, sample_preds, sample_probs, ckpt_dir):
    """Create comprehensive visualization plots without overlap"""

    # Create multiple separate figures instead of subplots within subplots
    plt.style.use('default')

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Real", "Tampered"], yticklabels=["Real", "Tampered"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(true_labels, probs)
    auc_score = roc_auc_score(true_labels, probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(true_labels, probs)
    avg_precision = np.trapz(precision, recall)
    plt.plot(recall, precision, color='green', lw=2, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve'); plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 4. Probability Distribution
    plt.figure(figsize=(8, 6))
    real_probs = [p for p, l in zip(probs, true_labels) if l == 0]
    tampered_probs = [p for p, l in zip(probs, true_labels) if l == 1]
    plt.hist(real_probs, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    plt.hist(tampered_probs, bins=50, alpha=0.7, label='Tampered', color='red', density=True)
    plt.xlabel('Probability of Tampered'); plt.ylabel('Density')
    plt.title('Probability Distribution'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "probability_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 5. Sample Predictions (if available)
    if sample_images and len(sample_images) >= 4:
        plt.figure(figsize=(12, 8))
        for i in range(min(4, len(sample_images))):
            plt.subplot(2, 2, i+1)
            plt.imshow(sample_images[i])
            pred_class = "Tampered" if sample_preds[i] == 1 else "Real"
            true_class = "Tampered" if true_labels[i] == 1 else "Real"
            conf = sample_probs[i][1] if pred_class == "Tampered" else sample_probs[i][0]
            color = 'green' if pred_class == true_class else 'red'
            plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {conf:.2f}", color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "sample_predictions.png"), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
# Enhanced K-Fold with balanced sampling
def enhanced_kfold_evaluation(model_class, all_samples, device, hyperparams, k=5, ckpt_dir=None):
    """Enhanced K-Fold with proper patient-wise splitting and balanced sampling"""

    # Group by patient
    patient_to_samples = defaultdict(list)
    for s in tqdm(all_samples, desc="Mapping patients to samples"):
        patient_to_samples[s['patient_id']].append(s)

    patient_ids = list(patient_to_samples.keys())
    patient_labels = [max(set([s['label'] for s in samples]), key=[s['label'] for s in samples].count)
                     for samples in patient_to_samples.values()]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_metrics = defaultdict(list)
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n--- Fold {fold+1}/{k} ---")

        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]

        # Collect samples
        train_samples_fold = [s for pid in train_patients for s in patient_to_samples[pid]]
        val_samples_fold = [s for pid in val_patients for s in patient_to_samples[pid]]

        # Balance training set
        train_samples_fold = create_balanced_dataset(train_samples_fold, target_real_per_class=1000)

        # Create datasets and loaders
        train_dataset = FastPreprocessedCTDataset(train_samples_fold, transform=TrainTransforms(224))
        val_dataset = FastPreprocessedCTDataset(val_samples_fold, transform=EvalTransforms(224))

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        # Train model
        model = model_class(num_classes=2, pretrained=True).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=10 * len(train_loader))
        criterion = EnhancedCombinedLoss()

        # Quick training
        model.train()
        for epoch in range(10):
            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        fold_probs, fold_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                fold_probs.extend(probs)
                fold_labels.extend(labels.cpu().numpy())

        fold_preds = (np.array(fold_probs) > 0.5).astype(int)
        fold_metrics['accuracy'].append(accuracy_score(fold_labels, fold_preds))
        fold_metrics['precision'].append(precision_score(fold_labels, fold_preds, zero_division=0))
        fold_metrics['recall'].append(recall_score(fold_labels, fold_preds, zero_division=0))
        fold_metrics['f1'].append(f1_score(fold_labels, fold_preds, zero_division=0))
        fold_metrics['auc'].append(roc_auc_score(fold_labels, fold_probs))

        print(f"Fold {fold+1}: Acc={fold_metrics['accuracy'][-1]:.4f}, "
              f"Prec={fold_metrics['precision'][-1]:.4f}, F1={fold_metrics['f1'][-1]:.4f}, "
              f"AUC={fold_metrics['auc'][-1]:.4f}")

        all_fold_results.append({
            'fold': fold+1,
            'probs': fold_probs,
            'labels': fold_labels,
            'metrics': {k: v[-1] for k, v in fold_metrics.items()}
        })

    # Calculate averages
    avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    std_metrics = {k: np.std(v) for k, v in fold_metrics.items()}

    print(f"\nK-Fold Cross Validation Results ({k}-fold):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    if ckpt_dir:
        kfold_results = {
            'averages': avg_metrics,
            'std_devs': std_metrics,
            'folds': all_fold_results
        }
        with open(os.path.join(ckpt_dir, "enhanced_kfold_results.json"), 'w') as f:
            json.dump(kfold_results, f, indent=2)

    return avg_metrics, std_metrics, all_fold_results

# Load Data (same as before)
def load_preprocessed_data(preprocessed_path):
    all_samples = []
    class_mapping = {"0": 0, "1": 0, "2": 1, "3": 1}

    label_dirs = glob(os.path.join(preprocessed_path, "[0-3]"))
    for label_dir in tqdm(label_dirs, desc="Processing label directories"):
        original_label = os.path.basename(label_dir)
        binary_label = class_mapping.get(original_label)
        if binary_label is None:
            continue

        patient_dirs = glob(os.path.join(label_dir, "*"))
        for patient_dir in tqdm(patient_dirs, desc=f"Processing patients in label {original_label}", leave=False):
            if not os.path.isdir(patient_dir):
                continue
            patient_id = os.path.basename(patient_dir)
            npy_files = glob(os.path.join(patient_dir, "*.npy"))
            for fpath in tqdm(npy_files, desc=f"Collecting .npy files for patient {patient_id}", leave=False):
                all_samples.append({
                    "fpath": fpath,
                    "label": binary_label,
                    "patient_id": f"{original_label}_{patient_id}",
                    "original_label": int(original_label)
                })

    print(f"Loaded {len(all_samples)} samples")
    return all_samples

# Enhanced Main Function
def main():
    PREPROCESSED_PATH = "/content/drive/MyDrive/preprocessed_data_v3"

    # Enhanced hyperparameters
    hyperparams = {
        'img_size': 224,
        'use_ml_head': True,
        'use_attention': True,
        'total_epochs': 50,
        'adamw_lr': 4e-4,  # Increased learning rate
        'sgd_lr': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'batch_size': 16,
        'resume': True,
        'target_real_per_class': 1200,  # Balanced sampling
    }

    ckpt_dir = "/content/drive/MyDrive/capstone_models/binary_ct_enhanced_v5"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("🚀 Starting Enhanced CT Classification Training")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Hyperparameters: {hyperparams}")

    # Load data
    all_samples = load_preprocessed_data(PREPROCESSED_PATH)

    # Enhanced patient-wise split with balanced sampling
    train_samples, val_samples, test_samples, split_info = enhanced_patient_wise_split(
        all_samples,
        test_size=0.3,
        split_dir=ckpt_dir,
        target_real_per_class=hyperparams['target_real_per_class'],
        create_test_set=False  # Use val as test
    )

    # Create data loaders with weighted sampling
    labels = np.array([s['label'] for s in train_samples])
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    sample_weights = compute_sample_weights(labels, class_counts)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)

    num_workers = min(8, cpu_count())
    train_dataset = FastPreprocessedCTDataset(train_samples, transform=TrainTransforms(224))
    val_dataset = FastPreprocessedCTDataset(val_samples, transform=EvalTransforms(224))

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams['batch_size'],
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparams['batch_size'] * 2,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    # Initialize model
    model = EnhancedCTModel(
        num_classes=2,
        pretrained=True,
        use_ml_head=hyperparams['use_ml_head'],
        use_attention=hyperparams['use_attention']
    ).to(device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    history, val_probs_history = enhanced_train_model(
        model, train_loader, val_loader, device, ckpt_dir, hyperparams, resume=hyperparams['resume']
    )

    # Load best model for evaluation
    best_path = os.path.join(ckpt_dir, "enhanced_best_model.pth")
    if os.path.exists(best_path):
        print(f"📊 Loading best model from {best_path} for evaluation")
        checkpoint = torch.load(best_path, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Best model stats: Val F1={checkpoint['val_f1']:.4f}, "
              f"Precision={checkpoint.get('val_precision', 0):.4f}")

        # Enhanced evaluation
        eval_results = enhanced_evaluate_model(model, val_loader, device, ckpt_dir)
    else:
        print("❌ Best model not found, using current model for evaluation")
        eval_results = enhanced_evaluate_model(model, val_loader, device, ckpt_dir)

    # K-Fold evaluation
    print("\n🔍 Starting Enhanced K-Fold Evaluation")
    kfold_metrics, kfold_std, kfold_results = enhanced_kfold_evaluation(
        EnhancedCTModel, all_samples, device, hyperparams, k=5, ckpt_dir=ckpt_dir
    )

    # Plot training history
    plot_enhanced_training_history(history, ckpt_dir)

    print("✅ Enhanced Training Completed!")
    print("📈 Key Improvements:")
    print("   - Fixed checkpoint resume logic")
    print("   - Balanced sampling from all original classes")
    print("   - Enhanced loss with class weights")
    print("   - Attention mechanism in model")
    print("   - Comprehensive metrics tracking")
    print("   - Improved early stopping (F1 + Precision)")
    print("   - Enhanced visualization and analysis")

def plot_enhanced_training_history(history, ckpt_dir):
    """Plot enhanced training history with multiple metrics"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Loss
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0,0].set_title('Training & Validation Loss')
    axes[0,0].set_xlabel('Epochs'); axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # F1 Score
    axes[0,1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[0,1].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[0,1].set_title('F1 Score')
    axes[0,1].set_xlabel('Epochs'); axes[0,1].set_ylabel('F1 Score')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    # Precision
    axes[0,2].plot(epochs, history['train_precision'], 'b-', label='Train Precision', linewidth=2)
    axes[0,2].plot(epochs, history['val_precision'], 'r-', label='Val Precision', linewidth=2)
    axes[0,2].set_title('Precision')
    axes[0,2].set_xlabel('Epochs'); axes[0,2].set_ylabel('Precision')
    axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

    # Recall
    axes[1,0].plot(epochs, history['train_recall'], 'b-', label='Train Recall', linewidth=2)
    axes[1,0].plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2)
    axes[1,0].set_title('Recall')
    axes[1,0].set_xlabel('Epochs'); axes[1,0].set_ylabel('Recall')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # Accuracy
    axes[1,1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1,1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1,1].set_title('Accuracy')
    axes[1,1].set_xlabel('Epochs'); axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    # AUC
    if 'val_auc' in history:
        axes[1,2].plot(epochs, history['val_auc'], 'g-', label='Val AUC', linewidth=2)
        axes[1,2].set_title('Validation AUC')
        axes[1,2].set_xlabel('Epochs'); axes[1,2].set_ylabel('AUC')
        axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "enhanced_training_history.png"), dpi=300, bbox_inches='tight')
    plt.show()
def save_json_serializable(data, path):
    def convert(o):
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [convert(i) for i in o]
        return o
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
if __name__ == "__main__":
    main()