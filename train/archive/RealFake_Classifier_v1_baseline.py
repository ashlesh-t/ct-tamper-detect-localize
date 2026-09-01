# binary_ct_training_balanced_v3.py
import os
import random
import numpy as np
import json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # ← Fixed: no warning
from multiprocessing import cpu_count
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Speed & Seed
# -------------------------
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

# -------------------------
# Dataset
# -------------------------
class FastPreprocessedCTDataset(Dataset):
    def __init__(self, samples, transform=None, img_size=224, mmap_mode='r'):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size
        self.mmap_mode = mmap_mode
        self._cache = {}

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
                if len(self._cache) > 128:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[fpath] = multi_channel
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                multi_channel = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel, torch.tensor(label, dtype=torch.long), patient_id


# -------------------------
# Transforms
# -------------------------
DATASET_MEAN = [0.5, 0.5, 0.5]
DATASET_STD = [0.5, 0.5, 0.5]

class TrainTransforms:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3))], p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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


# -------------------------
# Model
# -------------------------
class EnhancedCTModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, use_ml_head=True):
        super().__init__()
        self.use_ml_head = use_ml_head
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.backbone = densenet.features
        num_backbone_features = 1024

        if use_ml_head:
            self.projection_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.BatchNorm1d(num_backbone_features), nn.Dropout(0.3),
                nn.Linear(num_backbone_features, 512), nn.GELU(),
                nn.BatchNorm1d(512), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.SiLU(),
                nn.BatchNorm1d(256), nn.Dropout(0.2),
            )
            classifier_input = 256
        else:
            self.projection_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            classifier_input = num_backbone_features

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128), nn.GELU(),
            nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        p = self.projection_head(f)
        return self.classifier(p)


# -------------------------
# Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, focal_alpha=1, focal_gamma=2):
        super().__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss()
    def forward(self, preds, targets):
        return self.alpha * self.focal_loss(preds, targets) + (1 - self.alpha) * self.ce_loss(preds, targets)


# -------------------------
# Optimizers
# -------------------------
def get_optimizer_scheduler(model, phase, lr, weight_decay, max_epochs, current_epoch=0):
    if phase == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=max_epochs, pct_start=0.1, anneal_strategy='cos')
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    for _ in range(current_epoch):
        scheduler.step()
    return optimizer, scheduler


# -------------------------
# DOWN SAMPLE REAL SLICES (Before Dataset)
# -------------------------
def downsample_real_slices(train_samples, target_real=4800, seed=42):
    np.random.seed(seed)
    real_samples = [s for s in train_samples if s['label'] == 0]
    tampered_samples = [s for s in train_samples if s['label'] == 1]

    n_tampered = len(tampered_samples)
    print(f"Before downsampling → Real: {len(real_samples)}, Tampered: {n_tampered}")

    if len(real_samples) > target_real:
        real_samples = np.random.choice(real_samples, target_real, replace=False).tolist()
        print(f"Downsampled real → {target_real}")
    else:
        print(f"Real already ≤ {target_real}, keeping all")

    balanced_samples = real_samples + tampered_samples
    np.random.shuffle(balanced_samples)
    print(f"Final training set: {len(balanced_samples)} slices (Real: {len(real_samples)}, Tampered: {n_tampered})")
    return balanced_samples


# -------------------------
# Patient-wise Split (Cached)
# -------------------------
def patient_wise_split(all_samples, test_size=0.3, seed=42, split_dir=None):
    if split_dir and os.path.exists(os.path.join(split_dir, "split_info.json")):
        print(f"Loading cached split from {split_dir}")
        with open(os.path.join(split_dir, "split_info.json"), 'r') as f:
            info = json.load(f)
        patient_to_samples = defaultdict(list)
        for s in all_samples:
            patient_to_samples[s['patient_id']].append(s)
        train_samples = [s for pid in info['train_patients'] for s in patient_to_samples[pid]]
        val_samples = [s for pid in info['val_patients'] for s in patient_to_samples[pid]]
        return train_samples, val_samples, info['train_patients'], info['val_patients']

    print("Computing new patient-wise split...")
    patient_ids = np.array([s['patient_id'] for s in all_samples])
    unique_patients = np.unique(patient_ids)
    train_patients, val_patients = train_test_split(unique_patients, test_size=test_size, random_state=seed)

    train_mask = np.isin(patient_ids, train_patients)
    val_mask = ~train_mask
    train_samples = [s for s, m in zip(all_samples, train_mask) if m]
    val_samples = [s for s, m in zip(all_samples, val_mask) if m]

    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
        json.dump({
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist()
        }, open(os.path.join(split_dir, "split_info.json"), 'w'), indent=2)

    return train_samples, val_samples, train_patients.tolist(), val_patients.tolist()


# -------------------------
# Training Loop
# -------------------------
def train_enhanced_model(model, train_loader, val_loader, device, ckpt_dir, hyperparams, resume=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    total_epochs = hyperparams['adam_epochs'] + hyperparams['sgd_epochs']
    criterion = CombinedLoss(alpha=hyperparams['comb_alpha'])
    scaler = GradScaler()

    # Optional: Disable torch.compile if unstable
    # model = torch.compile(model, mode="max-autotune")
    model = model.to(device)

    optimizer_adam, scheduler_adam = get_optimizer_scheduler(model, "adam", hyperparams['adam_lr'], hyperparams['weight_decay'], hyperparams['adam_epochs'])
    optimizer_sgd, scheduler_sgd = get_optimizer_scheduler(model, "sgd", hyperparams['sgd_lr'], hyperparams['weight_decay'], hyperparams['sgd_epochs'])

    start_epoch = 0
    best_val_f1 = 0.0
    latest_ckpt = os.path.join(ckpt_dir, "latest_model.pth")
    best_ckpt = os.path.join(ckpt_dir, "best_model.pth")

    if resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_f1 = ckpt.get('best_val_f1', 0.0)

    for epoch in range(start_epoch, total_epochs):
        phase = "adam" if epoch < hyperparams['adam_epochs'] else "sgd"
        optimizer = optimizer_adam if phase == "adam" else optimizer_sgd
        scheduler = scheduler_adam if phase == "adam" else scheduler_sgd

        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()
        train_loss = epoch_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.inference_mode(), autocast():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        print(f"\nEpoch {epoch+1} | {phase.upper()} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_val_f1': best_val_f1
        }, latest_ckpt)

    return best_val_f1


# -------------------------
# Load Data
# -------------------------
def load_preprocessed_data(preprocessed_path):
    all_samples = []
    class_mapping = {"0": 0, "1": 0, "2": 1, "3": 1}
    for label_dir in glob(os.path.join(preprocessed_path, "[0-3]")):
        original_label = os.path.basename(label_dir)
        binary_label = class_mapping.get(original_label)
        if binary_label is None: continue
        for patient_dir in glob(os.path.join(label_dir, "*")):
            if not os.path.isdir(patient_dir): continue
            patient_id = os.path.basename(patient_dir)
            for fpath in glob(os.path.join(patient_dir, "*.npy")):
                all_samples.append({
                    "fpath": fpath,
                    "label": binary_label,
                    "patient_id": f"{original_label}_{patient_id}",
                    "original_label": int(original_label)
                })
    print(f"Loaded {len(all_samples)} samples")
    return all_samples


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    PREPROCESSED_PATH = "/content/drive/MyDrive/preprocessed_data_v3"
    BASE_CKPT_DIR = "/content/drive/MyDrive/capstone_models"
    hyperparams = {
        'experiment_name': 'balanced_weighted_v3',
        'img_size': 224,
        'adam_epochs': 30,
        'sgd_epochs': 20,
        'adam_lr': 1e-4,
        'sgd_lr': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'comb_alpha': 0.7,
        'batch_size': 16,
        'resume': True
    }

    ckpt_dir = os.path.join(BASE_CKPT_DIR, hyperparams['experiment_name'])
    print(f"Experiment: {hyperparams['experiment_name']}")
    print(f"Checkpoint dir: {ckpt_dir}")

    all_samples = load_preprocessed_data(PREPROCESSED_PATH)
    train_samples, val_samples, _, _ = patient_wise_split(all_samples, test_size=0.3, seed=42, split_dir=ckpt_dir)

    # DOWN SAMPLE REAL TO ~4.8k
    train_samples = downsample_real_slices(train_samples, target_real=4800, seed=42)

    # LIGHT WEIGHTED SAMPLER (for stability)
    labels = [s['label'] for s in train_samples]
    class_counts = np.bincount(labels, minlength=2)
    sample_weights = 1.0 / (class_counts[np.array(labels)] + 1e-6)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = min(8, cpu_count())
    train_dataset = FastPreprocessedCTDataset(train_samples, transform=TrainTransforms(224), mmap_mode='r')
    val_dataset = FastPreprocessedCTDataset(val_samples, transform=EvalTransforms(224), mmap_mode='r')

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams['batch_size'],
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4, drop_last=True  # ← FIXED
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparams['batch_size']*2,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    model = EnhancedCTModel(num_classes=2, pretrained=True).to(device)
    train_enhanced_model(model, train_loader, val_loader, device, ckpt_dir, hyperparams)

    print("Training completed! Perfect balance, no crashes.")