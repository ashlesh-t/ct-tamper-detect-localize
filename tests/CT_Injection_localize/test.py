# =============================================================
# FULL TEST ON 100 EXTRACTED SAMPLES (fixed: grad, resizing, visualization)
# =============================================================
import os
import json
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------ CONFIG ------------------
IMG_SIZE       = 320
RADIUS_PX      = 48
TOLERANCE_PX   = 48
TTA_FLIPS      = ['horizontal']
OUT_DIR        = Path("main/tests/CT_Injection_localize/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================
# UPDATE THESE PATHS
# =============================================================
EXTRACTED_FOLDER = "main/tests/CT_Injection_localize/data/val_100_samples"   # <-- Folder where you extracted val_100_samples.zip
BEST_CHECKPOINT  = "main/tests/CT_Injection_localize/model/CT_Injection_finetune_phases_v3.3/best_model.pth"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================

# ------------------ HELPERS ------------------
def window_image(img, wc, ww, to_uint8=True):
    img_min = wc - ww // 2
    img_max = wc + ww // 2
    w = np.clip(img, img_min, img_max)
    w = (w - img_min) / (img_max - img_min + 1e-6)
    return (w * 255).astype(np.uint8) if to_uint8 else w

def apply_clahe(img, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img)

def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
        ToTensorV2()
    ])

def draw_disk(mask, cx, cy, r):
    H, W = mask.shape
    y, x = np.ogrid[:H, :W]
    mask[(x-cx)**2 + (y-cy)**2 <= r*r] = 1

def tta_predict(model, img_tensor):
    """
    img_tensor: C,H,W on target device
    returns: torch tensor shape (1, H_out, W_out) (channel-first)
    """
    preds = []
    # base
    out = torch.sigmoid(model(img_tensor.unsqueeze(0)))  # [1,1,H,W]
    preds.append(out.squeeze(0))                         # [1,H,W]
    for flip in TTA_FLIPS:
        if flip == 'horizontal':
            flipped = torch.flip(img_tensor, dims=[2])
            p = torch.sigmoid(model(flipped.unsqueeze(0))).squeeze(0)
            p = torch.flip(p, dims=[2])
        else:
            flipped = torch.flip(img_tensor, dims=[1])
            p = torch.sigmoid(model(flipped.unsqueeze(0))).squeeze(0)
            p = torch.flip(p, dims=[1])
        preds.append(p)
    stacked = torch.stack(preds, dim=0)   # [n_augment, 1, H, W]
    mean_pred = torch.mean(stacked, dim=0)  # [1, H, W]
    return mean_pred

def compute_metrics(pred_bool, gt_bool):
    # pred_bool, gt_bool: boolean ndarray (H,W)
    pred = pred_bool.astype(bool)
    gt   = gt_bool.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    eps = 1e-7
    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    return {'dice': float(dice), 'iou': float(iou), 'precision': float(prec), 'recall': float(rec)}

def center_of_mass(mask_bool):
    y, x = np.where(mask_bool)
    if len(y) == 0: 
        return None, None
    return int(x.mean()), int(y.mean())

# ------------------ LOAD MODEL ------------------
print(f"Loading model from: {BEST_CHECKPOINT}")
ckpt = torch.load(BEST_CHECKPOINT, map_location=DEVICE)
cfg = ckpt.get('config', {})

model = smp.UnetPlusPlus(
    encoder_name=cfg.get('encoder', 'resnet34'),
    encoder_weights=None,
    decoder_attention_type='scse',
    classes=1,
    activation=None
)
model.load_state_dict(ckpt['model'])
model.to(DEVICE)
model.eval()
print("Model loaded")

# ------------------ LOAD 100 SAMPLES ------------------
npy_dir = Path(EXTRACTED_FOLDER) / "npy"
csv_dir = Path(EXTRACTED_FOLDER) / "csv"

samples = []
for npy_file in sorted(npy_dir.glob("*.npy")):
    idx = npy_file.stem
    csv_file = csv_dir / f"{idx}.csv"
    if not csv_file.exists():
        continue
    df_row = pd.read_csv(csv_file).iloc[0]
    samples.append({
        "npy_path": str(npy_file),
        "gt_x": int(df_row["x"]) if pd.notna(df_row["x"]) else None,
        "gt_y": int(df_row["y"]) if pd.notna(df_row["y"]) else None,
    })

print(f"Loaded {len(samples)} samples from extracted folder")

# ------------------ INFERENCE ------------------
transform = get_transform(IMG_SIZE)
results = []
prob_map = {}  # store resized probability maps per sample for visualization

with torch.no_grad():   # ensure no gradient tracking for entire inference
    for s in tqdm(samples, desc="Inference"):
        # Load & preprocess raw (original resolution)
        raw = np.load(s["npy_path"]).astype(np.float32)
        # original H,W
        H_orig, W_orig = raw.shape[0], raw.shape[1]

        # windowing + CLAHE on channels (uint8 0-255)
        ch1 = window_image(raw, -600, 1500, True); ch1 = apply_clahe(ch1)
        ch2 = window_image(raw, 40, 400, True)
        ch3 = window_image(raw, 400, 1800, True)
        img = np.stack([ch1, ch2, ch3], axis=-1)  # H_orig x W_orig x 3

        # augmentation / resize to IMG_SIZE
        aug = transform(image=img)
        img_tensor = aug['image'].to(DEVICE)  # C x IMG_SIZE x IMG_SIZE

        # TTA & predict (returns tensor shape [1, H_model, W_model])
        prob_t = tta_predict(model, img_tensor)  # torch tensor on DEVICE, shape [1,H,W]
        prob_np = prob_t.detach().cpu().numpy()  # (1, H_model, W_model)

        # binary prediction at model resolution
        pred_bin = (prob_np > 0.5).astype(np.uint8)  # (1,H_model,W_model)
        pred_bin_squeezed = np.squeeze(pred_bin, axis=0)  # (H_model, W_model)

        # Resize prediction back to original resolution using nearest for masks
        pred_resized = cv2.resize(pred_bin_squeezed, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        pred_bool = pred_resized.astype(bool)  # (H_orig, W_orig)

        # GT mask (built at original resolution)
        gt_mask = np.zeros((H_orig, W_orig), dtype=np.uint8)
        if s["gt_x"] is not None and s["gt_y"] is not None:
            draw_disk(gt_mask, s["gt_x"], s["gt_y"], RADIUS_PX)
        gt_bool = (gt_mask > 0).astype(bool)

        # Metrics
        mets = compute_metrics(pred_bool, gt_bool)

        # Centers (use resized pred)
        pred_x, pred_y = center_of_mass(pred_bool)
        correct = False
        if s["gt_x"] is not None and pred_x is not None:
            dist = np.hypot(pred_x - s["gt_x"], pred_y - s["gt_y"])
            correct = dist <= TOLERANCE_PX

        # Save per-sample prob (resized to original) to use later for visualization
        prob_orig = cv2.resize(np.squeeze(prob_np), (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)  # float map
        prob_map[s["npy_path"]] = prob_orig

        results.append({
            "path": s["npy_path"],
            "gt_x": s["gt_x"],
            "gt_y": s["gt_y"],
            "pred_x": pred_x,
            "pred_y": pred_y,
            "dice": mets['dice'],
            "iou": mets['iou'],
            "precision": mets['precision'],
            "recall": mets['recall'],
            "correct_32px": correct
        })

# ------------------ SAVE CSV ------------------
df_results = pd.DataFrame(results)
csv_out = OUT_DIR / "predictions_100.csv"
df_results.to_csv(csv_out, index=False)
print(f"CSV saved: {csv_out}")

# ------------------ GLOBAL METRICS ------------------
global_metrics = {
    "dice": df_results['dice'].mean() if not df_results.empty else 0.0,
    "iou": df_results['iou'].mean() if not df_results.empty else 0.0,
    "precision": df_results['precision'].mean() if not df_results.empty else 0.0,
    "recall": df_results['recall'].mean() if not df_results.empty else 0.0,
    "center_accuracy_32px": df_results['correct_32px'].mean() if not df_results.empty else 0.0
}
print("\n" + "="*50)
print("FINAL METRICS ON 100 SAMPLES")
print("="*50)
for k, v in global_metrics.items():
    print(f"{k:20}: {v:.4f}")
print("="*50)

# ------------------ VISUALIZE BEST/WORST ------------------
def save_overlay_from_prob(prob_orig, raw, row, tag, rank):
    """
    prob_orig: float map at original resolution (H,W), values [0,1]
    raw: original single-channel raw array (H,W)
    row: pandas Series with fields including pred_x/pred_y/gt_x/gt_y/dice
    """
    ch1 = window_image(raw, -600, 1500, True); ch1 = apply_clahe(ch1)
    img_rgb = np.stack([ch1]*3, axis=-1)

    pred_bin_orig = (prob_orig > 0.5).astype(np.uint8) * 255
    overlay = img_rgb.copy()
    overlay[pred_bin_orig > 0] = [0, 255, 0]  # Green

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb[..., 0], cmap='gray')
    axes[0].set_title("CT Slice")
    axes[1].imshow(prob_orig, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Prediction Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Green = Pred)")

    for ax in axes:
        if row["gt_x"] is not None:
            ax.plot(row["gt_x"], row["gt_y"], 'rx', markersize=12, markeredgewidth=2, label="GT")
        if row["pred_x"] is not None:
            ax.plot(row["pred_x"], row["pred_y"], 'wo', markersize=10, markeredgewidth=2, label="Pred")
        ax.legend()
        ax.axis('off')

    plt.suptitle(f"{tag} #{rank} | Dice: {row['dice']:.3f} | "
                 f"GT({row['gt_x']},{row['gt_y']}) → Pred({row['pred_x']},{row['pred_y']})")
    plt.tight_layout()
    fname = OUT_DIR / f"{tag}_{rank:02d}_{Path(row['path']).stem}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

# Best 10
for rank, idx in enumerate(df_results.nlargest(10, 'dice').index, 1):
    row = df_results.loc[idx]
    prob_orig = prob_map.get(row['path'])
    if prob_orig is None:
        continue
    raw = np.load(row['path']).astype(np.float32)
    save_overlay_from_prob(prob_orig, raw, row, "BEST", rank)

# Worst 10
for rank, idx in enumerate(df_results.nsmallest(10, 'dice').index, 1):
    row = df_results.loc[idx]
    prob_orig = prob_map.get(row['path'])
    if prob_orig is None:
        continue
    raw = np.load(row['path']).astype(np.float32)
    save_overlay_from_prob(prob_orig, raw, row, "WORST", rank)

print(f"\nVisualizations saved in: {OUT_DIR}")
print(f"   - predictions_100.csv")
print(f"   - BEST_*.png")
print(f"   - WORST_*.png")
