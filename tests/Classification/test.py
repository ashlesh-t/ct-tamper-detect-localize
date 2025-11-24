import os
import random
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2, densenet121
from PIL import Image
from tqdm.auto import tqdm
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
DRIVE_ROOT = "/content/drive/MyDrive/preprocessed_data_v3"
CKPT_DIR = f"{DRIVE_ROOT}/checkpoints/efficientnet-b2-v13"
BEST_MODEL_PATH = os.path.join(CKPT_DIR, "best_classifier_v2.pth")

# New Test Dataset Location
NEW_TEST_DIR = os.path.join(DRIVE_ROOT, "Injected_remove_classifier_test")

# Output for Results
OUTPUT_DIR = "/content/drive/MyDrive/capstone_models/balanced_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET CREATION LOGIC
# ==========================================
def load_all_patients(ckpt_dir):
    """Loads BOTH Train and Validation patients"""
    split_path = os.path.join(ckpt_dir, "data_split_v2.pkl")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found at {split_path}")
    
    with open(split_path, "rb") as f:
        split = pickle.load(f)
    
    # Combine Train + Val
    all_pts = split["train"] + split["val"]
    print(f"Loaded source pool: {len(split['train'])} Train + {len(split['val'])} Val = {len(all_pts)} Total Patients")
    return all_pts

def create_balanced_dataset(source_patients, target_dir):
    """Creates a physical 50/50 balanced dataset on disk"""
    
    # 1. Separate by class
    inj_pts = [p for p in source_patients if p['label'] == 0]
    rem_pts = [p for p in source_patients if p['label'] == 1]
    
    # 2. Determine balanced count (Limited by the smaller class)
    # limit = 200 # Uncomment if you want a fixed small number like 200 total
    limit = min(len(inj_pts), len(rem_pts)) # Use maximum possible balanced amount
    
    print(f"\n--- Balancing Data ---")
    print(f"Available Injected: {len(inj_pts)}")
    print(f"Available Removed:  {len(rem_pts)}")
    print(f"Target per class:   {limit}")
    
    sampled_inj = random.sample(inj_pts, limit)
    sampled_rem = random.sample(rem_pts, limit)
    
    # 3. Create Directories
    if os.path.exists(target_dir):
        print(f"Cleaning existing test dir: {target_dir}")
        shutil.rmtree(target_dir)
    
    dir_inj = os.path.join(target_dir, "injected")
    dir_rem = os.path.join(target_dir, "removed")
    os.makedirs(dir_inj, exist_ok=True)
    os.makedirs(dir_rem, exist_ok=True)
    
    # 4. Copy 1 Slice Per Patient
    print(f"Copying files to {target_dir}...")
    
    manifest = [] # To keep track of what we saved
    
    # Helper to copy
    def copy_samples(patients, dest_folder, label_name):
        for p in tqdm(patients, desc=f"Copying {label_name}"):
            # Pick 1 random slice
            src_path = random.choice(p['slices'])
            fname = os.path.basename(src_path)
            dst_path = os.path.join(dest_folder, fname)
            
            shutil.copy2(src_path, dst_path)
            
            manifest.append({
                "path": dst_path,
                "label": p['label'], # 0 or 1
                "original_patient_id": p['patient_id']
            })

    copy_samples(sampled_inj, dir_inj, "Injected")
    copy_samples(sampled_rem, dir_rem, "Removed")
    
    print(f"✅ Created Balanced Dataset with {len(manifest)} images.")
    return manifest

# ==========================================
# 3. MODEL ARCHITECTURE & UTILS
# ==========================================
# (Standard classes required for loading)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class EnhancedDualStreamFakeCT(nn.Module):
    def __init__(self):
        super().__init__()
        backbone1 = efficientnet_b2(weights=None)
        self.stream1_features = backbone1.features
        self.stream1_attention = ChannelAttention(1408)
        self.stream1_pool = nn.AdaptiveAvgPool2d(1)

        backbone2 = densenet121(weights=None)
        self.stream2_features = backbone2.features
        self.stream2_attention = ChannelAttention(1024)
        self.stream2_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion = nn.Sequential(
            nn.Linear(1408 + 1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x1 = x[:, 0:1].repeat(1, 3, 1, 1)
        f1 = self.stream1_features(x1)
        f1 = self.stream1_pool(f1 * self.stream1_attention(f1)).view(f1.size(0), -1)
        
        x2 = torch.cat([x[:, 1:3], x[:, 1:2]], dim=1)
        f2 = self.stream2_features(x2)
        f2 = self.stream2_pool(f2 * self.stream2_attention(f2)).view(f2.size(0), -1)
        
        return self.fusion(torch.cat([f1, f2], dim=1))

class StackedChannelsToRGB:
    def __call__(self, arr):
        arr = arr.astype(np.float32)
        if arr.shape[2] == 3:
            for c in range(3):
                ch = arr[:, :, c]
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
                arr[:, :, c] = ch
            rgb = (arr * 255).astype(np.uint8)
        else:
            rgb = (arr[:, :, 0] * 255).astype(np.uint8)
            rgb = np.stack([rgb] * 3, axis=-1)
        return Image.fromarray(rgb)

def process_image(fpath, transform, stats):
    try:
        hu = np.load(fpath).astype(np.float32)
        vis_img = StackedChannelsToRGB()(hu.copy())
        hu_tensor = transform(vis_img)
        for c in range(3):
            hu_tensor[c] = (hu_tensor[c] - stats[c]["mean"]) / (stats[c]["std"] + 1e-6)
        return hu_tensor.unsqueeze(0), vis_img
    except: return None, None

# ==========================================
# 4. EXECUTION FLOW
# ==========================================

# --- STEP 1: CREATE THE DATASET ---
print(">>> Step 1: Creating New Dataset...")
all_patients = load_all_patients(CKPT_DIR)
manifest = create_balanced_dataset(all_patients, NEW_TEST_DIR)

# --- STEP 2: LOAD MODEL ---
print("\n>>> Step 2: Loading Model...")
model = EnhancedDualStreamFakeCT().to(device)
checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint.get('channel_stats')
else:
    model.load_state_dict(checkpoint)
    stats = None

if not stats:
    stats = {0: {'mean': 0.5, 'std': 0.5}, 1: {'mean': 0.5, 'std': 0.5}, 2: {'mean': 0.5, 'std': 0.5}}

model.eval()

# --- STEP 3: RUN INFERENCE ON NEW DATASET ---
print("\n>>> Step 3: Testing on New Dataset...")
val_tf = transforms.Compose([transforms.Resize((288, 288)), transforms.ToTensor()])

results = []
# Iterate through the manifest we just created
for item in tqdm(manifest, desc="Inferencing"):
    img_tensor, vis_img = process_image(item['path'], val_tf, stats)
    if img_tensor is None: continue
    
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        probs = torch.softmax(logits, 1)
        pred = logits.argmax(1).item()
        conf = probs[0][pred].item() * 100
        
    results.append({
        "path": item['path'],
        "vis": vis_img,
        "true": item['label'],
        "pred": pred,
        "conf": conf,
        "correct": item['label'] == pred
    })

# ==========================================
# 5. METRICS & VISUALIZATION
# ==========================================
y_true = [r['true'] for r in results]
y_pred = [r['pred'] for r in results]
classes = ["Injected", "Removed"]

# Metrics
acc = accuracy_score(y_true, y_pred)
print("\n" + "="*40)
print(f" NEW DATASET RESULTS (Size: {len(results)})")
print("="*40)
print(f"Accuracy: {acc:.2%}")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Balanced Test Set)')
plt.savefig(os.path.join(OUTPUT_DIR, "balanced_cm.png"))
plt.show()

# Top 10 Visuals
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
axes = axes.flatten()
print("\nTop 10 Predictions:")
for i, res in enumerate(results[:10]):
    ax = axes[i]
    ax.imshow(res['vis'])
    col = 'green' if res['correct'] else 'red'
    ax.set_title(f"GT: {classes[res['true']]}\nPred: {classes[res['pred']]}\nConf: {res['conf']:.1f}%", color=col, fontweight='bold')
    ax.axis('off')
    print(f"Img: {os.path.basename(res['path'])} | GT: {classes[res['true']]} | Pred: {classes[res['pred']]} | {res['conf']:.1f}%")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_balanced.png"))
plt.show()

print(f"\n✅ All results saved to: {OUTPUT_DIR}")
print(f"✅ New Dataset saved at: {NEW_TEST_DIR}")