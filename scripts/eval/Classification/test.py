from torch.cuda.amp import GradScaler
import os, random, json, math, time
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

# sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.isotonic import IsotonicRegression

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ===== Import your model classes =====

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_ch, in_ch//reduction), nn.ReLU(), nn.Linear(in_ch//reduction, in_ch))
        self.sig = nn.Sigmoid()
        self.spatial = nn.Conv2d(2,1,kernel_size=7,padding=3,bias=False)
    def forward(self, x):
        b,c,_,_ = x.shape
        avg = self.avg(x).view(b,c); maxv = self.max(x).view(b,c)
        att = self.sig(self.fc(avg)+self.fc(maxv)).view(b,c,1,1)
        x = x * att
        a = torch.mean(x, dim=1, keepdim=True); m,_ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sig(self.spatial(torch.cat([a,m], dim=1)))
        return x

def ensure_3ch(arr, img_size=384):
    if arr is None: return np.zeros((img_size,img_size,3),dtype=np.float32)
    if arr.ndim==2: return np.stack([arr]*3,axis=-1)
    if arr.shape[-1] >= 3: return arr[:, :, :3]
    ch = arr.shape[-1]
    pad = np.zeros((arr.shape[0], arr.shape[1], 3-ch), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)

class MultiStreamModel(nn.Module):
    def __init__(self, pretrained=True, feature_dim=1280, num_classes=2):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        # helper to make single-channel backbone
        def make_stream():
            m = efficientnet_v2_m(weights=weights)
            # replace first conv to accept 1 channel:
            first = list(m.features.children())[0][0]
            new_conv = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
            with torch.no_grad():
                w = first.weight; w_mean = w.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(w_mean)
            m.features[0][0] = new_conv
            feat = nn.Sequential(*list(m.features.children()))
            return feat
        self.ct_stream = make_stream(); self.roi_stream = make_stream(); self.fft_stream = make_stream()
        self.cbam1 = CBAM(feature_dim); self.cbam2 = CBAM(feature_dim); self.cbam3 = CBAM(feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.fusion = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.BatchNorm1d(feature_dim*3),
                                    nn.Linear(feature_dim*3, 1024), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(1024,512), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(512, num_classes)
        self.aux_ct = nn.Linear(feature_dim, num_classes)
        self.aux_roi = nn.Linear(feature_dim, num_classes)
        self.aux_fft = nn.Linear(feature_dim, num_classes)
    def forward(self, x, return_aux=False):
        ct = x[:,0:1,:,:]; roi = x[:,1:2,:,:]; fft = x[:,2:3,:,:]
        fct = self.ct_stream(ct); froi = self.roi_stream(roi); ffft = self.fft_stream(fft)
        fct = self.cbam1(fct); froi = self.cbam2(froi); ffft = self.cbam3(ffft)
        b,c,h,w = fct.shape
        def to_seq(t): return t.view(b,c,-1).permute(0,2,1)
        seq_ct = to_seq(fct); seq_roi = to_seq(froi); seq_fft = to_seq(ffft)
        ct_att,_ = self.cross_attn(seq_ct, seq_roi, seq_roi)
        roi_att,_ = self.cross_attn(seq_roi, seq_fft, seq_fft)
        fft_att,_ = self.cross_attn(seq_fft, seq_ct, seq_ct)
        def from_seq(seq): return seq.permute(0,2,1).view(b,c,h,w)
        ct_att = from_seq(ct_att); roi_att = from_seq(roi_att); fft_att = from_seq(fft_att)
        fused = torch.cat([ct_att, roi_att, fft_att], dim=1)
        out_feat = self.fusion(fused)
        logits = self.classifier(out_feat)
        if return_aux:
            aux_ct = self.aux_ct(F.adaptive_avg_pool2d(fct,1).flatten(1))
            aux_roi = self.aux_roi(F.adaptive_avg_pool2d(froi,1).flatten(1))
            aux_fft = self.aux_fft(F.adaptive_avg_pool2d(ffft,1).flatten(1))
            return logits, (aux_ct, aux_roi, aux_fft), out_feat
        return logits, out_feat

# ---------------------------
# DenseNet model wrapper
# ---------------------------
class DenseNetBinary(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        m = densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        self.model = m
    def forward(self, x): return self.model(x)
        
# ------------------------
# Dataset for Inference
# ------------------------
class InferenceDataset(Dataset):
    def __init__(self, file_list, transform, img_size=384):
        self.file_list = file_list
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        arr = np.load(fpath).astype(np.float32)
        arr = ensure_3ch(arr, img_size=self.img_size)

        img = self.transform(arr)
        return img, os.path.basename(fpath)

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------------------
# Model Loading Utility
# --------------------------------
def load_model(best_ckpt_path, model_type="multistream"):
    if model_type == "multistream":
        model = MultiStreamModel(pretrained=False)
    else:
        model = DenseNetBinary(pretrained=False)

    ck = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state'])
    model.to(DEVICE)
    model.eval()
    return model

# --------------------------------
# Prediction Function
# --------------------------------
def predict_patient(patient_dir, model, model_type="multistream"):
    # print(os.listdir(patient_dir))
    files = sorted([os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith(".npy")])
    # print("Files",files)
    ds = InferenceDataset(files, eval_transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    slice_preds = []
    slice_probs = []

    with torch.inference_mode():
        for img, fname in loader:
            img = img.to(DEVICE).float()
            
            if model_type == "multistream":
                out, _ = model(img)
            else:
                out = model(img)

            prob = F.softmax(out, dim=1)[0, 1].item()  # Tampered probability
            pred = int(prob > 0.5)

            slice_preds.append(pred)
            slice_probs.append(prob)

            print(f"Slice {fname}: ProbTampered={prob:.3f} → {'Tampered' if pred else 'Real'}")
    print("slicePreds ",slice_preds)
    print("sliceProbs ",slice_probs)
    avg_prob = np.mean(slice_probs)
    majority_vote = 1 if slice_preds.count(1) >= len(slice_preds) / 2 else 0

    print("\n=========== FINAL PATIENT RESULT ===========")
    print(f"Slices: {len(slice_preds)}")
    print(f"Avg Tampered Probability: {avg_prob:.3f}")
    print(f"Majority Vote: {'Tampered' if majority_vote else 'Real'}")
    print("===========================================\n")

    return slice_preds, slice_probs, majority_vote, avg_prob


# --------------------------------
# Example Run
# --------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run classifier predictions over sample patient directories.")
    parser.add_argument("--checkpoint", required=True, help="Path to the classifier checkpoint (e.g. .../pipeline/models/classifier1/dn_phase1_best.pth)")
    parser.add_argument("--injection", help="Path to a CT_Injection patient directory")
    parser.add_argument("--removal", help="Path to a CT_Removal patient directory")
    parser.add_argument("--real", action="append", default=[], dest="real_dirs", help="Path to a real/untampered patient directory (repeatable)")
    parser.add_argument("--model-type", default="densenet", choices=["densenet", "efficientnet"])
    args = parser.parse_args()

    model = load_model(args.checkpoint, model_type=args.model_type)

    paths = [p for p in [args.injection, args.removal, *args.real_dirs] if p]
    if not paths:
        parser.error("Provide at least one of --injection, --removal, or --real")

    for idx, path in enumerate(paths):
        print("PREDICTING!!! for class :: ", idx)
        predict_patient(path, model, model_type=args.model_type)
