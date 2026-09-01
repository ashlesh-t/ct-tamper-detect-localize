# Fine-tunes the checkpoint produced by RealFakeClissifier.py (run that script first).
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from multiprocessing import cpu_count
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
)
import copy

from RealFakeClissifier import (
    device, compute_sample_weights, FastPreprocessedCTDataset,
    TrainTransforms, EvalTransforms, EnhancedCTModel, EnhancedCombinedLoss,
    calculate_detailed_metrics, enhanced_evaluate_model, save_json_serializable,
)

def mixup_data(x, y, alpha=0.2):
    """Simple mixup for binary augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def find_optimal_threshold(labels, probs, target_metric='f1'):
    """Optimize threshold on val for max F1 or precision"""
    precs, recs, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precs * recs / (precs + recs + 1e-8)
    if target_metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    else:  # precision
        optimal_idx = np.argmax(precs * recs)  # Balanced
    return thresholds[optimal_idx]

def tta_predict(model, loader, device, num_tta=3):
    """Test-Time Augmentation: Average predictions over flips"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            batch_probs = []
            for _ in range(num_tta):
                aug_imgs = transforms.RandomHorizontalFlip(p=0.5)(imgs)
                aug_imgs = transforms.RandomRotation(degrees=5)(aug_imgs)
                outputs = model(aug_imgs)
                probs = F.softmax(outputs, dim=1)[:, 1]
                batch_probs.append(probs)
            avg_probs = torch.stack(batch_probs).mean(0).cpu().numpy()
            all_probs.extend(avg_probs)
    return np.array(all_probs)

def run_finetune():
    # === FINE-TUNE EXECUTION ===
    print("🔥 Starting Fine-Tune to 0.95+ F1 (Minimal FP/FN)")

    # Load existing split/data (reuse from your main())
    PREPROCESSED_PATH = "/content/drive/MyDrive/preprocessed_data_v3"
    ckpt_dir = "/content/drive/MyDrive/capstone_models/binary_ct_enhanced_v5"
    best_ckpt_path = "/content/drive/MyDrive/capstone_models/binary_ct_enhanced_v5/enhanced_best_model.pth"

    # Reload split (from your cached JSON)
    with open(os.path.join(ckpt_dir, "enhanced_split_info.json"), 'r') as f:
        split_info = json.load(f)
    train_samples = split_info['train_samples']
    val_samples = split_info['val_samples']

    # Reuse loaders (same as main, batch=16 for T4)
    labels = np.array([s['label'] for s in train_samples])
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    sample_weights = compute_sample_weights(labels, class_counts)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True)

    num_workers = min(4, cpu_count())  # Lower for T4 stability
    train_dataset = FastPreprocessedCTDataset(train_samples, transform=TrainTransforms(224))
    val_dataset = FastPreprocessedCTDataset(val_samples, transform=EvalTransforms(224))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)  # Smaller batch for eval

    # Init model & load best
    model = EnhancedCTModel(num_classes=2, pretrained=False).to(device)  # pretrained=False since loading
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model: Val F1={checkpoint['val_f1']:.4f}")

    # Load history
    history_path = os.path.join(ckpt_dir, "enhanced_training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            old_history = json.load(f)
        history = defaultdict(list, {k: [float(v) for v in old_history[k]] for k in old_history if isinstance(old_history[k], list)})
        start_epoch = len(history['train_loss'])
        print(f"Resuming history from epoch {start_epoch}")
    else:
        history = defaultdict(list)
        start_epoch = 0

    # Fine-tune config (append to history)
    fine_tune_config = {
        "type": "fine-tune",
        "start_epoch": start_epoch + 1,
        "base_f1": 0.813,
        "unfrozen_initial": ["projection_head", "attention", "classifier"],
        "unfrozen_later": ["denseblock4"],
        "lr": 1e-5,
        "mixup_alpha": 0.2,
        "label_smoothing": 0.05,
        "tta_num": 3,
        "patience": 3  # Shorter for fine-tune
    }
    history['fine_tune_config'] = fine_tune_config

    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Unfreeze head/attention/classifier
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["projection_head", "attention", "classifier"]):
            param.requires_grad = True

    # Enhanced loss with more smoothing
    train_labels = [s['label'] for s in train_samples]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor([1.0 / count if count > 0 else 1.0 for count in class_counts], dtype=torch.float32).to(device)
    criterion = EnhancedCombinedLoss(alpha=0.7, focal_gamma=2.0, class_weights=class_weights)
    criterion.ce_loss.label_smoothing = 0.05  # Increase smoothing

    # Fine-tune optimizer/scheduler (low LR, simple)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = GradScaler()

    # Fine-tune loop: 8 epochs total
    fine_tune_epochs = 8
    head_only_epochs = 5  # Then unfreeze denseblock4
    best_val_f1 = checkpoint['val_f1']
    epochs_no_improve = 0
    val_probs_list = []  # For threshold opt

    for epoch in range(fine_tune_epochs):
        global_epoch = start_epoch + epoch + 1
        is_head_only = epoch < head_only_epochs

        if not is_head_only:
            # Unfreeze denseblock4 after head-only
            for name, module in model.backbone.named_modules():
                if 'denseblock4' in name:
                    for param in module.parameters():
                        param.requires_grad = True
            print(f"🔓 Unfroze denseblock4 at epoch {global_epoch}")
            # Re-init optimizer with new params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer.param_groups[0]['lr'] * 0.5, weight_decay=1e-4)

        # Train
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        train_pbar = tqdm(train_loader, desc=f"Fine-Tune Epoch {global_epoch} [{'Head-Only' if is_head_only else 'Dense4-Unfrozen'}] Train")
        for imgs, labels, _ in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Mixup
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=fine_tune_config['mixup_alpha'])

            with autocast():
                outputs = model(imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Tighter clip
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{current_lr:.2e}'})

        train_metrics = calculate_detailed_metrics(all_labels, all_preds, epoch_loss / len(train_loader.dataset))

        # Val with TTA
        tta_probs = tta_predict(model, val_loader, device, num_tta=fine_tune_config['tta_num'])
        val_labels = [s['label'] for s in val_samples]
        optimal_thresh = find_optimal_threshold(val_labels, tta_probs, 'f1')
        val_preds_bin = (tta_probs > optimal_thresh).astype(int)

        # Metrics at optimal thresh
        val_accuracy = accuracy_score(val_labels, val_preds_bin)
        val_precision = precision_score(val_labels, val_preds_bin, zero_division=0)
        val_recall = recall_score(val_labels, val_preds_bin, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds_bin, zero_division=0)
        val_auc = roc_auc_score(val_labels, tta_probs)
        cm = confusion_matrix(val_labels, val_preds_bin)
        tn, fp, fn, tp = cm.ravel()
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        val_loss = 0.0  # Skip loss for TTA

        val_metrics = {
            'loss': val_loss, 'accuracy': val_accuracy, 'precision': val_precision,
            'recall': val_recall, 'f1': val_f1, 'auc': val_auc, 'specificity': val_specificity
        }

        # Append to history
        for k, v in train_metrics.items():
            history[f'train_{k}'].append(v)
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)
        history['learning_rates'].append(current_lr)
        history['val_thresholds'].append(optimal_thresh)

        val_probs_list.append({'probs': tta_probs, 'labels': val_labels, 'preds': val_preds_bin})

        # Log
        print(f"\nFine-Tune Epoch {global_epoch}/{start_epoch + fine_tune_epochs} | LR: {current_lr:.2e}")
        print(f"Train: Loss={train_metrics['loss']:.4f} F1={train_metrics['f1']:.4f} Prec={train_metrics['precision']:.4f}")
        print(f"Val (TTA+Thresh={optimal_thresh:.3f}): F1={val_f1:.4f} Prec={val_precision:.4f} Rec={val_recall:.4f} AUC={val_auc:.4f}")
        print(f"Val CM: TN={tn} FP={fp} FN={fn} TP={tp}")

        # Scheduler step (on val F1)
        scheduler.step(val_f1)

        # Best model save (F1 + low FP priority)
        if val_f1 > best_val_f1 or (abs(val_f1 - best_val_f1) < 0.005 and fp < history.get('best_fp', float('inf'))):
            best_val_f1 = val_f1
            print(f"🎉 New best! Val F1: {val_f1:.4f}, FP: {fp}, Thresh: {optimal_thresh:.3f}")

            # Save with fine-tune flag
            torch.save({
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1, 'val_precision': val_precision, 'val_auc': val_auc,
                'val_threshold': optimal_thresh, 'val_cm': cm.tolist(),
                'history': dict(history),
                'hyperparams': {},  # Reuse old
                'fine_tune': fine_tune_config
            }, os.path.join(ckpt_dir, "enhanced_best_finetuned.pth"))

            history['best_fp'] = fp
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save latest
        latest_fine_ckpt = os.path.join(ckpt_dir, "enhanced_latest_finetuned.pth")
        torch.save({
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': dict(history),
            'fine_tune': fine_tune_config
        }, latest_fine_ckpt)

        # Save history
        save_json_serializable(dict(history), os.path.join(ckpt_dir, "enhanced_training_history_finetune.json"))

        if epochs_no_improve >= fine_tune_config['patience']:
            print(f"🛑 Fine-tune early stop at epoch {global_epoch}")
            break

    print(f"✅ Fine-Tune Complete! Best Val F1: {best_val_f1:.4f}")
    print("📊 Run enhanced_evaluate_model(model, val_loader, device, ckpt_dir, threshold=optimal_thresh) for full eval")
    print("🔍 Re-run plot_enhanced_training_history(history, ckpt_dir) to see extended curve")

    # Quick post-fine-tune eval (with optimal thresh)
    optimal_thresh = history['val_thresholds'][-1]
    eval_results = enhanced_evaluate_model(model, val_loader, device, ckpt_dir, threshold=optimal_thresh)
    print(f"Final Val F1 with TTA+Optimal Thresh: {eval_results['f1']:.4f}")

if __name__ == "__main__":
    run_finetune()
