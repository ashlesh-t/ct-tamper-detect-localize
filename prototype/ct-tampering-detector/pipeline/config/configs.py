# pipeline/config/configs.py
import os
import torch
# config/configs.py
import os

# config/configs.py
import os
import torch

class Config:
    # Model paths
    BEST_CHECKPOINT = "models/CT_Injection_finetune_phases_v3.3/best_model.pth"
    REAL_FAKE_MODEL_PATH = os.path.join("models", "binary_ct_kaggle_v1", "finetune", "ms_finetuned.pth")
    INJECTED_REMOVED_MODEL_PATH = os.path.join("models", "efficientnet-b2-v13", "best_classifier_v2.pth")
    
    # Removal Localization paths
    REMOVAL_LOCALIZATION_DIR = "/content/drive/MyDrive/capstone_models/MultiChannelUNet-v1"
    REMOVAL_BEST_DICE_MODEL = os.path.join(REMOVAL_LOCALIZATION_DIR, "best_dice_model.pth")
    REMOVAL_BEST_LOSS_MODEL = os.path.join(REMOVAL_LOCALIZATION_DIR, "best_loss_model.pth")
    REMOVAL_SPLIT_INDICES = os.path.join(REMOVAL_LOCALIZATION_DIR, "split_indices.pth")
    
    # Data paths for removal localization
    REMOVAL_DATA_ROOT = '/content/drive/MyDrive/Capstone/main/CT_Removal'
    REMOVAL_CSV_PATH = os.path.join(REMOVAL_DATA_ROOT, 'data_v1.csv')
    
    # Localization parameters
    LOCALIZATION_IMG_SIZE = 320
    REMOVAL_IMG_SIZE = 512  # Different from injection localization
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

class ModelConfig:
    # Model paths - update these based on your actual saved models
    REAL_FAKE_MODEL_PATH = os.path.join("models", "binary_ct_kaggle_v1", "finetune", "ms_finetuned.pth")
    INJECTED_REMOVED_MODEL_PATH = os.path.join("models", "injected_removed_model.pth")  # Update if you have this
    
    # Training parameters that match your training script
    IMG_SIZE = 384
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    
    INJ_REM_IMG_SIZE = 288  # Different from real/fake model
    INJ_REM_MEAN = [0.5, 0.5, 0.5]
    INJ_REM_STD = [0.5, 0.5, 0.5]
    # Inference parameters
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model architecture parameters
    NUM_CLASSES = 2
    FEATURE_DIM = 1280