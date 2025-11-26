import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "pipeline", "models")

class Config:
    REAL_FAKE_MODEL_PATH = os.path.join(MODELS_DIR, "classifier1", "dn_finetune_best.pth")
    INJECTED_REMOVED_MODEL_PATH = os.path.join(MODELS_DIR, "classifier2", "best_classifier_v2.pth")
    BEST_CHECKPOINT = os.path.join(MODELS_DIR, "Inject_localize", "best_model.pth")

    REMOVAL_LOCALIZATION_DIR = os.path.join(MODELS_DIR, "Remove_localize")
    REMOVAL_BEST_DICE_MODEL = os.path.join(REMOVAL_LOCALIZATION_DIR, "best_dice_model.pth")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Localization parameters
    LOCALIZATION_IMG_SIZE = 320
    REMOVAL_IMG_SIZE = 288 
    BATCH_SIZE = 4
 

class ModelConfig:
    # Model paths - update these based on your actual saved models
    REAL_FAKE_MODEL_PATH = Config.REAL_FAKE_MODEL_PATH
    INJECTED_REMOVED_MODEL_PATH = Config.INJECTED_REMOVED_MODEL_PATH
    
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