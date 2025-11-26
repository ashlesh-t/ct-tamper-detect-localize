# pipeline/classifierPipe/Real_Fake.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from torch.utils.data import DataLoader
import os

from pipeline.architectures.achitecture import MultiStreamCTModel, DenseNetBinary
from pipeline.data_loaders.inference_dataset import MultiChannelCTDataset, EvalTransforms
from pipeline.config.configs import ModelConfig
from pipeline.types.types import Types  # Import Types for consistent labeling

logger = logging.getLogger(__name__)

class RealFake:
    def __init__(self, data: List[Dict[str, Any]], length: int):
        self.data = data
        self.length = length
        self.config = ModelConfig()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.last_slice_details = None
        self.last_volume_stats = None
        self.types = Types()  # For consistent classification labels
        
    def load_model(self):
        """Load the trained model with proper weight handling"""
        try:
            # Use DenseNet to match checkpoint (dn_finetune_best.pth)
            self.model = DenseNetBinary(pretrained=False, num_classes=self.config.NUM_CLASSES).to(self.device)
            
            # Check if model file exists
            if not os.path.exists(self.config.REAL_FAKE_MODEL_PATH):
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {self.config.REAL_FAKE_MODEL_PATH}\n"
                    f"Please update REAL_FAKE_MODEL_PATH in config.py"
                )
            
            checkpoint = torch.load(
                self.config.REAL_FAKE_MODEL_PATH, 
                map_location=self.device,
                weights_only=False
            )
            
            # Handle different checkpoint formats from your training script
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load state dict with strict=True (now matches arch)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            
            logger.info(f"DenseNet model loaded successfully from {self.config.REAL_FAKE_MODEL_PATH}")
            if 'best_f1' in checkpoint:
                logger.info(f"Model was trained with best F1: {checkpoint['best_f1']:.4f}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_dataloader(self) -> DataLoader:
        """Create dataloader for inference with proper transforms"""
        dataset = MultiChannelCTDataset(
            slice_data=self.data,
            transform=EvalTransforms(img_size=self.config.IMG_SIZE),
            img_size=self.config.IMG_SIZE
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def run_inference(self) -> Tuple[List[str], List[float], List[int]]:
        """Run inference on all slices"""
        if self.model is None:
            self.load_model()
        
        dataloader = self.create_dataloader()
        
        all_filenames = []
        all_probs_fake = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device, non_blocking=True).float()
                fnames = batch['fnames']
                
                # DenseNet forward (no split, no autocast needed)
                outputs = self.model(images)  # logits (B, 2)
                outputs = outputs.float()  # Ensure float32 for consistency
                
                probs = torch.softmax(outputs, dim=1)
                probs_fake = probs[:, 1].cpu().numpy()  # Probability of class 1 (Fake)
                predictions = (probs_fake > 0.5).astype(int)  # Threshold match standalone
                
                all_filenames.extend(fnames)
                all_probs_fake.extend(probs_fake)
                all_predictions.extend(predictions)
        
        return all_filenames, all_probs_fake, all_predictions
    
    def aggregate_volume_prediction(self, 
                                  filenames: List[str], 
                                  probs_fake: List[float],
                                  predictions: List[int]) -> Dict[str, Any]:
        """Aggregate slice-level predictions to volume-level prediction"""
        
        # Calculate volume-level confidence (mean probability of fake)
        volume_confidence_fake = np.mean(probs_fake)
        volume_confidence_real = 1 - volume_confidence_fake
        
        # Determine volume classification (using 0.5 threshold as in training/standalone)
        if volume_confidence_fake > 0.5:
            volume_classification = self.types.type2  # "FAKE"
            volume_confidence = volume_confidence_fake
        else:
            volume_classification = self.types.type1  # "REAL"
            volume_confidence = volume_confidence_real
        
        # Identify affected slices (slices predicted as fake)
        affected_slices = [
            filenames[i] for i in range(len(filenames)) 
            if predictions[i] == 1  # Fake prediction
        ]
        
        # Calculate slice-level statistics
        slice_stats = {
            'total_slices': len(filenames),
            'slices_predicted_real': sum(1 for p in predictions if p == 0),
            'slices_predicted_fake': sum(1 for p in predictions if p == 1),
            'mean_fake_confidence': float(np.mean(probs_fake)),
            'std_fake_confidence': float(np.std(probs_fake)),
            'min_fake_confidence': float(np.min(probs_fake)),
            'max_fake_confidence': float(np.max(probs_fake))
        }
        
        # Store slice details for detailed reporting
        slice_details = [
            {
                'filename': filenames[i],
                'prediction': self.types.type2 if predictions[i] == 1 else self.types.type1,  # "FAKE" or "REAL"
                'fake_confidence': float(probs_fake[i]),
                'prediction_binary': int(predictions[i])
            }
            for i in range(len(filenames))
        ]
        
        return {
            'volume_classification': volume_classification,
            'volume_confidence': float(volume_confidence),
            'affected_slices': affected_slices,
            'slice_statistics': slice_stats,
            'slice_details': slice_details
        }
    
    def get_results(self) -> Tuple[int, Any, List[str], Exception]:
        """Main method to get classification results"""
        try:
            logger.info("Starting Real-Fake classification with DenseNetBinary")
            
            # Run inference
            filenames, probs_fake, predictions = self.run_inference()
            
            # Aggregate results
            results = self.aggregate_volume_prediction(filenames, probs_fake, predictions)
            
            # Store for detailed reporting
            self.last_slice_details = results['slice_details']
            self.last_volume_stats = results['slice_statistics']
            
            # Format return values to match your existing pipeline
            status = 200
            classification_result = (
                results['volume_classification'], 
                results['volume_confidence']
            )
            affected_filenames = results['affected_slices']
            
            logger.info(
                f"Classification complete: {results['volume_classification']} "
                f"(confidence: {results['volume_confidence']:.3f}, "
                f"affected slices: {len(affected_filenames)}/{len(filenames)})"
            )
            
            return status, classification_result, affected_filenames, None
            
        except Exception as e:
            logger.error(f"Real-Fake classification failed: {e}")
            return 500, None, [], e