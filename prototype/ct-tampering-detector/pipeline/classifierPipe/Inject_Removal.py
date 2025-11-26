# pipeline/classifierPipe/Inject_Removal.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from torch.utils.data import DataLoader
import os

from pipeline.architectures.injected_removed_arch import EnhancedDualStreamFakeCT
from pipeline.data_loaders.inference_dataset import MultiChannelCTDataset, InjRemEvalTransforms
from pipeline.config.configs import ModelConfig

logger = logging.getLogger(__name__)

class InjectRemoval:
    def __init__(self, data: List[Dict[str, Any]], length: int):
        self.data = data
        self.length = length
        self.config = ModelConfig()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.last_slice_details = None
        self.last_volume_stats = None
        
    def load_model(self):
        """Load the injected/removed classification model"""
        try:
            self.model = EnhancedDualStreamFakeCT(num_classes=2).to(self.device)
            
            # Check if model file exists
            if not os.path.exists(self.config.INJECTED_REMOVED_MODEL_PATH):
                raise FileNotFoundError(
                    f"Injected/Removed model checkpoint not found at: {self.config.INJECTED_REMOVED_MODEL_PATH}\n"
                    f"Please update INJECTED_REMOVED_MODEL_PATH in config.py"
                )
            
            checkpoint = torch.load(
                self.config.INJECTED_REMOVED_MODEL_PATH, 
                map_location=self.device,
                weights_only=False
            )
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.channel_stats = checkpoint.get('channel_stats')
            else:
                state_dict = checkpoint
                self.channel_stats = None
                
            # Load state dict
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Set default stats if not found
            if not self.channel_stats:
                self.channel_stats = {
                    0: {'mean': 0.5, 'std': 0.5}, 
                    1: {'mean': 0.5, 'std': 0.5}, 
                    2: {'mean': 0.5, 'std': 0.5}
                }
            
            logger.info(f"Injected/Removed model loaded successfully from {self.config.INJECTED_REMOVED_MODEL_PATH}")
            
        except Exception as e:
            logger.error(f"Error loading injected/removed model: {e}")
            raise
    
    def create_dataloader(self) -> DataLoader:
        """Create dataloader for injected/removed inference"""
        dataset = MultiChannelCTDataset(
            slice_data=self.data,
            transform=InjRemEvalTransforms(
                img_size=self.config.INJ_REM_IMG_SIZE,
                channel_stats=self.channel_stats
            ),
            img_size=self.config.INJ_REM_IMG_SIZE
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    def run_inference(self) -> Tuple[List[str], List[float], List[int], List[float]]:
        """Run inference on all slices for injected/removed classification"""
        if self.model is None:
            self.load_model()
        
        dataloader = self.create_dataloader()
        
        all_filenames = []
        all_probs_injected = []  # Probability of class 0 (Injected)
        all_probs_removed = []   # Probability of class 1 (Removed)
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device, non_blocking=True).float()
                fnames = batch['fnames']
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                probs_injected = probs[:, 0].cpu().numpy()  # Class 0: Injected
                probs_removed = probs[:, 1].cpu().numpy()   # Class 1: Removed
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_filenames.extend(fnames)
                all_probs_injected.extend(probs_injected)
                all_probs_removed.extend(probs_removed)
                all_predictions.extend(predictions)
        
        return all_filenames, all_probs_injected, all_probs_removed, all_predictions
    
    def aggregate_predictions(self, 
                            filenames: List[str], 
                            probs_injected: List[float],
                            probs_removed: List[float],
                            predictions: List[int]) -> Dict[str, Any]:
        """Aggregate slice-level predictions for injected/removed classification"""
        
        # Separate slices by prediction
        injected_slices = [
            filenames[i] for i in range(len(filenames)) 
            if predictions[i] == 0  # Injected prediction
        ]
        
        removed_slices = [
            filenames[i] for i in range(len(filenames)) 
            if predictions[i] == 1  # Removed prediction
        ]
        
        # Calculate volume-level confidence
        volume_confidence_injected = np.mean(probs_injected)
        volume_confidence_removed = np.mean(probs_removed)
        
        # Determine dominant class
        if volume_confidence_injected > volume_confidence_removed:
            volume_classification = "Injected"
            volume_confidence = volume_confidence_injected
        else:
            volume_classification = "Removed"
            volume_confidence = volume_confidence_removed
        
        # Calculate slice-level statistics
        slice_stats = {
            'total_slices': len(filenames),
            'slices_predicted_injected': sum(1 for p in predictions if p == 0),
            'slices_predicted_removed': sum(1 for p in predictions if p == 1),
            'mean_injected_confidence': float(np.mean(probs_injected)),
            'mean_removed_confidence': float(np.mean(probs_removed)),
            'std_injected_confidence': float(np.std(probs_injected)),
            'std_removed_confidence': float(np.std(probs_removed))
        }
        
        # Store slice details for detailed reporting
        slice_details = [
            {
                'filename': filenames[i],
                'prediction': 'Injected' if predictions[i] == 0 else 'Removed',
                'injected_confidence': float(probs_injected[i]),
                'removed_confidence': float(probs_removed[i]),
                'prediction_binary': int(predictions[i])
            }
            for i in range(len(filenames))
        ]
        
        return {
            'volume_classification': volume_classification,
            'volume_confidence': float(volume_confidence),
            'affected_slices': [injected_slices, removed_slices],  # Format: [injected, removed]
            'slice_statistics': slice_stats,
            'slice_details': slice_details,
            'class_breakdown': {
                'injected_count': len(injected_slices),
                'removed_count': len(removed_slices),
                'injected_confidence': float(volume_confidence_injected),
                'removed_confidence': float(volume_confidence_removed)
            }
        }
    
    def get_results(self) -> Tuple[int, Any, List[str], Exception]:
        """Main method to get injected/removed classification results"""
        try:
            logger.info("Starting Injected/Removed classification with EnhancedDualStreamFakeCT")
            
            # Run inference
            filenames, probs_injected, probs_removed, predictions = self.run_inference()
            
            # Aggregate results
            results = self.aggregate_predictions(filenames, probs_injected, probs_removed, predictions)
            
            # Store for detailed reporting
            self.last_slice_details = results['slice_details']
            self.last_volume_stats = results['slice_statistics']
            
            # Format return values to match pipeline expectations
            status = 200
            classification_result = (
                results['volume_classification'], 
                results['volume_confidence']
            )
            # Return both injected and removed slices in expected format
            affected_filenames = results['affected_slices']  # [injected_slices, removed_slices]
            
            logger.info(
                f"Injected/Removed classification complete: {results['volume_classification']} "
                f"(confidence: {results['volume_confidence']:.3f}, "
                f"injected: {len(affected_filenames[0])}, removed: {len(affected_filenames[1])})"
            )
            
            return status, classification_result, affected_filenames, None
            
        except Exception as e:
            logger.error(f"Injected/Removed classification failed: {e}")
            return 500, None, [], e