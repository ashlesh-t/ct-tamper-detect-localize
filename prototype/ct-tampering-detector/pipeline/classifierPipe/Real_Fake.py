# pipeline/classifierPipe/Real_Fake.py
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from torch.utils.data import DataLoader
import os
import json

from pipeline.architectures.architecture import MultiStreamCTModel
from pipeline.data_loaders.inference_dataset import MultiChannelCTDataset, EvalTransforms

logger = logging.getLogger(__name__)

class RealFake:
    def __init__(self, data: List[Dict[str, Any]], length: int):
        self.data = data
        self.length = length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        return {
            'img_size': 384,
            'batch_size': 16,
            'num_classes': 2,
            'model_checkpoint': self._find_model_checkpoint(),
            'device': self.device
        }
    
    def _find_model_checkpoint(self) -> str:
        """Find the appropriate model checkpoint"""
        # You can modify this path based on your model location
        possible_paths = [
            "models/binary_ct_ultimate_v6.0_finetune/ultimate_best_model.pth",
            "capstone_models/binary_ct_ultimate_v6.0_finetune/ultimate_best_model.pth",
            "/content/drive/MyDrive/capstone_models/binary_ct_ultimate_v6.0_finetune/ultimate_best_model.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found model checkpoint at: {path}")
                return path
        
        raise FileNotFoundError("Could not find model checkpoint. Please update the path in Real_Fake.py")
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = MultiStreamCTModel(num_classes=self.config['num_classes']).to(self.config['device'])
            
            checkpoint = torch.load(self.config['model_checkpoint'], 
                                  map_location=self.config['device'],
                                  weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_dataloader(self) -> DataLoader:
        """Create dataloader for inference"""
        dataset = MultiChannelCTDataset(
            slice_data=self.data,
            transform=EvalTransforms(img_size=self.config['img_size']),
            img_size=self.config['img_size']
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
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
                images = batch['images'].to(self.config['device'])
                fnames = batch['fnames']
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                probs_fake = probs[:, 1].cpu().numpy()  # Probability of class 1 (Fake)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
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
        
        # Determine volume classification
        if volume_confidence_fake > 0.5:
            volume_classification = "Fake"
            volume_confidence = volume_confidence_fake
        else:
            volume_classification = "Real" 
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
            'std_fake_confidence': float(np.std(probs_fake))
        }
        
        return {
            'volume_classification': volume_classification,
            'volume_confidence': float(volume_confidence),
            'affected_slices': affected_slices,
            'slice_statistics': slice_stats,
            'slice_details': [
                {
                    'filename': filenames[i],
                    'prediction': 'Fake' if predictions[i] == 1 else 'Real',
                    'fake_confidence': float(probs_fake[i])
                }
                for i in range(len(filenames))
            ]
        }
    
    def get_results(self) -> Tuple[int, Any, List[str], Exception]:
        """Main method to get classification results"""
        try:
            logger.info("Starting Real-Fake classification")
            
            # Run inference
            filenames, probs_fake, predictions = self.run_inference()
            
            # Aggregate results
            results = self.aggregate_volume_prediction(filenames, probs_fake, predictions)
            
            # Format return values to match your existing pipeline
            status = 200
            classification_result = (
                results['volume_classification'], 
                results['volume_confidence']
            )
            affected_filenames = results['affected_slices']
            
            logger.info(f"Classification complete: {results['volume_classification']} "
                       f"(confidence: {results['volume_confidence']:.3f})")
            
            return status, classification_result, affected_filenames, None
            
        except Exception as e:
            logger.error(f"Real-Fake classification failed: {e}")
            return 500, None, [], e