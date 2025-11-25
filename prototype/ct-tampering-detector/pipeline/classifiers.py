# pipeline/classifiers.py
from typing import Dict, List
from logs.logger import get_logger
from pipeline.classifierPipe.Inject_Removal import InjectRemoval
from pipeline.classifierPipe.Real_Fake import RealFake

logger = get_logger(__name__)

class TamperClassifiers:
    def __init__(self, data: List[Dict], length: int, type: int = 1):
        """
        Args:
            data: List of preprocessed data dicts
            length: Number of files
            type: 1 for Real-Fake, 2 for Injected-Removed
        """
        self.data = data
        self.length = length
        self.type = type
        
        if type == 1:
            self.obj = RealFake(data, length)
            logger.info("Started Enhanced Real-Fake classifier with multi-channel input")
        elif type == 2:
            self.obj = InjectRemoval(data, length)
            logger.info("Started Injected-Removed classifier")
        else:
            raise ValueError(f"Unknown classifier type: {type}")
    
    def get_results(self):
        """Get classification results"""
        return self.obj.get_results()