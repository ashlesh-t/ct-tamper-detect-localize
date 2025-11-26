# pipeline/localize.py
from logs.logger import get_logger
from pipeline.localizePipe.Injected import Injected
from pipeline.localizePipe.Removed import Removed
from typing import List, Dict, Any

logger = get_logger(__name__)

class Localize:
    def __init__(self, data: List[Dict[str, Any]], length: int, fnames: List[str], type: int = 1):
        """
        Args:
            data: List of slice data dicts
            length: Number of slices
            fnames: Affected filenames to localize
            type: 1 for Injected, 2 for Removed
        """
        self.data = data
        self.length = length
        self.fnames = fnames
        self.type = type
        
        if type == 1:
            self.obj = Injected(data, length)
            logger.info("Started Injected Localization")
        elif type == 2:
            self.obj = Removed(data, length)
            logger.info("Started Removed Localization")
        else:
            raise ValueError(f"Invalid localization type: {type}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get localization results for affected filenames"""
        return self.obj.get_results(self.fnames)