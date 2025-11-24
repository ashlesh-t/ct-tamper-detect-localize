# localize.py (updated)
from logs.logger import get_logger
from pipeline.localizePipe import Injected, Removed
from typing import List, Dict, Any

logger = get_logger(__name__)

class Localize:
    def __init__(self, data: List[Dict[str, Any]], length: int, fnames: List[str], type: int = 1):
        self.data = data
        self.length = length
        self.fnames = fnames
        self.type = type
        if type == 1:
            self.obj = Injected(data, length)
            logger.info("Started Injected Localization")
        elif type == 2:
            self.obj = Removed(data, length)  # Implement similarly for Removed
            logger.info("Started Removed Localization")
        else:
            raise ValueError(f"Invalid type: {type}")

    def get_results(self) -> List[Dict[str, Any]]:
        return self.obj.get_results(self.fnames)