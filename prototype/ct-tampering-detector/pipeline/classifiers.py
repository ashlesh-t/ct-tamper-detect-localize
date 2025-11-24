from logs.logger import get_logger
from pipeline.classifierPipe.Inject_Removal import InjectRemoval
from pipeline.classifierPipe.Real_Fake import RealFake

logger = get_logger(__name__)
class TamperClassifiers:
    def __init__(self,data : dict,length : int ,type = 1):
        #Type 1: Real-Fake classifier
        #Type Injected Removed classifier
        if type ==1 :
            self.obj = RealFake(data,length)
            logger.info("Started Real-Fake classifier")
        elif type ==2:
            self.obj = InjectRemoval(data,length)
            logger.info("Started Injected-Removed classifier")
    def get_results(self):
        pass
            