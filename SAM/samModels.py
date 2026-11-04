

from transformers import Sam3Model, Sam3Processor
from torch import nn
import torchvision.models as models




token = ""

class SamEnsemble: # Regular Sam 3 with Neophyte classifier
    def __init___(self):
        self.samModel = Sam3Model.from_pretrained()
        self.samProcessor = Sam3Processor.from_pretrained()
        self.classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        return
    

'''
class SamSegHead: # Vanilla Sam3 Segmentation Pipeline 
    def __init___():
        return


class SamCustom: #Example encoder + Segmentation head
    def __init___(self):

        return

    def forward(self):
        return
    

class CustomSam3(Sam3Model):
    def __init__(self):

        super.__init__()

        self.backbone = ...
        self.promptEncoder = ...
        self.semanticHead= ...

        return
'''
