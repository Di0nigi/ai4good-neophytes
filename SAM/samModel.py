import torch
from torch import nn


import os
os.environ["HF_TOKEN"] = ""


from PIL import Image
#from sam3.model_builder import build_sam3_image_model
#from sam3.model.sam3_image_processor import Sam3Processor

from transformers import AutoProcessor, AutoModel


processor = AutoProcessor.from_pretrained("facebook/sam3")
model = AutoModel.from_pretrained("facebook/sam3", device_map="auto")

#model = build_sam3_image_model()
#processor = Sam3Processor(model)

image = Image.open("C:\\Users\\dionigi\\Pictures\\elephant.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]



class CustomSam(nn.Module):
    def __init__():

        return