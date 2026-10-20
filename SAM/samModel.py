import torch
from torch import nn
import numpy as np
import matplotlib


import os
os.environ["HF_TOKEN"] = ""


from PIL import Image
from transformers import Sam3Model, Sam3Processor


class CustomSam(nn.Module):
    def __init__(self):

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
      
        for p in self.model.parameters():
            p.requires_grad = False

        
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True

        for p in self.model.dot_product_scoring.parameters():
            p.requires_grad = True

        # semantic segmentation layer 

        return

    @torch.inference_mode()
    def runInference(self, image, prompt=None):

        self.model.eval()

        with Image.open(image) as source:
            image = source.convert("RGB")

        inputs = self.processor(
            images=image,
            text=self.prompt if prompt is None else prompt,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        return result["masks"]

    def overlay_masks(self,image, masks):
        image = image.convert("RGBA")
        masks = 255 * masks.cpu().numpy().astype(np.uint8)
        
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image

    
    def trainModel(self):




        return