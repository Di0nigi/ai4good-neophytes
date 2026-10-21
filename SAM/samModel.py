import torch
from torch import nn
import numpy as np
import matplotlib

from pathlib import Path


import os
os.environ["HF_TOKEN"] = ""


from PIL import Image
from transformers import Sam3Model, Sam3Processor

#from sam_loss import sam_object_loss
CLASS_PROMPTS = {
    1: "butterfly bush",      # Buddleja davidii
    2: "staghorn sumac",      # Rhus typhina
    3: "Japanese knotweed",   # Reynoutria japonica
    4: "Turkish wartycabbage",# Bunias orientalis
    5: "narrow-leaved ragwort", # Senecio inaequidens
    6: "tree of heaven",      # Ailanthus altissima
}



class CustomSam(nn.Module):
    def __init__(self):

        super().__init__()

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
      
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True

        for p in self.model.detr_decoder.parameters():
            p.requires_grad=True

        for p in self.model.dot_product_scoring.parameters():
            p.requires_grad = True

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

    '''@torch.inference_mode()
    def runSemanticInference(self, image, threshold=0.5):
            """Per-class prompt + argmax; IDs match NeophyteDataset exactly."""
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if isinstance(image, (str, Path)):
                with Image.open(image) as source:
                    image = source.convert("RGB")
            else:
                image = image.convert("RGB")
            was_training = self.training
            self.eval()
            try:
                maps = []
                for prompt in CLASS_PROMPTS.values():
                    outputs = self([image], [prompt])
                    scores = outputs.pred_logits[0].sigmoid().reshape(-1)
                    if outputs.presence_logits is not None:
                        scores = scores * outputs.presence_logits[0].sigmoid().reshape(())
                    probabilities = F.interpolate(outputs.pred_masks[0, :, None],
                        size=(image.height, image.width), mode="bilinear", align_corners=False)[:, 0].sigmoid()
                    maps.append((probabilities * scores[:, None, None]).amax(0).cpu())
                confidence, index = torch.stack(maps).max(0)
                ids = torch.tensor(list(CLASS_PROMPTS))
                return ids[index].masked_fill(confidence < threshold, 0)
            finally:
                self.train(was_training)'''

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

    
    def trainModel(self, train_loader, val_loader=None, epochs=10, lr=1e-4,
                   checkpoint_dir="SAM/checkpoints", resume=None):
        
        if epochs < 1 or len(train_loader) == 0:
            raise ValueError("Need positive epochs and a nonempty training loader")
        if val_loader is not None and len(val_loader) == 0:
            raise ValueError("Validation loader is empty")
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=lr)
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        start, best, history = 0, float("inf"), []
        if resume is not None:
            state = torch.load(resume, map_location=next(self.parameters()).device, weights_only=True)
            if state["prompts"] != CLASS_PROMPTS:
                raise ValueError("Checkpoint class prompts do not match")
            if state.get("objective") != "sam_object_v1":
                raise ValueError("Checkpoint uses a different training objective")
            self.model.load_state_dict(state["trainable"], strict=False)
            optimizer.load_state_dict(state["optimizer"])
            start, best, history = state["epoch"], state["best_loss"], state["history"]

        for epoch in range(start, start + epochs):
            train_loss = self.run_epoch(train_loader, True,optimizer,parameters)
            val_loss = self.run_epoch(val_loader, False,optimizer,parameters) if val_loader is not None else None
            monitored = train_loss if val_loss is None else val_loss
            improved = monitored < best
            best = min(best, monitored)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            names = {name for name, p in self.model.named_parameters() if p.requires_grad}
            state = {"objective": "sam_object_v1",
                     "trainable": {name: value.detach().cpu() for name, value in self.model.state_dict().items() if name in names},
                     "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                     "best_loss": best, "history": history, "prompts": CLASS_PROMPTS,
                     "base_model": "facebook/sam3"}
            torch.save(state, directory / "latest.pt")
            if improved:
                torch.save(state, directory / "best.pt")
            print(history[-1], flush=True)
        self.eval()
        return history

    def run_epoch(self, loader, training, optimizer, parameters):
        self.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch in loader:
                keep = [i for i, valid in enumerate(batch["valid"]) if valid.any()]
                if not keep:
                    continue
                
                # 1. Prepare segmentation_targets (list of dicts with "masks")
                #    Each element corresponds to one image; "masks" is a list of mask paths/tensors
                segmentation_targets = [
                    {"masks": batch["mask"][i]}  # or a list of mask paths
                    for i in keep
                ]
                
                # 2. Call the processor with images, text prompts, and targets
                inputs = self.processor(
                    images=[batch["image"][i] for i in keep],
                    text=[batch["prompt"][i] for i in keep],
                    segmentation_targets=segmentation_targets,
                    return_tensors="pt",
                ).to(self.model.device)
                
                if training:
                    optimizer.zero_grad(set_to_none=True)
                
                # 3. Forward pass — loss is computed automatically
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                if loss is None:
                    raise ValueError("Model did not return a loss — check segmentation_targets format")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                
                total += loss.item() * len(keep)
                count += len(keep)
        
        if not count:
            raise ValueError("Loader contains no valid labeled pixels")
        return total / count

    def train(self, mode=True):
            super().train(mode)
            self.model.eval()
            for name in ("mask_decoder", "detr_decoder", "dot_product_scoring"):
                getattr(self.model, name).train(mode)
            return self
    
    def forward(self, images, prompts):
            inputs = self.processor(images=images, text=prompts, return_tensors="pt",
                                    padding=True).to(next(self.parameters()).device)
            # Do not use inference_mode/no_grad: decoder gradients are required.
            return self.model(**inputs)