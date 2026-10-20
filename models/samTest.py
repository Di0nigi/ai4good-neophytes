import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor

class SAMNeophyte(torch.nn.Module):
    def __init__(self, checkpoint, model_type='vit_b', device='cuda'):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device)
        # Freeze image encoder and prompt encoder; train only mask decoder
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = False
        # Mask decoder stays trainable

    def forward(self, image, boxes):
        """
        image: (B, 3, H, W) preprocessed tensor (SAM preprocessing applied)
        boxes: list of (N_i, 4) tensors, one per image in batch
        """
        # SAM's forward expects sparse embeddings from prompts.
        # The official API is not batched for variable prompts, so loop here.
        # For production, use the SamPredictor's transform and batching utilities.
        raise NotImplementedError("Use SamPredictor for single-prompt inference during development.")

def train_step(model, predictor, image_np, prompts, optimizer):
    """One gradient step for a single image with all its prompts."""
    # Preprocess image with SAM's transforms
    predictor.set_image(image_np)  # caches image embeddings; image encoder frozen
    total_loss = 0.0
    for box, target_mask in prompts:
        # Predict with box prompt
        masks, iou_pred, low_res_logits = predictor.predict(
            box=box[None, :],
            multimask_output=False,
            return_logits=True,
        )
        # masks: (1, H, W) boolean or logits depending on return_logits
        pred = torch.from_numpy(low_res_logits).float().to(predictor.device)
        target = torch.from_numpy(target_mask).float().to(predictor.device)
        target_low = F.interpolate(
            target[None, None], size=pred.shape[-2:], mode='bilinear'
        ).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, target_low)
        total_loss += loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item() / len(prompts)