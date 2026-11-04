"""Differentiable foreground supervision for SAM 3 proposal outputs."""

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def foreground_seg_loss(pred_masks, pred_logits, targets, ignore_index=-1, bce_weight=1.0, dice_weight=1.0):

    if pred_logits.ndim == 3:
        pred_logits = pred_logits.squeeze(-1)

    # resize predictions to ground truth mask resolution
    mask_logits = F.interpolate(pred_masks.float(), size=targets.shape[-2:], mode="bilinear", align_corners=False)

    # Get confidence-weighted mask probabilities
    probabilities = mask_logits.sigmoid() * pred_logits.float().reshape(*mask_logits.shape[:2], 1, 1).sigmoid()
    probabilities = probabilities.clamp(1e-6, 1 - 1e-6)

    foreground = 1 - (1-probabilities).prod(dim=1)
    foreground = foreground.clamp(1e-6, 1 - 1e-6)

    # Keep valid pixels and merge all species into foreground.
    valid = targets != ignore_index
    truth = (targets > 0).float()

    if not valid.any():
        zero = foreground.sum() * 0
        return {"loss": zero, "bce": zero, "dice": zero}

    # Binary cross-entropy on valid pixels.
    bce = F.binary_cross_entropy(foreground[valid], truth[valid])

    # Dice loss on valid pixels, calculated per image.
    prediction = foreground * valid
    truth = truth * valid

    intersection = (prediction * truth).sum(dim=(-2, -1))
    total = prediction.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    dice_per_image = 1 - (2 * intersection + 1) / (total + 1)

    dice = dice_per_image[valid.flatten(1).any(dim=1)].mean()

    return {"loss": bce + dice, "bce": bce, "dice": dice}






