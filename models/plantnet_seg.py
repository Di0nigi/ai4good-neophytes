"""Frozen Pl@ntNet SimDINOv2 backbone with a linear segmentation probe."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenPlantNetLinearSegmenter(nn.Module):
    """Turn spatial ViT patch tokens into dense seven-class logits.

    The Pl@ntNet SimDINOv2 checkpoint emits one CLS token, register tokens, and
    spatial patch tokens. Only the spatial tokens are reshaped into a feature map.
    The backbone stays frozen and in evaluation mode; the 1x1 convolution is the
    only trainable component.

    ``backbone`` is injectable to make shape/gradient tests fully offline. Normal
    training leaves it unset and loads ``model_id`` from Hugging Face.
    """

    def __init__(self, model_id: str, num_classes: int = 7,
                 backbone: Optional[nn.Module] = None):
        super().__init__()
        if backbone is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "PlantNetLinear requires transformers>=4.48 and safetensors. "
                    "Install the dependencies from environment.yml."
                ) from exc
            backbone = AutoModel.from_pretrained(model_id)

        self.model_id = model_id
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        hidden_size = int(self.backbone.config.hidden_size)
        self.patch_size = self._as_pair(self.backbone.config.patch_size)
        self.num_register_tokens = int(self.backbone.config.num_register_tokens)
        self.head = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    @staticmethod
    def _as_pair(value):
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def train(self, mode: bool = True):
        """Train the probe while keeping the frozen backbone in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]
        patch_h, patch_w = self.patch_size
        if input_h % patch_h or input_w % patch_w:
            raise ValueError(
                f"Input size {(input_h, input_w)} must be divisible by the "
                f"backbone patch size {(patch_h, patch_w)}."
            )

        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)

        # Token order: CLS, register tokens, then row-major spatial patch tokens.
        patch_tokens = outputs.last_hidden_state[:, 1 + self.num_register_tokens:]
        grid_h, grid_w = input_h // patch_h, input_w // patch_w
        expected_tokens = grid_h * grid_w
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Backbone returned {patch_tokens.shape[1]} spatial tokens, expected "
                f"{expected_tokens} for input {(input_h, input_w)} and patch size "
                f"{(patch_h, patch_w)}."
            )

        features = patch_tokens.transpose(1, 2).reshape(
            x.shape[0], patch_tokens.shape[2], grid_h, grid_w
        )
        logits = self.head(features)
        return F.interpolate(
            logits, size=(input_h, input_w), mode="bilinear", align_corners=False
        )
