import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.plantnet_seg import FrozenPlantNetLinearSegmenter


class SyntheticRegisterBackbone(nn.Module):
    """Small offline stand-in; no Hugging Face or network access required."""

    def __init__(self, hidden_size=32, patch_size=16, num_register_tokens=4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
        )
        self.patch_projection = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.prefix_value = 0.0

    def forward(self, pixel_values):
        patches = self.patch_projection(pixel_values).flatten(2).transpose(1, 2)
        prefix = torch.full(
            (pixel_values.shape[0], 1 + self.config.num_register_tokens,
             self.config.hidden_size), fill_value=self.prefix_value,
            device=pixel_values.device, dtype=patches.dtype,
        )
        return SimpleNamespace(last_hidden_state=torch.cat((prefix, patches), dim=1))


class FrozenPlantNetLinearSegmenterTest(unittest.TestCase):
    def test_shape_freezing_eval_mode_and_head_gradients(self):
        backbone = SyntheticRegisterBackbone(
            hidden_size=32, patch_size=16, num_register_tokens=4
        )
        model = FrozenPlantNetLinearSegmenter(
            model_id="synthetic/offline", num_classes=7, backbone=backbone
        )
        model.train()

        x = torch.randn(2, 3, 64, 64)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 7, 64, 64))

        # CLS/register tokens must not influence spatial logits.
        backbone.prefix_value = 1_000_000.0
        logits_with_changed_prefix = model(x)
        self.assertTrue(torch.equal(logits, logits_with_changed_prefix))

        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

        target = torch.randint(0, 7, (2, 64, 64))
        target[:, :4, :4] = -1
        loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-1)
        loss.backward()

        self.assertIsNotNone(model.head.weight.grad)
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
