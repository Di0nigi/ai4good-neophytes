"""CPU contract tests using lightweight stand-ins for unavailable model packages.

These test adapter logic, not pretrained accuracy or a real Lightning Trainer.
"""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class LightningStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_step = 0
        self.logged = {}

    def save_hyperparameters(self):
        values = dict(inspect.currentframe().f_back.f_locals)
        values.pop('self'); values.pop('__class__', None)
        self.hparams = types.SimpleNamespace(**values)

    def log(self, key, value, **kwargs):
        self.logged[key] = value


class EnsembleStub(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.sam_model = torch.nn.Identity()
        self.prompt_generator = torch.nn.Module()
        self.prompt_generator.clipModel = torch.nn.Identity()
        self.calls = []

    def forward(self, images, texts, heights):
        self.calls.append((images.detach(), texts, heights.detach()))
        # Species 3 always wins, independent of ground truth.
        foreground = torch.tensor([4. if t == 'species3' else -4. for t in texts])
        foreground = foreground.to(images).view(-1, 1, 1).expand(-1, *images.shape[-2:]) * self.weight
        return torch.stack([torch.zeros_like(foreground), foreground], dim=1)


def load_adapter():
    stubs = {
        'pytorch_lightning': types.SimpleNamespace(LightningModule=LightningStub),
        'models.sam3_model': types.SimpleNamespace(Sam3EnsembleModel=EnsembleStub),
        'SAM.samPromptsDescriptions': types.SimpleNamespace(speciesDictById={i: f'species{i}' for i in range(1, 7)}),
        'utils.scheduler_utils': types.SimpleNamespace(get_scheduler=None),
    }
    spec = importlib.util.spec_from_file_location('sam3_adapter_test', ROOT / 'models/sam3_plm.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.Sam3SegmentationModel


class Sam3LightningTests(unittest.TestCase):
    def setUp(self):
        self.model = load_adapter()(img_size=16)

    def test_preprocessing_and_ignore_labels(self):
        rgb = torch.full((1, 3, 16, 16), .5)
        normalized = (rgb - self.model.rgb_mean) / self.model.rgb_std
        height = torch.full((1, 1, 16, 16), (5.-1.43)/3.10)
        decoded, heights = self.model._decode_inputs(normalized, height)
        self.assertTrue(torch.allclose(decoded, rgb))
        self.assertTrue(torch.allclose(heights, torch.full_like(rgb, .5)))
        masks = torch.tensor([[[-1, 254], [255, 3]]])
        targets = self.model._targets(masks, torch.tensor([3]))
        self.assertEqual(targets.tolist(), [[[-1, -1], [-1, 1]]])
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        loss = self.model._loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(self.model._loss(logits, torch.full_like(targets, -1)).item(), 0.)

    def test_prompt_independent_evaluation_and_arbitrary_tile_shape(self):
        images = torch.zeros(1, 3, 19, 27)
        probs = self.model(images)
        self.assertEqual(tuple(probs.shape), (1, 7, 19, 27))
        self.assertTrue((probs.argmax(1) == 3).all())
        self.assertEqual(len(self.model.ensemble.calls), 24)  # 4 patches x 6 species
        batch = {'image': images, 'mask': torch.full((1, 19, 27), 3)}
        self.model.validation_step(batch, 0)
        self.model.on_validation_epoch_end()
        self.assertEqual(self.model.logged['val_f1'].item(), 1.)
        batch['mask'].zero_()
        self.model.validation_step(batch, 0)
        self.assertEqual(self.model.val_confusion[0, 3].item(), 19*27)

    def test_training_gradient_and_frozen_modes(self):
        self.model.train()
        self.assertFalse(self.model.ensemble.sam_model.training)
        self.assertFalse(self.model.ensemble.prompt_generator.clipModel.training)
        batch = {'image': torch.zeros(2, 3, 16, 16), 'ndsm': torch.zeros(2, 1, 16, 16),
                 'mask': torch.ones(2, 16, 16, dtype=torch.long)}
        loss = self.model.training_step(batch, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.ensemble.weight.grad))
        optimizer = self.model.configure_optimizers()
        optimizer.step()
        self.assertIn('train_loss', self.model.logged)

    def test_core_keeps_sam_1008_and_classifier_512(self):
        tree = ast.parse((ROOT / 'models/sam3_model.py').read_text())
        core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sam3EnsembleModel')
        forward = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {'torch': torch, 'F': F, 'BaseModelOutputWithPooling': types.SimpleNamespace}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<core_forward>', 'exec'), namespace)
        class Sam:
            config = types.SimpleNamespace(vision_config=types.SimpleNamespace(image_size=1008))
            def __call__(self, pixel_values, **kwargs):
                assert tuple(pixel_values.shape) == (1, 3, 1008, 1008)
                return types.SimpleNamespace(pred_masks=torch.zeros(1, 2, 8, 8),
                                             pred_logits=torch.zeros(1, 2), presence_logits=None)
        def classifier(images, masks):
            assert images.shape[-2:] == masks.shape[-2:] == (512, 512)
            return torch.zeros(1, 2, 512, 512)
        stub = types.SimpleNamespace(
            sam_model=Sam(), sam_processor=types.SimpleNamespace(image_processor=types.SimpleNamespace(
                image_mean=[.5]*3, image_std=[.5]*3)),
            prompt_generator=lambda inputs: torch.zeros(1, 256), classifier=classifier, max_proposals=2)
        result = namespace['forward'](stub, torch.zeros(1, 3, 512, 512), ['plant'])
        self.assertEqual(tuple(result.shape), (1, 2, 512, 512))


if __name__ == '__main__':
    unittest.main()
