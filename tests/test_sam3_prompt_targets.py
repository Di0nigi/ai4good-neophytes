"""Test real target preparation without loading pretrained backbones."""
import ast
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SAM.samPromptsDescriptions import speciesDict, speciesDictById


def prepare_inputs():
    tree = ast.parse((ROOT / 'models/sam3_model.py').read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'Sam3EnsembleModel')
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'prepare_inputs')
    namespace = {'torch': torch, 'speciesDict': speciesDictById}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<prepare_inputs>', 'exec'), namespace)
    return types.MethodType(namespace['prepare_inputs'], types.SimpleNamespace(ignore_index=-1))


class PromptTargetTests(unittest.TestCase):
    def test_descriptions_unchanged(self):
        names = ['B. davidii', 'R. typhina', 'R. japonica', 'B. orientalis',
                 'S. inaequidens', 'A. altissima']
        for class_id, name in enumerate(names, 1):
            self.assertIs(speciesDictById[class_id], speciesDict[name])

    def test_all_species_generate_positive_binary_targets(self):
        masks = torch.tensor([[[i, 0], [-1, i]] for i in range(1, 7)])
        images = torch.rand(6, 3, 2, 2)
        _, _, texts, labels, targets = prepare_inputs()(images, masks)
        self.assertEqual(labels, list(range(1, 7)))
        self.assertEqual(texts, list(speciesDictById.values()))
        self.assertTrue(torch.equal(targets[:, 0, 0], torch.ones(6, dtype=torch.long)))
        self.assertTrue((targets[:, 1, 0] == -1).all())

    def test_mixed_batches_preserve_background_in_either_order(self):
        for masks in [torch.tensor([[[0, 0], [-1, 0]], [[1, 2], [-1, 0]]]),
                      torch.tensor([[[1, 2], [-1, 0]], [[0, 0], [-1, 0]]])]:
            images = torch.arange(24.).reshape(2, 3, 2, 2)
            heights = images + 100
            expanded, expanded_heights, _, labels, targets = prepare_inputs()(images, masks, heights)
            self.assertEqual(sorted(labels), [0, 1, 2])
            self.assertTrue(torch.equal(expanded_heights, expanded + 100))
            for label, target in zip(labels, targets):
                self.assertEqual(int((target == 1).sum()), 0 if label == 0 else 1)
                self.assertEqual(int((target == -1).sum()), 1)

    def test_multiple_background_tiles_are_preserved(self):
        masks = torch.zeros(3, 2, 2, dtype=torch.long)
        _, _, _, labels, targets = prepare_inputs()(torch.zeros(3, 3, 2, 2), masks)
        self.assertEqual(labels, [0, 0, 0])
        self.assertEqual(tuple(targets.shape), (3, 2, 2))


if __name__ == '__main__':
    unittest.main()
