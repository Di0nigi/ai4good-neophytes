"""CPU checks for prompt selection, map fusion, and combined-map metrics."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalSam3 import combine_scores, confusion_matrix, evaluate, metrics_from_confusion


class EvalSam3Tests(unittest.TestCase):
    def test_argmax_threshold_and_ties(self):
        scores = torch.full((1, 6, 1, 4), 0.1)
        scores[0, 3, 0, 0] = 0.9
        scores[0, 5, 0, 1] = 0.49
        scores[0, 1, 0, 2] = 0.5
        scores[0, :2, 0, 3] = 0.8
        predictions, confidence = combine_scores(scores, 0.5)
        self.assertEqual(predictions.tolist(), [[[4, 0, 2, 1]]])
        self.assertTrue(torch.allclose(confidence, torch.tensor([[[.9, .49, .5, .8]]])))

    def test_nothing_can_be_disabled(self):
        prediction, _ = combine_scores(torch.full((1, 6, 2, 2), .1), 0.)
        self.assertTrue((prediction == 1).all())

    def test_combined_metrics_ignore_pixels_and_count_background_false_positives(self):
        targets = torch.tensor([[[0, 1, 1, 2, -1, 254, 255]]])
        prediction = torch.tensor([[[1, 1, 2, 0, 1, 1, 1]]])
        result = metrics_from_confusion(confusion_matrix(prediction, targets))
        self.assertEqual(result['valid_pixels'], 4)
        first, second = result['per_species'][:2]
        self.assertEqual((first['true_positive'], first['false_positive'], first['false_negative']), (1, 1, 1))
        self.assertAlmostEqual(first['iou'], 1 / 3)
        self.assertAlmostEqual(first['f1'], .5)
        self.assertEqual((second['true_positive'], second['false_positive'], second['false_negative']), (0, 1, 1))
        self.assertAlmostEqual(result['macro']['f1'], .25)
        self.assertIsNone(result['per_species'][2]['f1'])

    def test_all_species_are_prompted_and_model_is_in_eval_mode(self):
        prompts = {i: f'species {i}' for i in range(1, 7)}
        calls = []

        class FakeModel(torch.nn.Module):
            def forward(self, images, texts, heights):
                self.assertions(images, texts, heights)
                calls.append(texts[0])
                logits = torch.zeros(len(images), 2, *images.shape[-2:])
                logits[:, 1] = 5 if texts[0] == 'species 4' else -5
                return logits

            def assertions(self, images, texts, heights):
                assert not self.training
                assert not torch.is_grad_enabled()
                assert heights.shape == images.shape
                assert torch.allclose(heights, torch.full_like(images, .5))

        batch = dict(image=torch.zeros(1, 3, 2, 2), ndsm=torch.full((1, 1, 2, 2), 5.),
                     mask=torch.full((1, 2, 2), 4))
        model = FakeModel()
        result, _ = evaluate(model, [batch], 'cpu', prompts, .5, 10.)
        self.assertEqual(calls, list(prompts.values()))
        self.assertEqual(result['per_species'][3]['f1'], 1.)
        # Changing ground truth cannot change which prompts or predictions are used.
        calls.clear()
        batch['mask'].zero_()
        result, _ = evaluate(model, [batch], 'cpu', prompts, .5, 10.)
        self.assertEqual(calls, list(prompts.values()))
        self.assertEqual(result['per_species'][3]['false_positive'], 4)


if __name__ == '__main__':
    unittest.main()
