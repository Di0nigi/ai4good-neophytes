"""Vanilla SAM3 baseline using Transformers (requires checkpoint access).

From the repository root:
    python SAM/SAM_baseline.py --data neophytes_split_cv1_test
    python SAM/SAM_baseline.py --checkpoint facebook/sam3 --device cpu
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.neophyte_datamodule import NeophyteDataModule
from utils.transform_utils import get_transforms


@torch.inference_mode()
def predict(model, processor, image, class_names):
    image = Image.fromarray(image.permute(1, 2, 0).numpy())
    class_scores = []
    for name in class_names:
        inputs = processor(images=image, text=name, return_tensors='pt').to(model.device)
        result = processor.post_process_instance_segmentation(
            model(**inputs), threshold=0.5, mask_threshold=0.5,
            target_sizes=[(image.height, image.width)])[0]
        # Merge instances into a class map weighted by detection confidence.
        masks = result['masks']
        scores = (masks * result['scores'][:, None, None]).amax(0) if len(masks) else torch.zeros(
            (image.height, image.width), device=model.device)
        class_scores.append(scores)
    # All-zero ties resolve to class 0 (Background).
    return torch.stack(class_scores).argmax(0).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='neophytes_split_cv1_test')
    parser.add_argument('--checkpoint', default='facebook/sam3', help='Hugging Face model ID or local pretrained directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='SAM/results', help='Results root, relative to the repository or absolute')
    args = parser.parse_args()
    # Split paths and stats CSV entries are relative to the repository root.
    os.chdir(Path(__file__).resolve().parents[1])
    output_dir = Path(args.output_dir) / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{args.data}'
    output_dir.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'configs')):
        cfg = compose(config_name=None, overrides=[f'+data={args.data}']).data
    for path in [cfg.dataset_stats, *cfg.image_dir.test]:
        if not Path(path).exists():
            raise FileNotFoundError(f'{path}: expected dataset at ../data/Neophytes (see SAM/README.md)')

    # Preserve spatial test transforms; SAM3 normalizes the original uint8 RGB.
    transform = get_transforms([t for t in cfg.test_transforms if t.type != 'Normalize'],
                               cfg.transforms_additional_targets)
    data = NeophyteDataModule(
        cfg.image_dir, cfg.classes, transform, transform, transform,
        dataset_stats=cfg.dataset_stats, value_mapping=cfg.value_mapping,
        ignore_index=cfg.ignore_index, batch_size_test=1, num_workers=cfg.num_workers)
    data.prepare_data()
    data.setup('test')
    print(f'{args.data}: {len(data.test_dataset)} test images on {args.device}', flush=True)
    class_names = [cfg.classes[i].name for i in sorted(cfg.classes, key=int)]
    n = len(class_names)
    model = Sam3Model.from_pretrained(args.checkpoint).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    confusion = torch.zeros(n, n, dtype=torch.int64)
    for batch in tqdm(data.test_dataloader(), desc='SAM3'):
        for image, target in zip(batch['image'], batch['mask']):
            prediction = predict(model, processor, image, class_names)
            valid = target != cfg.ignore_index
            confusion += torch.bincount(n * target[valid] + prediction[valid], minlength=n * n).reshape(n, n)

    tp = confusion.diag().double()
    support, predicted = confusion.sum(1), confusion.sum(0)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    recall = tp / support.clamp_min(1)
    for name, f, r in zip(class_names, f1, recall):
        print(f'{name}: F1={f:.4f}  Recall={r:.4f}')
    print(f'Accuracy: {tp.sum() / confusion.sum().clamp_min(1):.4f}')
    print(f'Macro recall: {recall.mean():.4f}')
    results = {
        'data': args.data, 'checkpoint': args.checkpoint, 'device': args.device,
        'num_images': len(data.test_dataset), 'ignore_index': cfg.ignore_index,
        'threshold': 0.5, 'mask_threshold': 0.5,
        'per_class': {name: {'f1': f.item(), 'recall': r.item(), 'support': s.item()}
                      for name, f, r, s in zip(class_names, f1, recall, support)},
        'accuracy': (tp.sum() / confusion.sum().clamp_min(1)).item(),
        'macro_recall': recall.mean().item(),
        'class_names': class_names,
        'confusion_matrix': confusion.tolist(),  # rows: true, columns: predicted
    }
    (output_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Results saved to {output_dir.resolve() / "metrics.json"}')


if __name__ == '__main__':
    main()
