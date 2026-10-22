"""Evaluate SAM3 mask coverage against all labeled species merged as foreground.

python SAM/SAM_binary.py --data neophytes_split_cv1_test --device cuda
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from tqdm import tqdm

from SAM_baseline import NeophyteDataModule, get_transforms, save_preview


@torch.inference_mode()
def infer_masks(model, processor, image, prompts=('plant'), threshold=0.5):
    """Return raw postprocessed instance masks/scores and their binary union.

    image: uint8 CPU tensor [3, H, W]. No class competition or argmax.
    """
    rgb = Image.fromarray(image.permute(1, 2, 0).numpy())
    instances = []
    foreground = torch.zeros(image.shape[-2:], dtype=torch.bool)
    for prompt in prompts:
        inputs = processor(images=rgb, text=prompt, return_tensors='pt').to(model.device)
        result = processor.post_process_instance_segmentation(
            model(**inputs), threshold=threshold, mask_threshold=0.5,
            target_sizes=[(rgb.height, rgb.width)])[0]
        masks = result['masks'].bool().cpu()
        # Some Transformers versions return empty masks at decoder resolution.
        # Explicitly enforce the image grid, including when no instances remain.
        if masks.numel() == 0:
            masks = torch.zeros((0, *image.shape[-2:]), dtype=torch.bool)
        elif masks.shape[-2:] != image.shape[-2:]:
            masks = torch.nn.functional.interpolate(
                masks[:, None].float(), size=image.shape[-2:], mode='nearest')[:, 0].bool()
        foreground |= masks.any(dim=0)
        instances.append({'prompt': prompt, 'masks': masks, 'scores': result['scores'].cpu()})
    return foreground, instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='neophytes_split_cv1_test')
    parser.add_argument('--checkpoint', default='facebook/sam3')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prompts', nargs='+', default=['plant'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', default='SAM/binary_results')
    parser.add_argument('--previews', type=int, default=50, help='Number of plant-rich comparison PNGs to save')
    parser.add_argument('--max-images', type=int, default=None, help='Optional smoke-test limit; default evaluates the full test split')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.previews < 0 or (args.max_images is not None and args.max_images < 1):
        parser.error('Threshold must be in [0, 1], previews >= 0, and max-images >= 1')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    with initialize_config_dir(version_base=None, config_dir=str(root / 'configs')):
        cfg = compose(config_name=None, overrides=[f'+data={args.data}']).data
    transform = get_transforms([t for t in cfg.test_transforms if t.type != 'Normalize'],
                               cfg.transforms_additional_targets)
    data = NeophyteDataModule(
        cfg.image_dir, cfg.classes, transform, transform, transform,
        dataset_stats=cfg.dataset_stats, value_mapping=cfg.value_mapping,
        ignore_index=cfg.ignore_index, batch_size_test=1, num_workers=cfg.num_workers)
    data.prepare_data()
    data.setup('test')
    output = Path(args.output_dir) / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{args.data}'
    (output / 'masks').mkdir(parents=True)
    model = Sam3Model.from_pretrained(args.checkpoint).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    classes = {0: SimpleNamespace(name='Background', color='#000000'),
               1: SimpleNamespace(name='Labeled plant', color='#00cc66')}
    confusion = torch.zeros(2, 2, dtype=torch.int64)
    manifest = []
    num_images = min(len(data.test_dataset), args.max_images or len(data.test_dataset))
    # Round-robin across species, taking their largest labeled areas first.
    # Select previews only; evaluation still includes every configured test tile.
    stats = data.test_dataset.stats_df.iloc[:num_images].reset_index(drop=True)
    columns = data.test_dataset.class_px_cols[1:]
    ranked = [stats.loc[stats[col] > 0, col].sort_values(ascending=False).index.tolist()
              for col in columns]
    preview_indices = set()
    for rank in range(max(map(len, ranked), default=0) if args.previews else 0):
        for candidates in ranked:
            if rank < len(candidates):
                preview_indices.add(candidates[rank])
            if len(preview_indices) >= args.previews:
                break
        if len(preview_indices) >= args.previews:
            break
    for batch in tqdm(data.test_dataloader(), desc='SAM3 binary'):
        image, target = batch['image'][0], batch['mask'][0]
        prediction, instances = infer_masks(model, processor, image, args.prompts, args.threshold)
        valid = target != cfg.ignore_index
        truth = (target > 0).long()
        confusion += torch.bincount(2 * truth[valid] + prediction[valid].long(), minlength=4).reshape(2, 2)
        name = f'{len(manifest):05d}_{Path(batch["name"][0]).stem}.png'
        Image.fromarray(prediction.numpy().astype('uint8') * 255).save(output / 'masks' / name)
        preview = name if len(manifest) in preview_indices else None
        if preview is not None:
            truth[~valid] = cfg.ignore_index
            save_preview(image, truth, prediction.long(), classes, output / name, cfg.ignore_index)
        manifest.append({'source': batch['name'][0], 'mask': f'masks/{name}', 'preview': preview,
                         'detections': {r['prompt']: r['scores'].tolist() for r in instances}})
        if args.max_images is not None and len(manifest) >= args.max_images:
            break
    tn, fp, fn, tp = confusion.flatten().tolist()
    metrics = {'foreground_f1': 2 * tp / max(2 * tp + fp + fn, 1),
               'foreground_iou': tp / max(tp + fp + fn, 1),
               'foreground_precision': tp / max(tp + fp, 1),
               'foreground_recall': tp / max(tp + fn, 1),
               'accuracy': (tp + tn) / max(tp + tn + fp + fn, 1)}
    results = {**vars(args), 'num_images': len(manifest), 'ignore_index': cfg.ignore_index,
               'mask_threshold': 0.5, 'preview_selection': 'largest labeled areas, round-robin across species',
               'class_names': ['Background', 'Labeled plant'],
               'confusion_matrix': confusion.tolist(), **metrics, 'images': manifest}
    (output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    print(f'Results saved to {output.resolve()}')


if __name__ == '__main__':
    main()
