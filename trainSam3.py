"""Train the SAM3 ensemble. Paths are resolved relative to this script.

Examples:
    python trainSam3.py --paths-file ../data/Neophytes/subset_paths.yaml --check-data
    python trainSam3.py --train-dirs /data/site/train --val-dirs /data/site/val

The model is kept unchanged. Model contract violations stop training explicitly.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
IGNORE_INDEX = -1
CLASSES = {
    0: {"name": "Background", "color": "#000000", "stats_name": "background"},
    1: {"name": "B. davidii", "color": "#dce14c", "stats_name": "B_davidii"},
    2: {"name": "R. typhina", "color": "#b76e44", "stats_name": "R_typhina"},
    3: {"name": "R. japonica", "color": "#305dc6", "stats_name": "R_japonica"},
    4: {"name": "B. orientalis", "color": "#e2256a", "stats_name": "B_orientalis"},
    5: {"name": "S. inaequidens", "color": "#50e3c2", "stats_name": "S_inaequidens"},
    6: {"name": "A. altissima", "color": "#dd60f6", "stats_name": "A_altissima"},
}


def resolve_path(value, base=ROOT):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def read_config(path):
    content = path.read_text()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML (or use a JSON paths file).") from exc
        return yaml.safe_load(content)


def get_dirs(args):
    # Explicit overrides win; otherwise use the downloaded subset or local split.
    paths_file = resolve_path(args.paths_file) if args.paths_file else ROOT.parent / 'data/Neophytes/subset_paths.yaml'
    config = read_config(paths_file) if paths_file.is_file() else None
    if args.paths_file and config is None:
        raise FileNotFoundError(paths_file)
    if config is None and (not args.train_dirs or not args.val_dirs):
        config = read_config(ROOT / 'configs/data/split_local.yaml')
    splits = (config or {}).get('image_dir', {})
    train = [resolve_path(p) for p in (args.train_dirs or splits.get('train', []))]
    val = [resolve_path(p) for p in (args.val_dirs or splits.get('val', []))]
    for name, dirs in [('train', train), ('val', val)]:
        if not dirs:
            raise ValueError(f'No {name} directories configured.')
        for path in dirs:
            if not (path / 'images').is_dir() or not any((path / d).is_dir() for d in ['masks_prep', 'labels']):
                raise FileNotFoundError(f'{name}: {path} must contain images/ and masks_prep/ or labels/. '
                                        'Supply --paths-file or --train-dirs and --val-dirs.')
    if set(train) & set(val):
        raise ValueError('Training and validation directories overlap.')
    return train, val


def dice_cce_loss(logits, targets, ignore_index=IGNORE_INDEX):
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0
    cce = F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
    pred_fg = logits.softmax(dim=1)[:, 1]
    foreground = (targets == 1).float()
    intersection = (pred_fg * foreground * valid).sum()
    denominator = (pred_fg * valid).sum() + (foreground * valid).sum()
    return cce + 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)


def build_dataset(dirs, size, training):
    from samDataSet import SamNeophyteDataset
    from utils.transform_utils import get_transforms

    transforms = [{'type': 'Resize', 'params': {'height': size, 'width': size}}]
    if training:
        transforms += [{'type': 'HorizontalFlip', 'params': {'p': 0.5}}]
    # Keep RGB in [0, 1]; backbone-specific preprocessing is a model concern.
    transforms += [{'type': 'Normalize', 'params': {'mean': [0, 0, 0], 'std': [1, 1, 1]}}]
    dataset = SamNeophyteDataset(
        image_dirs=[str(p) for p in dirs], classes=CLASSES,
        transform=get_transforms(transforms), ignore_index=IGNORE_INDEX,
    )
    if not len(dataset):
        raise ValueError(f'No image/mask pairs found in {dirs}')
    # The dataset sorts images and masks separately. Reject mismatched pairing.
    for image, mask in zip(dataset.image_files, dataset.mask_files):
        # Prepared labels insert this marker between the site and tile indices.
        mask_stem = Path(mask).stem.replace('_json_to_mask_', '_')
        if Path(image).stem != mask_stem:
            raise ValueError(f'Image/mask filenames do not match: {image}, {mask}')
    return dataset


def collate_samples(samples):
    # Raster metadata can contain None or affine objects; the model needs tensors only.
    return {key: torch.stack([sample[key] for sample in samples]) for key in ['image', 'mask']}


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    model.sam_model.eval()  # model.train() also switches the frozen SAM backbone.
    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            masks[(masks == 254) | (masks == 255)] = IGNORE_INDEX
            if not (((masks >= 0) & (masks <= 6)) | (masks == IGNORE_INDEX)).all():
                raise ValueError('Masks must contain class IDs 0–6, 254, 255, or -1.')
            if not (masks != IGNORE_INDEX).any():
                continue
            enhanced, heights, texts, labels, targets = model.prepare_inputs(images, masks)
            expected = sum(max(1, len(torch.unique(m[(m > 0) & (m <= 6)]))) for m in masks)
            if len(targets) != expected:
                raise RuntimeError('Model prepare_inputs dropped background tiles; fix the model before training. Model file was not modified.')
            expected_ignored = sum(int((m == IGNORE_INDEX).sum()) * max(1, len(torch.unique(m[(m > 0) & (m <= 6)]))) for m in masks)
            if int((targets == IGNORE_INDEX).sum()) != expected_ignored:
                raise RuntimeError('Model prepare_inputs lost ignore-index pixels; fix the model before training. Model file was not modified.')
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(enhanced, texts, heights)
            if logits.shape != (len(targets), 2, *targets.shape[-2:]):
                raise RuntimeError(f'Model returned unexpected logits shape: {tuple(logits.shape)}')
            loss = dice_cce_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping before saving an invalid checkpoint.')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
    if not total_samples:
        raise ValueError('Epoch contained no valid training targets.')
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-file', help='Subset paths JSON/YAML; relative paths use the script directory.')
    parser.add_argument('--train-dirs', nargs='+')
    parser.add_argument('--val-dirs', nargs='+')
    parser.add_argument('--output-dir', default='checkpoints/sam3')
    parser.add_argument('--sam-checkpoint', default='facebook/sam3')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=1008)
    parser.add_argument('--max-proposals', type=int, default=32)
    parser.add_argument('--max-batches', type=int, help='Limit each epoch for a smoke run.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-data', action='store_true', help='Load one batch per split without downloading model weights.')
    args = parser.parse_args()
    for key in ['batch_size', 'epochs', 'image_size', 'max_proposals', 'lr']:
        if getattr(args, key) <= 0:
            parser.error(f'{key} must be positive')
    if args.num_workers < 0 or (args.max_batches is not None and args.max_batches <= 0):
        parser.error('num-workers must be nonnegative and max-batches must be positive')
    torch.manual_seed(args.seed)
    train_dirs, val_dirs = get_dirs(args)
    loaders = []
    for training, dirs in [(True, train_dirs), (False, val_dirs)]:
        dataset = build_dataset(dirs, args.image_size, training)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                            num_workers=args.num_workers, pin_memory=args.device.startswith('cuda'),
                            collate_fn=collate_samples)
        loaders.append(loader)
        print(f'{"Train" if training else "Validation"}: {len(dataset)} tiles')
        if args.check_data:
            batch = next(iter(loader))
            print({k: tuple(v.shape) for k, v in batch.items()})
    if args.check_data:
        return
    from models.sam3_model import Sam3EnsembleModel
    device = torch.device(args.device)
    model = Sam3EnsembleModel(num_classes=2, sam_checkpoint=args.sam_checkpoint,
                              max_proposals=args.max_proposals, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders[0], device, optimizer, args.max_batches)
        val_loss = run_epoch(model, loaders[1], device, max_batches=args.max_batches)
        print(f'Epoch {epoch:03d}/{args.epochs}: train={train_loss:.5f}, val={val_loss:.5f}')
        checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), train_loss=train_loss,
                          val_loss=val_loss, config=vars(args))
        torch.save(checkpoint, output / 'sam3_checkpoint_last.pth')
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, output / 'sam3_checkpoint_best.pth')


if __name__ == '__main__':
    main()
