
import argparse
import getpass
import json
import math
import os
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


class HeightAlignedTransform:
    """Resize paired height tiles to RGB resolution before joint augmentation.

    Assumes paired tiles cover the same footprint, as in the dataset pipeline.
    This resizes pixel arrays; it does not perform geospatial reprojection.
    """

    def __init__(self, transform):
        self.transform = transform
        # Expose the pixel pipeline to the dataset's normalization inspection.
        self.pixel = getattr(transform, 'pixel', transform)

    def set_tile_context(self, **context):
        from utils.transform_utils import set_tile_context
        set_tile_context(self.transform, **context)

    @property
    def transforms(self):
        # Let the existing context helper forward per-tile information.
        return [self]

    def __call__(self, **sample):
        import cv2
        height, width = sample['image'].shape[:2]
        ndsm = sample.get('ndsm')
        if ndsm is not None and ndsm.shape[:2] != (height, width):
            sample['ndsm'] = cv2.resize(ndsm, (width, height), interpolation=cv2.INTER_LINEAR)
        return self.transform(**sample)


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
        transform=HeightAlignedTransform(
            get_transforms(transforms, additional_targets={'ndsm': 'float_mask'})),
        load_ndsm=True, ignore_index=IGNORE_INDEX,
    )
    if not len(dataset):
        raise ValueError(f'No image/mask pairs found in {dirs}')
    # Trust the dataset's pairing; elevation tiles may have lower resolution.
    for kind, files in [('DSM', dataset.dsm_files), ('DTM', dataset.dtm_files)]:
        if len(files) != len(dataset.image_files):
            raise ValueError(f'{kind}: found {len(files)} files for {len(dataset.image_files)} images. '
                             'Each split directory must contain matching dsm/ and dtm/ GeoTIFFs.')
    return dataset


def collate_samples(samples):
    # Raster metadata can contain None or affine objects; the model needs tensors only.
    return {key: torch.stack([sample[key] for sample in samples]) for key in ['image', 'mask', 'ndsm']}


def prepare_height_maps(ndsm, images, height_max_m):
    """Encode metre-valued nDSM as three identical channels in [0, 1]."""
    if ndsm.shape != (images.shape[0], 1, *images.shape[-2:]):
        raise ValueError(f'nDSM must match RGB spatial dimensions; got {tuple(ndsm.shape)}')
    heights = ndsm.to(device=images.device, dtype=images.dtype)
    heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
    return (heights.clamp(0, height_max_m) / height_max_m).repeat(1, 3, 1, 1)


def run_epoch(model, loader, device, optimizer=None, max_batches=None, height_max_m=10.0):
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
            height_maps = prepare_height_maps(batch['ndsm'], images, height_max_m)
            enhanced, heights, texts, labels, targets = model.prepare_inputs(
                images, masks, height_maps=height_maps)
            expected = sum(max(1, len(torch.unique(m[(m > 0) & (m <= 6)]))) for m in masks)
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


def login_to_huggingface():
    """Read a token without echoing it; keep it out of checkpoint arguments."""
    from huggingface_hub import login

    token = getpass.getpass('Hugging Face token (hidden): ').strip()
    if not token:
        raise ValueError('Hugging Face token cannot be empty.')
    login(token=token, add_to_git_credential=False)
    # HF_TOKEN overrides cached credentials; use the entered token for this run.
    os.environ['HF_TOKEN'] = token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-file', help='Subset paths JSON/YAML; relative paths use the script directory.')
    parser.add_argument('--train-dirs', nargs='+')
    parser.add_argument('--val-dirs', nargs='+')
    parser.add_argument('--output-dir', default='checkpoints/sam3')
    parser.add_argument('--sam-checkpoint', default='facebook/sam3')
    parser.add_argument('--hf-login', action='store_true',
                        help='Prompt for a hidden Hugging Face token and save login locally.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=1008)
    parser.add_argument('--max-proposals', type=int, default=32)
    parser.add_argument('--height-max-m', type=float, default=10.0,
                        help='nDSM clipping height in metres; DSM and DTM must use metres.')
    parser.add_argument('--max-batches', type=int, help='Limit each epoch for a smoke run.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-data', action='store_true', help='Load one batch per split without downloading model weights.')
    args = parser.parse_args()
    for key in ['batch_size', 'epochs', 'image_size', 'max_proposals', 'lr', 'height_max_m']:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
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
            heights = prepare_height_maps(batch['ndsm'], batch['image'], args.height_max_m)
            print(f'Encoded height: shape={tuple(heights.shape)}, '
                  f'range=[{heights.min().item():.3f}, {heights.max().item():.3f}]')
    if args.check_data:
        return
    if args.hf_login:
        login_to_huggingface()
    from models.sam3_model import Sam3EnsembleModel
    device = torch.device(args.device)
    model = Sam3EnsembleModel(num_classes=2, sam_checkpoint=args.sam_checkpoint,
                              max_proposals=args.max_proposals, ignore_index=IGNORE_INDEX,
                              height_max_m=args.height_max_m).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        print("got here")
        train_loss = run_epoch(model, loaders[0], device, optimizer, args.max_batches,
                               height_max_m=args.height_max_m)
        val_loss = run_epoch(model, loaders[1], device, max_batches=args.max_batches,
                             height_max_m=args.height_max_m)
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
