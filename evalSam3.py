"""Evaluate SAM3 with six prompts and a combined species/nothing prediction map.

Run from any directory; relative paths use the repository directory.
Predictions and metrics use the resized training input grid, not native resolution.
"""
import argparse
import csv
import json
import math

import torch
from torch.utils.data import DataLoader

from trainSam3 import (
    CLASSES, IGNORE_INDEX, ROOT, build_dataset, collate_samples,
    login_to_huggingface, prepare_height_maps, read_config, resolve_path,
)

SPECIES_IDS = tuple(range(1, 7))


def get_eval_dirs(args):
    if args.test_dirs:
        dirs = args.test_dirs
    else:
        path = resolve_path(args.paths_file) if args.paths_file else ROOT.parent / 'data/Neophytes/subset_paths.yaml'
        if args.paths_file or path.is_file():
            config = read_config(path)
        else:
            config = read_config(ROOT / 'configs/data/split_local.yaml')
        dirs = config.get('image_dir', {}).get(args.split, [])
    if not dirs:
        raise ValueError(f'No {args.split} directories configured; use --test-dirs or --paths-file.')
    return [resolve_path(path) for path in dirs]


@torch.inference_mode()
def predict_species(model, images, heights, prompts):
    """Run each prompt separately to avoid expanding the GPU batch sixfold.

    No ground-truth masks enter this function or prompt selection.
    Scores are binary foreground probabilities, not SAM proposal confidences.
    """
    scores = []
    for class_id in SPECIES_IDS:
        logits = model(images, [prompts[class_id]] * len(images), heights)
        expected_shape = (len(images), 2, *images.shape[-2:])
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f'Expected logits {expected_shape}, got {tuple(logits.shape)}')
        scores.append(logits.float().softmax(dim=1)[:, 1].cpu())
    return torch.stack(scores, dim=1)  # [B, 6, H, W], CPU


def combine_scores(scores, threshold):
    """Argmax over six species; assign 0 when every species is below threshold.

    Exact ties select the first (lowest numbered) species, following torch.max.
    Thresholding does not alter the winning class for accepted pixels.
    """
    confidence, winner = scores.max(dim=1)
    prediction = winner + 1
    prediction[confidence < threshold] = 0
    return prediction, confidence


def confusion_matrix(predictions, targets):
    targets = targets.long().clone()
    targets[(targets == 254) | (targets == 255)] = IGNORE_INDEX
    valid = targets != IGNORE_INDEX
    if not ((targets[valid] >= 0) & (targets[valid] <= 6)).all():
        raise ValueError('Ground truth must contain class IDs 0–6, 254, 255 or -1.')
    indices = targets[valid] * 7 + predictions[valid].long()
    return torch.bincount(indices, minlength=49).reshape(7, 7)


def metrics_from_confusion(matrix):
    """Rows=true, columns=predicted; background contributes false positives."""
    rows = []
    def divide(numerator, denominator):
        return numerator / denominator if denominator else None
    for class_id in SPECIES_IDS:
        tp = int(matrix[class_id, class_id])
        support = int(matrix[class_id].sum())
        predicted = int(matrix[:, class_id].sum())
        fp, fn = predicted - tp, support - tp
        rows.append(dict(class_id=class_id, name=CLASSES[class_id]['name'],
                         true_positive=tp, false_positive=fp, false_negative=fn,
                         support_pixels=support, predicted_pixels=predicted,
                         precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                         iou=divide(tp, tp + fp + fn), f1=divide(2 * tp, 2 * tp + fp + fn)))
    macro = {}
    included = {}
    for key in ['precision', 'recall', 'iou', 'f1']:
        values = [row[key] for row in rows if row[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
        included[key] = len(values)
    return dict(per_species=rows, macro=macro, macro_class_counts=included,
                undefined_metric_policy='null; excluded from that metric macro mean',
                confusion_matrix=matrix.tolist(),
                confusion_matrix_order=['nothing'] + [CLASSES[i]['name'] for i in SPECIES_IDS],
                valid_pixels=int(matrix.sum()))


def save_maps(output, index, prediction, confidence, color):
    from PIL import Image
    import numpy as np

    name = f'{index:06d}'
    Image.fromarray(prediction.numpy().astype(np.uint8)).save(output / f'{name}_labels.png')
    # Store the winning foreground probability without quantizing it to 8 bits.
    torch.save(confidence, output / f'{name}_confidence.pt')
    if color:
        palette = torch.tensor([
            [int(CLASSES[i]['color'][j:j + 2], 16) for j in (1, 3, 5)]
            for i in range(7)
        ], dtype=torch.uint8)
        Image.fromarray(palette[prediction].numpy()).save(output / f'{name}_color.png')
    return name


def evaluate(model, loader, device, prompts, threshold, height_max_m,
             output=None, image_files=None, save_previews=10, max_batches=None):
    model.eval()
    matrix = torch.zeros(7, 7, dtype=torch.long)
    count = 0
    manifest = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch['image'].to(device)
            heights = prepare_height_maps(batch['ndsm'], images, height_max_m)
            scores = predict_species(model, images, heights, prompts)
            prediction, confidence = combine_scores(scores, threshold)
            matrix += confusion_matrix(prediction, batch['mask'].cpu())
            for item in range(len(prediction)):
                if output is not None:
                    name = save_maps(output, count, prediction[item], confidence[item], count < save_previews)
                    manifest.append(dict(id=name, source=image_files[count] if image_files else None))
                count += 1
            print(f'Evaluated {count} tiles', flush=True)
    if count == 0:
        raise ValueError('No evaluation tiles were processed.')
    result = metrics_from_confusion(matrix)
    result['tiles'] = count
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--paths-file', help='JSON/YAML containing image_dir.test (or image_dir.val).')
    parser.add_argument('--test-dirs', nargs='+', help='Explicit dataset split folders.')
    parser.add_argument('--split', choices=['test', 'val'], default='test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='outputs/sam3_eval')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum winning species probability; below this becomes nothing.')
    parser.add_argument('--image-size', type=int, help='Defaults to checkpoint config (otherwise 1008).')
    parser.add_argument('--height-max-m', type=float, help='Defaults to checkpoint config (otherwise 10).')
    parser.add_argument('--max-proposals', type=int, help='Defaults to checkpoint config (otherwise 32).')
    parser.add_argument('--sam-checkpoint', help='Defaults to checkpoint config (otherwise facebook/sam3).')
    parser.add_argument('--hf-login', action='store_true')
    parser.add_argument('--max-batches', type=int, help='Limit evaluation for a smoke run.')
    parser.add_argument('--save-previews', type=int, default=10)
    parser.add_argument('--no-save-maps', action='store_true', help='Write metrics only.')
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error('threshold must be between 0 and 1')
    if args.batch_size <= 0 or args.num_workers < 0 or args.save_previews < 0:
        parser.error('batch-size must be positive; num-workers and save-previews must be nonnegative')
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error('max-batches must be positive')
    checkpoint_path = resolve_path(args.checkpoint)
    # Training saves a dictionary of tensors and primitive configuration values.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {})
    for key, default in [('image_size', 1008), ('height_max_m', 10.),
                         ('max_proposals', 32), ('sam_checkpoint', 'facebook/sam3')]:
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, default))
    for key in ['image_size', 'height_max_m', 'max_proposals']:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    if args.hf_login:
        login_to_huggingface()
    from models.sam3_model import Sam3EnsembleModel
    from SAM.samPromptsDescriptions import speciesDictById as speciesDict

    model = Sam3EnsembleModel(num_classes=2, sam_checkpoint=args.sam_checkpoint,
                              height_max_m=args.height_max_m,
                              max_proposals=args.max_proposals, ignore_index=IGNORE_INDEX)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    model.to(args.device)
    dirs = get_eval_dirs(args)
    dataset = build_dataset(dirs, args.image_size, training=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.device.startswith('cuda'),
                        collate_fn=collate_samples)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    maps = output / 'maps'
    if not args.no_save_maps:
        maps.mkdir(exist_ok=True)
    result, manifest = evaluate(
        model, loader, args.device, speciesDict, args.threshold, args.height_max_m,
        output=None if args.no_save_maps else maps,
        image_files=dataset.image_files, save_previews=args.save_previews,
        max_batches=args.max_batches)
    result['settings'] = dict(vars(args), checkpoint=str(checkpoint_path),
                              evaluation_dirs=[str(p) for p in dirs],
                              score='binary softmax foreground probability for each species',
                              resolution='resized input grid; not native tile resolution')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'per_species.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['per_species'][0]))
        writer.writeheader()
        writer.writerows(result['per_species'])
    if not args.no_save_maps:
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for row in result['per_species']:
        print(f"{row['name']}: IoU={row['iou']} F1={row['f1']}")
    print('Macro (species only):', result['macro'])
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
