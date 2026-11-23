#!/usr/bin/env python3
"""Fine-tune the PlantNet ViT-B classifier on Neophytes image crops.

The grouped-CV runner sets ``TRAINING_SITES``, ``TEST_SITES``,
``VALIDATION_SITES`` and ``ALL_SITE_GROUPS`` before calling :func:`main`. When
run directly, the defaults reproduce the exploratory CV3 setup: CV3 is held
out for testing and CV5 supplies validation sites.

Each crop receives one of seven labels (six target species plus
other/background), derived from its aligned pixel annotation. This is crop
classification; the model does not output a pixel mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("NEOPHYTES_DATA_ROOT", "/zfs/ai4good/datasets/Neophytes")
)

# Import crop and label utilities from the shared pretrained evaluator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_pretrained_vitb as evaluator  # noqa: E402


TEST_SITES = {"Basel_2", "Waltenschwil_1", "Dornach_1", "Rivera_1"}
VALIDATION_SITES = {"Chamoson_1", "Ardon_2", "Basel_4", "Rueti_1"}
TRAINING_SITES: set[str] | None = None
ALL_SITE_GROUPS = TEST_SITES | VALIDATION_SITES

CLASS_NAMES = [evaluator.CLASS_NAMES[class_id] for class_id in range(7)]
MODEL_ID = evaluator.MODEL_ID
MANIFEST_COLUMNS = [
    "image_path", "mask_path", "y0", "x0", "y1", "x1", "target_id", "site_split"
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read run settings; the evaluation manifest must be chosen explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        required=True,
        help="Frozen held-out crop manifest created in the prepare stage.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    return parser.parse_args(argv)


def site_matches(flight_name: str, site_name: str) -> bool:
    """Match a site group such as ``Basel_2`` to flights like ``Basel_2_3``."""
    return flight_name == site_name or flight_name.startswith(site_name + "_")


def find_site_partitions(data_root: Path) -> list[Path]:
    """Return available per-flight ``train`` folders under the dataset root."""
    return sorted(data_root.glob("*/*/train"))


def select_training_crops(
    split_dirs: Iterable[Path], max_per_class: int, seed: int
) -> list[tuple[str, str, int, int, int, int, int, str]]:
    """Select up to ``max_per_class`` accepted crops for each of seven labels.

    Candidate crops are non-overlapping 512-pixel windows. The fixed label
    rule is shared with the unchanged-model evaluation. The supplied seed
    makes the selection reproducible for a fixed dataset and file ordering.
    """
    split_dirs = sorted(split_dirs)
    rng = random.Random(seed)
    candidates_by_class = {class_id: [] for class_id in range(7)}
    label_rule = SimpleNamespace(
        min_species_pixel_fraction=0.02,
        min_dominant_foreground_fraction=0.80,
        max_foreground_fraction_for_background=0.02,
        min_background_fraction=0.90,
    )
    image_mask_pairs = 0

    for split_dir in split_dirs:
        try:
            pairs = evaluator.list_pairs(split_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            # Keep a clear log of a missing partition; do not silently skip it.
            print(f"Skipping {split_dir}: {exc}", flush=True)
            continue

        for image_path, mask_path in pairs:
            image_mask_pairs += 1
            mask = evaluator.load_mask(mask_path)
            height, width = mask.shape

            for y0, x0, y1, x1 in evaluator.windows(height, width, 512):
                # Training uses full 512 x 512 crops, matching the model input
                # preparation used in the reported experiments.
                if y1 - y0 != 512 or x1 - x0 != 512:
                    continue
                class_id, _reason = evaluator.crop_target(mask[y0:y1, x0:x1], label_rule)
                if class_id is not None:
                    candidates_by_class[class_id].append(
                        (
                            str(image_path), str(mask_path), y0, x0, y1, x1,
                            class_id, str(split_dir),
                        )
                    )

    selected = []
    for class_id, rows in candidates_by_class.items():
        rng.shuffle(rows)
        selected.extend(rows[:max_per_class])
        print(
            f"{CLASS_NAMES[class_id]}: selected {min(len(rows), max_per_class)} "
            f"of {len(rows)} accepted candidates",
            flush=True,
        )

    rng.shuffle(selected)
    print(
        f"Scanned {len(split_dirs)} partitions and {image_mask_pairs} "
        f"image/mask pairs; selected {len(selected)} training crops.",
        flush=True,
    )
    return selected


class CropDataset(Dataset):
    """Load crop pixels lazily so the full aerial dataset stays off RAM."""

    def __init__(self, rows, transform, augment: bool = False):
        self.rows = rows
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, _mask_path, y0, x0, y1, x1, class_id, _site_split = self.rows[index]
        rgb = evaluator.load_rgb_crop(Path(image_path), y0, x0, y1, x1)

        # Horizontal and vertical flips are the only augmentations in this run.
        if self.augment and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        if self.augment and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[::-1])

        return self.transform(Image.fromarray(rgb)), int(class_id)


def evaluate(model, loader, device: torch.device) -> tuple[list[int], list[int]]:
    """Return true and predicted crop labels without updating model weights."""
    model.eval()
    true_labels, predicted_labels = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return true_labels, predicted_labels


def read_manifest(path: Path):
    """Load rows from a frozen crop manifest in the dataset tuple format."""
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                (
                    row["image_path"], row["mask_path"], int(row["y0"]),
                    int(row["x0"]), int(row["y1"]), int(row["x1"]),
                    int(row["target_id"]), row["site_split"],
                )
            )
    if not rows:
        raise ValueError(f"Evaluation manifest contains no crops: {path}")
    return rows


def write_manifest(path: Path, rows) -> None:
    """Save crop coordinates and labels for later review and exact reuse."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    """Return a content hash so fold manifests can be compared exactly."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_per_class) < 1:
        raise ValueError("epochs, batch size, and max-per-class must be positive")
    if args.class_weight_power < 0:
        raise ValueError("class-weight-power must be non-negative")

    args.data_root = args.data_root.expanduser().resolve()
    args.eval_manifest = args.eval_manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")
    if not args.eval_manifest.is_file():
        raise FileNotFoundError(f"Frozen evaluation manifest not found: {args.eval_manifest}")
    # Refuse to overwrite a previous run: results should have a unique folder.
    args.output.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    partitions = find_site_partitions(args.data_root)
    if not partitions:
        raise RuntimeError(f"No */*/train partitions found under {args.data_root}")

    # Test and validation sites are excluded from the training partition list.
    test_dirs = [
        path for path in partitions
        if any(site_matches(path.parent.name, site) for site in TEST_SITES)
    ]
    validation_dirs = [
        path for path in partitions
        if any(site_matches(path.parent.name, site) for site in VALIDATION_SITES)
    ]
    training_dirs = [
        path for path in partitions
        if path not in test_dirs and path not in validation_dirs
        and (
            TRAINING_SITES is None
            or any(site_matches(path.parent.name, site) for site in TRAINING_SITES)
        )
    ]
    excluded_dirs = [
        path
        for path in partitions
        if path not in test_dirs
        and path not in validation_dirs
        and path not in training_dirs
    ]
    if not training_dirs or not validation_dirs or not test_dirs:
        raise RuntimeError(
            "The grouped split is incomplete: "
            f"train={len(training_dirs)}, validation={len(validation_dirs)}, "
            f"test={len(test_dirs)} partitions. Check the configured site names."
        )

    print(
        f"Site partitions: train={len(training_dirs)}, "
        f"validation={len(validation_dirs)}, test={len(test_dirs)}",
        flush=True,
    )
    train_rows = select_training_crops(training_dirs, args.max_per_class, args.seed)
    validation_rows = select_training_crops(
        validation_dirs, max(100, args.max_per_class // 2), args.seed + 1
    )
    test_rows = read_manifest(args.eval_manifest)

    # Audit separation at both levels used by the data: source image and site.
    def site_group(path: Path) -> str:
        flight_name = path.parent.name
        return next(
            (site for site in ALL_SITE_GROUPS if site_matches(flight_name, site)),
            flight_name,
        )

    train_images = {str(Path(row[0]).resolve()) for row in train_rows}
    validation_images = {str(Path(row[0]).resolve()) for row in validation_rows}
    test_images = {str(Path(row[0]).resolve()) for row in test_rows}
    train_groups = {site_group(path) for path in training_dirs}
    validation_groups = {site_group(path) for path in validation_dirs}
    test_groups = set(TEST_SITES)
    manifest_test_groups = {site_group(Path(row[7])) for row in test_rows}
    unexpected_test_groups = sorted(manifest_test_groups - test_groups)
    if unexpected_test_groups:
        raise RuntimeError(
            "The frozen test manifest contains sites outside this fold: "
            f"{unexpected_test_groups}; expected only {sorted(test_groups)}"
        )
    tile_overlap = {
        "train_test": len(train_images & test_images),
        "validation_test": len(validation_images & test_images),
        "train_validation": len(train_images & validation_images),
    }
    site_overlap = {
        "train_test": sorted(train_groups & test_groups),
        "validation_test": sorted(validation_groups & test_groups),
        "train_validation": sorted(train_groups & validation_groups),
    }
    if any(tile_overlap.values()) or any(site_overlap.values()):
        raise RuntimeError(
            "Split leakage detected; refusing to train. "
            f"Source-image overlap={tile_overlap}; site overlap={site_overlap}"
        )

    split_audit = {
        "test_sites": sorted(test_groups),
        "validation_sites": sorted(validation_groups),
        "training_sites": sorted(train_groups),
        "train_crops": len(train_rows),
        "validation_crops": len(validation_rows),
        "test_crops": len(test_rows),
        "unassigned_site_partitions_excluded": [
            path.relative_to(args.data_root).as_posix() for path in excluded_dirs
        ],
        "train_source_images": len(train_images),
        "validation_source_images": len(validation_images),
        "test_source_images": len(test_images),
        "test_manifest_sites": sorted(manifest_test_groups),
        "source_image_overlap": tile_overlap,
        "site_group_overlap": site_overlap,
    }
    (args.output / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_audit, indent=2), flush=True)

    train_manifest = args.output / "crop_manifest_train.csv"
    validation_manifest = args.output / "crop_manifest_val.csv"
    test_manifest = args.output / "crop_manifest_test.csv"
    write_manifest(train_manifest, train_rows)
    write_manifest(validation_manifest, validation_rows)
    write_manifest(test_manifest, test_rows)

    train_counts = Counter(row[6] for row in train_rows)
    validation_counts = Counter(row[6] for row in validation_rows)
    missing_train = [CLASS_NAMES[i] for i in range(7) if train_counts[i] == 0]
    missing_validation = [CLASS_NAMES[i] for i in range(7) if validation_counts[i] == 0]
    if missing_train or missing_validation:
        raise RuntimeError(
            "Every class must appear in training and validation; "
            f"missing train={missing_train}, missing validation={missing_validation}"
        )

    run_config = {
        "model_id": MODEL_ID,
        "test_sites": sorted(test_groups),
        "validation_sites": sorted(validation_groups),
        "data_root": str(args.data_root),
        "evaluation_manifest": str(args.eval_manifest),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "class_weight_power": args.class_weight_power,
        "max_training_crops_per_class": args.max_per_class,
        "train_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "test_manifest_sha256": sha256_file(test_manifest),
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    # Let timm supply the normalization/resize preprocessing associated with
    # the exact Hugging Face checkpoint.
    model = timm.create_model(
        f"hf-hub:{MODEL_ID}", pretrained=True, num_classes=7
    ).to(device)
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        CropDataset(train_rows, transform, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        CropDataset(validation_rows, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Inverse square-root class frequency reduces imbalance without assigning
    # the rarest classes the very large weights produced by full inversion.
    counts = np.array([train_counts[class_id] for class_id in range(7)], dtype=float)
    class_weights = np.power(np.maximum(counts, 1), -args.class_weight_power)
    class_weights /= class_weights.mean()
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    epoch_history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)

        validation_true, validation_pred = evaluate(
            model, validation_loader, device
        )
        validation_f1 = f1_score(
            validation_true,
            validation_pred,
            average="macro",
            labels=list(range(7)),
            zero_division=0,
        )
        validation_accuracy = accuracy_score(validation_true, validation_pred)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        }
        epoch_history.append(record)
        print(json.dumps(record), flush=True)

        # The test set is never used to choose an epoch or checkpoint.
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_id": MODEL_ID,
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "validation_macro_f1": validation_f1,
                    "seed": args.seed,
                },
                args.output / "best_model.pt",
            )

    (args.output / "epoch_metrics.json").write_text(
        json.dumps(epoch_history, indent=2) + "\n", encoding="utf-8"
    )

    # Evaluate the validation-selected checkpoint once on the frozen test list.
    checkpoint = torch.load(
        args.output / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = DataLoader(
        CropDataset(test_rows, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_true, test_pred = evaluate(model, test_loader, device)
    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(7)),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "model_id": MODEL_ID,
        "n_crops": len(test_true),
        "accuracy": accuracy_score(test_true, test_pred),
        "macro_f1": f1_score(
            test_true, test_pred, average="macro", labels=list(range(7)), zero_division=0
        ),
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            test_true, test_pred, labels=list(range(7))
        ).tolist(),
        "best_epoch": best_epoch,
        "validation_best_macro_f1": best_validation_f1,
        "elapsed_seconds": time.time() - started,
        "evaluation_manifest_sha256": run_config["test_manifest_sha256"],
        "note": "Crop-level classification; this model does not output pixel masks.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    prediction_path = args.output / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(MANIFEST_COLUMNS + ["predicted_id", "predicted_name", "correct"])
        for row, predicted in zip(test_rows, test_pred):
            writer.writerow(row + (predicted, CLASS_NAMES[predicted], row[6] == predicted))

    print(
        json.dumps(
            {
                "saved_to": str(args.output),
                "test_crops": len(test_true),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "best_epoch": best_epoch,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
