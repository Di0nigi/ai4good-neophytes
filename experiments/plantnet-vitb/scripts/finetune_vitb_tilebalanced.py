#!/usr/bin/env python3
"""Tile-balanced fine-tuning ablation for the PlantNet ViT-B classifier.

This controlled comparison reuses the baseline train/validation manifests and
the same frozen held-out test manifest. Only the training-crop contributions
to the loss change: crops from a source image share equal total weight within
their class.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from timm.data import create_transform, resolve_data_config

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    os.environ.get("PLANTNET_RUN_ROOT", PROJECT_ROOT / "results/plantnet-vitb-cv5")
)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "juliand942-uzh")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "PlantNet")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_vitb_weighted as ft

BASELINE_NAME = "finetune_weighted_sqrt_seed43_strictsplit"
FOLDS = ("cv1", "cv2", "cv3", "cv4", "cv5")

def read_rows(path):
    """Read a saved crop manifest in the tuple format expected by the dataset."""
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["image_path"], r["mask_path"], int(r["y0"]), int(r["x0"]),
                         int(r["y1"]), int(r["x1"]), int(r["target_id"]), r["site_split"]))
    return rows

def sha256(path):
    """Hash a manifest's bytes to prove two comparisons used the same list."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

class TileWeightedDataset(Dataset):
    """Return each baseline crop together with its precomputed loss weight."""

    def __init__(self, rows, transform, weights):
        self.base = ft.CropDataset(rows, transform, augment=True)
        self.weights = weights
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, torch.tensor(self.weights[index], dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=FOLDS, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--baseline-name",
        default=BASELINE_NAME,
        help="Baseline output directory name containing the saved crop manifests.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_root = args.run_root.expanduser().resolve()
    base_dir = run_root / args.fold / args.baseline_name
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Baseline manifests are missing: {base_dir}")
    train_src = base_dir / "crop_manifest_train.csv"
    val_src = base_dir / "crop_manifest_val.csv"
    test_src = run_root / args.fold / "pretrained_eval" / "crop_manifest.csv"
    output = run_root / args.fold / f"finetune_tilebalanced_seed{args.seed}_strictsplit"
    output.mkdir(parents=True, exist_ok=False)

    train_rows, val_rows, test_rows = read_rows(train_src), read_rows(val_src), read_rows(test_src)
    for src, name in ((train_src, "crop_manifest_train.csv"),
                      (val_src, "crop_manifest_val.csv"),
                      (test_src, "crop_manifest_test.csv")):
        shutil.copy2(src, output / name)
    for name in ("split_audit.json", "fold_protocol.json"):
        shutil.copy2(base_dir / name, output / name)

    train_counts = Counter(row[6] for row in train_rows)
    tile_counts = Counter((row[0], row[6]) for row in train_rows)
    unique_tiles_per_class = {c: len({row[0] for row in train_rows if row[6] == c}) for c in range(7)}
    cw = np.power(np.maximum(np.array([train_counts[c] for c in range(7)], dtype=np.float64), 1.0),
                  -args.class_weight_power)
    cw = cw / cw.mean()
    class_weights_tensor = torch.as_tensor(cw, dtype=torch.float32, device=device)
    # Within each class, normalize crop weights so every source tile has equal
    # total loss contribution while preserving the baseline class-level weight.
    sample_weights = []
    for row in train_rows:
        label = row[6]
        within_class_tile_weight = train_counts[label] / (
            unique_tiles_per_class[label] * tile_counts[(row[0], label)]
        )
        sample_weights.append(float(cw[label] * within_class_tile_weight))

    train_hash, val_hash, test_hash = sha256(train_src), sha256(val_src), sha256(test_src)
    config = {
        "fold": args.fold,
        "model_id": ft.MODEL_ID,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "class_weight_power": args.class_weight_power,
        "train_manifest_sha256": train_hash,
        "validation_manifest_sha256": val_hash,
        "test_manifest_sha256": test_hash,
        "train_crop_count": len(train_rows),
        "train_crops_per_class": {str(k): train_counts[k] for k in range(7)},
        "unique_training_tiles_per_class": {str(k): unique_tiles_per_class[k] for k in range(7)},
        "tile_weighting": "within each class, each source image has equal total loss contribution",
        "validation_and_test_manifests": "reused byte-for-byte from the strict-split baseline",
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model = timm.create_model(f"hf-hub:{ft.MODEL_ID}", pretrained=True, num_classes=7).to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    train_loader = DataLoader(TileWeightedDataset(train_rows, transform, sample_weights),
                              batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(ft.CropDataset(val_rows, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    best_f1, best_epoch, history = -1.0, 0, []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = seen = 0
        for x, y, sample_weight in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            # Keep the baseline's class-weight denominator. Tile balancing
            # changes each crop's contribution in the numerator, while the
            # average within-class tile multiplier remains one over the full
            # training set. This makes the ablation differ in crop weighting,
            # not in the overall scale of the weighted loss.
            per_crop_loss = F.cross_entropy(logits, y, reduction="none")
            class_weight_denominator = class_weights_tensor[y].sum()
            loss = (per_crop_loss * sample_weight).sum() / class_weight_denominator
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            seen += len(y)

        val_y, val_pred = ft.evaluate(model, val_loader, device)
        val_f1 = f1_score(val_y, val_pred, average="macro", labels=list(range(7)), zero_division=0)
        val_acc = accuracy_score(val_y, val_pred)
        record = {"epoch": epoch, "train_loss": total_loss / seen,
                  "train_accuracy": correct / seen, "val_accuracy": val_acc,
                  "val_macro_f1": val_f1}
        history.append(record)
        (output / "epoch_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"model_state_dict": model.state_dict(), "model_id": ft.MODEL_ID,
                        "class_names": ft.CLASS_NAMES, "epoch": epoch, "val_macro_f1": val_f1,
                        "seed": args.seed}, output / "best_model.pt")

    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = DataLoader(ft.CropDataset(test_rows, transform), batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
    test_y, test_pred = ft.evaluate(model, test_loader, device)
    report = classification_report(test_y, test_pred, labels=list(range(7)),
                                   target_names=ft.CLASS_NAMES, zero_division=0, output_dict=True)
    metrics = {
        "fold": args.fold,
        "n_crops": len(test_y),
        "accuracy": accuracy_score(test_y, test_pred),
        "macro_f1": f1_score(test_y, test_pred, average="macro", labels=list(range(7)), zero_division=0),
        "class_names": ft.CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_y, test_pred, labels=list(range(7))).tolist(),
        "validation_best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "note": "Crop-level classification diagnostic, not pixel segmentation.",
        "tile_weighting": config["tile_weighting"],
        "train_manifest_sha256": train_hash,
        "validation_manifest_sha256": val_hash,
        "test_manifest_sha256": test_hash,
    }
    (output / "fold_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "target_id", "predicted_id", "correct"])
        writer.writerows((row[0], y, p, y == p) for row, y, p in zip(test_rows, test_y, test_pred))
    print(json.dumps({"saved_to": str(output), "fold": args.fold, "n": len(test_y),
                      "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "best_val_macro_f1": best_f1, "best_epoch": best_epoch}, indent=2), flush=True)

    try:
        import wandb
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                         name=f"groupedcv-vitb-tilebalanced-{args.fold}-seed{args.seed}",
                         group="plantnet-grouped-cv5-tilebalance",
                         job_type="tile-balanced-finetuning",
                         tags=["PlantNet", "ViT-B", "grouped-CV", "tile-balanced"],
                         config=config, settings=wandb.Settings(init_timeout=30))
        for rec in history:
            run.log({"epoch": rec["epoch"], "train/loss": rec["train_loss"],
                     "train/accuracy": rec["train_accuracy"], "val/accuracy": rec["val_accuracy"],
                     "val/macro_f1": rec["val_macro_f1"]})
        payload = {"test/accuracy": metrics["accuracy"], "test/macro_f1": metrics["macro_f1"],
                   "test/n_crops": metrics["n_crops"], "best_epoch": best_epoch,
                   "validation/best_macro_f1": best_f1}
        for name, values in report.items():
            if isinstance(values, dict):
                slug = "_".join(name.lower().replace(".", "").split())
                for source, suffix in (("precision", "precision"), ("recall", "recall"),
                                       ("f1-score", "f1"), ("support", "support")):
                    if source in values:
                        payload[f"test/class_{slug}_{suffix}"] = values[source]
        run.log(payload)
        run.finish()
        print(f"W&B logged: {run.url}", flush=True)
    except Exception as exc:
        print(f"W&B logging unavailable; local results are saved ({type(exc).__name__}: {exc})", flush=True)

if __name__ == "__main__":
    main()
