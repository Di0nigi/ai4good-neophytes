#!/usr/bin/env python3
"""Evaluate the unchanged PlantNet classifier on held-out Neophytes crops.

The script does not train or modify the checkpoint. It extracts non-overlapping
512x512 RGB crops from the configured held-out sites, assigns crop labels from the
project's pixel masks using the fixed rule below, and evaluates top-1 predictions.
When run directly, the configured sites are CV3; the grouped-CV runner replaces
that list for each fold.

The crop-label rule is intentionally explicit and configurable:
  * background/other: >= 90% of valid annotated pixels are background and
    foreground species together occupy <= 2%;
  * a species: it occupies >= 2% of valid pixels and >= 80% of all annotated
    foreground pixels;
  * mixed/uncertain crops are skipped.

Mask values 254 and 255 are ignored, matching the project data config.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from timm.data import create_transform, resolve_data_config


MODEL_ID = "PlantNet/vit_base_patch14_reg4_dinov2_lvd142m.pc24_all"
SPECIES = {
    1: "Buddleja davidii",
    2: "Rhus typhina",
    3: "Reynoutria japonica",
    4: "Bunias orientalis",
    5: "Senecio inaequidens",
    6: "Ailanthus altissima",
}
CLASS_NAMES = {0: "other_background", **SPECIES}

# These are the CV3 held-out sites from configs/data/split_cv3.yaml.
# The config deliberately uses each held-out site's /train partition as test;
# its /test partition is retained for separate few-shot experiments.
HELD_OUT_PARTITIONS = (
    "2024/Basel_2_1/train",
    "2024/Basel_2_2/train",
    "2024/Basel_2_3/train",
    "2024/Basel_2_4/train",
    "2024/Basel_2_5/train",
    "2024/Basel_2_6/train",
    "2024/Waltenschwil_1_1/train",
    "2024/Waltenschwil_1_2/train",
    "2025/Dornach_1_1/train",
    "2025/Dornach_1_2/train",
    "2025/Rivera_1_1/train",
    "2025/Rivera_1_2/train",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", type=Path, required=True,
        help="Directory containing 2024/ and 2025/ (the Neophytes dataset root).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/plantnet_cv3_pretrained"),
        help="Output folder for manifest, predictions, and metrics.",
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--min-species-pixel-fraction", type=float, default=0.02)
    p.add_argument("--min-dominant-foreground-fraction", type=float, default=0.80)
    p.add_argument("--max-foreground-fraction-for-background", type=float, default=0.02)
    p.add_argument("--min-background-fraction", type=float, default=0.90)
    p.add_argument(
        "--max-crops", type=int, default=None,
        help="Optional deterministic smoke-test limit. Omit to evaluate all accepted crops.",
    )
    p.add_argument("--balanced-crops-per-class", type=int, default=None, help="Select an equal, site-balanced number of unique-image crops for each of background and six species.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-threads", type=int, default=4)
    return p.parse_args(argv)


def build_mask_index(split_dir: Path) -> dict[tuple[str, ...], Path]:
    image_dir = split_dir / "images"
    mask_dir = next(
        (split_dir / name for name in ("masks_prep", "labels", "masks") if (split_dir / name).is_dir()),
        split_dir / "masks_prep",
    )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing masks_prep/, labels/, or masks/ under {split_dir}")

    masks = sorted(p for p in mask_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png"})
    index: dict[tuple[str, ...], Path] = {}
    for path in masks:
        parts = list(path.relative_to(mask_dir).with_suffix("").parts)
        # Mask exports add this marker; paired RGB filenames do not.
        parts[-1] = parts[-1].replace("_json_to_mask", "").replace("_registered_resampled", "").replace("_mask_", "_")
        key = tuple(parts)
        if key in index:
            raise RuntimeError(f"Duplicate mask key {key} under {mask_dir}")
        index[key] = path
    return index


def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    image_dir = split_dir / "images"
    mask_index = build_mask_index(split_dir)
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        parts = list(image.relative_to(image_dir).with_suffix("").parts); parts[-1] = parts[-1].replace("_registered_resampled", ""); key = tuple(parts)
        mask = mask_index.get(key)
        if mask is None:
            matches = [v for k, v in mask_index.items() if k[-1] == image.stem]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Could not uniquely match annotation for {image}; found {len(matches)} basename matches"
                )
            mask = matches[0]
        pairs.append((image, mask))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {split_dir}")
    return pairs


def crop_target(mask: np.ndarray, args: argparse.Namespace) -> tuple[int | None, str]:
    """Return (class id, reason); None means the crop is ambiguous/ignored."""
    values, counts = np.unique(mask, return_counts=True)
    count_by_value = {int(v): int(c) for v, c in zip(values, counts)}
    valid_total = sum(c for v, c in count_by_value.items() if v in range(7))
    if valid_total == 0:
        return None, "no_valid_annotation"

    background_fraction = count_by_value.get(0, 0) / valid_total
    species_counts = {class_id: count_by_value.get(class_id, 0) for class_id in SPECIES}
    foreground_total = sum(species_counts.values())
    foreground_fraction = foreground_total / valid_total

    if (
        foreground_fraction <= args.max_foreground_fraction_for_background
        and background_fraction >= args.min_background_fraction
    ):
        return 0, "accepted_background"

    if foreground_total == 0:
        return None, "ambiguous_no_species_or_background"
    best_id = max(species_counts, key=species_counts.get)
    best_pixel_fraction = species_counts[best_id] / valid_total
    best_foreground_fraction = species_counts[best_id] / foreground_total
    if (
        best_pixel_fraction >= args.min_species_pixel_fraction
        and best_foreground_fraction >= args.min_dominant_foreground_fraction
    ):
        return best_id, "accepted_species"
    return None, "ambiguous_mixed_or_sparse_species"


def windows(height: int, width: int, size: int) -> Iterable[tuple[int, int, int, int]]:
    """Yield non-overlapping windows, keeping smaller edge windows as-is."""
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield y, x, min(y + size, height), min(x + size, width)


def load_rgb_crop(path: Path, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Expected at least 3 RGB bands in {path}; found {src.count}")
        arr = src.read((1, 2, 3), window=((y0, y1), (x0, x1)))
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB data like the project pipeline, got {arr.dtype} in {path}; "
            "inspect the TIFF scaling before evaluating."
        )
    return arr


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def normalize_label_name(value: str) -> str:
    # PlantNet labels append botanical authorities; compare canonical binomials.
    words = value.casefold().replace("_", " ").split()
    return " ".join(words[:2])


def get_model_labels() -> tuple[list[dict], dict[str, int]]:
    import csv

    class_map_path = hf_hub_download(MODEL_ID, "class_mapping.txt")
    species_path = hf_hub_download(MODEL_ID, "species_id_to_name.txt")
    with open(class_map_path, encoding="utf-8") as f:
        species_ids = [line.strip() for line in f if line.strip()]
    with open(species_path, encoding="utf-8", newline="") as f:
        species_by_id = {
            row["species_id"]: row["species"]
            for row in csv.DictReader(f, delimiter=";", quotechar='"')
        }

    labels = [{"species": species_by_id.get(species_id, species_id)} for species_id in species_ids]
    name_to_index = {
        normalize_label_name(item["species"]): idx
        for idx, item in enumerate(labels)
    }
    missing = [name for name in SPECIES.values() if normalize_label_name(name) not in name_to_index]
    if missing:
        raise RuntimeError(f"Selected model has no exact labels for project species: {missing}")
    return labels, name_to_index


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    n_classes = len(CLASS_NAMES)
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for actual, predicted in zip(y_true, y_pred):
        cm[actual][predicted] += 1
    per_class = {}
    all_class_f1s = []
    species_f1s = []
    for class_id, name in CLASS_NAMES.items():
        tp = cm[class_id][class_id]
        fp = sum(cm[r][class_id] for r in range(n_classes) if r != class_id)
        fn = sum(cm[class_id][c] for c in range(n_classes) if c != class_id)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else 0.0
        per_class[str(class_id)] = {
            "name": name,
            "support_crops": sum(cm[class_id]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        # Macro-F1 over all seven labels is directly comparable with the
        # seven-output fine-tuned classifier, including other/background.
        all_class_f1s.append(f1)
        # Keep the earlier foreground-only measure as a secondary diagnostic.
        if class_id != 0 and sum(cm[class_id]) > 0:
            species_f1s.append(f1)
    total = len(y_true)
    accuracy = sum(cm[i][i] for i in range(n_classes)) / total if total else None
    return {
        "n_evaluated_crops": total,
        "accuracy": accuracy,
        "macro_f1_all_classes": float(np.mean(all_class_f1s)) if all_class_f1s else None,
        "macro_f1_species_present": float(np.mean(species_f1s)) if species_f1s else None,
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": cm,
        "class_order": [CLASS_NAMES[i] for i in range(n_classes)],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1 or args.crop_size < 1:
        raise ValueError("batch-size and crop-size must be positive")
    torch.set_num_threads(args.num_threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    args.data_root = args.data_root.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading unchanged checkpoint {MODEL_ID} on {device} ...", flush=True)
    model = timm.create_model(f"hf-hub:{MODEL_ID}", pretrained=True).eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model_labels, name_to_index = get_model_labels()
    model_index_to_project = {
        name_to_index[normalize_label_name(name)]: class_id
        for class_id, name in SPECIES.items()
    }
    print(f"Verified all six target species among {len(model_labels)} PlantNet labels.", flush=True)

    candidates: list[dict] = []
    audit = Counter()
    for relative_split in HELD_OUT_PARTITIONS:
        split_dir = args.data_root / relative_split
        print(f"Scanning {split_dir}", flush=True)
        for image_path, mask_path in list_pairs(split_dir):
            with rasterio.open(image_path) as src:
                height, width = src.height, src.width
            mask = load_mask(mask_path)
            if mask.shape != (height, width):
                raise ValueError(f"Image/mask dimensions differ: {image_path} {height}x{width}, {mask_path} {mask.shape}")
            unexpected = sorted(int(v) for v in np.unique(mask) if int(v) not in {*range(7), 254, 255})
            if unexpected:
                raise ValueError(f"Unexpected annotation values {unexpected} in {mask_path}; expected 0..6, 254, 255")
            for y0, x0, y1, x1 in windows(height, width, args.crop_size):
                patch_mask = mask[y0:y1, x0:x1]
                target, reason = crop_target(patch_mask, args)
                audit[reason] += 1
                if target is None:
                    continue
                candidates.append({
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "site_split": relative_split,
                    "y0": y0,
                    "x0": x0,
                    "y1": y1,
                    "x1": x1,
                    "target_id": target,
                    "target_name": CLASS_NAMES[target],
                })

    if args.max_crops is not None and args.balanced_crops_per_class is not None:
        raise ValueError("Use either --max-crops or --balanced-crops-per-class, not both.")
    if args.balanced_crops_per_class is not None:
        if args.balanced_crops_per_class < 1:
            raise ValueError("--balanced-crops-per-class must be at least 1.")
        rng = np.random.default_rng(args.seed)
        class_ids = list(range(7))
        by_class: dict[int, list[dict]] = {class_id: [] for class_id in class_ids}
        for crop in candidates:
            by_class[int(crop["target_id"])].append(crop)
        counts = {class_id: len({str(crop["image_path"]) for crop in by_class[class_id]}) for class_id in class_ids}
        if any(counts[class_id] == 0 for class_id in class_ids):
            raise RuntimeError(f"Cannot form a balanced set; candidate counts by class id: {counts}")
        per_class = min(args.balanced_crops_per_class, min(counts.values()))
        selected_candidates = []
        selected_per_class = {}
        for class_id in class_ids:
            pools: dict[str, list[dict]] = {}
            for crop in by_class[class_id]:
                pools.setdefault(str(crop["site_split"]), []).append(crop)
            sites = list(pools)
            rng.shuffle(sites)
            for pool in pools.values():
                rng.shuffle(pool)
            chosen = []
            seen_images = set()
            while len(chosen) < per_class:
                advanced = False
                for site in sites:
                    pool = pools[site]
                    while pool:
                        crop = pool.pop()
                        image_key = str(crop["image_path"])
                        if image_key not in seen_images:
                            seen_images.add(image_key)
                            chosen.append(crop)
                            advanced = True
                            break
                    if len(chosen) == per_class:
                        break
                if not advanced:
                    break
            if len(chosen) != per_class:
                raise RuntimeError(f"Could not select {per_class} unique-image crops for class {class_id}; selected {len(chosen)}")
            selected_candidates.extend(chosen)
            selected_per_class[class_id] = chosen
        candidates = selected_candidates
        print(f"Balanced selection: {per_class} crops per class (7 classes; {len(candidates)} total), seed={args.seed}.", flush=True)
        for class_id in class_ids:
            site_counts = {}
            for crop in selected_per_class[class_id]:
                site = str(crop["site_split"])
                site_counts[site] = site_counts.get(site, 0) + 1
            print(f"  class {class_id}: {counts[class_id]} eligible; selected sites={site_counts}", flush=True)
    elif args.max_crops is not None:
        if args.max_crops < 1:
            raise ValueError("--max-crops must be at least 1.")
        rng = np.random.default_rng(args.seed)
        if len(candidates) > args.max_crops:
            selected = np.sort(rng.choice(len(candidates), args.max_crops, replace=False))
            candidates = [candidates[int(i)] for i in selected]
        print(f"Smoke-test limit selected {len(candidates)} crops with seed {args.seed}.", flush=True)
    if args.balanced_crops_per_class is not None and candidates:
        with (args.output / "selected_crops.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
    if not candidates:
        raise RuntimeError("No crops passed the label rule. Check paths, masks, and thresholds.")
    print(f"Evaluating {len(candidates)} accepted crops (ambiguous crops are excluded).", flush=True)

    predictions: list[dict] = []
    started = time.time()
    batch_images: list[torch.Tensor] = []
    batch_rows: list[dict] = []

    def run_batch() -> None:
        if not batch_rows:
            return
        batch = torch.stack(batch_images).to(device)
        with torch.inference_mode():
            probs = model(batch).softmax(dim=-1).cpu()
        for row, prob in zip(batch_rows, probs):
            model_id = int(prob.argmax())
            project_id = model_index_to_project.get(model_id, 0)
            species = model_labels[model_id].get("species", str(model_id))
            predictions.append({
                **row,
                "predicted_id": project_id,
                "predicted_name": CLASS_NAMES[project_id],
                "plantnet_top1_species": species,
                "plantnet_top1_probability": float(prob[model_id]),
            })
        batch_images.clear()
        batch_rows.clear()

    for index, row in enumerate(candidates, 1):
        rgb = load_rgb_crop(Path(row["image_path"]), row["y0"], row["x0"], row["y1"], row["x1"])
        batch_images.append(transform(Image.fromarray(rgb)))
        batch_rows.append(row)
        if len(batch_rows) >= args.batch_size:
            run_batch()
            if index % (args.batch_size * 10) == 0:
                print(f"Predicted {index}/{len(candidates)} crops", flush=True)
    run_batch()
    elapsed = time.time() - started

    y_true = [int(r["target_id"]) for r in predictions]
    y_pred = [int(r["predicted_id"]) for r in predictions]
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "elapsed_seconds": elapsed,
        "crops_per_second": len(predictions) / elapsed if elapsed else None,
        "crop_label_audit_counts": dict(audit),
        "label_rule": {
            "ignored_mask_values": [254, 255],
            "min_species_pixel_fraction": args.min_species_pixel_fraction,
            "min_dominant_foreground_fraction": args.min_dominant_foreground_fraction,
            "max_foreground_fraction_for_background": args.max_foreground_fraction_for_background,
            "min_background_fraction": args.min_background_fraction,
            "ambiguous_crops": "excluded",
        },
        "split": "held-out site partitions; see held_out_site_partitions",
        "held_out_site_partitions": list(HELD_OUT_PARTITIONS),
        "model_id": MODEL_ID,
        "checkpoint_modified": False,
        "device": str(device),
        "crop_size": args.crop_size,
        "seed": args.seed,
        "note": "Crop-level classification, not pixel-level segmentation. PlantNet labels outside the six target species map to other_background.",
    })

    manifest_path = args.output / "crop_manifest.csv"
    prediction_path = args.output / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (args.output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (args.output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "data_root": str(args.data_root),
            "output": str(args.output.resolve()),
            "device": str(device),
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "seed": args.seed,
            "max_crops": args.max_crops,
            "balanced_crops_per_class": args.balanced_crops_per_class,
            "selection": "equal per-class; round-robin across held-out site partitions; unique source image per class" if args.balanced_crops_per_class is not None else "random smoke-test subset" if args.max_crops is not None else "all accepted candidates",
            "held_out_site_partitions": list(HELD_OUT_PARTITIONS),
        }, f, indent=2)

    print(f"Done in {elapsed / 60:.1f} min. Results: {args.output.resolve()}", flush=True)
    print(json.dumps({k: metrics[k] for k in ("n_evaluated_crops", "accuracy", "macro_f1_species_present", "per_class")}, indent=2))


if __name__ == "__main__":
    main()
