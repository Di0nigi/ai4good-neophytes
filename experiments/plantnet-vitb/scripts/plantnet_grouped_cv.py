#!/usr/bin/env python3
"""Prepare and run one site-grouped PlantNet ViT-B cross-validation fold.

``prepare`` freezes the held-out crop list, evaluates the unchanged PlantNet
checkpoint on it, saves metrics/manifests, and logs the baseline to W&B.
``train`` trains on the other site groups, selects an epoch using validation
sites, checks for site/source-image overlap, and evaluates on that same frozen
test manifest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("NEOPHYTES_DATA_ROOT", "/zfs/ai4good/datasets/Neophytes")
)
DEFAULT_RUN_ROOT = Path(
    os.environ.get("PLANTNET_RUN_ROOT", PROJECT_ROOT / "results/plantnet-vitb-cv5")
)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "juliand942-uzh")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "PlantNet")

# A fold names four independent site groups; all flights with that site prefix
# are assigned together to prevent nearby tiles crossing the split boundary.
FOLDS = {
    "cv1": ("Arlesheim_1", "Dietikon_1", "Basel_3", "Aarau_1"),
    "cv2": ("Aarau_3", "Olten_1", "Aarau_2", "Ardon_3"),
    "cv3": ("Dornach_1", "Rivera_1", "Basel_2", "Waltenschwil_1"),
    "cv4": ("Sion_1", "Rivera_2", "Dornach_2", "Ardon_1"),
    "cv5": ("Chamoson_1", "Ardon_2", "Basel_4", "Rueti_1"),
}
VALIDATION_FOLD = {
    "cv1": "cv2",
    "cv2": "cv3",
    "cv3": "cv4",
    "cv4": "cv5",
    "cv5": "cv1",
}


def site_partitions(data_root: Path, site_names: Sequence[str]) -> tuple[str, ...]:
    """Return relative ``.../<flight>/train`` paths belonging to these sites."""
    matches = []
    for path in sorted(data_root.glob("*/*/train")):
        flight_name = path.parent.name
        if any(
            flight_name == site or flight_name.startswith(site + "_")
            for site in site_names
        ):
            matches.append(path.relative_to(data_root).as_posix())
    return tuple(matches)


def refuse_busy_gpu() -> None:
    """Stop rather than start a cluster job on a GPU already in use."""
    try:
        used_memory = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).splitlines()[0]
        )
        process_ids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
    except FileNotFoundError as exc:
        raise SystemExit("nvidia-smi is unavailable; cannot verify GPU availability") from exc

    if used_memory > 1024 or process_ids:
        raise SystemExit(
            f"GPU appears occupied ({used_memory} MiB; compute PIDs: "
            f"{process_ids or 'none'}). Wait before starting this fold."
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log_wandb(name: str, group: str, job_type: str, config: dict, metrics: dict) -> None:
    """Log a fold summary if W&B is installed and authenticated.

    A logging failure does not discard local metrics or stop a completed run.
    Credentials are read by the W&B client from its normal login configuration.
    """
    try:
        import wandb

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=name,
            group=group,
            job_type=job_type,
            tags=["PlantNet", "ViT-B", "grouped-CV"],
            config=config,
            settings=wandb.Settings(init_timeout=30),
        )
        payload = {
            f"test/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for class_id, values in metrics.get("per_class", {}).items():
            if not isinstance(values, dict):
                continue
            for metric_name in ("precision", "recall", "f1"):
                value = values.get(metric_name)
                if isinstance(value, (int, float)):
                    payload[f"test/class_{class_id}_{metric_name}"] = value
        for class_name, values in metrics.get("classification_report", {}).items():
            if not isinstance(values, dict):
                continue
            label = "_".join(str(class_name).lower().replace(".", "").split())
            for source_name, suffix in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1-score", "f1"),
                ("support", "support"),
            ):
                value = values.get(source_name)
                if isinstance(value, (int, float)):
                    payload[f"test/class_{label}_{suffix}"] = value
        if payload:
            run.log(payload)
        run_url = run.url
        run.finish()
        print(f"W&B logged: {run_url}", flush=True)
    except Exception as exc:  # noqa: BLE001 - local run outputs remain authoritative
        print(
            f"W&B logging unavailable; local results are saved "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )


def add_macro_f1_from_saved_predictions(eval_dir: Path, metrics: dict) -> dict:
    """Add seven-class macro-F1 to older saved raw-model metrics without inference."""
    if "macro_f1_all_classes" in metrics:
        return metrics

    predictions_path = eval_dir / "predictions.csv"
    if not predictions_path.is_file():
        return metrics

    import evaluate_pretrained_vitb as evaluator

    predictions = read_csv_rows(predictions_path)
    metrics.update(
        evaluator.compute_metrics(
            [int(row["target_id"]) for row in predictions],
            [int(row["predicted_id"]) for row in predictions],
        )
    )
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train"), required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--crops-per-class", type=int, default=50)
    parser.add_argument("--crop-seed", type=int, default=31415)
    parser.add_argument("--train-seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device == "cuda":
        refuse_busy_gpu()

    args.data_root = args.data_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    test_sites = FOLDS[args.fold]
    test_dirs = site_partitions(args.data_root, test_sites)
    if not test_dirs:
        raise SystemExit(f"No data directories matched {args.fold}: {test_sites}")

    # The evaluator is shared by raw prediction and fine-tuning to ensure the
    # crop-label rule and RGB preprocessing stay identical.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evaluate_pretrained_vitb as evaluator

    evaluator.HELD_OUT_PARTITIONS = test_dirs
    fold_dir = args.run_root / args.fold
    eval_dir = fold_dir / "pretrained_eval"
    weight_tag = (
        "sqrt" if args.class_weight_power == 0.5
        else "power" + str(args.class_weight_power).replace(".", "p")
    )
    baseline_dir = fold_dir / (
        f"finetune_weighted_{weight_tag}_seed{args.train_seed}_strictsplit"
    )

    if args.stage == "prepare":
        eval_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = eval_dir / "crop_manifest.csv"
        metrics_path = eval_dir / "metrics.json"
        predictions_path = eval_dir / "predictions.csv"
        run_config_path = eval_dir / "run_config.json"

        # Reuse a complete saved evaluation. New code can add the comparable
        # seven-class macro-F1 from the prediction CSV without rerunning ViT-B.
        saved_outputs = (
            manifest_path,
            metrics_path,
            predictions_path,
            run_config_path,
        )
        if not all(path.is_file() for path in saved_outputs):
            evaluator.main(
                [
                    "--data-root", str(args.data_root),
                    "--output", str(eval_dir),
                    "--balanced-crops-per-class", str(args.crops_per_class),
                    "--seed", str(args.crop_seed),
                    "--device", args.device,
                ]
            )
        else:
            with run_config_path.open(encoding="utf-8") as file:
                saved_config = json.load(file)
            saved_sites = saved_config.get(
                "held_out_site_partitions",
                saved_config.get("cv3_held_out_sites", []),
            )
            expected_config = {
                "data_root": str(args.data_root),
                "seed": args.crop_seed,
                "crop_size": 512,
                "balanced_crops_per_class": args.crops_per_class,
                "held_out_site_partitions": list(test_dirs),
                "model_id": evaluator.MODEL_ID,
            }
            saved_relevant_config = {
                "data_root": saved_config.get("data_root"),
                "seed": saved_config.get("seed"),
                "crop_size": saved_config.get("crop_size"),
                "balanced_crops_per_class": saved_config.get(
                    "balanced_crops_per_class"
                ),
                "held_out_site_partitions": saved_sites,
                "model_id": saved_config.get("model_id"),
            }
            if saved_relevant_config != expected_config:
                raise SystemExit(
                    "A frozen evaluation already exists with different settings. "
                    "Keep it unchanged and choose a new --run-root for a new "
                    f"evaluation. Saved={saved_relevant_config}; "
                    f"requested={expected_config}"
                )
            print(f"Reusing frozen crop list: {manifest_path}", flush=True)

        if not manifest_path.is_file() or not predictions_path.is_file():
            raise SystemExit("Pretrained evaluation did not save its manifest and predictions")
        with metrics_path.open(encoding="utf-8") as file:
            metrics = json.load(file)
        metrics = add_macro_f1_from_saved_predictions(eval_dir, metrics)

        rows = read_csv_rows(manifest_path)
        counts = collections.Counter(row["target_id"] for row in rows)
        source_images_per_class = {
            class_id: len(
                {row["image_path"] for row in rows if row["target_id"] == class_id}
            )
            for class_id in counts
        }
        site_counts = collections.Counter(row["site_split"] for row in rows)
        metadata = {
            "fold": args.fold,
            "test_sites": list(test_sites),
            "test_partitions": list(test_dirs),
            "requested_crops_per_class": args.crops_per_class,
            "selected_crops_per_class": dict(counts),
            "unique_source_images_per_class": source_images_per_class,
            "selected_crops_by_site_partition": dict(site_counts),
            "manifest_sha256": sha256_file(manifest_path),
            "crop_seed": args.crop_seed,
        }
        (eval_dir / "frozen_manifest_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2), flush=True)
        log_wandb(
            f"groupedcv-vitb-pretrained-{args.fold}-seed{args.crop_seed}",
            "plantnet-grouped-cv5",
            "pretrained-baseline",
            metadata | {"model_id": evaluator.MODEL_ID},
            metrics,
        )
        return

    manifest_path = eval_dir / "crop_manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Frozen manifest missing for {args.fold}; run --stage prepare first")

    # Configure the fine-tuning module for this fold, then pass its CLI options
    # directly instead of changing process-wide command-line state.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import finetune_vitb_weighted as fine_tuner

    validation_fold = VALIDATION_FOLD[args.fold]
    training_sites = {
        site
        for fold_name, sites in FOLDS.items()
        if fold_name not in (args.fold, validation_fold)
        for site in sites
    }
    fine_tuner.TRAINING_SITES = training_sites
    fine_tuner.TEST_SITES = set(test_sites)
    fine_tuner.VALIDATION_SITES = set(FOLDS[validation_fold])
    fine_tuner.ALL_SITE_GROUPS = {
        site for fold_sites in FOLDS.values() for site in fold_sites
    }
    fine_tuner.main(
        [
            "--data-root", str(args.data_root),
            "--output", str(baseline_dir),
            "--eval-manifest", str(manifest_path),
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--seed", str(args.train_seed),
            "--class-weight-power", str(args.class_weight_power),
        ]
    )

    metrics_path = baseline_dir / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Fine-tuning did not save its metrics: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    train_rows = read_csv_rows(baseline_dir / "crop_manifest_train.csv")
    validation_rows = read_csv_rows(baseline_dir / "crop_manifest_val.csv")
    test_rows = read_csv_rows(baseline_dir / "crop_manifest_test.csv")
    train_images = {row["image_path"] for row in train_rows}
    validation_images = {row["image_path"] for row in validation_rows}
    test_images = {row["image_path"] for row in test_rows}
    protocol = {
        "fold": args.fold,
        "test_sites": list(test_sites),
        "validation_fold": validation_fold,
        "validation_sites": list(FOLDS[validation_fold]),
        "training_sites": sorted(training_sites),
        "test_manifest_sha256": sha256_file(baseline_dir / "crop_manifest_test.csv"),
        "train_test_source_image_overlap": len(train_images & test_images),
        "validation_test_source_image_overlap": len(validation_images & test_images),
    }
    if protocol["train_test_source_image_overlap"] or protocol["validation_test_source_image_overlap"]:
        raise RuntimeError(f"Source-image leakage detected after training: {protocol}")

    (baseline_dir / "fold_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    metrics.update(protocol)
    (baseline_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"protocol": protocol, "metrics": metrics}, indent=2), flush=True)
    log_wandb(
        f"groupedcv-vitb-weighted-{weight_tag}-{args.fold}-seed{args.train_seed}-strictsplit",
        "plantnet-grouped-cv5",
        "finetuned-holdout",
        protocol
        | {
            "epochs": args.epochs,
            "class_weight_power": args.class_weight_power,
            "test_crop_count": len(test_rows),
            "train_crop_count": len(train_rows),
            "validation_crop_count": len(validation_rows),
        },
        metrics,
    )


if __name__ == "__main__":
    main()
