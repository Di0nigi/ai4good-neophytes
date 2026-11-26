#!/usr/bin/env python3
"""Benchmark the real CV3 PlantNet pipeline without starting a training run.

Measures end-to-end data loading, forward/backward, the same training/validation
metrics used by ``SegmentationModel``, patch-wise full-tile validation, and peak
CUDA memory. Runtime projections are reported both directly from measurements and
with an explicitly labelled overhead buffer.
"""

import argparse
import math
import os
import time

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.semseg_plm import SegmentationModel, compute_patch_params
from samDataModule import SamNeophyteDataModule
from utils.transform_utils import get_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batches", type=int, default=10)
    parser.add_argument("--val-tiles", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--overhead-fraction", type=float, default=0.15,
        help="Labelled projection buffer for checkpointing/logging/plotting.",
    )
    return parser.parse_args()


def compose_config():
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[
                "data=neophytes_split_cv3_train_512_plantnet",
                "model=model_neophytes_plantnet_simdinov2_vitl_linear",
                "logger=csv",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def build_datamodule(cfg):
    train_tf = get_transforms(cfg.data.train_transforms, cfg.data.transforms_additional_targets)
    val_tf = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    test_tf = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_tf = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_tf = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    return SamNeophyteDataModule(
        cfg.data.image_dir, cfg.data.classes, train_tf, val_tf, test_tf,
        dsm_transform=dsm_tf,
        ndsm_transform=ndsm_tf,
        dataset_stats=cfg.data.dataset_stats,
        weighted_sampling=cfg.data.weighted_sampling,
        cap_quantile=cfg.data.cap_quantile,
        lift_quantile=cfg.data.lift_quantile,
        bg_fraction=cfg.data.bg_fraction,
        min_strat_images=cfg.data.min_strat_images,
        value_mapping=cfg.data.value_mapping,
        ignore_index=cfg.data.ignore_index,
        load_auxiliary=cfg.data.load_auxiliary,
        load_dsm=cfg.data.load_dsm,
        load_dtm=cfg.data.load_dtm,
        load_ndsm=cfg.data.load_ndsm,
        few_shot_k=cfg.data.few_shot_k,
        few_shot_mode=cfg.data.few_shot_mode,
        few_shot_bg_k=cfg.data.few_shot_bg_k,
        max_nodata_frac=cfg.data.max_nodata_frac,
        val_few_shot_k=cfg.data.val_few_shot_k,
        val_few_shot_mode=cfg.data.val_few_shot_mode,
        val_few_shot_bg_k=cfg.data.val_few_shot_bg_k,
        load_into_ram=False,
        batch_size_train=cfg.data.batch_size_train,
        batch_size_val=cfg.data.batch_size_val,
        batch_size_test=cfg.data.batch_size_test,
        num_workers=cfg.data.num_workers,
        citizen_cutouts_csv=cfg.data.citizen_cutouts_csv,
        citizen_k=cfg.data.citizen_k,
        citizen_cache=cfg.data.citizen_cache,
        citizen_max_gsd_ratio=cfg.data.citizen_max_gsd_ratio,
        citizen_min_mask_cov=cfg.data.citizen_min_mask_cov,
        citizen_max_border_touch=cfg.data.citizen_max_border_touch,
    )


def build_model(cfg):
    return SegmentationModel(
        cfg.model.model_name,
        cfg.model.encoder_name,
        cfg.model.img_size,
        cfg.model.num_classes,
        cfg.model.lr,
        in_channels=cfg.model.in_channels,
        ignore_index=cfg.data.ignore_index,
        optimizer=cfg.model.optimizer,
        lr_scheduler=cfg.model.lr_scheduler,
        loss=cfg.model.loss,
        weight=cfg.model.weight,
        patch_2_img_size=cfg.model.patch_2_img_size,
        fusion_mode=cfg.model.fusion_mode,
        patch_batch_size=cfg.model.get("patch_batch_size"),
    )


def optimizer_from(model):
    configured = model.configure_optimizers()
    return configured["optimizer"] if isinstance(configured, dict) else configured


def update_metrics(model, stage, logits, masks):
    preds = torch.argmax(logits, dim=1)
    getattr(model, f"{stage}_acc").update(preds, masks)
    getattr(model, f"{stage}_iou").update(preds, masks)
    getattr(model, f"{stage}_f1").update(preds, masks)
    getattr(model, f"{stage}_auc").update(logits, masks)
    getattr(model, f"{stage}_f1_per_class").update(preds, masks)


def reset_metrics(model, stage):
    for name in ("acc", "iou", "f1", "auc", "f1_per_class"):
        getattr(model, f"{stage}_{name}").reset()


def metric_summary(model, stage):
    return {
        "accuracy": float(getattr(model, f"{stage}_acc").compute()),
        "iou": float(getattr(model, f"{stage}_iou").compute()),
        "f1": float(getattr(model, f"{stage}_f1").compute()),
        "auc": float(getattr(model, f"{stage}_auc").compute()),
    }


def move_inputs(model, batch, device):
    moved = {}
    for key in model.in_channels:
        moved[key] = batch[key].to(device, non_blocking=True)
    inputs, aux = model.prepare_inputs(moved)
    masks = batch["mask"].to(device, non_blocking=True).long()
    return inputs, aux, masks


def train_batch(model, optimizer, batch, device):
    inputs, aux, masks = move_inputs(model, batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model._forward(inputs, aux)
        loss = model.criterion(logits, masks)
    loss.backward()
    optimizer.step()
    update_metrics(model, "train", logits.detach(), masks)
    return inputs.shape[0], loss.detach()


def val_batch(model, batch, device):
    inputs, aux, masks = move_inputs(model, batch, device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model._forward(inputs, aux)
        loss = model.criterion(logits, masks)
    update_metrics(model, "val", logits, masks)
    return inputs.shape[-2:], logits.shape, loss.detach()


def timed_training(model, optimizer, loader, device, warmup, requested):
    model.train()
    iterator = iter(loader)
    for _ in range(min(warmup, len(loader))):
        train_batch(model, optimizer, next(iterator), device)
    reset_metrics(model, "train")

    measured = min(requested, max(0, len(loader) - warmup))
    if measured == 0:
        raise RuntimeError("Not enough training batches for the requested warm-up.")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    samples = 0
    losses = []
    for _ in range(measured):
        count, loss = train_batch(model, optimizer, next(iterator), device)
        samples += count
        losses.append(loss)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "batches": measured,
        "seconds_per_batch": elapsed / measured,
        "samples_per_second": samples / elapsed,
        "mean_loss": float(torch.stack(losses).mean()),
        "metrics": metric_summary(model, "train"),
        "allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def timed_validation(model, loader, device, warmup, requested):
    model.eval()
    iterator = iter(loader)
    for _ in range(min(warmup, len(loader))):
        val_batch(model, next(iterator), device)
    reset_metrics(model, "val")

    measured = min(requested, max(0, len(loader) - warmup))
    if measured == 0:
        raise RuntimeError("Not enough validation tiles for the requested warm-up.")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    shapes = None
    for _ in range(measured):
        input_shape, output_shape, loss = val_batch(model, next(iterator), device)
        shapes = (input_shape, tuple(output_shape))
        losses.append(loss)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "tiles": measured,
        "seconds_per_tile": elapsed / measured,
        "mean_loss": float(torch.stack(losses).mean()),
        "metrics": metric_summary(model, "val"),
        "shapes": shapes,
        "allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def print_result(title, result):
    print(f"\n=== {title} ===")
    for key, value in result.items():
        print(f"{key}: {value}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PlantNet benchmark.")

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda")
    cfg = compose_config()

    setup_started = time.perf_counter()
    datamodule = build_datamodule(cfg)
    datamodule.setup()
    setup_seconds = time.perf_counter() - setup_started

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    sampler_draws = getattr(train_loader.sampler, "num_samples", len(datamodule.train_dataset))

    print("=== DATA ===")
    print(f"dataset setup seconds: {setup_seconds:.2f}")
    print(f"training tiles: {len(datamodule.train_dataset)}")
    print(f"validation tiles: {len(datamodule.val_dataset)}")
    print(f"test tiles: {len(datamodule.test_dataset)}")
    print(f"weighted sampler draws/epoch: {sampler_draws}")
    print(f"training batches/epoch: {len(train_loader)}")
    print(f"validation batches/epoch: {len(val_loader)}")
    print(f"test batches after fit: {len(test_loader)}")
    print(f"train/val/test batch sizes: {cfg.data.batch_size_train}/"
          f"{cfg.data.batch_size_val}/{cfg.data.batch_size_test}")

    model_started = time.perf_counter()
    model = build_model(cfg).to(device)
    model_seconds = time.perf_counter() - model_started
    optimizer = optimizer_from(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("\n=== MODEL ===")
    print(f"model load seconds (includes download if uncached): {model_seconds:.2f}")
    print(f"trainable parameters: {trainable:,}")
    print(f"frozen parameters: {frozen:,}")
    print(f"backbone training mode: {model.model.backbone.training}")
    print(f"precision: {cfg.model.precision}")
    print(f"patch micro-batch size: {cfg.model.patch_batch_size}")

    stride, pad = compute_patch_params(2048, int(cfg.model.img_size))
    patches_axis = (2048 + 2 * pad - int(cfg.model.img_size)) // stride + 1
    print(f"2048px patches/tile: {patches_axis ** 2} ({patches_axis}x{patches_axis})")

    train_result = timed_training(
        model, optimizer, train_loader, device,
        args.warmup_batches, args.train_batches,
    )
    print_result("MEASURED TRAINING", train_result)

    val_result = timed_validation(
        model, val_loader, device,
        min(1, args.warmup_batches), args.val_tiles,
    )
    print_result("MEASURED VALIDATION", val_result)

    train_epoch = len(train_loader) * train_result["seconds_per_batch"]
    val_epoch = len(val_loader) * val_result["seconds_per_tile"]
    final_test = len(test_loader) * val_result["seconds_per_tile"]
    buffer_multiplier = 1.0 + args.overhead_fraction

    print("\n=== RUNTIME PROJECTIONS ===")
    print("Measured projection = training + validation each epoch + final Lightning test.")
    print("Buffered projection adds an explicitly assumed "
          f"{args.overhead_fraction:.0%} for checkpointing/logging/other overhead.")
    print("Detailed test.py plots and 200-threshold AUROC are not measured here.")
    for epochs in (1, 5, 10, 50):
        measured_seconds = epochs * (train_epoch + val_epoch) + final_test
        print(
            f"{epochs:>2} epochs: measured={measured_seconds / 3600:.2f}h, "
            f"buffered={measured_seconds * buffer_multiplier / 3600:.2f}h"
        )


if __name__ == "__main__":
    main()
