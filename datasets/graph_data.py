# graph_data.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def read_aligned_raster(path, reference, bands=None):
    """Read a raster and check that it matches the image tile's grid."""
    with rasterio.open(path) as src:
        if (
            src.height != reference.height
            or src.width != reference.width
            or src.transform != reference.transform
            or src.crs != reference.crs
        ):
            raise ValueError(f"{path} is not aligned with the image tile.")

        if bands is None:
            arr = src.read(masked=True).astype(np.float32).filled(np.nan)
        else:
            arr = src.read(bands, masked=True).astype(np.float32).filled(np.nan)

    return arr


def extract_node_features(image, segments, extra_rasters):
    """
    image: (H, W, 3), float
    segments: (H, W), integer IDs from 0 to N-1
    extra_rasters: list of (H, W) arrays, e.g. DSM and DSTM

    Feature order:
      RGB means, RGB standard deviations, RGB covariance eigenvalues,
      log area, aspect ratio, eccentricity,
      then mean/std/valid fraction for each extra raster.
    """
    props = regionprops(segments + 1)
    features = []

    for prop in props:
        rows, cols = prop.coords[:, 0], prop.coords[:, 1]
        pixels = image[rows, cols, :]  # (pixels in segment, 3)

        rgb_mean = np.nanmean(pixels, axis=0)
        rgb_std = np.nanstd(pixels, axis=0)

        valid_rgb = pixels[np.isfinite(pixels).all(axis=1)]
        if len(valid_rgb) > 1:
            cov = np.cov(valid_rgb, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
        else:
            eigvals = np.zeros(3, dtype=np.float32)

        aspect_ratio = (
            prop.major_axis_length / prop.minor_axis_length
            if prop.minor_axis_length > 0
            else 0.0
        )

        values = [
            *rgb_mean,
            *rgb_std,
            *eigvals,
            np.log1p(prop.area),
            aspect_ratio,
            prop.eccentricity,
        ]

        for raster in extra_rasters:
            v = raster[rows, cols]
            v = v[np.isfinite(v)]
            if len(v):
                values.extend([float(v.mean()), float(v.std()), len(v) / prop.area])
            else:
                values.extend([0.0, 0.0, 0.0])

        features.append(values)

    return np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def make_node_labels(mask, segments, class_to_id, min_labeled_fraction=0.25):
    """Assign each segment its majority class; unlabeled/unknown nodes get -1."""
    num_nodes = int(segments.max()) + 1
    labels = np.full(num_nodes, -1, dtype=np.int64)

    for node_id in range(num_nodes):
        pixels = mask[segments == node_id]
        mapped = np.array(
            [class_to_id.get(int(v), -1) for v in pixels],
            dtype=np.int64,
        )
        mapped = mapped[mapped >= 0]

        if len(mapped) == 0:
            continue

        # Avoid training on segments with only a small labeled portion.
        if len(mapped) / len(pixels) < min_labeled_fraction:
            continue

        counts = np.bincount(mapped, minlength=len(class_to_id))
        labels[node_id] = int(counts.argmax())

    return labels


def make_adjacency(segments):
    """Connect segments sharing a horizontal or vertical pixel boundary."""
    pairs = []

    for a, b in (
        (segments[:, :-1], segments[:, 1:]),
        (segments[:-1, :], segments[1:, :]),
    ):
        boundary = a != b
        if boundary.any():
            pairs.append(np.stack([a[boundary], b[boundary]], axis=1))

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
    # Make edges bidirectional.
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return torch.from_numpy(edges.T.copy()).long()


def build_tile_graph(
    record,
    class_values,
    *,
    image_bands=(1, 2, 3),
    image_scale=255.0,
    felz_scale=100,
    felz_sigma=0.5,
    felz_min_size=20,
    min_labeled_fraction=0.25,
):
    """
    Build one PyG Data graph from a tile record.

    class_values must be the SAME global class list for every tile,
    e.g. [1, 3, 7], not classes discovered independently per tile.
    """
    class_to_id = {int(value): i for i, value in enumerate(class_values)}

    with rasterio.open(record["image"]) as src:
        image_chw = src.read(list(image_bands)).astype(np.float32)
        reference = src

        image = np.moveaxis(image_chw, 0, -1)

        if image_scale is not None:
            image = image / float(image_scale)

        mask = read_aligned_raster(record["mask"], reference)[0].astype(np.int64)

        extra_rasters = []
        for key in ("dsm", "dstm"):
            if record.get(key):
                extra = read_aligned_raster(record[key], reference)[0]
                extra_rasters.append(extra)

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask dimensions differ for {record['image']}.")

    # Keep your dataset's ignore values here.
    mask[np.isin(mask, [254, 255])] = -1

    # Felzenszwalb expects an HWC image. Segmentation IDs are relabeled
    # contiguously because downstream node IDs must be 0, ..., N-1.
    raw_segments = felzenszwalb(
        image,
        scale=felz_scale,
        sigma=felz_sigma,
        min_size=felz_min_size,
        channel_axis=-1,
    )
    _, segments = np.unique(raw_segments, return_inverse=True)
    segments = segments.reshape(raw_segments.shape).astype(np.int64)

    x = extract_node_features(image, segments, extra_rasters)
    y = make_node_labels(
        mask,
        segments,
        class_to_id,
        min_labeled_fraction=min_labeled_fraction,
    )
    edge_index = make_adjacency(segments)

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(y),
        num_nodes=x.shape[0],
    )


class TileGraphDataset(Dataset):
    def __init__(
        self,
        records,
        class_values,
        cache_dir,
        graph_kwargs=None,
    ):
        self.records = list(records)
        self.class_values = list(class_values)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.graph_kwargs = graph_kwargs or {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def _cache_path(self, record):
        # Include paths, class mapping, and graph settings in the cache key.
        key_data = {
            "record": {k: str(v) for k, v in record.items()},
            "classes": self.class_values,
            "kwargs": self.graph_kwargs,
        }
        key = hashlib.sha1(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def __getitem__(self, index):
        record = self.records[index]

        if self.cache_dir:
            cache_path = self._cache_path(record)
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu", weights_only=False)

        graph = build_tile_graph(
            record,
            self.class_values,
            **self.graph_kwargs,
        )

        if self.cache_dir:
            cache_path = self._cache_path(record)
            temp_path = cache_path.with_suffix(".tmp")
            torch.save(graph, temp_path)
            temp_path.replace(cache_path)

        return graph


class TileGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_records,
        val_records,
        class_values,
        cache_dir,
        batch_size=4,
        num_workers=4,
        graph_kwargs=None,
    ):
        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.class_values = class_values
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.graph_kwargs = graph_kwargs or {}

    def setup(self, stage=None):
        self.train_dataset = TileGraphDataset(
            self.train_records,
            self.class_values,
            Path(self.cache_dir) / "train",
            self.graph_kwargs,
        )
        self.val_dataset = TileGraphDataset(
            self.val_records,
            self.class_values,
            Path(self.cache_dir) / "val",
            self.graph_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
