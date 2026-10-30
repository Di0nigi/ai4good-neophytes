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
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datasets.neophyte_datamodule import NeophyteDataModule
from datasets.neophyte_dataset import NeophyteDataset
from torch_geometric.loader import DataListLoader as GraphDataLoader

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
        key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
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
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc(value):
    """Convert a 3-channel image tensor/array to HWC."""
    image = _as_numpy(value)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image.shape}.")

    # ToTensorV2 commonly returns CHW; the dataset may also return HWC.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")

    return image.astype(np.float32, copy=False)


def _raster_hw(value):
    """Convert a single-band raster tensor/array to HW."""
    raster = _as_numpy(value)

    if raster.ndim == 3:
        if raster.shape[0] == 1:  # CHW
            raster = raster[0]
        elif raster.shape[-1] == 1:  # HWC
            raster = raster[..., 0]
        else:
            raise ValueError(f"Expected a single-band raster, got {raster.shape}.")

    if raster.ndim != 2:
        raise ValueError(f"Expected a 2-D raster, got shape {raster.shape}.")

    return raster.astype(np.float32, copy=False)


class NeophyteGraphDataset(NeophyteDataset):
    """
    Uses NeophyteDataset's file discovery, loading, mapping, and transforms,
    then converts each transformed tile into a PyG graph.
    """

    def __init__(
        self,
        *args,
        graph_class_values=None,
        graph_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # These must match the pixel values in the mask returned by
        # NeophyteDataset, after its value_mapping has been applied.
        self.graph_class_values = (
            list(range(self.num_classes))
            if graph_class_values is None
            else list(graph_class_values)
        )
        self.graph_kwargs = graph_kwargs or {}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = _image_hwc(sample["image"])
        mask = _as_numpy(sample["mask"]).astype(np.int64, copy=False)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask dimensions differ for {sample.get('name', idx)}: "
                f"{image.shape[:2]} vs {mask.shape}."
            )

        # Add whichever optional raster channels the parent returned.
        extra_rasters = []
        for key in ("dsm", "dtm", "ndsm"):
            if key in sample:
                raster = _raster_hw(sample[key])
                if raster.shape != mask.shape:
                    raise ValueError(
                        f"{key} shape {raster.shape} does not match mask "
                        f"shape {mask.shape} for {sample.get('name', idx)}."
                    )
                extra_rasters.append(raster)

        segments = felzenszwalb(
            image,
            scale=self.graph_kwargs.get("felz_scale", 100),
            sigma=self.graph_kwargs.get("felz_sigma", 0.5),
            min_size=self.graph_kwargs.get("felz_min_size", 20),
            channel_axis=-1,
        )

        # Ensure node IDs are exactly 0 ... N-1.
        _, segments = np.unique(segments, return_inverse=True)
        segments = segments.reshape(mask.shape).astype(np.int64)

        x = extract_node_features(image, segments, extra_rasters)
        y = make_node_labels(
            mask,
            segments,
            {int(v): i for i, v in enumerate(self.graph_class_values)},
            min_labeled_fraction=self.graph_kwargs.get("min_labeled_fraction", 0.25),
        )
        edge_index = make_adjacency(segments)

        return Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            y=torch.from_numpy(y),
            num_nodes=x.shape[0],
        )


class NeophyteGraphDataModule(NeophyteDataModule):
    def __init__(self, *args, graph_class_values=None, graph_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_class_values = graph_class_values
        self.graph_kwargs = graph_kwargs or {}

    def _make_graph_dataset(self, image_dirs, transform, training=False):
        kwargs = dict(
            transform=transform,
            dsm_transform=self.dsm_transform,
            ndsm_transform=self.ndsm_transform,
            dataset_stats=self.dataset_stats,
            value_mapping=self.value_mapping,
            ignore_index=self.ignore_index,
            load_auxiliary=self.load_auxiliary,
            load_dsm=self.load_dsm,
            load_dtm=self.load_dtm,
            load_ndsm=self.load_ndsm,
            load_into_ram=self.load_into_ram,
            preload_workers=self.num_workers,
            graph_class_values=self.graph_class_values,
            graph_kwargs=self.graph_kwargs,
        )

        if training:
            kwargs.update(
                weighted_sampling=self.weighted_sampling,
                cap_quantile=self.cap_quantile,
                lift_quantile=self.lift_quantile,
                bg_fraction=self.bg_fraction,
                min_strat_images=self.min_strat_images,
                few_shot_k=self.few_shot_k,
                few_shot_mode=self.few_shot_mode,
                few_shot_bg_k=self.few_shot_bg_k,
                max_nodata_frac=self.max_nodata_frac,
            )

        return NeophyteGraphDataset(image_dirs, self.classes, **kwargs)

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = self._make_graph_dataset(
                self.image_dirs.train, self.transform_train, training=True
            )
            self.val_dataset = self._make_graph_dataset(
                self.image_dirs.val, self.transform_val
            )

            if self.weighted_sampling:
                self.sampler = WeightedRandomSampler(
                    self.train_dataset.sample_weights,
                    num_samples=len(self.train_dataset),
                    replacement=True,
                )

        if stage in ("test", None):
            self.test_dataset = self._make_graph_dataset(
                self.image_dirs.test, self.transform_test
            )

    def train_dataloader(self):
        kwargs = {
            'batch_size': self.batch_size_train,
            'num_workers': self.num_workers,
            'persistent_workers': self.num_workers > 0,
            'pin_memory': False,
        }
        if self.num_workers > 0:
            kwargs['prefetch_factor'] = 2

        if self.weighted_sampling:
            return 
            GraphDataLoader(
                self.train_dataset,
                sampler=self.sampler,
                **kwargs
            )
        return GraphDataLoader(
            self.train_dataset,
            shuffle=True,
            **kwargs
        )

    def val_dataloader(self):
        return GraphDataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return GraphDataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )
