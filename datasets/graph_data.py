from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from skimage.measure import regionprops
from skimage.segmentation import felzenszwalb
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Batch, Data

from datasets.neophyte_datamodule import NeophyteDataModule
from datasets.neophyte_dataset import NeophyteDataset

""" 
Custom data modules adapted from the original data modules from the repository. 
Key extensions:
- Graph format 
- Node feature extraction
- Graph generation (either kdtree with radius or neighbor edge creation)
- Additional function to sample images from dataset
"""


GRAPH_CACHE_VERSION = "3"


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
    num_nodes = int(segments.max()) + 1
    num_classes = len(class_to_id)
    max_mask = int(mask.max()) if mask.size else -1
    max_key = max(class_to_id) if class_to_id else -1
    lut = np.full(max(max_mask, max_key) + 1, -1, dtype=np.int64)
    for v, i in class_to_id.items():
        lut[v] = i

    safe = np.clip(mask, 0, lut.shape[0] - 1)
    mapped = lut[safe]
    mapped[(mask < 0) | (mask >= lut.shape[0])] = -1  # ignore + unknown values

    seg = segments.ravel()
    ok = mapped.ravel() >= 0
    counts = np.bincount(
        seg[ok] * num_classes + mapped.ravel()[ok],
        minlength=num_nodes * num_classes,
    ).reshape(num_nodes, num_classes)
    total = np.bincount(seg, minlength=num_nodes)
    labeled = counts.sum(axis=1)
    frac = labeled / np.maximum(total, 1)
    labels = np.where(frac >= min_labeled_fraction, counts.argmax(axis=1), -1)
    return labels.astype(np.int64)


def make_node_phenology(mask, phenology, segments, num_classes, num_stages: int = 3):
    """One-hot of the dominant phenology stage per node over its target-class pixels.

    mask: (H, W) class ids, 0 = background
    phenology: (H, W) stage ids 1..num_stages (band 3 of the mask raster)
    """
    num_nodes = int(segments.max()) + 1
    one_hot_encoded_phenology = np.zeros((num_nodes, num_stages), dtype=np.float32)

    found_phenology = (
        (mask > 0) & (mask < num_classes) & (phenology > 0) & (phenology <= num_stages)
    )
    if not found_phenology.any():
        return one_hot_encoded_phenology

    counts = np.bincount(
        segments[found_phenology] * num_stages + (phenology[found_phenology] - 1),
        minlength=num_nodes * num_stages,
    ).reshape(num_nodes, num_stages)
    dominant = counts.argmax(axis=1)
    has_stage = counts.sum(axis=1) > 0
    one_hot_encoded_phenology[has_stage, dominant[has_stage]] = 1.0
    return one_hot_encoded_phenology


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


def make_adjacency_kdtree(segments, radius=5):
    """Connect segments whose centroids are within `radius` pixels (KDTree)."""
    props = regionprops(segments + 1)
    if not props:
        return torch.empty((2, 0), dtype=torch.long)

    centroids = np.zeros((len(props), 2), dtype=np.float64)
    for prop in props:
        centroids[prop.label - 1] = prop.centroid

    tree = KDTree(centroids)
    neighbors = tree.query_radius(centroids, r=radius)

    pairs = [
        (i, j)
        for i, neighbors_list in enumerate(neighbors)
        for j in neighbors_list
        if i != j
    ]
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    pairs = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
    edges = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return torch.from_numpy(edges.T.copy()).long()


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc(value):
    """Convert a 3-channel image tensor/array to HWC."""
    image = _as_numpy(value)

    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image.shape}.")
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
        graph_cache_dir=None,
        graph_kdtree_radius: int = 0,
        graph_use_phenology: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.graph_class_values = (
            list(range(self.num_classes))
            if graph_class_values is None
            else list(graph_class_values)
        )
        self.graph_kwargs = graph_kwargs or {}
        self.graph_kdtree_radius = graph_kdtree_radius
        self.graph_use_phenology = graph_use_phenology
        self.graph_cache_dir = Path(graph_cache_dir) if graph_cache_dir else None
        if self.graph_cache_dir:
            self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, idx):
        files = [self.image_files[idx], self.mask_files[idx]]
        if self.load_dsm:
            files.append(self.dsm_files[idx])
        if self.load_dtm:
            files.append(self.dtm_files[idx])
        file_info = []
        for f in files:
            st = os.stat(f)
            file_info.append(
                {"path": str(f), "mtime_ns": st.st_mtime_ns, "size": st.st_size}
            )
        key_data = {
            "version": GRAPH_CACHE_VERSION,
            "files": file_info,
            "classes": self.graph_class_values,
            "graph_kwargs": self.graph_kwargs,
            "graph_kdtree_radius": self.graph_kdtree_radius,
            "value_mapping": str(self.value_mapping),
            "ignore_index": self.ignore_index,
            "load_auxiliary": self.load_auxiliary,
            "load_dsm": self.load_dsm,
            "load_dtm": self.load_dtm,
            "load_ndsm": self.load_ndsm,
            "transform": repr(self.transform),
            "dsm_transform": repr(self.dsm_transform),
            "ndsm_transform": repr(self.ndsm_transform),
        }
        if self.graph_use_phenology:
            key_data["graph_use_phenology"] = True
        key = hashlib.sha1(
            json.dumps(key_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        return self.graph_cache_dir / f"{key}.pt"

    def __getitem__(self, idx):
        if self.graph_cache_dir:
            cache_path = self._cache_path(idx)
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu", weights_only=False)

        graph = self._build_graph(idx)

        if self.graph_cache_dir:
            temp_path = cache_path.with_suffix(f".tmp{os.getpid()}")
            torch.save(graph, temp_path)
            temp_path.replace(cache_path)

        return graph

    def _unpack_sample(self, sample, idx):
        """Image (HWC), mask (HW) and extra rasters from a parent sample."""
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

        phenology = None
        if self.graph_use_phenology:
            if "phenology" not in sample:
                raise ValueError(
                    "graph_use_phenology needs the phenology band: set "
                    "data.load_auxiliary=True."
                )
            phenology = _raster_hw(sample["phenology"]).astype(np.int64, copy=False)
            if phenology.shape != mask.shape:
                raise ValueError(
                    f"phenology shape {phenology.shape} does not match mask "
                    f"shape {mask.shape} for {sample.get('name', idx)}."
                )

        return image, mask, extra_rasters, phenology

    def _make_segments(self, image, shape):
        segments = felzenszwalb(
            image,
            scale=self.graph_kwargs.get("felz_scale", 100),
            sigma=self.graph_kwargs.get("felz_sigma", 0.5),
            min_size=self.graph_kwargs.get("felz_min_size", 20),
            channel_axis=-1,
        )

        # Ensure node IDs are exactly 0 ... N-1.
        _, segments = np.unique(segments, return_inverse=True)
        return segments.reshape(shape).astype(np.int64)

    def _build_graph(self, idx):
        sample = super().__getitem__(idx)
        image, mask, extra_rasters, phenology = self._unpack_sample(sample, idx)
        segments = self._make_segments(image, mask.shape)

        x = extract_node_features(image, segments, extra_rasters)
        if phenology is not None:
            x = np.concatenate(
                [x, make_node_phenology(mask, phenology, segments, self.num_classes)],
                axis=1,
            )
        y = make_node_labels(
            mask,
            segments,
            {int(v): i for i, v in enumerate(self.graph_class_values)},
            min_labeled_fraction=self.graph_kwargs.get("min_labeled_fraction", 0.25),
        )
        if self.graph_kdtree_radius > 0:
            edge_index = make_adjacency_kdtree(segments, self.graph_kdtree_radius)
        else:
            edge_index = make_adjacency(segments)
        return Data(
            x=torch.from_numpy(x),
            edge_index=edge_index,
            y=torch.from_numpy(y),
            num_nodes=x.shape[0],
        )

    def get_visualization_sample(self, idx):
        """One tile as image, mask, Felzenszwalb segments and its PyG graph.

        For drawing qualitative figures: unlike ``__getitem__`` it also
        exposes the pixels behind the nodes. The segmentation is rebuilt
        with the same parameters as ``_build_graph``, so its node IDs line
        up with the (possibly cached) graph.
        """
        sample = super().__getitem__(idx)
        image, mask, _, _ = self._unpack_sample(sample, idx)
        segments = self._make_segments(image, mask.shape)

        graph = self[idx]  # uses the graph cache when present

        if graph.num_nodes != int(segments.max()) + 1:
            raise RuntimeError(
                f"Graph for {sample.get('name', idx)} has {graph.num_nodes} nodes "
                f"but the current segmentation yields {int(segments.max()) + 1} "
                "segments. Delete the graph cache and rerun."
            )

        return {
            "image": image,
            "mask": mask,
            "segments": segments,
            "graph": graph,
            "name": sample.get("name", str(idx)),
        }


class NeophyteGraphDataModule(NeophyteDataModule):
    def __init__(
        self,
        *args,
        graph_class_values=None,
        graph_kwargs=None,
        graph_cache_dir=None,
        graph_cache_train=False,
        graph_kdtree_radius: int = 0,
        graph_use_phenology: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.graph_class_values = graph_class_values
        self.graph_kwargs = graph_kwargs or {}
        self.graph_cache_dir = Path(graph_cache_dir) if graph_cache_dir else None
        self.graph_cache_train = graph_cache_train
        self.graph_kdtree_radius = graph_kdtree_radius
        self.graph_use_phenology = graph_use_phenology

    def _make_graph_dataset(
        self, image_dirs, transform, training=False, cache_dir=None
    ):
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
            graph_cache_dir=cache_dir,
            graph_kdtree_radius=self.graph_kdtree_radius,
            graph_use_phenology=self.graph_use_phenology,
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
            train_cache = None
            if self.graph_cache_dir and self.graph_cache_train:
                train_cache = self.graph_cache_dir / "train"
            self.train_dataset = self._make_graph_dataset(
                self.image_dirs.train,
                self.transform_train,
                training=True,
                cache_dir=train_cache,
            )
            self.val_dataset = self._make_graph_dataset(
                self.image_dirs.val,
                self.transform_val,
                cache_dir=self.graph_cache_dir / "val"
                if self.graph_cache_dir
                else None,
            )

            if self.weighted_sampling:
                self.sampler = WeightedRandomSampler(
                    self.train_dataset.sample_weights,
                    num_samples=len(self.train_dataset),
                    replacement=True,
                )

        if stage in ("test", None):
            self.test_dataset = self._make_graph_dataset(
                self.image_dirs.test,
                self.transform_test,
                cache_dir=self.graph_cache_dir / "test"
                if self.graph_cache_dir
                else None,
            )

    def _get_loader_kwargs(self, shuffle=False, sampler=None, batch_size=None):
        """Get DataLoader kwargs with proper multiprocessing support."""
        bs = batch_size if batch_size is not None else self.batch_size_train
        kwargs = {
            "batch_size": bs,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": False,
            "shuffle": shuffle and sampler is None,
        }

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
            import sys

            kwargs["multiprocessing_context"] = (
                "spawn" if sys.platform != "linux" else "fork"
            )

        if sampler is not None:
            kwargs["sampler"] = sampler

        return kwargs

    def _collate_graphs(self, batch):
        """Collate function that batches PyG Data objects into a Batch."""
        # Batch can handle variable-sized graphs
        return Batch.from_data_list(batch)

    def train_dataloader(self):
        if self.weighted_sampling:
            return TorchDataLoader(
                self.train_dataset,
                collate_fn=self._collate_graphs,
                **self._get_loader_kwargs(
                    sampler=self.sampler, batch_size=self.batch_size_train
                ),
            )
        return TorchDataLoader(
            self.train_dataset,
            collate_fn=self._collate_graphs,
            **self._get_loader_kwargs(shuffle=True, batch_size=self.batch_size_train),
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            collate_fn=self._collate_graphs,
            **self._get_loader_kwargs(batch_size=self.batch_size_val),
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            collate_fn=self._collate_graphs,
            **self._get_loader_kwargs(batch_size=self.batch_size_test),
        )
