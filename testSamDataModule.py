"""Smoke test for the citizen-science support set.

Bypasses Hydra and the DataModule so failures point at the dataset, not the
config. Run from the repo root:

    python smoke_test_citizen.py

Checks, in order:
  1. cutouts.csv exists and resolves
  2. NeophyteDataset builds an index with non-zero counts per class
  3. one tile with a present species returns a well-formed support dict
  4. one tile without a present species returns the empty-tensor fallback
  5. a full DataLoader batch collates without None/tensor mix errors
  6. disable switch works (citizen_cutouts_csv=None)
"""

import os
import sys

import torch
from omegaconf import OmegaConf

from samDataSet import SamNeophyteDataset
from utils.transform_utils import get_transforms

CUTOUTS_CSV = "../data/NeophytesCSCutouts/cutouts.csv"
IMAGE_DIR = "../data/Neophytes/2025"  # adjust to a split that exists on your machine

# minimal class dict matching configs/data/neophytes_names_colors.yaml
CLASSES = {
    0: {"name": "Background",     "color": "#000000", "stats_name": "background"},
    1: {"name": "B. davidii",     "color": "#dce14c", "stats_name": "B_davidii"},
    2: {"name": "R. typhina",     "color": "#b76e44", "stats_name": "R_typhina"},
    3: {"name": "R. japonica",    "color": "#305dc6", "stats_name": "R_japonica"},
    4: {"name": "B. orientalis",  "color": "#e2256a", "stats_name": "B_orientalis"},
    5: {"name": "S. inaequidens", "color": "#50e3c2", "stats_name": "S_inaequidens"},
    6: {"name": "A. altissima",   "color": "#dd60f6", "stats_name": "A_altissima"},
}

VALUE_MAPPING = {"Neophytes": {254: -1, 255: -1}}


def banner(msg):
    print(f"\n=== {msg} ===")


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_cutouts_csv():
    banner("1. cutouts.csv")
    path = os.path.normpath(CUTOUTS_CSV)
    if not os.path.isfile(path):
        fail(f"cutouts.csv not found at {path} -- adjust CUTOUTS_CSV")
    print(f"  found: {path}")

    # verify the path column resolves against repo-root-relative semantics
    import pandas as pd
    df = pd.read_csv(path)
    first = os.path.normpath(str(df.iloc[0]["path"]))
    if os.path.isfile(first):
        print(f"  path column is repo-root-relative: {first}")
        return "root"
    alt = os.path.normpath(os.path.join(os.path.dirname(path), "..", "..", first))
    if os.path.isfile(alt):
        print(f"  path column resolves via ../..: {alt}")
        return "csv_relative"
    fail(f"path column does not resolve either way. First row: {first}")


def build_dataset(**kwargs):
    transforms = get_transforms(
        OmegaConf.create({
            "crop_size_train": 128, "crop_size_val": 128, "ds2": False,
            "augmentations": {}, "normalize": {"mean": [0.485, 0.456, 0.406],
                                               "std":  [0.229, 0.224, 0.225]},
        }),
        [],
    ) if False else None  # smoke test runs without augmentation; dataset handles transform=None

    return SamNeophyteDataset(
        image_dirs=[IMAGE_DIR],
        classes=CLASSES,
        transform=None,
        value_mapping=VALUE_MAPPING,
        ignore_index=-1,
        **kwargs,
    )


def check_index():
    banner("2. citizen index")
    ds = build_dataset(
        citizen_cutouts_csv=CUTOUTS_CSV,
        citizen_k=4,
        citizen_cache=False,
    )
    if not ds.citizen_index:
        fail("citizen_index is empty -- path resolution is wrong or CSV columns "
             "were filtered out entirely. Check the per-class print above.")
    total = sum(len(v) for v in ds.citizen_index.values())
    print(f"  classes indexed: {sorted(ds.citizen_index.keys())}")
    print(f"  total cutouts: {total}")
    for cid in sorted(ds.citizen_index):
        if cid > 6:
            print(f"  WARNING: class_id {cid} > 6, will never match a mapped mask")
    return ds


def find_tile_with_species(ds, k=20):
    """Scan up to k tiles for one that contains a present species."""
    banner("3. tile with present species")
    for i in range(min(k, len(ds))):
        sample = ds[i]
        if sample.get("citizen_examples") is not None and sample["citizen_examples"].shape[0] > 0:
            print(f"  found at index {i}: {sample['name']}")
            return i, sample
    fail(f"none of the first {k} tiles contained a target species. This could be "
         f"legitimate (background-heavy split) or the class_ids in cutouts.csv "
         f"do not match the mask values after value_mapping. Try a different "
         f"IMAGE_DIR (a train split, not val).")


def verify_sample(sample):
    exp = {
        "citizen_examples":   (4, 3),
        "enhanced_examples":  (4, 2, 3),
    }
    for key, prefix in exp.items():
        if key not in sample:
            fail(f"missing key: {key}")
        shape = tuple(sample[key].shape[:len(prefix)])
        if shape != prefix:
            fail(f"{key} shape {tuple(sample[key].shape)} does not start with {prefix}")
        if not torch.isfinite(sample[key]).all():
            fail(f"{key} contains NaN/Inf")
        print(f"  {key}: {tuple(sample[key].shape)}  range "
              f"[{sample[key].min():.3f}, {sample[key].max():.3f}]")

    for key in ("citizen_labels", "enhanced_labels"):
        if key not in sample:
            fail(f"missing key: {key}")
        if sample[key].shape != (4,):
            fail(f"{key} shape {tuple(sample[key].shape)} != (4,)")
        if not all(1 <= int(v) <= 6 for v in sample[key]):
            fail(f"{key} contains out-of-range class IDs: {sample[key].tolist()}")
        print(f"  {key}: {sample[key].tolist()}")

    if len(sample["enhanced_types"]) != 4:
        fail(f"enhanced_types length {len(sample['enhanced_types'])} != 4")


def check_empty_fallback(ds):
    banner("4. tile without species (empty-tensor fallback)")
    for i in range(len(ds)):
        sample = ds[i]
        if sample.get("citizen_examples") is not None and sample["citizen_examples"].shape[0] == 0:
            print(f"  found at index {i}: {sample['name']}")
            if sample["citizen_labels"].shape != (0,):
                fail(f"citizen_labels shape {sample['citizen_labels'].shape} != (0,)")
            return
    print("  no background-only tile in this split; skipping (not a failure)")


def check_dataloader(ds):
    banner("5. DataLoader collate")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    try:
        batch = next(iter(loader))
    except Exception as e:
        fail(f"collate failed: {type(e).__name__}: {e}\n"
             f"This usually means some samples return None and others return "
             f"tensors -- apply the empty-tensor fallback in _build_citizen_support.")
    print(f"  image:  {tuple(batch['image'].shape)}")
    print(f"  mask:   {tuple(batch['mask'].shape)}")
    if "citizen_examples" in batch:
        print(f"  citizen_examples:  {tuple(batch['citizen_examples'].shape)}")
        print(f"  citizen_labels:    {tuple(batch['citizen_labels'].shape)}")
        print(f"  enhanced_examples: {tuple(batch['enhanced_examples'].shape)}")
    else:
        print("  no citizen keys in batch (all 4 tiles were background-only)")
    print("  collate OK")


def check_disable(ds_no_citizen):
    banner("6. disable switch")
    if ds_no_citizen.citizen_index:
        fail("citizen_index is non-empty when citizen_cutouts_csv=None")
    if "citizen_examples" in ds_no_citizen[0]:
        fail("sample contains citizen keys when feature is disabled")
    print("  citizen_cutouts_csv=None leaves samples unchanged")


def main():
    print(f"python: {sys.version.split()[0]}  torch: {torch.__version__}")
    print(f"cwd: {os.getcwd()}")

    check_cutouts_csv()

    ds = check_index()
    idx, sample = find_tile_with_species(ds)
    verify_sample(sample)

    check_empty_fallback(ds)
    check_dataloader(ds)

    ds_off = build_dataset(citizen_cutouts_csv=None, citizen_k=4)
    check_disable(ds_off)

    banner("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()