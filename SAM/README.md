# SAM3 baseline on the GPU node

Use the project's environment on a node with an allocated GPU. From the repository root:

```bash
conda activate ai4good-neophytes
python -m pip install --upgrade transformers
hf auth login
python SAM/SAM_baseline.py --data neophytes_split_cv1_test --device cuda
```

This uses Transformers' SAM3 implementation; Meta's `sam3` package is unnecessary.
The environment must already contain the project dependencies from `environment.yml`.
Authenticate with the Hugging Face account granted access to `facebook/sam3`.
`--checkpoint` also accepts a local Transformers model directory for offline nodes.

## Dataset layout

The split YAML files and `stats_imagewise.csv` both use paths starting with
`../data/Neophytes/`. Keep that layout on the node:

```text
parent/
├── ai4good-neophytes/
└── data/Neophytes/
    ├── stats_imagewise.csv
    ├── 2024/<flight>/<split>/images/*.tif
    ├── 2024/<flight>/<split>/masks_prep/*.tif
    └── 2025/<flight>/<split>/...
```

If the dataset is mounted elsewhere, create a symlink from the repository root
(replace the target with the actual cluster path; only needed if the link is absent):

```bash
mkdir -p ../data
ln -s /actual/cluster/path/Neophytes ../data/Neophytes
```

This preserves the CSV's image-path matching. The script anchors relative paths
to the repository root, including when launched from another working directory.

`neophytes_split_cv1_test` composes `data.yaml`, `neophytes_names_colors.yaml`,
`split_cv1.yaml`, and `transforms_orig.yaml`. Its test set is the `/train` tiles of
Aarau_1 (two flights), Arlesheim_1, Basel_3, and Dietikon_1 (two flights), as defined
by the existing held-out-site protocol. Use `neophytes_split_cv2_test` through
`neophytes_split_cv5_test` for the other folds.

The baseline loads RGB and semantic labels, keeps whole-tile test geometry,
lets SAM3 normalize RGB, and ignores label values 254/255 through the configured
mapping to -1. Elevation and auxiliary labels are not passed to SAM3.
It prints F1 and recall for all seven classes, overall pixel accuracy, and macro
recall. Species prompts use full scientific names; Background remains a prompt.
Use `--original-prompts` to reproduce the abbreviated prompts.

Before a full run, use `--diagnostic` to evaluate up to ten foreground-containing
tiles and print detections and pixel counts. `--threshold 0.1` can test whether
confidence filtering causes empty predictions; the default remains 0.5.
Choose prompt/threshold settings on validation data before reporting test results.

Each completed run saves `metrics.json` in `SAM/results/<timestamp>_<data>/`,
containing full-precision metrics, per-class support, the confusion matrix
(true classes in rows, predictions in columns), and run settings. Set
`--output-dir /path/to/results` to change the results root. Pixel predictions
are not saved.
