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

## Binary segmentation without class argmax

```bash
python SAM/SAM_binary.py --data neophytes_split_cv1_test --device cuda
```

`infer_masks` in `SAM_binary.py` returns a binary union and the retained instance
masks/scores. The script evaluates the entire selected test split, merging ground
truth classes 1–6 into foreground and excluding ignored labels. By default it
uses one generic `plant` prompt. Other vegetation is still dataset background,
so detecting it counts as a false positive; this is not an all-vegetation ground truth.
Use `--prompts "Buddleja davidii" "Rhus typhina" "Reynoutria japonica" "Bunias orientalis" "Senecio inaequidens" "Ailanthus altissima"`
to union species-prompt detections instead, still without class competition.

Results go to `SAM/binary_results/<timestamp>_<data>/`: `metrics.json` contains
binary F1, IoU, precision, recall, accuracy, confusion matrix (true rows,
predicted columns), source paths and detection confidences. `masks/` contains
one full-resolution binary PNG per evaluated image (0 background, 255 predicted
foreground). Up to fifty plant-rich images also receive RGB / ground-truth / prediction
comparison previews. Selection cycles across species, prioritizing the largest
labeled areas for each species and excluding background-only tiles. Change the
count with `--previews`. If fewer foreground tiles are available, fewer previews
are saved. This changes preview selection only, not the evaluated dataset. Each manifest
entry records its preview filename when selected. Use `--max-images 10` for a short smoke test. Preview selection here
does not filter for labeled foreground, and the full run includes background-only tiles.
