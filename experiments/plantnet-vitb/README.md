# PlantNet ViT-B crop-classification experiment

This folder contains the PlantNet ViT-B experiment code. It is a separate
crop-classification experiment within the Neophytes segmentation repository.
The model assigns one class to each image crop; it does **not** produce a
pixel-level species mask.

## Model

The exact Hugging Face checkpoint is
[`PlantNet/vit_base_patch14_reg4_dinov2_lvd142m.pc24_all`](https://huggingface.co/PlantNet/vit_base_patch14_reg4_dinov2_lvd142m.pc24_all),
a PlantNet ViT-Base classifier with a DINOv2 ViT-B/14 backbone.

- **Pretrained baseline:** use the published checkpoint unchanged. Its
  7,806-way species output is mapped to the six project species; every other
  PlantNet class maps to `other/background`.
- **Fine-tuned baseline:** start from the same checkpoint, replace the output
  layer with seven project classes, then fine-tune on annotated Neophytes crops.
- **Tile-balanced ablation:** repeat the same fine-tuning with the same crop
  manifests, but reduce the total loss contribution from source tiles that
  provide many crops of the same class.

## Evaluation protocol

Each CV fold holds out its site group as test data. The next fold in the
protocol supplies validation sites; the remaining three folds supply training
sites. The same saved held-out crop manifest is used for the unchanged and
fine-tuned predictions, so the comparison is paired on the same crops.

| Test fold | Held-out sites | Validation fold | Training folds |
|---|---|---|---|
| CV1 | Arlesheim_1, Dietikon_1, Basel_3, Aarau_1 | CV2 | CV3, CV4, CV5 |
| CV2 | Aarau_3, Olten_1, Aarau_2, Ardon_3 | CV3 | CV1, CV4, CV5 |
| CV3 | Dornach_1, Rivera_1, Basel_2, Waltenschwil_1 | CV4 | CV1, CV2, CV5 |
| CV4 | Sion_1, Rivera_2, Dornach_2, Ardon_1 | CV5 | CV1, CV2, CV3 |
| CV5 | Chamoson_1, Ardon_2, Basel_4, Rueti_1 | CV1 | CV2, CV3, CV4 |

The crop sampler requests up to 50 crops per class per test fold. It selects
across held-out site partitions and avoids selecting the same source image
twice for one class. The number actually available varies by fold. The primary
summary excludes CV3 because those sites had already been used in an earlier
exploratory pilot; CV3 results remain in the saved output for transparency.

The label for each crop comes from its aligned pixel annotation. A crop is
accepted as a species when that species covers at least 2% of valid pixels and
at least 80% of annotated foreground. It is accepted as background when at
least 90% of valid pixels are background and foreground covers no more than
2%. Mixed or sparse crops are omitted. The selection rules and thresholds are
saved with each evaluation run.

The training recipe used for the reported runs was five epochs, AdamW at
2e-5, square-root class-frequency weights, random horizontal and vertical
flips, and best-checkpoint selection by validation macro-F1. The tile-balanced
variant reused the exact train, validation, and test manifests from the
baseline. Each run writes manifest hashes and a split audit.

## Data and privacy

The course Neophytes imagery and annotations are **not included** here and
must not be added to GitHub. Set `NEOPHYTES_DATA_ROOT` to the local or cluster
dataset root. Crop manifests, predictions, checkpoints, logs, and W&amp;B
files are experiment outputs and are ignored by this folder's `.gitignore`.

## Environment

On the Ribeiro cluster, activate the dedicated PlantNet environment (do not
use the separate SAM SegHead environment). To create an equivalent environment
where Conda is available:

```bash
conda env create -f experiments/plantnet-vitb/environment.yml
conda activate plantnet-vitb
```

The training scripts can log to the existing Weights &amp; Biases project. Sign
in in the terminal with `wandb login`; never put an API key in a script or
notebook. Set `WANDB_ENTITY` and `WANDB_PROJECT` if your team uses a different
destination.

## Run

From the repository root, set the data location and a new output directory:

```bash
export NEOPHYTES_DATA_ROOT=/zfs/ai4good/datasets/Neophytes
export PLANTNET_RUN_ROOT="$PWD/results/plantnet-vitb-cv5-$(date +%Y%m%d)"
```

Prepare the frozen evaluation lists and log the unchanged pretrained ViT-B
baseline, then run fine-tuning and the tile-balanced comparison:

```bash
bash experiments/plantnet-vitb/scripts/run_grouped_cv.sh prepare
bash experiments/plantnet-vitb/scripts/run_grouped_cv.sh baseline
bash experiments/plantnet-vitb/scripts/run_grouped_cv.sh tile-balanced
```

Or run all three phases in order with:

```bash
bash experiments/plantnet-vitb/scripts/run_grouped_cv.sh all
```

The launcher waits for the GPU to become available between folds. Set
`PLANTNET_DEVICE=cpu` to run without a GPU. Each fine-tuning output directory
must be new; use a new `PLANTNET_RUN_ROOT` for a fresh experiment. To compare a
new method fairly, reuse the frozen evaluation manifests from the chosen run
root and do not select or tune on its test labels.

## Code map

- `scripts/evaluate_pretrained_vitb.py` — reads image/annotation pairs, builds
  deterministic 512 × 512 crops, applies the fixed crop-label rules, and
  evaluates the original PlantNet classifier.
- `scripts/finetune_vitb_weighted.py` — trains the seven-class baseline,
  audits site/tile separation, selects the checkpoint using validation data,
  and evaluates on the frozen test manifest.
- `scripts/finetune_vitb_tilebalanced.py` — runs the tile-balanced ablation
  using the baseline's saved manifests.
- `scripts/plantnet_grouped_cv.py` — coordinates one fold, persists protocol
  information, and logs fold metrics to W&amp;B.
- `scripts/run_grouped_cv.sh` — runs preparation, baseline training, or the
  tile-balanced phase across folds.

All scores are crop-classification scores, not segmentation metrics. Review
per-class precision/recall and confusion matrices alongside accuracy and
macro-F1, especially because the test crops are class-balanced rather than a
representative sample of landscape prevalence.
