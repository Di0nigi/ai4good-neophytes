#!/usr/bin/env bash
# Run one or all phases of the five-fold PlantNet ViT-B experiment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$PROJECT_ROOT/../.." && pwd)"

MODE="${1:-}"
if [[ "$MODE" != "prepare" && "$MODE" != "baseline" && "$MODE" != "tile-balanced" && "$MODE" != "all" ]]; then
  echo "Usage: $0 {prepare|baseline|tile-balanced|all}" >&2
  exit 2
fi

DATA_ROOT="${NEOPHYTES_DATA_ROOT:-/zfs/ai4good/datasets/Neophytes}"
RUN_ROOT="${PLANTNET_RUN_ROOT:-$REPO_ROOT/results/plantnet-vitb-cv5}"
DEVICE="${PLANTNET_DEVICE:-cuda}"
CROP_SEED="${PLANTNET_CROP_SEED:-31415}"
TRAIN_SEED="${PLANTNET_TRAIN_SEED:-43}"
EPOCHS="${PLANTNET_EPOCHS:-5}"
CROPS_PER_CLASS="${PLANTNET_CROPS_PER_CLASS:-50}"
CLASS_WEIGHT_POWER="${PLANTNET_CLASS_WEIGHT_POWER:-0.5}"
CONDA_SH="${CONDA_SH:-/zfs/ai4good/groups/neophytes/miniforge3/etc/profile.d/conda.sh}"
PLANTNET_CONDA_ENV="${PLANTNET_CONDA_ENV:-/zfs/ai4good/groups/neophytes/mkorch/envs/plantnet}"

# Preserve an already active environment; on Ribeiro, activate the dedicated
# PlantNet environment automatically. This script never selects the SAM env.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda is not active and CONDA_SH was not found: $CONDA_SH" >&2
    echo "Activate the environment from environment.yml or set CONDA_SH." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$PLANTNET_CONDA_ENV"
fi

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
  echo "PLANTNET_DEVICE must be cuda or cpu, got: $DEVICE" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"

wait_for_gpu() {
  [[ "$DEVICE" == "cuda" ]] || return 0
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required for a CUDA run." >&2
    exit 1
  }

  while true; do
    local used_memory process_ids
    used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    process_ids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')"
    if [[ "$used_memory" -lt 1024 && -z "$process_ids" ]]; then
      return 0
    fi
    echo "GPU is occupied (${used_memory} MiB; PIDs=${process_ids:-none}); checking again in 30 seconds."
    sleep 30
  done
}

run_prepare() {
  for fold in cv1 cv2 cv3 cv4 cv5; do
    wait_for_gpu
    echo "Preparing frozen evaluation list and pretrained baseline for $fold"
    python "$SCRIPT_DIR/plantnet_grouped_cv.py" \
      --stage prepare --fold "$fold" \
      --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" \
      --crops-per-class "$CROPS_PER_CLASS" --crop-seed "$CROP_SEED" \
      --device "$DEVICE" \
      2>&1 | tee -a "$RUN_ROOT/logs/prepare_all.log"
  done
  echo "ALL_PREPARES_FINISHED" | tee -a "$RUN_ROOT/logs/prepare_all.log"
}

run_baseline() {
  for fold in cv1 cv2 cv3 cv4 cv5; do
    wait_for_gpu
    echo "Fine-tuning the baseline for $fold"
    python "$SCRIPT_DIR/plantnet_grouped_cv.py" \
      --stage train --fold "$fold" \
      --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" \
      --train-seed "$TRAIN_SEED" --epochs "$EPOCHS" \
      --class-weight-power "$CLASS_WEIGHT_POWER" --device "$DEVICE" \
      2>&1 | tee -a "$RUN_ROOT/logs/baseline_all.log"
  done
  echo "ALL_BASELINE_FOLDS_FINISHED" | tee -a "$RUN_ROOT/logs/baseline_all.log"
}

run_tile_balanced() {
  if [[ "$CLASS_WEIGHT_POWER" == "0.5" ]]; then
    weight_tag="sqrt"
  else
    weight_tag="power${CLASS_WEIGHT_POWER//./p}"
  fi
  baseline_name="finetune_weighted_${weight_tag}_seed${TRAIN_SEED}_strictsplit"

  for fold in cv1 cv2 cv3 cv4 cv5; do
    wait_for_gpu
    echo "Fine-tuning the tile-balanced ablation for $fold"
    python "$SCRIPT_DIR/finetune_vitb_tilebalanced.py" \
      --fold "$fold" --device "$DEVICE" --seed "$TRAIN_SEED" \
      --epochs "$EPOCHS" --class-weight-power "$CLASS_WEIGHT_POWER" \
      --run-root "$RUN_ROOT" --baseline-name "$baseline_name" \
      2>&1 | tee -a "$RUN_ROOT/logs/tilebalanced_all.log"
  done
  echo "ALL_TILEBALANCED_FOLDS_FINISHED" | tee -a "$RUN_ROOT/logs/tilebalanced_all.log"
}

case "$MODE" in
  prepare) run_prepare ;;
  baseline) run_baseline ;;
  tile-balanced) run_tile_balanced ;;
  all)
    run_prepare
    run_baseline
    run_tile_balanced
    ;;
esac
