#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUNNER="${ROOT_DIR}/exp_03_fast_gpu/run_exp03_fast_gpu.py"
BASE_OUT="${ROOT_DIR}/exp_03_fast_gpu/significance_exp03/results"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset-key> [extra runner args...]" >&2
  echo "Dataset keys: movielens-1m, amazon-digital_music, amazon-video_games, amazon-arts_crafts_and_sewing" >&2
  exit 2
fi

DATASET_KEY="$1"
shift
EXTRA_ARGS=("$@")

SEEDS=(42 2024 3407 1234 777)
ABLATION_VARIANTS=(Ours-Plain Ours-NoTemp Ours-NoStruct)
FULL_VARIANTS=(Ours-Full)

COMMON_ARGS=(
  --num-windows 5
  --rounds-per-window 120
  --eval-every 10
  --participation-rate 0.1
  --microbatch-size 512
  --eval-batch-size 2048
  --local-epochs 1
  --candidate-count 99
  --window-mode slice
  --lambda-temp 0.01
  --lambda-struct 0.20
  --edge-ema 0.80
  --feddieg-memory-momentum 0.80
  --no-plot
)

case "${DATASET_KEY}" in
  movielens-1m)
    DATASET_NAME="movielens-1m"
    ;;
  amazon-digital_music)
    DATASET_NAME="amazon-digital music"
    ;;
  amazon-video_games)
    DATASET_NAME="amazon-video games"
    ;;
  amazon-arts_crafts_and_sewing)
    DATASET_NAME="amazon-arts crafts and sewing"
    ;;
  *)
    echo "Unknown dataset key: ${DATASET_KEY}" >&2
    exit 2
    ;;
esac

for seed in "${SEEDS[@]}"; do
  SEED_DIR="${BASE_OUT}/${DATASET_KEY}/seed_${seed}"
  ABLATION_OUT="${SEED_DIR}/ablations"
  FULL_OUT="${SEED_DIR}/full_refresh"
  mkdir -p "${ABLATION_OUT}" "${FULL_OUT}"

  echo "=== Exp03 ${DATASET_KEY} ablations seed=${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${DATASET_NAME}" \
    --seed "${seed}" \
    --variants "${ABLATION_VARIANTS[@]}" \
    --out-dir "${ABLATION_OUT}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "=== Exp03 ${DATASET_KEY} Ours-Full seed=${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${DATASET_NAME}" \
    --seed "${seed}" \
    --variants "${FULL_VARIANTS[@]}" \
    --out-dir "${FULL_OUT}" \
    "${COMMON_ARGS[@]}" \
    --feddieg-boundary-refresh-rate 0.30 \
    --feddieg-boundary-refresh-momentum 0.20 \
    "${EXTRA_ARGS[@]}"

  "${PYTHON_BIN}" "${ROOT_DIR}/exp_03_fast_gpu/significance_exp03/scripts/merge_exp03_seed_outputs.py" \
    --seed-dir "${SEED_DIR}"
done

"${PYTHON_BIN}" "${ROOT_DIR}/exp_03_fast_gpu/significance_exp03/scripts/analyze_exp03_significance.py" \
  --root "${ROOT_DIR}" \
  --datasets "${DATASET_KEY}"
