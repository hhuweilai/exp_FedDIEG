#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUNNER="${ROOT_DIR}/exp_01_fast_gpu/run_exp01_fast_gpu.py"
MERGER="${ROOT_DIR}/exp_01_fast_gpu/significance_exp01/scripts/replace_method_rows.py"
ANALYZER="${ROOT_DIR}/exp_01_fast_gpu/significance_exp01/scripts/analyze_exp01_significance.py"
BASE_OUT="${ROOT_DIR}/exp_01_fast_gpu/significance_exp01/results"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset-key> [extra FedDIEG runner args...]" >&2
  echo "Dataset keys: movielens-1m, amazon-digital_music, amazon-video_games, amazon-arts_crafts_and_sewing" >&2
  exit 2
fi

DATASET_KEY="$1"
shift

case "${DATASET_KEY}" in
  movielens-1m)
    DATASET_NAME="movielens-1m"
    FEDDIEG_ARGS=(
      --feddieg-lr 0.0012
      --feddieg-graph-mix 0.02
      --feddieg-final-loss-weight 0
      --feddieg-short-score-mix 0
    )
    ;;
  amazon-digital_music)
    DATASET_NAME="amazon-digital music"
    FEDDIEG_ARGS=()
    ;;
  amazon-video_games)
    DATASET_NAME="amazon-video games"
    FEDDIEG_ARGS=()
    ;;
  amazon-arts_crafts_and_sewing)
    DATASET_NAME="amazon-arts crafts and sewing"
    FEDDIEG_ARGS=()
    ;;
  *)
    echo "Unknown dataset key: ${DATASET_KEY}" >&2
    exit 2
    ;;
esac

SEEDS=(42 2024 3407 1234 777)
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --rounds 600
  --eval-every 10
  --participation-rate 0.10
  --microbatch-size 512
  --eval-batch-size 1024
  --local-epochs 1
  --lr 0.001
)

for seed in "${SEEDS[@]}"; do
  OUT_DIR="${BASE_OUT}/${DATASET_KEY}/seed_${seed}"
  RERUN_DIR="${OUT_DIR}/feddieg_only_${RUN_TAG}"
  mkdir -p "${OUT_DIR}" "${RERUN_DIR}"

  echo "=== Exp01 ${DATASET_KEY} FedDIEG-only seed=${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${DATASET_NAME}" \
    --seed "${seed}" \
    --methods FedDIEG \
    --out-dir "${RERUN_DIR}" \
    "${COMMON_ARGS[@]}" \
    "${FEDDIEG_ARGS[@]}" \
    "$@"

  if [[ -f "${OUT_DIR}/summary.csv" ]]; then
    cp "${OUT_DIR}/summary.csv" "${OUT_DIR}/summary.csv.bak_feddieg_only_${RUN_TAG}"
    "${PYTHON_BIN}" "${MERGER}" \
      --target "${OUT_DIR}/summary.csv" \
      --source "${RERUN_DIR}/summary.csv" \
      --method FedDIEG
  else
    cp "${RERUN_DIR}/summary.csv" "${OUT_DIR}/summary.csv"
  fi

  if [[ -f "${OUT_DIR}/round_metrics.csv" ]]; then
    cp "${OUT_DIR}/round_metrics.csv" "${OUT_DIR}/round_metrics.csv.bak_feddieg_only_${RUN_TAG}"
    "${PYTHON_BIN}" "${MERGER}" \
      --target "${OUT_DIR}/round_metrics.csv" \
      --source "${RERUN_DIR}/round_metrics.csv" \
      --method FedDIEG
  else
    cp "${RERUN_DIR}/round_metrics.csv" "${OUT_DIR}/round_metrics.csv"
  fi
done

"${PYTHON_BIN}" "${ANALYZER}" \
  --root "${ROOT_DIR}" \
  --datasets "${DATASET_KEY}"
