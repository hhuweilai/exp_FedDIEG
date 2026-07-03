#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUNNER="${ROOT_DIR}/exp_02_fast_gpu/run_exp02_fast_gpu.py"
BASE_OUT="${ROOT_DIR}/exp_02_fast_gpu/significance_exp02/results"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset-key> [extra runner args...]" >&2
  echo "Dataset keys: movielens-1m, amazon-digital_music, amazon-video_games, amazon-arts_crafts_and_sewing" >&2
  exit 2
fi

DATASET_KEY="$1"
shift
EXTRA_BASELINE_ARGS=("$@")

BASELINE_METHODS=(FedMF FedSASRec GPFedRec FCUCR F3CRec DGFedRS)
FEDDIEG_METHODS=(FedDIEG)
SEEDS=(42 2024 3407 1234 777)

COMMON_ARGS=(
  --num-windows 5
  --rounds-per-window 120
  --eval-every 10
  --participation-rate 0.10
  --microbatch-size 512
  --eval-batch-size 2048
  --local-epochs 1
  --candidate-count 99
  --window-mode slice
  --no-plot
)

BASELINE_ARGS=(
  --lr 0.001
)

case "${DATASET_KEY}" in
  movielens-1m)
    DATASET_NAME="movielens-1m"
    BASELINE_ARGS+=(--clients-per-round 604)
    FEDDIEG_ARGS=(
      --lr 0.0012
      --clients-per-round 604
      --feddieg-neighbors 10
      --feddieg-graph-mix 0.02
      --feddieg-warmup-rounds 100
      --feddieg-neighbor-temp 0.05
      --feddieg-min-sim 0.2
      --feddieg-tail-power 4.0
      --feddieg-final-loss-weight 0.0
      --feddieg-memory-momentum 0.8
      --feddieg-short-score-mix 0.0
    )
    ;;
  amazon-digital_music)
    DATASET_NAME="amazon-digital music"
    BASELINE_ARGS+=(--clients-per-round 1238)
    FEDDIEG_ARGS=(
      --lr 0.0012
      --feddieg-neighbors 10
      --feddieg-graph-mix 0.03
      --feddieg-warmup-rounds 80
      --feddieg-neighbor-temp 0.07
      --feddieg-min-sim 0.2
      --feddieg-tail-power 4.0
      --feddieg-final-loss-weight 0.05
      --feddieg-memory-momentum 0.8
      --feddieg-short-score-mix 0.0
    )
    ;;
  amazon-video_games)
    DATASET_NAME="amazon-video games"
    BASELINE_ARGS+=(--clients-per-round 5062)
    FEDDIEG_ARGS=(
      --lr 0.0016
      --feddieg-neighbors 10
      --feddieg-graph-mix 0.02
      --feddieg-warmup-rounds 80
      --feddieg-neighbor-temp 0.07
      --feddieg-min-sim 0.2
      --feddieg-tail-power 4.0
      --feddieg-final-loss-weight 0.0
      --feddieg-memory-momentum 0.8
      --feddieg-short-score-mix 0.0
    )
    ;;
  amazon-arts_crafts_and_sewing)
    DATASET_NAME="amazon-arts crafts and sewing"
    BASELINE_ARGS+=(--clients-per-round 4534)
    FEDDIEG_ARGS=(
      --lr 0.0022
      --feddieg-neighbors 8
      --feddieg-graph-mix 0.015
      --feddieg-warmup-rounds 80
      --feddieg-neighbor-temp 0.07
      --feddieg-min-sim 0.2
      --feddieg-tail-power 4.0
      --feddieg-final-loss-weight 0.02
      --feddieg-memory-momentum 0.90
      --feddieg-short-score-mix 0.0
    )
    ;;
  *)
    echo "Unknown dataset key: ${DATASET_KEY}" >&2
    exit 2
    ;;
esac

for seed in "${SEEDS[@]}"; do
  SEED_DIR="${BASE_OUT}/${DATASET_KEY}/seed_${seed}"
  BASELINE_OUT="${SEED_DIR}/baselines"
  FEDDIEG_OUT="${SEED_DIR}/feddieg"
  mkdir -p "${BASELINE_OUT}" "${FEDDIEG_OUT}"

  echo "=== Exp02 ${DATASET_KEY} baselines seed=${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${DATASET_NAME}" \
    --seed "${seed}" \
    --methods "${BASELINE_METHODS[@]}" \
    --out-dir "${BASELINE_OUT}" \
    "${COMMON_ARGS[@]}" \
    "${BASELINE_ARGS[@]}" \
    "${EXTRA_BASELINE_ARGS[@]}"

  echo "=== Exp02 ${DATASET_KEY} FedDIEG seed=${seed} ==="
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${DATASET_NAME}" \
    --seed "${seed}" \
    --methods "${FEDDIEG_METHODS[@]}" \
    --out-dir "${FEDDIEG_OUT}" \
    "${COMMON_ARGS[@]}" \
    "${FEDDIEG_ARGS[@]}"

  "${PYTHON_BIN}" "${ROOT_DIR}/exp_02_fast_gpu/significance_exp02/scripts/merge_exp02_seed_outputs.py" \
    --seed-dir "${SEED_DIR}"
done

"${PYTHON_BIN}" "${ROOT_DIR}/exp_02_fast_gpu/significance_exp02/scripts/analyze_exp02_significance.py" \
  --root "${ROOT_DIR}" \
  --datasets "${DATASET_KEY}"
