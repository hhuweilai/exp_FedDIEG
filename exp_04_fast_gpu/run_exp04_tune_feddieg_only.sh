#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUNNER="${ROOT_DIR}/exp_04_fast_gpu/run_exp04_tune_feddieg_only.py"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/exp_04_fast_gpu/final}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/exp_04_fast_gpu/tune_feddieg_only}"

"${PYTHON_BIN}" "${RUNNER}" \
  --base-out "${BASE_OUT}" \
  --work-dir "${WORK_DIR}" \
  --candidate-names balanced proto_strong_light_graph proto_edge_strong stable_conservative \
  --selection-metric auc_ndcg \
  --pretrain-rounds 50 \
  --adapt-rounds 10 \
  --new-client-ratio 0.25 \
  --new-user-strategy low_history \
  --min-new-user-history 5 \
  --d-init 3 \
  --new-interactions-per-round 2 \
  --participation-rate 0.10 \
  --microbatch-size 512 \
  --eval-batch-size 1024 \
  --local-epochs 1 \
  --adapt-epochs 1 \
  --lr 0.001 \
  --adapt-lr 0.0002 \
  "$@"
