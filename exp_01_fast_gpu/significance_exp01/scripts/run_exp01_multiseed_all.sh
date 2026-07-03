#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DATASETS=(
  movielens-1m
  amazon-digital_music
  amazon-video_games
  amazon-arts_crafts_and_sewing
)

for dataset in "${DATASETS[@]}"; do
  "${SCRIPT_DIR}/run_exp01_multiseed_dataset.sh" "${dataset}" "$@"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_exp01_significance.py" --root "${ROOT_DIR}"
