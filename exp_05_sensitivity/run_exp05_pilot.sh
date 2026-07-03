#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" exp_05_sensitivity/run_exp05_sensitivity.py \
  --python-bin "${PYTHON_BIN}" \
  --datasets "movielens-1m" "amazon-video games" \
  --execute \
  "$@"
