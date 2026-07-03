#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS=(
  movielens-1m
  amazon-digital_music
  amazon-video_games
  amazon-arts_crafts_and_sewing
)

for dataset in "${DATASETS[@]}"; do
  "${SCRIPT_DIR}/run_exp02_multiseed_dataset.sh" "${dataset}" "$@"
done
