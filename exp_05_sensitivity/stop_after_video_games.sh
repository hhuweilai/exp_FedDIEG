#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${ROOT_DIR}/exp_05_sensitivity/final/stop_after_video_games.log"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "[$(date '+%F %T')] Watching amazon-video games completion..." >>"${LOG_FILE}"

while true; do
  completed="$(find exp_05_sensitivity/final/amazon_video_games -name exp03_summary.csv 2>/dev/null | wc -l | tr -d ' ')"
  echo "[$(date '+%F %T')] amazon-video games completed ${completed}/16" >>"${LOG_FILE}"

  if [[ "${completed}" -ge 16 ]]; then
    echo "[$(date '+%F %T')] amazon-video games complete. Stopping Exp5 all-dataset scheduler." >>"${LOG_FILE}"

    pkill -TERM -f "run_exp03_fast_gpu.py --dataset amazon-arts crafts and sewing" || true
    pkill -TERM -f "exp_05_sensitivity/run_exp05_sensitivity.py --python-bin" || true
    pkill -TERM -f "bash exp_05_sensitivity/run_exp05_all.sh" || true

    sleep 10
    pkill -KILL -f "run_exp03_fast_gpu.py --dataset amazon-arts crafts and sewing" || true
    pkill -KILL -f "exp_05_sensitivity/run_exp05_sensitivity.py --python-bin" || true
    pkill -KILL -f "bash exp_05_sensitivity/run_exp05_all.sh" || true

    echo "[$(date '+%F %T')] Stop request issued." >>"${LOG_FILE}"
    exit 0
  fi

  sleep 20
done
