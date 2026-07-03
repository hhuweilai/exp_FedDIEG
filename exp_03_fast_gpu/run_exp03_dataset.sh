#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: bash exp_03_fast_gpu/run_exp03_dataset.sh <dataset-name> <output-slug>}"
SLUG="${2:?Usage: bash exp_03_fast_gpu/run_exp03_dataset.sh <dataset-name> <output-slug>}"

BASE="exp_03_fast_gpu/final/${SLUG}_base"
REFRESH="exp_03_fast_gpu/final/${SLUG}_full_refresh"
FINAL="exp_03_fast_gpu/final/${SLUG}"

mkdir -p exp_03_fast_gpu/final
rm -rf "$BASE" "$REFRESH" "$FINAL"

echo "===== Exp3 ablations without full model: ${DATASET} ====="
/root/miniconda3/bin/python exp_03_fast_gpu/run_exp03_fast_gpu.py \
  --dataset "$DATASET" \
  --variants Ours-Plain Ours-NoTemp Ours-NoStruct \
  --num-windows 5 \
  --rounds-per-window 120 \
  --eval-every 10 \
  --participation-rate 0.1 \
  --microbatch-size 512 \
  --eval-batch-size 2048 \
  --local-epochs 1 \
  --candidate-count 99 \
  --lambda-temp 0.01 \
  --lambda-struct 0.20 \
  --edge-ema 0.80 \
  --feddieg-memory-momentum 0.80 \
  --out-dir "$BASE"

echo "===== Exp3 final full model with boundary refresh: ${DATASET} ====="
/root/miniconda3/bin/python exp_03_fast_gpu/run_exp03_fast_gpu.py \
  --dataset "$DATASET" \
  --variants Ours-Full \
  --num-windows 5 \
  --rounds-per-window 120 \
  --eval-every 10 \
  --participation-rate 0.1 \
  --microbatch-size 512 \
  --eval-batch-size 2048 \
  --local-epochs 1 \
  --candidate-count 99 \
  --lambda-temp 0.01 \
  --lambda-struct 0.20 \
  --edge-ema 0.80 \
  --feddieg-memory-momentum 0.80 \
  --feddieg-boundary-refresh-rate 0.30 \
  --feddieg-boundary-refresh-momentum 0.20 \
  --out-dir "$REFRESH"

echo "===== Merge final: ${DATASET} ====="
SLUG_ENV="$SLUG" /root/miniconda3/bin/python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

slug = os.environ["SLUG_ENV"]
root = Path("exp_03_fast_gpu/final")
base = root / f"{slug}_base"
refresh = root / f"{slug}_full_refresh"
final = root / slug
final.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(base / "exp03_summary.csv")
rounds = pd.read_csv(base / "exp03_round_metrics.csv")
windows = pd.read_csv(base / "exp03_window_metrics.csv")
refresh_summary = pd.read_csv(refresh / "exp03_summary.csv")
refresh_rounds = pd.read_csv(refresh / "exp03_round_metrics.csv")
refresh_windows = pd.read_csv(refresh / "exp03_window_metrics.csv")

summary = pd.concat([summary, refresh_summary], ignore_index=True)
rounds = pd.concat([rounds, refresh_rounds], ignore_index=True)
windows = pd.concat([windows, refresh_windows], ignore_index=True)

order = {"Ours-Plain": 0, "Ours-NoTemp": 1, "Ours-NoStruct": 2, "Ours-Full": 3}
summary["_order"] = summary["variant"].map(order)
rounds["_order"] = rounds["variant"].map(order)
windows["_order"] = windows["variant"].map(order)
summary = summary.sort_values("_order").drop(columns=["_order"])
rounds = rounds.sort_values(["_order", "window", "global_round"]).drop(columns=["_order"])
windows = windows.sort_values(["_order", "window"]).drop(columns=["_order"])

summary.to_csv(final / "exp03_summary.csv", index=False, encoding="utf-8-sig")
rounds.to_csv(final / "exp03_round_metrics.csv", index=False, encoding="utf-8-sig")
windows.to_csv(final / "exp03_window_metrics.csv", index=False, encoding="utf-8-sig")

for src, dst in [
    (base / "config.json", "ablation_base_config.json"),
    (refresh / "config.json", "full_refresh_config.json"),
]:
    if src.exists():
        shutil.copy2(src, final / dst)

sys.path.insert(0, str(Path(".").resolve()))
from exp_03_fast_gpu.run_exp03_fast_gpu import make_plots

make_plots(summary.to_dict("records"), rounds.to_dict("records"), final)
cols = [
    "variant",
    "avg_window_ndcg",
    "best_test_ndcg",
    "final_test_ndcg",
    "max_drop",
    "boundary_max_drop",
    "loss_oscillation",
    "avg_ecr",
    "stability_score",
]
summary[cols].to_csv(final / "exp03_report_table.csv", index=False, encoding="utf-8-sig")
print(summary[cols].to_string(index=False))
PY

rm -rf "$BASE" "$REFRESH"
echo "===== Done: ${FINAL} ====="
