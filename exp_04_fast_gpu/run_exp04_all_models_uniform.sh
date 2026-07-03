#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUNNER="${ROOT_DIR}/exp_04_fast_gpu/run_exp04_fast_gpu.py"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/exp_04_fast_gpu/final_uniform_all_models}"
RUN_ABLATION="${RUN_ABLATION:-0}"
extra_args=("$@")

methods=(FedMF FedSASRec GPFedRec FCUCR F3CRec DGFedRS FedDIEG)

common_args=(
  --methods "${methods[@]}"
  --pretrain-rounds 50
  --adapt-rounds 10
  --new-client-ratio 0.25
  --new-user-strategy low_history
  --min-new-user-history 5
  --d-init 3
  --new-interactions-per-round 2
  --participation-rate 0.10
  --microbatch-size 512
  --eval-batch-size 1024
  --local-epochs 1
  --adapt-epochs 1
  --lr 0.001
  --adapt-lr 0.0002
  --gpfedrec-neighbors 8
  --gpfedrec-mix 0.12
  --fcucR-kb-len 5
  --fcucR-noise-std 0.01
  --fcucR-distill-weight 0.1
  --f3crec-replay-topk 8
  --f3crec-replay-mix 0.15
  --f3crec-temporal-beta 0.15
  --f3crec-memory-momentum 0.60
  --dgfedrs-aug-ratio 0.15
  --dgfedrs-aug-mix 0.30
  --dgfedrs-noise-std 0.02
  --dgfedrs-schedule-power 1.0
  --feddieg-neighbors 20
  --feddieg-graph-mix 0.30
  --feddieg-neighbor-temp 0.05
  --feddieg-min-sim 0.05
  --feddieg-tail-power 1.8
  --feddieg-final-loss-weight 0.15
  --feddieg-memory-momentum 0.45
  --feddieg-short-score-mix 0.20
  --feddieg-proto-topk 8
  --feddieg-proto-state-mix 0.35
  --feddieg-proto-memory-mix 0.75
  --feddieg-proto-edge-mix 0.45
  --feddieg-proto-score-mix 0.55
)

if [[ "${RUN_ABLATION}" != "1" ]]; then
  common_args+=(--no-ablation)
fi

run_dataset() {
  local dataset="$1"
  local folder="$2"
  "${PYTHON_BIN}" "${RUNNER}" \
    --dataset "${dataset}" \
    --out-dir "${BASE_OUT}/${folder}" \
    "${common_args[@]}" \
    "${extra_args[@]}"
}

mkdir -p "${BASE_OUT}"

run_dataset "movielens-1m" "movielens-1m"
run_dataset "amazon-digital music" "amazon-digital_music"
run_dataset "amazon-video games" "amazon-video_games"
run_dataset "amazon-arts crafts and sewing" "amazon-arts_crafts_and_sewing"

"${PYTHON_BIN}" - "${BASE_OUT}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

base_out = Path(sys.argv[1])
folders = [
    "movielens-1m",
    "amazon-digital_music",
    "amazon-video_games",
    "amazon-arts_crafts_and_sewing",
]
fields = [
    "folder",
    "method",
    "initial_ndcg",
    "growth_rate",
    "early_growth_rate",
    "auc_ndcg",
    "best_ndcg",
    "best_round",
    "final_ndcg",
    "initial_hr",
    "final_hr",
]

rows = []
for folder in folders:
    summary_path = base_out / folder / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    for _, row in summary.iterrows():
        rows.append({"folder": folder, **{field: row[field] for field in fields if field != "folder"}})

out = pd.DataFrame(rows, columns=fields)
out.to_csv(base_out / "exp04_all_models_main_metrics.csv", index=False)
print(f"[aggregate] wrote {base_out / 'exp04_all_models_main_metrics.csv'}")
PY

echo "Done. Unified all-model Experiment 4 outputs: ${BASE_OUT}"
