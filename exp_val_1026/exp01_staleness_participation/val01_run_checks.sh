#!/usr/bin/env bash
# 验证实验 1 · 三项前置检验（Linux / 服务器）
#
#   bash exp_val_1026/exp01_staleness_participation/val01_run_checks.sh
#   ROUNDS=60 SEEDS="42" bash .../val01_run_checks.sh        # 先探路
#   ONLY=drift bash .../val01_run_checks.sh                  # 只跑其中一项
#
# 三项检验都只跑 Movielens-1M 的低参与率（0.1 / 0.3）、3 个种子，写到
# results/checks/<检验名>/，不会覆盖正式结果 results/raw/。
# 跑完执行： python val01_checks_report.py
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${ROOT}/exp_val_1026/exp01_staleness_participation"
SCRIPT="${BASE}/val01_run_staleness.py"

DATASET="${DATASET:-movielens-1m}"
IFS=',' read -ra SEEDS <<< "${SEEDS:-42,777,2024}"
IFS=',' read -ra RATES <<< "${RATES:-0.1,0.3}"
WINDOWS="${WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
DRIFT_USERS="${DRIFT_USERS:-2000}"
ONLY="${ONLY:-}"

SLUG="$(printf '%s' "${DATASET}" | sed 's/[^[:alnum:]_-]/_/g')"

run_check() {
  local check="$1" mode="$2" rate="$3" seed="$4"; shift 4
  local out="${BASE}/results/checks/${check}"
  local rate_str done
  rate_str="$(printf '%.2f' "${rate}")"
  done="${out}/${SLUG}/${mode}/p${rate_str}/seed_${seed}/val01_summary.csv"
  if [ -f "${done}" ]; then echo "[skip] ${done} 已存在"; return; fi
  echo "=== ${check} | ${mode} | p=${rate_str} | seed=${seed} ==="
  "${PYTHON}" "${SCRIPT}" \
    --dataset "${DATASET}" --mode "${mode}" \
    --participation-rate "${rate}" --seed "${seed}" \
    --num-windows "${WINDOWS}" --rounds-per-window "${ROUNDS}" \
    --out-root "${out}" "$@"
}

for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [ -z "${ONLY}" ] || [ "${ONLY}" = "graphmix" ]; then
      for mode in upload_only upload_fresh_edges; do
        run_check graphmix0.2 "${mode}" "${rate}" "${seed}" --feddieg-graph-mix 0.2
      done
    fi
    if [ -z "${ONLY}" ] || [ "${ONLY}" = "edgescope" ]; then
      run_check edgescope_all upload_only "${rate}" "${seed}" --update-scope all
    fi
    if [ -z "${ONLY}" ] || [ "${ONLY}" = "drift" ]; then
      run_check drift upload_only "${rate}" "${seed}" --drift-sample-users "${DRIFT_USERS}"
    fi
  done
done

echo
echo "三项检验跑完。汇总："
echo "  ${PYTHON} ${BASE}/val01_checks_report.py"
