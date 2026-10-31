#!/usr/bin/env bash
# 验证实验 1（val01）批次脚本
#
# 交付物：results/aggregated/figures/val01_fig_staleness_rate.png
#         （摘要陈旧程度 × 客户端参与率 → 平均 NDCG@10）
#
# 网格（符合任务书"实验一"要求）：
#   upload_only        p = 100% / 50% / 30% / 10%   现状：陈旧摘要照样参与构图
#   upload_fresh_edges p = 30% / 10%                对照：信息永远新鲜（oracle 上界）
#   数据集：MovieLens-1M + 一个 Amazon 数据集
#   种子：3 个独立随机种子
#   合计 6 组 × 3 种子 × 2 数据集 = 36 次运行
#
# 用法：
#   bash exp_val_1026/exp01_staleness_participation/val01_run_basic.sh
#   DATASETS="movielens-1m" bash .../val01_run_basic.sh     # 只跑一个数据集（时间减半）
#   SEEDS="42" ROUNDS=60 bash .../val01_run_basic.sh        # 先探路
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT}/exp_val_1026/exp01_staleness_participation/val01_run_staleness.py"
OUT="${ROOT}/exp_val_1026/exp01_staleness_participation/results/raw"

IFS=',' read -ra DATASETS <<< "${DATASETS:-movielens-1m,amazon-digital music}"
IFS=',' read -ra SEEDS <<< "${SEEDS:-42,777,2024}"
WINDOWS="${WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
FULL_RATES=(1.0 0.5 0.3 0.1)
CTRL_RATES=(0.3 0.1)

run_one() {
  local dataset="$1" mode="$2" rate="$3" seed="$4"
  local slug rate_str done
  slug="$(printf '%s' "${dataset}" | sed 's/[^[:alnum:]_-]/_/g')"
  rate_str="$(printf '%.2f' "${rate}")"
  done="${OUT}/${slug}/${mode}/p${rate_str}/seed_${seed}/val01_summary.csv"
  if [ -f "${done}" ]; then
    echo "[skip] ${done} 已存在"
    return
  fi
  echo "=== ${dataset} | ${mode} | p=${rate_str} | seed=${seed} ==="
  "${PYTHON}" "${SCRIPT}" \
    --dataset "${dataset}" \
    --mode "${mode}" \
    --participation-rate "${rate}" \
    --seed "${seed}" \
    --num-windows "${WINDOWS}" \
    --rounds-per-window "${ROUNDS}" \
    --out-root "${OUT}"
}

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for RATE in "${FULL_RATES[@]}"; do
      run_one "${DATASET}" upload_only "${RATE}" "${SEED}"
    done
    for RATE in "${CTRL_RATES[@]}"; do
      run_one "${DATASET}" upload_fresh_edges "${RATE}" "${SEED}"
    done
  done
done

echo
echo "跑完。出图 + 结论："
echo "  ${PYTHON} ${ROOT}/exp_val_1026/exp01_staleness_participation/val01_aggregate.py"
