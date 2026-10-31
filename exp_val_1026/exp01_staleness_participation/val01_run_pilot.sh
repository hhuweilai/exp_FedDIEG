#!/usr/bin/env bash
# 验证实验 1（val01）· 冒烟测试：只验证流程能跑通，结果无统计意义。
# 用法： bash exp_val_1026/exp01_staleness_participation/val01_run_pilot.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${ROOT}/exp_val_1026/exp01_staleness_participation"
OUT="${BASE}/results/archive/pilot"

for MODE in upload_only upload_fresh_edges; do
  echo "=== pilot | ${MODE} ==="
  "${PYTHON}" "${BASE}/val01_run_staleness.py" \
    --dataset movielens-1m \
    --mode "${MODE}" \
    --participation-rate 0.3 \
    --seed 42 \
    --num-windows 1 \
    --rounds-per-window 20 \
    --eval-every 10 \
    --max-eval-users 2000 \
    --out-root "${OUT}"
done

echo
echo "冒烟完成。检查产物："
echo "  ${OUT}/movielens-1m/<mode>/p0.30/seed_42/"
echo "  应包含 config.json / val01_summary.csv / val01_staleness_buckets.csv / val01_run.log"
echo
echo "（可选）用冒烟数据试一下聚合："
echo "  ${PYTHON} ${BASE}/val01_aggregate.py --root ${OUT} --out ${BASE}/results/archive/pilot_agg"
