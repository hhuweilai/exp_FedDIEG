#!/usr/bin/env bash
# 用法：dg_wait.sh [目标完成数] [最长等待秒]
# 内部每 10 秒轮询，达成目标（或超时/全部完成）时打印进度 + 已完成 run 的关键指标。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
TARGET="${1:-24}"
MAX="${2:-570}"
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"

elapsed=0
while [ "$elapsed" -lt "$MAX" ]; do
    DONE=$(ls -1 results/raw/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
    FIN=$(grep -c "全部完成" results/dg_finish_all.out 2>/dev/null | head -1)
    if [ "$DONE" -ge "$TARGET" ] || [ "${FIN:-0}" -gt 0 ]; then break; fi
    sleep 10
    elapsed=$((elapsed + 10))
done

RUN=$(pgrep -cf "dg_run.py.*amazon" 2>/dev/null | head -1)
DONE=$(ls -1 results/raw/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' '/' | sed 's|/$||')
FIN=$(grep -c "全部完成" results/dg_finish_all.out 2>/dev/null | head -1)
printf "%s | 在跑 %s | 完成 %s/24 | GPU %s%% | 收尾 %s | 本次等了 %ss\n" \
  "$(date '+%H:%M:%S')" "${RUN:-0}" "$DONE" "$GPU" \
  "$([ "${FIN:-0}" -gt 0 ] && echo DONE || echo 等待)" "$elapsed"

echo "--- 已完成 run 的关键指标（Amazon，对比 ML 的 dynamic 基线 0.2379 / lift 1.56）---"
"$PY" - <<'PYEOF'
import csv
from pathlib import Path
files = sorted(Path("results/raw/amazon-digital_music").rglob("dg_summary.csv"))
if not files:
    print("  （还没有完成的 run）")
for f in files:
    r = next(csv.DictReader(open(f, encoding="utf-8-sig")))
    print(f"  {r['arm_dir']:26s} seed{r['seed']:>5} | {float(r['elapsed_seconds'])/60:5.1f} min | "
          f"NDCG {float(r['avg_window_ndcg']):.4f} → {float(r['final_window_ndcg']):.4f} | "
          f"lift {r['mean_neighbor_lift'][:5]:>5} | delta_rel {r['mean_delta_rel'][:5]:>5} | "
          f"edges {r['final_n_edges']:>5}")
PYEOF

if [ "${FIN:-0}" -gt 0 ]; then
    echo "--- 收尾日志尾部 ---"
    tail -12 results/dg_finish_all.out
fi
