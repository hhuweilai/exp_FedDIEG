#!/usr/bin/env bash
# 一行式进度快照（给本地监控 / 人工巡检用）。
#
#   bash val_DynamicGraph/dg_status.sh
#
# 输出三行：完成 run 数 / 进程数与 GPU 占用 / 各臂已完成计数 + 最近一条日志。
# 只读，不改任何文件；tags 目录可用 RAW 环境变量覆盖（默认 raw_formal + logs_formal）。
set -u
VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT" || exit 1

RAW="${RAW:-$VAL_DIR/results/raw_formal}"
LOGS="${LOGS:-$VAL_DIR/results/logs_formal}"

CT=$(find "$RAW" -name dg_summary.csv 2>/dev/null | wc -l)
PY=$(pgrep -cf "[p]ython.*dg_run.py" || true)
SH=$(pgrep -cf "[d]g_run_formal" || true)
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | paste -sd';')
ARMS=$(find "$RAW" -name dg_summary.csv 2>/dev/null \
       | sed 's#.*/raw_formal/##; s#/seed_.*##' | sort | uniq -c | paste -sd';')
LAST=$(ls -t "$LOGS"/*.log 2>/dev/null | head -1 | xargs -r tail -n 1 | cut -c1-130)

echo "CT=$CT PY=${PY:-0} SH=${SH:-0}"
echo "GPU=$GPU"
echo "ARMS=$ARMS"
echo "LAST=$LAST"
