#!/usr/bin/env bash
# 带自动补跑的正式批次驱动：反复执行指定批次（最多 3 轮），每轮都会跳过已完成的 run。
#
# 用法（在服务器上，detached 运行）：
#   setsid bash val_DynamicGraph/dg_run_formal_retry.sh all < /dev/null &
#   setsid bash val_DynamicGraph/dg_run_formal_retry.sh phase3 < /dev/null &
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT"

ONLY="${1:-all}"
LOG="${2:-$VAL_DIR/results/formal_${ONLY}.out}"
exec > "$LOG" 2>&1

echo "[$(date '+%F %T')] 开始：批次=$ONLY，最多 3 轮（每轮跳过已完成 run）"
for round in 1 2 3; do
    echo ""
    echo "=========== round $round / 3 ==========="
    bash "$VAL_DIR/dg_run_formal.sh" "$ONLY" || true
    echo "=========== round $round done ==========="
done
echo "[$(date '+%F %T')] 全部轮次结束"
