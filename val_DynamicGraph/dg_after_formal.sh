#!/usr/bin/env bash
# 等正式批次跑完，自动做阶段 4 的聚合 + 阶段 5 的统计表 + 验收，并把结果写进日志。
#
# 为什么要有它：正式批次（128 run、4 并发、含共享卡干扰）要跑 20 h 以上，
# 人不可能一直盯着。这个脚本只做**只读 + 写隔离目录**的事，失败了也不会污染 legacy：
#   results/agg_formal、results/{freeze,quality,genre}_formal、实验验证报告_动态图_2026-11-10.md
#
# 用法（放在服务器上，detached 跑）：
#   setsid bash val_DynamicGraph/dg_after_formal.sh < /dev/null > /dev/null 2>&1 &
#
# 手工执行也安全：批次还在跑时会先等，跑完才开始聚合。
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG="$VAL_DIR/results/phase4_auto.out"
RAW="$VAL_DIR/results/raw_formal"
AGG="${AGG_ROOT:-$VAL_DIR/results/agg_formal}"
REPORT="$VAL_DIR/实验验证报告_动态图_2026-11-10.md"
EXPECTED="${EXPECTED_RUNS:-128}"
POLL="${POLL_SECONDS:-300}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 等待正式批次结束（期望 $EXPECTED 个 run）==="
while pgrep -f "[d]g_run_formal_retry.sh" >/dev/null 2>&1 || pgrep -f "[v]al_DynamicGraph/dg_run_parallel.sh" >/dev/null 2>&1; do
    sleep "$POLL"
done

N=$(find "$RAW" -name dg_summary.csv 2>/dev/null | wc -l)
log "批次进程已退出；已完成 run = $N / $EXPECTED"
if [ "$N" -lt "$EXPECTED" ]; then
    log "[warn] run 数不足，仍继续聚合（缺的 run 会在报告里表现为某格缺失）"
fi

log "--- 阶段 4：聚合 + 诊断 + 报告 ---"
bash "$VAL_DIR/dg_phase4.sh" all >>"$LOG" 2>&1
RC4=$?

log "--- 阶段 5：统计表 ---"
"$PY" -X utf8 "$VAL_DIR/dg_stats_formal.py" --agg "$AGG" --legacy "$VAL_DIR/results/aggregated" \
    >>"$LOG" 2>&1
RC5=$?

log "--- 阶段 5：验收 ---"
bash "$VAL_DIR/dg_verify_formal.sh" >>"$LOG" 2>&1
RC6=$?

log "=== 结束：phase4=$RC4 stats=$RC5 verify=$RC6 ==="
log "产物：$AGG/tables/　$AGG/阶段5_统计表.md　$REPORT"
