#!/usr/bin/env bash
# 阶段 2b：用修正后的 ECR 定义（v2）对 4 条臂做逐位回归。
#
# 目的
#   1) 证明「ECR 定义修正」不改变任何推荐指标（NDCG/HR/逐用户）；
#   2) 产出 v2 定义下的 ECR / graph_turnover（旧的 raw 是 legacy 定义，不能混用）。
#
# 用法（服务器）：
#   bash val_DynamicGraph/dg_regress_v2.sh                 # 4 臂 × seed 42，4 并发
#   ARMS="static periodic" bash val_DynamicGraph/dg_regress_v2.sh
#   SEEDS="42 777" bash val_DynamicGraph/dg_regress_v2.sh
#
# 输出：results/raw_phase2b/、results/logs_phase2b/（不碰 results/raw）
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT"

ARMS="${ARMS:-nograph static periodic dynamic}"
SEEDS="${SEEDS:-42}"
RAW_ROOT="${RAW_ROOT:-$VAL_DIR/results/raw_phase2b}"
LOG_ROOT="${LOG_ROOT:-$VAL_DIR/results/logs_phase2b}"
AGG_ROOT="${AGG_ROOT:-$VAL_DIR/results/agg_phase2b}"
NO_AGGREGATE="${NO_AGGREGATE:-1}"

echo "ARMS      = $ARMS"
echo "SEEDS     = $SEEDS"
echo "RAW_ROOT  = $RAW_ROOT"
echo "LOG_ROOT  = $LOG_ROOT"

setsid env \
  ARMS="$ARMS" SEEDS="$SEEDS" NO_AGGREGATE="$NO_AGGREGATE" \
  RAW_ROOT="$RAW_ROOT" LOG_ROOT="$LOG_ROOT" AGG_ROOT="$AGG_ROOT" \
  bash "$VAL_DIR/dg_run_parallel.sh" \
  > "$VAL_DIR/results/phase2b_regress.out" 2>&1 < /dev/null &

echo "launched pid=$! -> $VAL_DIR/results/phase2b_regress.out"
