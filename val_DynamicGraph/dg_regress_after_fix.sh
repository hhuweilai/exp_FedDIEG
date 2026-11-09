#!/usr/bin/env bash
# 修复后（F7 摘要时点 + F8 因果负采样）的单种子回归：4 条臂 × seed 42。
#
# 对比基线：`results/raw_phase2b`（ECR v2 定义、但没有 F7/F8 修复）。
# 目的：量化修复对推荐指标的影响，决定正式实验的重跑范围。
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT"

ARMS="${ARMS:-nograph static periodic dynamic}"
SEEDS="${SEEDS:-42}"
TAG="${1:-fix1}"
RAW_ROOT="$VAL_DIR/results/raw_${TAG}"
LOG_ROOT="$VAL_DIR/results/logs_${TAG}"

echo "ARMS     = $ARMS"
echo "SEEDS    = $SEEDS"
echo "RAW_ROOT = $RAW_ROOT"

setsid env \
  ARMS="$ARMS" SEEDS="$SEEDS" NO_AGGREGATE=1 \
  RAW_ROOT="$RAW_ROOT" LOG_ROOT="$LOG_ROOT" AGG_ROOT="$VAL_DIR/results/agg_${TAG}" \
  bash "$VAL_DIR/dg_run_parallel.sh" \
  > "$VAL_DIR/results/${TAG}_regress.out" 2>&1 < /dev/null &

echo "launched pid=$! -> $VAL_DIR/results/${TAG}_regress.out"
