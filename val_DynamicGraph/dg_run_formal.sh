#!/usr/bin/env bash
# 正式重跑（F7/F8 修复后）：阶段 3 + 阶段 4 的全部批次。
#
# 用法：
#   bash val_DynamicGraph/dg_run_formal.sh            # 全部批次（预计 ~18 h，4 并发）
#   bash val_DynamicGraph/dg_run_formal.sh phase3     # 只跑阶段 3（12 run ≈ 1.6 h）
#   bash val_DynamicGraph/dg_run_formal.sh core       # 只跑主臂 5 种子（60 run）
#   bash val_DynamicGraph/dg_run_formal.sh grid       # 只跑冻结网格（40 run）
# 每次调用都会跳过已完成的 run，可安全重复执行 / 断点续跑。
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
export PERSONAL_ON_GPU=1 JOBS_PER_GPU=2 GPUS=0,1
export NO_AGGREGATE=1
TAG="formal"
export RAW_ROOT="$VAL_DIR/results/raw_${TAG}"
export LOG_ROOT="$VAL_DIR/results/logs_${TAG}"
export AGG_ROOT="$VAL_DIR/results/agg_${TAG}"
mkdir -p "$RAW_ROOT" "$LOG_ROOT"

SEEDS5="42 777 2024 1234 3407"
SEEDS3="42 777 2024"
BOTH="movielens-1m,amazon-digital music"
ML="movielens-1m"

# run <label> <datasets> <arms> <seeds> [K=V ...]
run() {
    local label="$1" ds="$2" arms="$3" seeds="$4"
    shift 4
    echo ""
    echo "################ $label ################"
    env DATASETS="$ds" ARMS="$arms" SEEDS="$seeds" "$@" \
        bash "$VAL_DIR/dg_run_parallel.sh" || true
    echo "################ $label done ################"
}

ONLY="${1:-all}"
want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

# ---------------- 阶段 3：新臂 3 种子（ML）----------------
# onboarding / fresh_only + static_build120 + periodic_boundary 一次跑完（12 run）
if want phase3; then
    run phase3_new "$ML" "onboarding fresh_only static periodic" "$SEEDS3" \
        BUILD_ONLY=1 FREEZE_ROUND=120 REFRESH_PHASE=boundary
fi

# ---------------- 阶段 4 核心：6 臂 × 5 种子 × 2 数据集（60 run）----------------
if want core; then
    run core6 "$BOTH" "nograph static periodic dynamic onboarding fresh_only" "$SEEDS5"
fi

# ---------------- 阶段 4 静态臂变体（ML）----------------
if want static; then
    run static_at120 "$ML" "static" "$SEEDS5" FREEZE_ROUND=120
    run static_build120 "$ML" "static" "$SEEDS5" FREEZE_ROUND=120 BUILD_ONLY=1
fi

# ---------------- 阶段 4 periodic 变体（ML）----------------
if want periodic; then
    run periodic_boundary "$ML" "periodic" "$SEEDS5" REFRESH_PHASE=boundary
    run periodic_p20 "$ML" "periodic" "$SEEDS3" REFRESH_ROUNDS=20
    run periodic_p240 "$ML" "periodic" "$SEEDS3" REFRESH_ROUNDS=240
fi

# ---------------- 阶段 4 genre 诊断（ML，用于问题三）----------------
if want genre; then
    run genre "$ML" "dynamic" "$SEEDS5" \
        GENRE_NPZ="$VAL_DIR/data/item_genres_ml1m.npz" RUN_TAG=genre
fi

# ---------------- 阶段 4 冻结 2x2 网格（ρ=0.8，两数据集，5 种子 = 40 run）----------------
if want grid; then
    for st in pool topk; do
        for ne in zero warm; do
            run "grid_${st}_${ne}" "$BOTH" "dynamic" "$SEEDS5" \
                RHO=0.8 GRAPH_STATE="$st" GRAPH_NEW_EDGE="$ne"
        done
    done
fi

# ---------------- 阶段 4 锚点（D1：中期报告口径的可比性）----------------
if want anchor; then
    run anchor_mix0005 "$ML" "dynamic" "42" GRAPH_MIX=0.005
    run anchor_mix005 "$ML" "dynamic" "42" GRAPH_MIX=0.05
fi

echo ""
echo "=============================================================="
echo "批次结束。结果目录：$RAW_ROOT"
echo "已完成 run 数：$(find "$RAW_ROOT" -name dg_summary.csv | wc -l)"
echo "=============================================================="
