#!/usr/bin/env bash
# =============================================================================
# val_DynamicGraph 冒烟（pilot）：三条臂先小规模跑通，约 8~12 分钟
#
#   bash val_DynamicGraph/dg_run_pilot.sh
#
# 做什么：nograph / static / dynamic × 1 个种子 × 40 轮/窗口 × 5 窗口
# 输出：  val_DynamicGraph/results/raw_pilot/        每次运行 5 个文件
#         val_DynamicGraph/results/aggregated_pilot/ 图 + 表 + 报告
#
# 目的不是拿结论（40 轮/窗口太短），而是确认：
#   · 三条臂都能跑通、都能写出结果；
#   · 聚合脚本能出表、出图、给出结论骨架；
#   · StaticGraph 的边确实冻结、DynamicGraph 的边确实在变、NoGraph 的 delta_rel 为 0。
#
# 常用覆盖：
#   PYTHON=/path/to/python bash val_DynamicGraph/dg_run_pilot.sh
#   SEEDS="42 777" ROUNDS=30 bash val_DynamicGraph/dg_run_pilot.sh
#   DEVICE=cpu bash val_DynamicGraph/dg_run_pilot.sh
# =============================================================================
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m}"
SEEDS="${SEEDS:-42}"
ARMS="${ARMS:-nograph static dynamic}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-40}"               # 正式是 120
PARTICIPATION="${PARTICIPATION:-0.1}"
RHO="${RHO:-0.2}"
NEIGHBORS="${NEIGHBORS:-20}"
GRAPH_MIX="${GRAPH_MIX:-1}"
PROBE_SIZE="${PROBE_SIZE:-128}"
DEVICE="${DEVICE:-auto}"
NO_AGGREGATE="${NO_AGGREGATE:-0}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "/home/wl/.venv-feddieg/bin/python" ]; then
        PYTHON="/home/wl/.venv-feddieg/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    else
        PYTHON="python3"
    fi
fi

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
RUNNER="$VAL_DIR/dg_run.py"
AGGREGATE="$VAL_DIR/dg_aggregate.py"
RAW="$VAL_DIR/results/raw_pilot"
AGG="$VAL_DIR/results/aggregated_pilot"

slug() { echo "$1" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//'; }

echo "=============================================================="
echo "val_DynamicGraph 冒烟（pilot）"
echo "  工作目录 : $ROOT"
echo "  解释器   : $PYTHON"
echo "  数据集   : $DATASETS"
echo "  臂       : $ARMS"
echo "  种子     : $SEEDS"
echo "  规模     : $NUM_WINDOWS 窗口 x $ROUNDS 轮 | 参与率 $PARTICIPATION | rho $RHO"
echo "  设备     : $DEVICE"
echo "=============================================================="

echo "=== 环境自检 ==="
if ! "$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"; then
    echo "环境自检失败：这个 Python 里缺少 numpy / pandas / torch。用 PYTHON=... 指定解释器。" >&2
    exit 1
fi

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
FAILED=()
START=$(date +%s)

for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    for SEED in $SEEDS; do
        for ARM in $ARMS; do
            DONE="$RAW/$S/$ARM/seed_$SEED/dg_summary.csv"
            if [ -f "$DONE" ]; then
                echo "[skip] $DATASET | $ARM | seed $SEED 已完成"
                continue
            fi
            echo ""
            echo "=== $DATASET | $ARM | seed $SEED ==="
            if ! "$PYTHON" -X utf8 "$RUNNER" \
                --dataset "$DATASET" \
                --arm "$ARM" \
                --seed "$SEED" \
                --rho "$RHO" \
                --neighbors "$NEIGHBORS" \
                --graph-mix "$GRAPH_MIX" \
                --num-windows "$NUM_WINDOWS" \
                --rounds-per-window "$ROUNDS" \
                --participation-rate "$PARTICIPATION" \
                --probe-size "$PROBE_SIZE" \
                --device "$DEVICE" \
                --out-root "$RAW"; then
                echo "[warn] 运行失败：$DATASET | $ARM | seed $SEED（继续）"
                FAILED+=("$DATASET|$ARM|seed$SEED")
            fi
        done
    done
done

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "训练阶段用时 $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
[ ${#FAILED[@]} -gt 0 ] && echo "失败的运行: ${FAILED[*]}"

if [ "$NO_AGGREGATE" != "1" ]; then
    echo ""
    echo "=== 聚合 -> $AGG ==="
    "$PYTHON" -X utf8 "$AGGREGATE" --root "$RAW" --out "$AGG" || echo "[warn] 聚合失败"
fi

echo ""
echo "=============================================================="
echo "冒烟结果："
echo "  $AGG/dg_report.md"
echo "  $AGG/tables/dg_verdict.csv"
echo "  $AGG/figures/dg_fig_arms_*.png"
echo ""
echo "三条臂都跑通后，跑正式实验："
echo "  bash val_DynamicGraph/dg_run_basic.sh"
echo "=============================================================="
