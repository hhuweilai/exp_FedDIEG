#!/usr/bin/env bash
# =============================================================================
# val_DynamicGraph 正式实验：NoGraph / StaticGraph / DynamicGraph 严格对照
#
#   bash val_DynamicGraph/dg_run_basic.sh
#
# 默认网格：MovieLens-1M × 3 条臂 × 3 个种子 = 9 次运行
#           每次 5 窗口 × 120 轮 × 参与率 0.1（论文 C=0.1）
#           单卡 3090 实测约 2.45 s/轮 ⇒ 每次约 28 分钟 ⇒ 全量约 4~4.5 小时
#
# 输出：
#   val_DynamicGraph/results/raw/<dataset>/<arm>/seed_<s>/   每次运行 5 个文件
#   val_DynamicGraph/results/aggregated/                    图 + 9 张表 + 报告
#   -> 直接看 results/aggregated/dg_report.md 与 tables/dg_verdict.csv
#
# 脚本行为：
#   · 已完成的运行自动跳过（中断后重跑同一条命令即可续跑）
#   · 单次失败不中断整轮，最后汇总失败的组合
#   · 末尾自动聚合（NO_AGGREGATE=1 可关掉）
#
# 常用变体（都是独立目录，不会覆盖已有结果）：
#   SEEDS="42" bash val_DynamicGraph/dg_run_basic.sh                 # 先跑一个种子（约 1.5 小时）
#   ARMS="nograph static dynamic periodic" bash .../dg_run_basic.sh  # 加"每窗口刷新一次"的补充臂
#   ROUNDS=60 bash .../dg_run_basic.sh                               # 半长窗口，快速看方向
#   DATASETS="amazon-digital music" bash .../dg_run_basic.sh         # 换数据集
#   GRAPH_MIX=5 bash .../dg_run_basic.sh                             # 剂量反应：把图分支放大 5 倍
#   RHO=0.8 bash .../dg_run_basic.sh                                 # 动态臂换 EMA 系数（自动独立目录）
# =============================================================================
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m}"
SEEDS="${SEEDS:-42 777 2024}"
ARMS="${ARMS:-nograph static dynamic}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
PARTICIPATION="${PARTICIPATION:-0.1}"     # 论文 C=0.1
RHO="${RHO:-0.2}"                         # 论文 RHO=0.2
NEIGHBORS="${NEIGHBORS:-20}"
GRAPH_MIX="${GRAPH_MIX:-1}"
PROBE_SIZE="${PROBE_SIZE:-256}"
DEVICE="${DEVICE:-auto}"                  # 有 GPU 就自动用 GPU
NO_AGGREGATE="${NO_AGGREGATE:-0}"
FREEZE_ROUND="${FREEZE_ROUND:-}"           # 留空 = 预热结束那一轮建一次图
REFRESH_ROUNDS="${REFRESH_ROUNDS:-}"       # 留空 = rounds_per_window

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
RAW="$VAL_DIR/results/raw"
AGG="$VAL_DIR/results/aggregated"

slug() { echo "$1" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//'; }

VARIANT_SUFFIX=""
[ "$GRAPH_MIX" != "1" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_mix${GRAPH_MIX}"
[ "$RHO" != "0.2" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_rho${RHO}"
[ "$NEIGHBORS" != "20" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_k${NEIGHBORS}"

echo "=============================================================="
echo "val_DynamicGraph 正式实验"
echo "  工作目录 : $ROOT"
echo "  解释器   : $PYTHON"
echo "  数据集   : $DATASETS"
echo "  臂       : $ARMS"
echo "  种子     : $SEEDS"
echo "  规模     : $NUM_WINDOWS 窗口 x $ROUNDS 轮 | 参与率 $PARTICIPATION"
echo "  图       : rho=$RHO neighbors=$NEIGHBORS graph_mix=$GRAPH_MIX"
echo "  设备     : $DEVICE | probe_size=$PROBE_SIZE"
echo "  输出     : $RAW"
echo "=============================================================="

echo "=== 环境自检 ==="
if ! "$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"; then
    echo "环境自检失败：这个 Python 里缺少 numpy / pandas / torch。用 PYTHON=... 指定解释器。" >&2
    exit 1
fi
"$PYTHON" -c "import torch;print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(cpu)')" || true

n_arms=$(echo $ARMS | wc -w)
n_seeds=$(echo $SEEDS | wc -w)
n_datasets=$(echo "$DATASETS" | tr ',' '\n' | grep -c . )
n_runs=$((n_arms * n_seeds * n_datasets))
per_run_min=$(awk -v r="$ROUNDS" -v w="$NUM_WINDOWS" 'BEGIN{printf "%.0f", (r*w*3.0)/60}')
echo ""
echo "计划 $n_runs 次运行；按 3090 实测 2.45 s/轮 + 评测/数据加载开销估算，每次约 ${per_run_min} 分钟，共约 $((n_runs * per_run_min / 60)) 小时。"
echo ""

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
FAILED=()
DONE_COUNT=0
SWEEP_START=$(date +%s)

for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    for SEED in $SEEDS; do
        for ARM in $ARMS; do
            ARM_DIR="$ARM$VARIANT_SUFFIX"
            DONE_FILE="$RAW/$S/$ARM_DIR/seed_$SEED/dg_summary.csv"
            if [ -f "$DONE_FILE" ]; then
                echo "[skip] $DATASET | $ARM_DIR | seed $SEED 已完成"
                DONE_COUNT=$((DONE_COUNT + 1))
                continue
            fi
            echo ""
            echo "=== $(date '+%H:%M:%S')  $DATASET | $ARM_DIR | seed $SEED ==="
            EXTRA_ARGS=()
            if [ -n "$FREEZE_ROUND" ] && [ "$ARM" = "static" ]; then
                EXTRA_ARGS+=(--freeze-round "$FREEZE_ROUND")
            fi
            if [ -n "$REFRESH_ROUNDS" ] && [ "$ARM" = "periodic" ]; then
                EXTRA_ARGS+=(--refresh-rounds "$REFRESH_ROUNDS")
            fi
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
                --out-root "$RAW" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
                echo "[warn] 运行失败：$DATASET | $ARM_DIR | seed $SEED（继续跑下一个）"
                FAILED+=("$DATASET|$ARM_DIR|seed$SEED")
            fi
        done
    done
done

ELAPSED=$(( $(date +%s) - SWEEP_START ))
echo ""
echo "=============================================================="
echo "训练阶段结束：用时 $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒（跳过已完成 $DONE_COUNT 次）"
[ ${#FAILED[@]} -gt 0 ] && echo "失败的运行: ${FAILED[*]}"

if [ "$NO_AGGREGATE" != "1" ]; then
    if ls "$RAW"/*/*/seed_*/dg_summary.csv >/dev/null 2>&1; then
        echo ""
        echo "=== 聚合 -> $AGG ==="
        "$PYTHON" -X utf8 "$AGGREGATE" --root "$RAW" --out "$AGG" || echo "[warn] 聚合失败"
    else
        echo "[skip] 没有任何结果，跳过聚合"
    fi
fi

echo ""
echo "=============================================================="
echo "结果位置："
echo "  $AGG/dg_report.md                                 <- 先看这个"
echo "  $AGG/tables/dg_verdict.csv                        <- ① ② ③ 三条结论"
echo "  $AGG/tables/dg_contrasts.csv                      <- 配对差值与显著性"
echo "  $AGG/tables/dg_hygiene.csv                        <- 严格性检查"
echo "  $AGG/figures/dg_fig_arms_movielens-1m.png         <- 三条臂的曲线"
echo ""
echo "接下来可以做的补充对照（各自独立目录，不会覆盖上面的结果）："
echo "  ARMS=\"nograph static dynamic periodic\" bash val_DynamicGraph/dg_run_basic.sh"
echo "      加「每窗口刷新一次」的慢刷新臂：区分「动态性有用」与「图建得太早」"
echo "  GRAPH_MIX=5 bash val_DynamicGraph/dg_run_basic.sh"
echo "      剂量反应：把图分支放大 5 倍，看 delta_rel 太小时是否才显出贡献"
echo "  RHO=0.8 bash val_DynamicGraph/dg_run_basic.sh"
echo "      只换动态臂的 EMA 系数（自动进 dynamic_rho0.8 目录并与 dynamic 配对）"
echo "=============================================================="
