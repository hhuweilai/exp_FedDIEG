#!/usr/bin/env bash
# =============================================================================
# Amazon 数据集上的动态图三组验证实验（训练部分）
#
#   bash val_DynamicGraph/dg_run_amazon.sh
#
# 任务表（8 个配置 × 3 个种子 = 24 个 run）
#   实验一（动态图是否有贡献）：nograph / static / periodic / dynamic
#   实验二（冻结归因 2x2，rho=0.8）：pool+zero / pool+warm / topk+zero / topk+warm
#   实验三（兴趣相似 vs 协同有效）：Amazon 无 item 类别元数据，改用「不需要标签」的
#       邻居质量 vs 收益分析（dg_neighbor_quality.py），复用实验一的逐用户探针结果。
#
# 4 并发（每卡 2 进程）—— 实测最优；Amazon N=12381，单 run 约 32~40 分钟。
# 已完成的自动跳过；单次失败不中断；每任务独立日志。
# =============================================================================
set -uo pipefail

DATASET="${DATASET:-amazon-digital music}"
SEEDS="${SEEDS:-42 777 2024}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
GPUS="${GPUS:-0,1}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
PARTICIPATION="${PARTICIPATION:-0.1}"
PROBE_SIZE="${PROBE_SIZE:-256}"
FORCE="${FORCE:-0}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "/home/wl/.venv-feddieg/bin/python" ]; then
        PYTHON="/home/wl/.venv-feddieg/bin/python"
    else
        PYTHON="python3"
    fi
fi

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$VAL_DIR/dg_run.py"
RAW="${RAW_ROOT:-$VAL_DIR/results/raw}"
LOGS="${LOG_ROOT:-$VAL_DIR/results/amazon_logs}"

n_gpus=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
SLOTS=$((JOBS_PER_GPU * n_gpus))
SLUG=$(echo "$DATASET" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//')

echo "=============================================================="
echo "Amazon 动态图验证实验（训练）"
echo "  数据集   : $DATASET（slug=$SLUG）"
echo "  种子     : $SEEDS"
echo "  规模     : $NUM_WINDOWS 窗口 x $ROUNDS 轮 | 参与率 $PARTICIPATION"
echo "  并行     : 每卡 $JOBS_PER_GPU 进程 x ${n_gpus} 卡 = $SLOTS 并发"
echo "  输出     : $RAW/$SLUG"
echo "=============================================================="

if [ "$FORCE" != "1" ] && pgrep -f "val_DynamicGraph/dg_run.py" >/dev/null 2>&1; then
    echo "!! 已有 dg_run.py 在跑，先停掉（或 FORCE=1）" >&2
    exit 1
fi
mkdir -p "$LOGS"

# ---- 按「预计耗时从长到短」排列，让 4 个槽位先吃最重的任务，缩短总墙钟
TASKS_FILE=$(mktemp)
cat > "$TASKS_FILE" <<'EOF'
dynamic|--arm dynamic
dynamic_rho0.8|--arm dynamic --rho 0.8 --graph-state pool --graph-new-edge zero
dynamic_rho0.8_warm|--arm dynamic --rho 0.8 --graph-state pool --graph-new-edge warm
dynamic_rho0.8_topk_warm|--arm dynamic --rho 0.8 --graph-state topk --graph-new-edge warm
dynamic_rho0.8_topk|--arm dynamic --rho 0.8 --graph-state topk --graph-new-edge zero
periodic|--arm periodic
static|--arm static
nograph|--arm nograph
EOF

TODO=()
while IFS='|' read -r ARM_DIR EXTRA; do
    [ -z "${ARM_DIR:-}" ] && continue
    for SEED in $SEEDS; do
        if [ -f "$RAW/$SLUG/$ARM_DIR/seed_$SEED/dg_summary.csv" ]; then
            echo "[skip] $ARM_DIR | seed $SEED 已完成"
            continue
        fi
        TODO+=("$ARM_DIR|$SEED|$EXTRA")
    done
done < "$TASKS_FILE"
rm -f "$TASKS_FILE"

if [ ${#TODO[@]} -eq 0 ]; then
    echo "没有需要跑的任务。"
    exit 0
fi
echo ""
echo "待跑 ${#TODO[@]} 个任务，$SLOTS 并发（预计 3.5~4 小时）。"
echo ""

START=$(date +%s)
IDX=0
for TASK in "${TODO[@]}"; do
    IFS='|' read -r ARM_DIR SEED EXTRA <<< "$TASK"
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
    GPU=$(echo "$GPUS" | cut -d, -f$(( (IDX % n_gpus) + 1 )))
    LOGF="$LOGS/${ARM_DIR}_seed${SEED}.log"
    echo "[$(date '+%H:%M:%S')] 启动 GPU$GPU | $ARM_DIR | seed $SEED"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -X utf8 "$RUNNER" \
        --dataset "$DATASET" \
        --seed "$SEED" \
        --num-windows "$NUM_WINDOWS" \
        --rounds-per-window "$ROUNDS" \
        --participation-rate "$PARTICIPATION" \
        --probe-size "$PROBE_SIZE" \
        --personal-on-gpu \
        --device cuda \
        --out-root "$RAW" \
        $EXTRA >"$LOGF" 2>&1 &
    IDX=$((IDX + 1))
    sleep 3
done
wait
END=$(date +%s)
echo ""
echo "全部进程结束，墙钟 $(( (END - START) / 60 )) 分 $(( (END - START) % 60 )) 秒"

echo "--- 结果核对 ---"
FAIL=0
for TASK in "${TODO[@]}"; do
    IFS='|' read -r ARM_DIR SEED EXTRA <<< "$TASK"
    if [ -f "$RAW/$SLUG/$ARM_DIR/seed_$SEED/dg_summary.csv" ]; then
        echo "  [ok  ] $ARM_DIR seed $SEED"
    else
        echo "  [FAIL] $ARM_DIR seed $SEED（日志 $LOGS/${ARM_DIR}_seed${SEED}.log）"
        FAIL=$((FAIL + 1))
    fi
done
[ "$FAIL" -gt 0 ] && echo "失败 $FAIL 个"
echo "日志目录：$LOGS"
