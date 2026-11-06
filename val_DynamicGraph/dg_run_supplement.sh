#!/usr/bin/env bash
# =============================================================================
# Q2（冻结归因）+ Q3（真实 genre 的邻居有效性）实验运行器
#
#   bash val_DynamicGraph/q23_run.sh
#
# Q2：图更新机制是否合理？—— 2x2 网格（图状态 x 新边初始化），rho 固定 0.8（冻结观测区）
#     pool=保留候选池（修复版） / topk=只存 Top-K（审计前的旧行为，会冻结）
#     zero=论文的 0 初值 / warm=首次观测直接取 Δe
#     => 如果 topk 两个格子都冻结、pool 两个格子都不冻结，就说明冻结来自「状态保留规则」
#        而不是 EMA 本身或新边初始化；再看 warm 相对 zero 改变了多少"适应速度"。
#
# Q3：兴趣相似是否意味着协同有效？—— 在 dynamic 臂上额外计算「邻居 vs 随机」的
#     真实 genre 一致度（ML-1M movies.dat，18 类，物品覆盖率 100%），逐用户落盘，
#     供 q3_analyze.py 与逐用户推荐收益做相关/分桶分析。
#
# 默认 4 并发（每卡 2 进程，实测最优），15 次运行约 80~90 分钟。
# 已完成的自动跳过；每个任务独立日志；启动前检查是否有别的 dg_run.py 在跑。
# =============================================================================
set -uo pipefail

SEEDS="${SEEDS:-42 777 2024}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
GPUS="${GPUS:-0,1}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
PARTICIPATION="${PARTICIPATION:-0.1}"
PROBE_SIZE="${PROBE_SIZE:-256}"
DATASET="${DATASET:-movielens-1m}"
FORCE="${FORCE:-0}"
DO_FREEZE="${DO_FREEZE:-1}"
DO_GENRE="${DO_GENRE:-1}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x "/home/wl/.venv-feddieg/bin/python" ]; then
        PYTHON="/home/wl/.venv-feddieg/bin/python"
    else
        PYTHON="python3"
    fi
fi

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
RUNNER="$VAL_DIR/dg_run.py"
RAW="${RAW_ROOT:-$VAL_DIR/results/raw}"
LOGS="${LOG_ROOT:-$VAL_DIR/results/supplement_logs}"
GENRE_NPZ="$VAL_DIR/data/item_genres_ml1m.npz"

n_gpus=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
SLOTS=$((JOBS_PER_GPU * n_gpus))

echo "=============================================================="
echo "冻结归因 + Genre 协同有效性"
echo "  数据集   : $DATASET   种子: $SEEDS"
echo "  规模     : $NUM_WINDOWS 窗口 x $ROUNDS 轮 | 参与率 $PARTICIPATION"
echo "  并行     : 每卡 $JOBS_PER_GPU 进程 x ${n_gpus} 卡 = $SLOTS 并发"
echo "  输出     : $RAW"
echo "  genre    : $([ -f "$GENRE_NPZ" ] && echo "$GENRE_NPZ（存在）" || echo '缺失！')"
echo "=============================================================="

if [ "$FORCE" != "1" ] && pgrep -f "val_DynamicGraph/dg_run.py" >/dev/null 2>&1; then
    echo "!! 已经有 dg_run.py 在跑，先停掉（或 FORCE=1）" >&2
    exit 1
fi
if [ "$DO_GENRE" = "1" ] && [ ! -f "$GENRE_NPZ" ]; then
    echo "!! 缺少 genre 矩阵 $GENRE_NPZ" >&2
    exit 1
fi

mkdir -p "$LOGS"

# ---- 任务表：目录名 | 额外参数
TASKS_FILE=$(mktemp)
if [ "$DO_FREEZE" = "1" ]; then
  cat >> "$TASKS_FILE" <<'EOF'
dynamic_rho0.8|--arm dynamic --rho 0.8 --graph-state pool --graph-new-edge zero
dynamic_rho0.8_warm|--arm dynamic --rho 0.8 --graph-state pool --graph-new-edge warm
dynamic_rho0.8_topk|--arm dynamic --rho 0.8 --graph-state topk --graph-new-edge zero
dynamic_rho0.8_topk_warm|--arm dynamic --rho 0.8 --graph-state topk --graph-new-edge warm
EOF
fi
if [ "$DO_GENRE" = "1" ]; then
  echo "dynamic_genre|--arm dynamic --rho 0.2 --genre-npz $GENRE_NPZ --run-tag genre" >> "$TASKS_FILE"
fi

# ---- 先算要跑哪些
TODO=()
while IFS='|' read -r ARM_DIR EXTRA; do
    [ -z "${ARM_DIR:-}" ] && continue
    for SEED in $SEEDS; do
        F="$RAW/$DATASET/$ARM_DIR/seed_$SEED/dg_summary.csv"
        if [ -f "$F" ]; then
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
echo "待跑 ${#TODO[@]} 个任务，$SLOTS 并发。"
echo ""

START=$(date +%s)
IDX=0
for TASK in "${TODO[@]}"; do
    IFS='|' read -r ARM_DIR SEED EXTRA <<< "$TASK"
    while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 2; done

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
    sleep 2
done
wait
END=$(date +%s)
echo ""
echo "全部进程结束，墙钟 $(( (END - START) / 60 )) 分 $(( (END - START) % 60 )) 秒"

echo "--- 结果核对 ---"
FAIL=0
while IFS='|' read -r ARM_DIR SEED EXTRA; do
    [ -z "${ARM_DIR:-}" ] && continue
    F="$RAW/$DATASET/$ARM_DIR/seed_$SEED/dg_summary.csv"
    if [ -f "$F" ]; then echo "  [ok  ] $ARM_DIR seed $SEED"; else
        echo "  [FAIL] $ARM_DIR seed $SEED（日志 $LOGS/${ARM_DIR}_seed${SEED}.log）"; FAIL=$((FAIL+1)); fi
done < <(for T in "${TODO[@]}"; do echo "$T"; done)
[ "$FAIL" -gt 0 ] && echo "失败 $FAIL 个"
echo "日志目录：$LOGS"
