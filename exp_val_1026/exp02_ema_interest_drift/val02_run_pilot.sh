#!/usr/bin/env bash
# 验证实验 2（val02）冒烟/探路（Linux / macOS / WSL）
#
#   bash exp_val_1026/exp02_ema_interest_drift/val02_run_pilot.sh
#
# 两个数据集 × 2 个 rho（0 和 0.95）× 1 个种子 × 40 轮/窗口，
# 结果写 results/archive/pilot_raw，末尾自动聚合到 results/archive/pilot_aggregated。
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m,amazon-digital music}"
SEED="${SEED:-42}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-40}"
RHOS="${RHOS:-0 0.95}"
PYTHON="${PYTHON:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAL_DIR="$ROOT/exp_val_1026/exp02_ema_interest_drift"
DETECT="$VAL_DIR/val02_detect_drift.py"
RUNNER="$VAL_DIR/val02_run_ema.py"
AGGREGATE="$VAL_DIR/val02_aggregate.py"
ARCHIVE="$VAL_DIR/results/archive"
PILOT_ROOT="$ARCHIVE/pilot_raw"
AGG_OUT="$ARCHIVE/pilot_aggregated"

slug() { echo "$1" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//'; }

# 环境自检：缺依赖就直接停
echo "=== 环境自检：$PYTHON ==="
if ! "$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"; then
    echo "环境自检失败：这个 Python 里缺少 numpy / pandas / torch。请把 PYTHON 指向跑实验用的解释器。" >&2
    exit 1
fi

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"

# ① cohort
for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    if [ -f "$VAL_DIR/results/drift/$S/val02_drift_users.csv" ]; then
        echo "[skip] cohort 已存在：$DATASET"
    else
        echo "=== ① 漂移识别：$DATASET ==="
        "$PYTHON" "$DETECT" --dataset "$DATASET" --num-windows "$NUM_WINDOWS" || echo "[warn] 漂移识别失败：$DATASET"
    fi
done

# ② 试跑
for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    for RHO in $RHOS; do
        RHOSTR="$(printf '%.2f' "$RHO")"
        OUT_DIR="$PILOT_ROOT/$S/rho$RHOSTR/seed_$SEED"
        if [ -f "$OUT_DIR/val02_summary.csv" ]; then
            echo "[skip] $OUT_DIR 已存在"
            continue
        fi
        echo "=== ② pilot | $DATASET | rho=$RHOSTR | seed=$SEED | $ROUNDS 轮/窗口 ==="
        "$PYTHON" "$RUNNER" \
            --dataset "$DATASET" \
            --rho "$RHOSTR" \
            --seed "$SEED" \
            --num-windows "$NUM_WINDOWS" \
            --rounds-per-window "$ROUNDS" \
            --participation-rate 1.0 \
            --out-dir "$OUT_DIR" || echo "[warn] pilot 运行失败：$DATASET rho$RHOSTR（继续）"
    done
done

# ③ 聚合
echo ""
echo "=== ③ 聚合 pilot 结果 ==="
"$PYTHON" "$AGGREGATE" --root "$PILOT_ROOT" --out "$AGG_OUT" --max-rel-round "$ROUNDS" \
    || echo "[warn] pilot 聚合失败"

echo ""
echo "试跑结束。正式跑用："
echo "  bash $VAL_DIR/val02_run_basic.sh"
