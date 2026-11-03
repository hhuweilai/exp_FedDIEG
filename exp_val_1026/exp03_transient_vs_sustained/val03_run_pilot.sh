#!/usr/bin/env bash
# 验证实验 3（val03）冒烟/探路（Linux / WSL）
#
#   bash exp_val_1026/exp03_transient_vs_sustained/val03_run_pilot.sh
#
# 两个数据集 × 两个场景 × ρ∈{0, 0.8} × 1 个种子 × 40 轮/窗口，
# 结果写到 results/archive/pilot_raw，末尾聚合到 results/archive/pilot_aggregated。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-movielens-1m|amazon-digital music}"
SEED="${SEED:-42}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-40}"
RHOS="${RHOS:-0,0.8}"

BUILDER="$HERE/val03_build_scenarios.py"
RUNNER="$HERE/val03_run_scenario.py"
AGGREGATE="$HERE/val03_aggregate.py"
PLAN_ROOT="$HERE/results/scenarios"
PILOT_ROOT="$HERE/results/archive/pilot_raw"
AGG_OUT="$HERE/results/archive/pilot_aggregated"

slug() { echo "$1" | sed 's/[^A-Za-z0-9_-]/_/g' | sed 's/^_*//; s/_*$//'; }

echo "=== 环境自检：$PYTHON ==="
"$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

IFS='|' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a RHO_LIST <<< "$RHOS"
SCENARIOS=(sustained transient)

for DATASET in "${DATASET_LIST[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    S="$(slug "$DATASET")"
    if [ -f "$PLAN_ROOT/$S/val03_injection_plan.csv" ]; then
        echo "[skip] 场景已存在：$PLAN_ROOT/$S"
    else
        echo "=== 场景构造：$DATASET ==="
        "$PYTHON" "$BUILDER" --dataset "$DATASET" --num-windows "$NUM_WINDOWS" || echo "[warn] 场景构造失败：$DATASET"
    fi
    for SCENARIO in "${SCENARIOS[@]}"; do
        for RHO in "${RHO_LIST[@]}"; do
            OUT_DIR="$PILOT_ROOT/$S/$SCENARIO/rho$RHO/seed_$SEED"
            if [ -f "$OUT_DIR/val03_summary.csv" ]; then
                echo "[skip] 已完成：$OUT_DIR"
                continue
            fi
            echo "=== pilot | $DATASET | $SCENARIO | rho=$RHO | seed=$SEED | $ROUNDS 轮/窗口 ==="
            "$PYTHON" "$RUNNER" \
                --dataset "$DATASET" \
                --scenario "$SCENARIO" \
                --rho "$RHO" \
                --seed "$SEED" \
                --num-windows "$NUM_WINDOWS" \
                --rounds-per-window "$ROUNDS" \
                --participation-rate 1.0 \
                --out-dir "$OUT_DIR" || echo "[warn] pilot 运行失败：$DATASET $SCENARIO rho$RHO"
        done
    done
done

echo ""
echo "=== 聚合 pilot 结果 ==="
"$PYTHON" "$AGGREGATE" --root "$PILOT_ROOT" --out "$AGG_OUT" --rounds-per-window "$ROUNDS" || echo "[warn] pilot 聚合失败"

echo ""
echo "试跑结束。正式跑用："
echo "  bash $HERE/val03_run_basic.sh"
