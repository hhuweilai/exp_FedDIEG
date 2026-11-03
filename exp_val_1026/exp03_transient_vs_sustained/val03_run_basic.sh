#!/usr/bin/env bash
# 验证实验 3（val03）一键跑完（Linux / WSL）
#
#   bash exp_val_1026/exp03_transient_vs_sustained/val03_run_basic.sh
#
# 等价于 val03_run_basic.ps1，默认：
#   两个数据集 × 两个场景（sustained / transient）× ρ∈{0, 0.5, 0.8} × 3 个种子
#   末尾自动聚合到 results/aggregated/
#
# 可用环境变量覆盖（与脚本参数同义）：
#   PYTHON / DATASETS / SEEDS / RHOS / NUM_WINDOWS / ROUNDS / GRAPH_STRENGTH / NEW_EDGE_INIT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-movielens-1m|amazon-digital music}"
SEEDS="${SEEDS:-42,777,2024}"
RHOS="${RHOS:-0,0.5,0.8}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
GRAPH_STRENGTH="${GRAPH_STRENGTH:-paper}"
NEW_EDGE_INIT="${NEW_EDGE_INIT:-ema}"
GATE="${GATE:-raw}"

BUILDER="$HERE/val03_build_scenarios.py"
RUNNER="$HERE/val03_run_scenario.py"
AGGREGATE="$HERE/val03_aggregate.py"
PLAN_ROOT="$HERE/results/scenarios"

slug() { echo "$1" | sed 's/[^A-Za-z0-9_-]/_/g' | sed 's/^_*//; s/_*$//'; }

case "${GRAPH_STRENGTH}" in
    paper)  TAG="" ;;
    strong) TAG="_strong" ;;
    *) echo "GRAPH_STRENGTH 只能是 paper / strong"; exit 1 ;;
esac
if [ "${NEW_EDGE_INIT}" != "ema" ]; then TAG="${TAG}_${NEW_EDGE_INIT}"; fi
OUT_ROOT="$HERE/results/raw${TAG}"
AGG_OUT="$HERE/results/aggregated${TAG}"

echo "=============================================================="
echo "验证实验 3（val03）一键运行"
echo "  数据集   : $DATASETS"
echo "  rho      : $RHOS"
echo "  种子     : $SEEDS"
echo "  窗口/轮数: $NUM_WINDOWS 窗口 x $ROUNDS 轮"
echo "  图强度/新边初值: $GRAPH_STRENGTH / $NEW_EDGE_INIT"
echo "=============================================================="

echo "=== 环境自检：$PYTHON ==="
"$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

IFS='|' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a RHO_LIST <<< "$RHOS"
SCENARIOS=(sustained transient)

# ① 场景构造
for DATASET in "${DATASET_LIST[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    S="$(slug "$DATASET")"
    if [ -f "$PLAN_ROOT/$S/val03_injection_plan.csv" ]; then
        echo "[skip] 场景已存在：$PLAN_ROOT/$S"
        continue
    fi
    echo "=== ① 场景构造：$DATASET ==="
    "$PYTHON" "$BUILDER" --dataset "$DATASET" --num-windows "$NUM_WINDOWS" || echo "[warn] 场景构造失败：$DATASET"
done

# ② 训练
FAILED=()
for DATASET in "${DATASET_LIST[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    S="$(slug "$DATASET")"
    if [ ! -f "$PLAN_ROOT/$S/val03_injection_plan.csv" ]; then
        echo "[skip] $DATASET 没有注入计划"
        continue
    fi
    for SCENARIO in "${SCENARIOS[@]}"; do
        for RHO in "${RHO_LIST[@]}"; do
            for SEED in "${SEED_LIST[@]}"; do
                OUT_DIR="$OUT_ROOT/$S/$SCENARIO/rho$RHO/seed_$SEED"
                if [ -f "$OUT_DIR/val03_summary.csv" ]; then
                    echo "[skip] 已完成：$OUT_DIR"
                    continue
                fi
                echo ""
                echo "=== $DATASET | $SCENARIO | rho=$RHO | seed=$SEED ==="
                if ! "$PYTHON" "$RUNNER" \
                        --dataset "$DATASET" \
                        --scenario "$SCENARIO" \
                        --rho "$RHO" \
                        --seed "$SEED" \
                        --num-windows "$NUM_WINDOWS" \
                        --rounds-per-window "$ROUNDS" \
                        --participation-rate 1.0 \
                        --graph-strength "$GRAPH_STRENGTH" \
                        --graph-gate "$GATE" \
                        --new-edge-init "$NEW_EDGE_INIT" \
                        --out-root "$OUT_ROOT"; then
                    echo "[warn] 运行失败：$DATASET | $SCENARIO | rho=$RHO | seed=$SEED"
                    FAILED+=("$DATASET|$SCENARIO|rho=$RHO|seed=$SEED")
                fi
            done
        done
    done
done

# ③ 聚合
echo ""
echo "=== ③ 聚合：$OUT_ROOT → $AGG_OUT ==="
"$PYTHON" "$AGGREGATE" --root "$OUT_ROOT" --out "$AGG_OUT" || echo "[warn] 聚合失败"

echo ""
echo "=============================================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失败的组合（重跑同一条命令会自动续跑）："
    printf '  - %s\n' "${FAILED[@]}"
else
    echo "没有任何失败组合。"
fi
echo "结果：$AGG_OUT"
echo "=============================================================="
