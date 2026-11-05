#!/usr/bin/env bash
# 验证实验 2（val02）一键跑完（Linux / macOS / WSL）
#
#   bash exp_val_1026/exp02_ema_interest_drift/val02_run_basic.sh
#
# 默认行为（一条命令跑完所有结果）：
#   ① 对两个数据集（MovieLens-1M + amazon-digital music）做离线漂移识别
#      （稀疏数据集的 cohort 会自动放宽阈值；实在不可用的数据集自动跳过）
#   ② 5 个 rho（0.2 / 0 / 0.5 / 0.8 / 0.95；0.2 = 论文默认）× 3 个种子 × 参与率 10%（论文 C=0.1）
#   ③ 自动聚合并出图 → results/aggregated/
#
#   已完成的任务自动跳过，中断后重跑同一条命令即可续跑；单个任务失败不会中断整轮。
#
# 常用变体：
#   DATASETS="movielens-1m" SEEDS="42" ROUNDS=40 bash .../val02_run_basic.sh   # 先探路
#   GRAPH_STATE=topk bash .../val02_run_basic.sh             # 复现旧的"图冻结"行为做对照
#   DEVICE=cuda bash .../val02_run_basic.sh                  # 用 GPU 跑
#   NO_AGGREGATE=1 bash .../val02_run_basic.sh               # 只跑训练不聚合
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m,amazon-digital music}"
SEEDS="${SEEDS:-42 777 2024}"
RHOS="${RHOS:-0.2 0 0.5 0.8 0.95}"   # 0.2 = 论文默认，务必包含
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
GRAPH_STATE="${GRAPH_STATE:-pool}"      # pool=修复版；topk=复现旧"冻结"行为
NEW_EDGE_INIT="${NEW_EDGE_INIT:-zero}"  # zero=论文 0 初值；warm=首次观测直接取 Δe
PARTICIPATION="${PARTICIPATION:-0.1}"   # 论文 C=0.1
DEVICE="${DEVICE:-auto}"                # auto / cuda / cpu
MIN_DRIFT_COHORT="${MIN_DRIFT_COHORT:-10}"
NO_AGGREGATE="${NO_AGGREGATE:-0}"
PYTHON="${PYTHON:-python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAL_DIR="$ROOT/exp_val_1026/exp02_ema_interest_drift"
DETECT="$VAL_DIR/val02_detect_drift.py"
RUNNER="$VAL_DIR/val02_run_ema.py"
AGGREGATE="$VAL_DIR/val02_aggregate.py"
RELATION="$VAL_DIR/val02_relation_quality.py"
DRIFT_ROOT="$VAL_DIR/results/drift"

slug() { echo "$1" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//'; }

# 目录后缀：pool+zero 用默认目录（与已跑结果兼容），其余组各自独立，避免互相覆盖/被误判为已完成
case_tag() {
    local state="$1" init="$2" tag=""
    [ "$state" != "pool" ] && tag="_${state}"
    [ "$init" != "zero" ] && tag="${tag}_${init}"
    echo "$tag"
}

out_root_for() {
    echo "$VAL_DIR/results/raw$(case_tag "$1" "$2")"
}
agg_out_for() {
    echo "$VAL_DIR/results/aggregated$(case_tag "$1" "$2")"
}

case "$GRAPH_STATE" in
    all) STATE_LIST="pool topk" ;;
    pool|topk) STATE_LIST="$GRAPH_STATE" ;;
    *) echo "GRAPH_STATE 只能是 pool / topk / all"; exit 1 ;;
esac

echo "=============================================================="
echo "验证实验 2（val02）一键运行"
echo "  数据集   : $DATASETS"
echo "  rho      : $RHOS"
echo "  种子     : $SEEDS"
echo "  窗口/轮数: $NUM_WINDOWS 窗口 x $ROUNDS 轮"
echo "  图状态   : $STATE_LIST（pool=修复版 / topk=旧冻结行为）"
echo "  新边初值 : $NEW_EDGE_INIT | 参与率: $PARTICIPATION | 设备: $DEVICE"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 环境自检："失败继续"的代价是环境不对时会一路跑到底，所以先确认依赖存在
# ---------------------------------------------------------------------------
echo "=== 环境自检：$PYTHON ==="
if ! "$PYTHON" -c "import sys, numpy, pandas, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"; then
    echo "环境自检失败：这个 Python 里缺少 numpy / pandas / torch。请把 PYTHON 指向跑实验用的解释器。" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# ① 离线漂移识别
# ---------------------------------------------------------------------------
IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
RUN_DATASETS=()
SKIPPED=()
for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    COHORT="$DRIFT_ROOT/$S/val02_drift_users.csv"
    if [ -f "$COHORT" ]; then
        echo "[skip] cohort 已存在：$COHORT"
    else
        echo ""
        echo "=== ① 漂移识别：$DATASET ==="
        if ! "$PYTHON" "$DETECT" --dataset "$DATASET" --num-windows "$NUM_WINDOWS"; then
            echo "[skip] 漂移识别失败：$DATASET"; SKIPPED+=("$DATASET"); continue
        fi
    fi
    if [ -f "$COHORT" ]; then
        N_DRIFT=$(awk -F, 'NR>1 && $3=="drift"' "$COHORT" | wc -l | tr -d ' ')
        echo "[cohort] $DATASET : drift = $N_DRIFT 人"
        if [ "$N_DRIFT" -lt "$MIN_DRIFT_COHORT" ]; then
            echo "[skip] $DATASET 的 drift cohort 只有 $N_DRIFT 人（< $MIN_DRIFT_COHORT），跳过"
            SKIPPED+=("$DATASET")
            continue
        fi
        RUN_DATASETS+=("$DATASET")
    else
        echo "[skip] $DATASET 没有生成 cohort，跳过"
        SKIPPED+=("$DATASET")
    fi
done

if [ ${#RUN_DATASETS[@]} -eq 0 ]; then
    echo "所有数据集的 cohort 都不可用，无法运行实验二。" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# ② 训练
# ---------------------------------------------------------------------------
START=$(date +%s)
FAILED=()
for STRENGTH in $STATE_LIST; do
    OUT_ROOT="$(out_root_for "$STRENGTH" "$NEW_EDGE_INIT")"
    for DATASET in "${RUN_DATASETS[@]}"; do
        S="$(slug "$DATASET")"
        for SEED in $SEEDS; do
            for RHO in $RHOS; do
                RHOSTR="$(printf '%.2f' "$RHO")"
                DONE_FILE="$OUT_ROOT/$S/rho$RHOSTR/seed_$SEED/val02_summary.csv"
                if [ -f "$DONE_FILE" ]; then
                    echo "[skip] $DATASET | $STRENGTH | rho=$RHOSTR | seed=$SEED 已完成"
                    continue
                fi
                echo ""
                echo "=== $DATASET | $STRENGTH | rho=$RHOSTR | seed=$SEED ==="
                if ! "$PYTHON" "$RUNNER" \
                    --dataset "$DATASET" \
                    --rho "$RHOSTR" \
                    --seed "$SEED" \
                    --num-windows "$NUM_WINDOWS" \
                    --rounds-per-window "$ROUNDS" \
                    --participation-rate "$PARTICIPATION" \
                    --graph-state "$STRENGTH" \
                    --graph-new-edge "$NEW_EDGE_INIT" \
                    --device "$DEVICE" \
                    --out-root "$OUT_ROOT"; then
                    echo "[warn] 运行失败：$DATASET | $STRENGTH | rho=$RHOSTR | seed=$SEED（继续）"
                    FAILED+=("$DATASET|$STRENGTH|rho$RHOSTR|seed$SEED")
                fi
            done
        done
    done
done

# ---------------------------------------------------------------------------
# ③ 自动聚合
# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=============================================================="
echo "训练阶段结束，用时 $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
[ ${#SKIPPED[@]} -gt 0 ] && echo "跳过的数据集: ${SKIPPED[*]}"
[ ${#FAILED[@]} -gt 0 ] && echo "失败的运行: ${FAILED[*]}"

if [ "$NO_AGGREGATE" != "1" ]; then
    for STRENGTH in $STATE_LIST; do
        OUT_ROOT="$(out_root_for "$STRENGTH" "$NEW_EDGE_INIT")"
        AGG_OUT="$(agg_out_for "$STRENGTH" "$NEW_EDGE_INIT")"
        if ! ls "$OUT_ROOT"/*/*/*/val02_summary.csv >/dev/null 2>&1; then
            echo "[skip] $STRENGTH 没有可用结果，跳过聚合"
            continue
        fi
        echo ""
        echo "=== ③ 聚合（$STRENGTH / new-edge-init=$NEW_EDGE_INIT）-> $AGG_OUT ==="
        "$PYTHON" "$AGGREGATE" --root "$OUT_ROOT" --out "$AGG_OUT" || echo "[warn] 聚合失败：$STRENGTH"
        echo "--- 关系质量分析（邻居兴趣一致度 vs 随机基线）---"
        "$PYTHON" "$RELATION" --root "$OUT_ROOT" --out "$AGG_OUT" || echo "[warn] 关系质量分析失败：$STRENGTH"
    done
fi

echo ""
echo "=============================================================="
echo "结果位置（本组：$GRAPH_STATE / new-edge-init=$NEW_EDGE_INIT）："
echo "  $(agg_out_for "$GRAPH_STATE" "$NEW_EDGE_INIT")/figures/val02_fig_ema_recovery.png   <- 任务书要的那张恢复曲线"
echo "  $(agg_out_for "$GRAPH_STATE" "$NEW_EDGE_INIT")/figures/val02_fig_edge_adjust.png"
echo "  $(agg_out_for "$GRAPH_STATE" "$NEW_EDGE_INIT")/tables/val02_lag_table.csv"
echo "  $(agg_out_for "$GRAPH_STATE" "$NEW_EDGE_INIT")/tables/val02_verdict.csv"
echo "  $(agg_out_for "$GRAPH_STATE" "$NEW_EDGE_INIT")/val02_report.md"
echo ""
echo "可选的后续对照（各自独立目录，不会覆盖已有结果，已完成的会跳过）："
echo "  GRAPH_STATE=topk ...            复现旧行为（只存 Top-K → 图冻结），用于对照"
echo "  NEW_EDGE_INIT=warm ...          新边首次观测直接取 Δe（会让图更灵敏）"
echo "  DEVICE=cuda ...                 用 GPU 跑"
echo "  DATASETS=\"movielens-1m\" ...     只跑一个数据集"
echo "=============================================================="
