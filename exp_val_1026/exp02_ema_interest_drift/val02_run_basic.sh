#!/usr/bin/env bash
# 验证实验 2（val02）一键跑完（Linux / macOS / WSL）
#
#   bash exp_val_1026/exp02_ema_interest_drift/val02_run_basic.sh
#
# 默认行为（一条命令跑完所有结果）：
#   ① 对两个数据集（MovieLens-1M + amazon-digital music）做离线漂移识别
#      （稀疏数据集的 cohort 会自动放宽阈值；实在不可用的数据集自动跳过）
#   ② 4 个 rho（0 / 0.5 / 0.8 / 0.95）× 3 个随机种子（42/777/2024）× 参与率 100%
#   ③ 自动聚合并出图 → results/aggregated/
#
#   已完成的任务自动跳过，中断后重跑同一条命令即可续跑；单个任务失败不会中断整轮。
#
# 常用变体：
#   DATASETS="movielens-1m" SEEDS="42" ROUNDS=40 bash .../val02_run_basic.sh   # 先探路
#   GRAPH_STRENGTH=strong bash .../val02_run_basic.sh        # 只跑放大图影响的敏感性组
#   GRAPH_STRENGTH=all bash .../val02_run_basic.sh           # 两组都跑（约 4 小时）
#   NO_AGGREGATE=1 bash .../val02_run_basic.sh               # 只跑训练不聚合
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m,amazon-digital music}"
SEEDS="${SEEDS:-42 777 2024}"
RHOS="${RHOS:-0 0.5 0.8 0.95}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
GRAPH_STRENGTH="${GRAPH_STRENGTH:-paper}"
GATE="${GATE:-raw}"
NEW_EDGE_INIT="${NEW_EDGE_INIT:-ema}"
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

# 目录后缀：paper+ema 用默认目录（与已跑结果兼容），其余组各自独立，避免互相覆盖/被误判为已完成
case_tag() {
    local strength="$1" init="$2" tag=""
    [ "$strength" != "paper" ] && tag="_${strength}"
    [ "$init" != "ema" ] && tag="${tag}_${init}"
    echo "$tag"
}

out_root_for() {
    echo "$VAL_DIR/results/raw$(case_tag "$1" "$2")"
}
agg_out_for() {
    echo "$VAL_DIR/results/aggregated$(case_tag "$1" "$2")"
}

case "$GRAPH_STRENGTH" in
    all) STRENGTH_LIST="paper strong mix" ;;
    paper|strong|mix) STRENGTH_LIST="$GRAPH_STRENGTH" ;;
    *) echo "GRAPH_STRENGTH 只能是 paper / strong / mix / all"; exit 1 ;;
esac

echo "=============================================================="
echo "验证实验 2（val02）一键运行"
echo "  数据集   : $DATASETS"
echo "  rho      : $RHOS"
echo "  种子     : $SEEDS"
echo "  窗口/轮数: $NUM_WINDOWS 窗口 x $ROUNDS 轮"
echo "  图强度   : $STRENGTH_LIST"
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
for STRENGTH in $STRENGTH_LIST; do
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
                    --participation-rate 1.0 \
                    --graph-strength "$STRENGTH" \
                    --graph-gate "$GATE" \
                    --new-edge-init "$NEW_EDGE_INIT" \
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
    for STRENGTH in $STRENGTH_LIST; do
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
echo "结果位置（本组：$GRAPH_STRENGTH / new-edge-init=$NEW_EDGE_INIT）："
echo "  $(agg_out_for "$GRAPH_STRENGTH" "$NEW_EDGE_INIT")/figures/val02_fig_ema_recovery.png   <- 任务书要的那张恢复曲线"
echo "  $(agg_out_for "$GRAPH_STRENGTH" "$NEW_EDGE_INIT")/figures/val02_fig_edge_adjust.png"
echo "  $(agg_out_for "$GRAPH_STRENGTH" "$NEW_EDGE_INIT")/tables/val02_lag_table.csv"
echo "  $(agg_out_for "$GRAPH_STRENGTH" "$NEW_EDGE_INIT")/tables/val02_verdict.csv"
echo "  $(agg_out_for "$GRAPH_STRENGTH" "$NEW_EDGE_INIT")/val02_report.md"
echo ""
echo "可选的后续对照（各自独立目录，不会覆盖已有结果，已完成的会跳过）："
echo "  GRAPH_STRENGTH=strong ...       放大图影响 + 去掉两个可靠性门控"
echo "  GRAPH_STRENGTH=mix ...          只提高图权重、保留门控（区分图权重与门控）"
echo "  NEW_EDGE_INIT=warm ...          新边不被惩罚（检验'冻结'是否来自初始化）"
echo "  DATASETS=\"movielens-1m\" ...     只跑一个数据集"
echo "=============================================================="
