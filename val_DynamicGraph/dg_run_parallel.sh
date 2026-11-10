#!/usr/bin/env bash
# =============================================================================
# val_DynamicGraph 并行版：多进程同时跑，把 9 次运行从 ~2.5 小时压到 ~40 分钟
#
#   bash val_DynamicGraph/dg_run_parallel.sh
#
# 为什么需要它（服务器实测，2x RTX 3090）
# --------------------------------------
# 单个进程只能把一张 3090 推到 **32% 利用率 / 113W / 452 MB**，另一张卡完全空闲 ——
# 因为训练循环是"逐客户端 batch=1 串行"，每轮 604 个客户端、每个都要把个性化层的
# 几十个小张量从 CPU 拷到 GPU（一轮上万次小拷贝），瓶颈是 launch/拷贝开销而不是算力。
#
# 实测并发扩容（nograph，2 窗口 x 20 轮）：
#   个性化层存 CPU： 1 进程 2000 ms/轮 | 2 进程并发墙钟 1.47x（单进程被拖到 2641 ms/轮）
#   个性化层存 GPU： 1 进程 1782 ms/轮 | 2 进程并发墙钟 **2.21x**（单进程反而不变慢）
#   个性化层存 GPU： 3 进程并发墙钟只有 1.89x（过度订阅，反而更差）
# => 最优配置 = **个性化层放 GPU + 每卡 2 个进程**，两张卡共 4 并发。
#
# 数值上安全吗
# ------------
# --personal-on-gpu 只改"个性化参数存在哪块内存"，不改任何计算；数值逐位不变
# （这是 A/B 版原始行为，见 feddieg_core/config.py 的注释）。但为了可追溯，
# dg_summary.csv 与聚合的严格性核对都把 personal_on_cpu 记成配置项 —— 三条臂必须一致。
#
# 常用覆盖：
#   JOBS_PER_GPU=1 bash .../dg_run_parallel.sh     # 保守：每卡 1 个（和串行一样快）
#   GPUS="0" bash .../dg_run_parallel.sh          # 只用 0 号卡
#   SEEDS="42" bash .../dg_run_parallel.sh        # 只跑一个种子
#   PERSONAL_ON_GPU=0 bash .../dg_run_parallel.sh # 回到 CPU 驻留（不推荐，见上）
# =============================================================================
set -uo pipefail

DATASETS="${DATASETS:-movielens-1m}"
SEEDS="${SEEDS:-42 777 2024}"
ARMS="${ARMS:-nograph static dynamic}"
NUM_WINDOWS="${NUM_WINDOWS:-5}"
ROUNDS="${ROUNDS:-120}"
PARTICIPATION="${PARTICIPATION:-0.1}"
RHO="${RHO:-0.2}"
NEIGHBORS="${NEIGHBORS:-20}"
GRAPH_MIX="${GRAPH_MIX:-1}"
PROBE_SIZE="${PROBE_SIZE:-256}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
PERSONAL_ON_GPU="${PERSONAL_ON_GPU:-1}"
NO_AGGREGATE="${NO_AGGREGATE:-0}"
FORCE="${FORCE:-0}"
FREEZE_ROUND="${FREEZE_ROUND:-}"
REFRESH_ROUNDS="${REFRESH_ROUNDS:-}"
BUILD_ONLY="${BUILD_ONLY:-0}"
REFRESH_PHASE="${REFRESH_PHASE:-warmup}"
GRAPH_STATE="${GRAPH_STATE:-pool}"
GRAPH_NEW_EDGE="${GRAPH_NEW_EDGE:-zero}"
RUN_TAG="${RUN_TAG:-}"
GENRE_NPZ="${GENRE_NPZ:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
GPUS="${GPUS:-0,1}"

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
# 可用环境变量覆盖输出位置（小规模试跑时避免写进正式结果目录）
RAW="${RAW_ROOT:-$VAL_DIR/results/raw}"
AGG="${AGG_ROOT:-$VAL_DIR/results/aggregated}"
LOGS="${LOG_ROOT:-$VAL_DIR/results/parallel_logs}"

slug() { echo "$1" | tr -c 'A-Za-z0-9_\-' '_' | sed 's/^_*//; s/_*$//'; }

n_gpus=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
SLOTS=$((JOBS_PER_GPU * n_gpus))

VARIANT_SUFFIX=""
[ "$GRAPH_MIX" != "1" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_mix${GRAPH_MIX}"
[ "$RHO" != "0.2" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_rho${RHO}"
[ "$NEIGHBORS" != "20" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_k${NEIGHBORS}"
[ "$GRAPH_STATE" != "pool" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_${GRAPH_STATE}"
[ "$GRAPH_NEW_EDGE" != "zero" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_${GRAPH_NEW_EDGE}"
[ -n "$RUN_TAG" ] && VARIANT_SUFFIX="${VARIANT_SUFFIX}_${RUN_TAG}"

#: 与 dg_common.arm_dir_name 保持一致：static+build_only 的目录名是 static_build<N>
arm_dir_for() {
    local arm="$1" d
    if [ "$arm" = "static" ] && [ -n "$FREEZE_ROUND" ]; then
        if [ "$BUILD_ONLY" = "1" ]; then d="static_build${FREEZE_ROUND}"; else d="static_at${FREEZE_ROUND}"; fi
    else
        d="$arm"
    fi
    if [ "$arm" = "periodic" ]; then
        if [ -n "$REFRESH_ROUNDS" ] && [ "$REFRESH_ROUNDS" != "120" ]; then d="${d}_p${REFRESH_ROUNDS}"; fi
        if [ "$REFRESH_PHASE" != "warmup" ]; then d="${d}_${REFRESH_PHASE}"; fi
    fi
    echo "${d}${VARIANT_SUFFIX}"
}

echo "=============================================================="
echo "val_DynamicGraph 并行版"
echo "  解释器   : $PYTHON"
echo "  数据集   : $DATASETS"
echo "  臂       : $ARMS"
echo "  种子     : $SEEDS"
echo "  规模     : $NUM_WINDOWS 窗口 x $ROUNDS 轮 | 参与率 $PARTICIPATION"
echo "  图       : rho=$RHO neighbors=$NEIGHBORS graph_mix=$GRAPH_MIX"
echo "  并行     : 每卡 $JOBS_PER_GPU 个进程 x ${n_gpus} 张卡（$GPUS）= $SLOTS 并发"
echo "  个性化层 : $([ "$PERSONAL_ON_GPU" = "1" ] && echo 'GPU 驻留（快）' || echo 'CPU 驻留（慢）')"
echo "  输出     : $RAW"
echo "=============================================================="

# ---- 安全检查：别和正在跑的批次抢同一个结果目录
if [ "$FORCE" != "1" ] && pgrep -f "[v]al_DynamicGraph/dg_run.py" >/dev/null 2>&1; then
    echo ""
    echo "!! 已经有 dg_run.py 在运行（可能就是你之前启动的串行批次）。" >&2
    echo "   两个批次会写同一批结果目录，互相覆盖。" >&2
    echo "   先停掉旧批次（Ctrl-C 或 kill），或者 FORCE=1 强行继续。" >&2
    echo "   查看： pgrep -af 'val_DynamicGraph/dg_run.py'" >&2
    exit 1
fi

echo "=== 环境自检 ==="
"$PYTHON" -c "import sys, torch, pandas; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" || exit 1
"$PYTHON" -c "import torch;print('  GPU 数:', torch.cuda.device_count())" || true

mkdir -p "$LOGS"

# ---- 收集任务（跳过已完成）
TASKS=()
IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
for DATASET in "${DATASET_ARR[@]}"; do
    DATASET="$(echo "$DATASET" | sed 's/^ *//; s/ *$//')"
    [ -z "$DATASET" ] && continue
    S="$(slug "$DATASET")"
    for SEED in $SEEDS; do
        for ARM in $ARMS; do
            ARM_DIR="$(arm_dir_for "$ARM")"
            if [ -f "$RAW/$S/$ARM_DIR/seed_$SEED/dg_summary.csv" ]; then
                echo "[skip] $DATASET | $ARM_DIR | seed $SEED 已完成"
                continue
            fi
            TASKS+=("$DATASET|$ARM|$SEED")
        done
    done
done

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "没有需要跑的任务。"
else
    echo ""
    echo "待跑 ${#TASKS[@]} 个任务，$SLOTS 并发 => 约 $(( (${#TASKS[@]} + SLOTS - 1) / SLOTS )) 轮。"
    echo "参考耗时：nograph 约 13~15 分钟/次，dynamic 约 25~30 分钟/次（单进程基准外推）。"
    echo ""
    T_START=$(date +%s)
    IDX=0
    for TASK in "${TASKS[@]}"; do
        IFS='|' read -r T_DS T_ARM T_SEED <<< "$TASK"
        while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 2; done

        GPU=$(echo "$GPUS" | cut -d, -f$(( (IDX % n_gpus) + 1 )))
        T_SLUG="$(slug "$T_DS")"
        LOGF="$LOGS/${T_SLUG}_${T_ARM}_seed${T_SEED}.log"
        EXTRA=()
        [ "$PERSONAL_ON_GPU" = "1" ] && EXTRA+=(--personal-on-gpu) || EXTRA+=(--personal-on-cpu)
        if [ -n "$FREEZE_ROUND" ] && [ "$T_ARM" = "static" ]; then
            EXTRA+=(--freeze-round "$FREEZE_ROUND")
        fi
        if [ -n "$REFRESH_ROUNDS" ] && [ "$T_ARM" = "periodic" ]; then
            EXTRA+=(--refresh-rounds "$REFRESH_ROUNDS")
        fi
        if [ "$BUILD_ONLY" = "1" ] && [ "$T_ARM" = "static" ]; then
            EXTRA+=(--build-only)
        fi
        if [ "$REFRESH_PHASE" != "warmup" ] && [ "$T_ARM" = "periodic" ]; then
            EXTRA+=(--refresh-phase "$REFRESH_PHASE")
        fi
        if [ "$GRAPH_STATE" != "pool" ]; then
            EXTRA+=(--graph-state "$GRAPH_STATE")
        fi
        if [ "$GRAPH_NEW_EDGE" != "zero" ]; then
            EXTRA+=(--graph-new-edge "$GRAPH_NEW_EDGE")
        fi
        if [ -n "$RUN_TAG" ]; then
            EXTRA+=(--run-tag "$RUN_TAG")
        fi
        if [ -n "$GENRE_NPZ" ]; then
            EXTRA+=(--genre-npz "$GENRE_NPZ")
        fi
        if [ -n "$EXTRA_ARGS" ]; then
            # shellcheck disable=SC2206
            EXTRA+=($EXTRA_ARGS)
        fi

        echo "[$(date '+%H:%M:%S')] 启动 GPU$GPU | $T_DS | $T_ARM | seed $T_SEED -> $LOGF"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -X utf8 "$RUNNER" \
            --dataset "$T_DS" \
            --arm "$T_ARM" \
            --seed "$T_SEED" \
            --rho "$RHO" \
            --neighbors "$NEIGHBORS" \
            --graph-mix "$GRAPH_MIX" \
            --num-windows "$NUM_WINDOWS" \
            --rounds-per-window "$ROUNDS" \
            --participation-rate "$PARTICIPATION" \
            --probe-size "$PROBE_SIZE" \
            --device cuda \
            --out-root "$RAW" \
            "${EXTRA[@]}" >"$LOGF" 2>&1 &
        IDX=$((IDX + 1))
    done
    wait
    T_END=$(date +%s)
    echo ""
    echo "全部进程结束，墙钟 $(( (T_END - T_START) / 60 )) 分 $(( (T_END - T_START) % 60 )) 秒"

    echo "--- 结果核对 ---"
    MISSING=()
    for TASK in "${TASKS[@]}"; do
        IFS='|' read -r T_DS T_ARM T_SEED <<< "$TASK"
        F="$RAW/$(slug "$T_DS")/$(arm_dir_for "$T_ARM")/seed_$T_SEED/dg_summary.csv"
        if [ -f "$F" ]; then
            echo "  [ok  ] $T_DS | $T_ARM | seed $T_SEED"
        else
            echo "  [FAIL] $T_DS | $T_ARM | seed $T_SEED（日志：$LOGS/$(slug "$T_DS")_${T_ARM}_seed${T_SEED}.log）"
            MISSING+=("$TASK")
        fi
    done
    [ ${#MISSING[@]} -gt 0 ] && echo "失败 ${#MISSING[@]} 个任务"
fi

if [ "$NO_AGGREGATE" != "1" ]; then
    if ls "$RAW"/*/*/seed_*/dg_summary.csv >/dev/null 2>&1; then
        echo ""
        echo "=== 聚合 -> $AGG ==="
        "$PYTHON" -X utf8 "$AGGREGATE" --root "$RAW" --out "$AGG" || echo "[warn] 聚合失败"
    fi
fi

echo ""
echo "=============================================================="
echo "结果："
echo "  $AGG/dg_report.md"
echo "  $AGG/tables/dg_verdict.csv"
echo "  $AGG/figures/dg_fig_arms_*.png"
echo "  逐 run 日志：$LOGS/"
echo "=============================================================="
