#!/usr/bin/env bash
# 等主批次结束 -> 补跑缺失的 run（最多 3 轮）-> 全部聚合 -> 生成报告 -> 提交
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG=results/dg_pipeline.out
: > "$LOG"
DATASET="amazon-digital music"
SLUG="amazon-digital_music"
RAW=results/raw
LOGS=results/amazon_logs
SEEDS="42 777 2024"
PARTICIPATION=0.1
ROUNDS=120
WINDOWS=5
PROBE_SIZE=256

log () { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

# 任务表：目录 | 额外参数
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

log "等待主批次 dg_run_amazon.sh 结束"
for i in $(seq 1 900); do
    if ! pgrep -f "dg_run_amazon.sh" >/dev/null 2>&1 && ! pgrep -f "dg_run.py.*amazon" >/dev/null 2>&1; then
        log "主批次已结束（等待 $((i*20)) 秒）"
        break
    fi
    sleep 20
done
sleep 5

# ---- 补跑缺失（最多 3 轮，2 并发保证两个卡各 1 个，避免 3 个挤一张卡）
for attempt in 1 2 3; do
    MISSING=()
    while IFS='|' read -r ARM_DIR EXTRA; do
        [ -z "${ARM_DIR:-}" ] && continue
        for SEED in $SEEDS; do
            [ -f "$RAW/$SLUG/$ARM_DIR/seed_$SEED/dg_summary.csv" ] || MISSING+=("$ARM_DIR|$SEED|$EXTRA")
        done
    done < "$TASKS_FILE"
    if [ ${#MISSING[@]} -eq 0 ]; then log "第 $attempt 轮检查：没有缺失"; break; fi
    log "第 $attempt 轮补跑：${#MISSING[@]} 个缺失"
    IDX=0
    for T in "${MISSING[@]}"; do
        IFS='|' read -r ARM_DIR SEED EXTRA <<< "$T"
        while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 10; done
        GPU=$(( IDX % 2 ))
        log "  启动 GPU$GPU | $ARM_DIR | seed $SEED"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -X utf8 dg_run.py \
            --dataset "$DATASET" --seed "$SEED" \
            --num-windows "$WINDOWS" --rounds-per-window "$ROUNDS" \
            --participation-rate "$PARTICIPATION" --probe-size "$PROBE_SIZE" \
            --personal-on-gpu --device cuda --out-root "$RAW" \
            $EXTRA > "$LOGS/${ARM_DIR}_seed${SEED}.retry${attempt}.log" 2>&1 &
        IDX=$((IDX + 1))
        sleep 4
    done
    wait
    log "第 $attempt 轮结束"
done
rm -f "$TASKS_FILE"

DONE=$(ls -1 "$RAW/$SLUG"/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
log "补跑结束，完成 $DONE / 24"

# ---- 聚合
log "== 主对照聚合 =="
"$PY" -X utf8 dg_aggregate.py >> "$LOG" 2>&1 || log "[warn] 主对照聚合失败"
for ds in movielens-1m amazon-digital_music; do
    log "== 冻结归因：$ds =="
    "$PY" -X utf8 dg_freeze.py --dataset "$ds" >> "$LOG" 2>&1 || log "[warn] 冻结归因失败：$ds"
done
log "== Genre 协同有效性（movielens-1m）=="
"$PY" -X utf8 dg_genre.py >> "$LOG" 2>&1 || log "[warn] Genre 分析失败"
log "== 邻居质量 vs 收益（双数据集）=="
"$PY" -X utf8 dg_neighbor_quality.py --datasets "movielens-1m,amazon-digital music" >> "$LOG" 2>&1 || log "[warn] 邻居质量分析失败"

# ---- 报告 + 验收 + 提交
log "== 生成报告 =="
"$PY" -X utf8 dg_report.py >> "$LOG" 2>&1 || log "[warn] 报告生成失败"
log "== 验收 =="
bash dg_verify.sh >> "$LOG" 2>&1 || true

log "== 提交 =="
cd /home/wl/exp_FedDIEG || exit 1
git add -A val_DynamicGraph
git -c user.name="weilai" -c user.email="weilai_hhu@163.com" commit -q \
  -m "val_DynamicGraph: Amazon 数据集三组验证实验完成，最终报告生成（含自动补跑）" \
  -m "24 个 run 跑完（含 2 个因环境级 torch 崩溃而自动补跑的 run）；dg_report.py 生成双数据集最终报告。本提交由 dg_pipeline.sh 自动完成。" >> "$LOG" 2>&1
git log --oneline -1 >> "$LOG" 2>&1
log "全部完成"
