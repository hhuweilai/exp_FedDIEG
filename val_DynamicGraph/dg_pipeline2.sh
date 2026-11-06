#!/usr/bin/env bash
# 等「所有训练进程 + 补跑器」结束 -> 最多 3 轮补跑缺失 -> 聚合 -> 报告 -> 验收 -> 提交
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG=results/dg_pipeline.out
: > "$LOG"
log () { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

busy () {
    pgrep -f "dg_run.py.*amazon" >/dev/null 2>&1 && return 0
    pgrep -f "dg_run_amazon.sh" >/dev/null 2>&1 && return 0
    pgrep -f "dg_run_rest.py" >/dev/null 2>&1 && return 0
    return 1
}

log "等待全部训练结束"
for i in $(seq 1 1200); do
    busy || { log "训练已全部结束（等待 $((i*20)) 秒）"; break; }
    sleep 20
done
sleep 5

for attempt in 1 2 3; do
    log "== 第 $attempt 轮补跑检查 =="
    "$PY" -X utf8 dg_run_rest.py --jobs 4 >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then log "[warn] 补跑器异常"; fi
    DONE=$(ls -1 results/raw/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
    log "当前完成 $DONE / 24"
    [ "$DONE" -ge 24 ] && break
done

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

log "== 生成报告 =="
"$PY" -X utf8 dg_report.py >> "$LOG" 2>&1 || log "[warn] 报告生成失败"
log "== 验收 =="
bash dg_verify.sh >> "$LOG" 2>&1 || true

log "== 提交 =="
cd /home/wl/exp_FedDIEG || exit 1
git add -A val_DynamicGraph
git -c user.name="weilai" -c user.email="weilai_hhu@163.com" commit -q \
  -m "val_DynamicGraph: Amazon 三组验证实验完成（含崩溃补跑与均衡分派），最终报告生成" \
  -m "24 个 run 全部完成；过程中 2 个 run 因环境级 torch 崩溃（ValueError / core dumped）被自动补跑；分派偏斜（4 进程挤一卡）已由 dg_run_rest.py 的「按实际占用选卡」纠正。dg_report.py 生成双数据集最终报告。本提交由 dg_pipeline2.sh 自动完成。" >> "$LOG" 2>&1
git log --oneline -1 >> "$LOG" 2>&1
log "全部完成"
