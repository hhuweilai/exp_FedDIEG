#!/usr/bin/env bash
# 等训练结束 -> 跑全部聚合（主对照 / 冻结归因 / Genre / 邻居质量）
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG=results/dg_finish_all.out
: > "$LOG"

DATASETS="movielens-1m amazon-digital_music"

echo "等待训练结束 $(date '+%H:%M:%S')" >> "$LOG"
for i in $(seq 1 600); do
    if ! pgrep -f "val_DynamicGraph/dg_run.py" >/dev/null 2>&1; then
        echo "训练已结束 $(date '+%H:%M:%S')（等待 $((i*30)) 秒）" >> "$LOG"
        break
    fi
    sleep 30
done
sleep 20

run () { echo "" >> "$LOG"; echo "=========== $1 ===========" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1 || echo "[warn] 失败" >> "$LOG"; }

run "主对照聚合（NoGraph/Static/Periodic/Dynamic，双数据集）" "$PY" -X utf8 dg_aggregate.py
for ds in $DATASETS; do
    run "冻结归因（$ds）" "$PY" -X utf8 dg_freeze.py --dataset "$ds"
done
run "Genre 协同有效性（movielens-1m，真实 18 类）" "$PY" -X utf8 dg_genre.py
run "邻居质量 vs 收益（无标签版，双数据集）" "$PY" -X utf8 dg_neighbor_quality.py --datasets "movielens-1m,amazon-digital music"

echo "" >> "$LOG"
echo "全部完成 $(date '+%H:%M:%S')" >> "$LOG"
