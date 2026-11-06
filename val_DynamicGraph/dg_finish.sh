#!/usr/bin/env bash
# 等训练结束 -> 把临时命名的日志/输出改成正式名 -> 跑冻结归因与 genre 两个分析。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG=results/dg_finish.out
: > "$LOG"

echo "等待训练结束 $(date '+%H:%M:%S')" >> "$LOG"
for i in $(seq 1 480); do
    if ! pgrep -f "val_DynamicGraph/dg_run.py" >/dev/null 2>&1; then
        echo "训练已结束 $(date '+%H:%M:%S')（等待 $((i*30)) 秒）" >> "$LOG"
        break
    fi
    sleep 30
done
sleep 15

# ---- 收尾：旧名字的日志/输出改成正式名；运行器内容里的旧名字也改掉（此时它已不在运行）
[ -d results/q23_logs ] && mv results/q23_logs results/supplement_logs && echo "日志目录已改名" >> "$LOG"
[ -f results/q23_run.out ] && mv results/q23_run.out results/dg_run_supplement.out && echo "运行输出已改名" >> "$LOG"
if [ -f dg_run_supplement.sh ]; then
    sed -i \
        -e 's/q23_logs/supplement_logs/g' \
        -e 's/q23_run\.out/dg_run_supplement.out/g' \
        -e 's/DO_Q2/DO_FREEZE/g' \
        -e 's/DO_Q3/DO_GENRE/g' \
        -e 's/Q2（冻结归因）+ Q3（genre 邻居有效性）/冻结归因 + Genre 协同有效性/g' \
        dg_run_supplement.sh
    echo "运行器内容已改名" >> "$LOG"
fi

echo "" >> "$LOG"
echo "=========== 冻结归因分析 ===========" >> "$LOG"
"$PY" -X utf8 dg_freeze.py >> "$LOG" 2>&1 || echo "[warn] 冻结归因分析失败" >> "$LOG"
echo "" >> "$LOG"
echo "=========== Genre 协同有效性分析 ===========" >> "$LOG"
"$PY" -X utf8 dg_genre.py >> "$LOG" 2>&1 || echo "[warn] Genre 分析失败" >> "$LOG"
echo "" >> "$LOG"
echo "全部完成 $(date '+%H:%M:%S')" >> "$LOG"
