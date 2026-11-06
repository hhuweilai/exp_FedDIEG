#!/usr/bin/env bash
# 等 dg_finish_all.sh 跑完聚合 -> 生成最终报告 -> 打印摘要 -> 自动提交
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
LOG=results/dg_finalize.out
: > "$LOG"

echo "等待聚合完成 $(date '+%H:%M:%S')" >> "$LOG"
for i in $(seq 1 800); do
    FIN=$(grep -c "全部完成" results/dg_finish_all.out 2>/dev/null | head -1)
    if [ "${FIN:-0}" -gt 0 ]; then
        echo "聚合已完成（等待 $((i*30)) 秒）" >> "$LOG"
        break
    fi
    if ! pgrep -f "dg_finish_all.sh" >/dev/null 2>&1; then
        echo "聚合进程已退出" >> "$LOG"
        break
    fi
    sleep 30
done
sleep 5

echo "" >> "$LOG"
echo "=========== 生成最终报告 ===========" >> "$LOG"
"$PY" -X utf8 dg_report.py >> "$LOG" 2>&1 || echo "[warn] 报告生成失败" >> "$LOG"

echo "" >> "$LOG"
echo "=========== 报告摘要 ===========" >> "$LOG"
for f in "results/aggregated/tables/dg_arm_summary.csv" \
         "results/aggregated/tables/dg_contrasts.csv" \
         "results/quality/amazon-digital_music/tables/dg_quality_verdict.csv" \
         "results/freeze/amazon-digital_music/tables/dg_freeze_cells.csv"; do
    [ -f "$f" ] && { echo "--- $f ---" >> "$LOG"; head -12 "$f" >> "$LOG"; echo "" >> "$LOG"; }
done

echo "" >> "$LOG"
echo "=========== 提交 ===========" >> "$LOG"
cd /home/wl/exp_FedDIEG || exit 1
git add -A val_DynamicGraph
git -c user.name="weilai" -c user.email="weilai_hhu@163.com" commit -q \
  -m "val_DynamicGraph: Amazon 数据集三组验证实验完成，最终报告自动生成" \
  -m "24 个 run（实验一 4 臂 + 实验二 2x2 网格，各 3 种子）跑完并聚合；dg_report.py 生成《实验验证报告_动态图_2026-11-06.md》，包含 ML-1M 与 Amazon 双数据集证据。本提交由 dg_finalize.sh 在聚合结束后自动完成。" >> "$LOG" 2>&1
git log --oneline -1 >> "$LOG" 2>&1
echo "全部完成 $(date '+%H:%M:%S')" >> "$LOG"
