#!/usr/bin/env bash
# 一行进度：时间 | 在跑 | 已完成 | GPU | 收尾状态
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
RUN=$(pgrep -cf "dg_run.py.*amazon" 2>/dev/null | head -1)
DONE=$(ls -1 results/raw/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' '/' | sed 's|/$||')
FIN=$(grep -c "全部完成" results/dg_finish_all.out 2>/dev/null | head -1)
if [ "${FIN:-0}" -gt 0 ]; then STATE=DONE; else STATE=等待; fi
printf "%s | 在跑 %s | 完成 %s/24 | GPU %s%% | 收尾 %s\n" \
  "$(date '+%H:%M:%S')" "${RUN:-0}" "$DONE" "$GPU" "$STATE"
