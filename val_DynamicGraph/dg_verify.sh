#!/usr/bin/env bash
# 最终验收：核验 24 个 Amazon run、四个聚合产出、报告完整性。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
REPORT="实验验证报告_动态图_2026-11-06.md"
PASS=0; FAIL=0
ok () { echo "  [PASS] $1"; PASS=$((PASS+1)); }
no () { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "================ 1) 训练 run 数 ================"
ML=$(ls -1 results/raw/movielens-1m/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
AZ=$(ls -1 results/raw/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
echo "  ML: $ML 个 | Amazon: $AZ 个 | 合计 $((ML+AZ))"
[ "$ML" -ge 30 ] && ok "MovieLens-1M 至少 30 个 run（主臂 + 变体 + periodic + static_at120 + 策略臂）" || no "MovieLens-1M run 数不足（$ML < 30）"
[ "$AZ" -ge 24 ] && ok "Amazon 至少 24 个 run（主臂 + 冻结网格）" || no "Amazon 只完成 $AZ/24"

echo
echo "================ 2) 四类聚合产出 ================"
for f in "results/aggregated/tables/dg_verdict.csv" \
         "results/aggregated/tables/dg_contrasts.csv" \
         "results/aggregated/tables/dg_hygiene.csv" \
         "results/freeze/movielens-1m/tables/dg_freeze_cells.csv" \
         "results/freeze/amazon-digital_music/tables/dg_freeze_cells.csv" \
         "results/genre/tables/dg_genre_verdict.csv" \
         "results/quality/movielens-1m/tables/dg_quality_verdict.csv" \
         "results/quality/amazon-digital_music/tables/dg_quality_verdict.csv"; do
  [ -f "$f" ] && ok "$f" || no "缺少 $f"
done

echo
echo "================ 3) 严格性检查（两个数据集）================"
"$PY" - <<'PYEOF'
import pandas as pd
from pathlib import Path
p = Path("results/aggregated/tables/dg_hygiene.csv")
if not p.exists():
    print("  [FAIL] 没有 dg_hygiene.csv")
else:
    d = pd.read_csv(p)
    for ds, g in d.groupby("dataset"):
        bad = g[g["ok"] == 0]
        print(f"  [{'PASS' if bad.empty else 'FAIL'}] {ds}: {len(g)-len(bad)}/{len(g)} 项通过")
        for _, r in bad.iterrows():
            print(f"         - {r['check']}: {r['detail']}")
PYEOF

echo
echo "================ 4) 报告完整性 ================"
if [ ! -f "$REPORT" ]; then
  no "报告文件不存在：$REPORT"
else
  LINES=$(wc -l < "$REPORT")
  BYTES=$(stat -c %s "$REPORT")
  echo "  $REPORT：$LINES 行 / $BYTES 字节"
  for sec in "## 一、实验设置与严格性" "## 二、问题一" "## 三、问题二" "## 四、问题三" \
             "## 五、对研究问题的含义" "## 六、局限" "## 七、复现与产出清单"; do
    grep -q "^$sec" "$REPORT" && ok "章节存在：$sec" || no "缺少章节：$sec"
  done
  N_EMPTY=$(grep -c "（无数据）" "$REPORT" || true)
  [ "${N_EMPTY:-0}" -eq 0 ] && ok "没有「（无数据）」占位符" || no "有 $N_EMPTY 处「（无数据）」占位符"
  # 跨数据集对比表应有 2 行数据
  ROWS=$(sed -n '/### 2.3 跨数据集对比/,/### 2.4/p' "$REPORT" | grep -c "^| MovieLens\|^| Amazon" || true)
  [ "${ROWS:-0}" -ge 2 ] && ok "跨数据集对比表有 $ROWS 行（ML + Amazon）" || no "跨数据集对比表只有 ${ROWS:-0} 行"
  # 图引用是否都存在
  MISS=0
  for img in $(grep -oE '!\[[^]]*\]\(([^)]+)\)' "$REPORT" | sed -E 's/.*\(([^)]+)\)/\1/'); do
    [ -f "$img" ] || { echo "         缺图：$img"; MISS=$((MISS+1)); }
  done
  [ "$MISS" -eq 0 ] && ok "所有引用的图都存在" || no "$MISS 张图缺失"
  # 关键数字是否出现在报告里
  grep -q "DynamicGraph 相对 NoGraph" "$REPORT" && ok "报告含问题一的核心对比句" || no "报告缺少问题一核心对比句"
fi

echo
echo "================ 5) 是否还有进程在跑 ================"
RUN=$(pgrep -cf "dg_run.py.*amazon" 2>/dev/null | head -1)
echo "  在跑 dg_run.py: ${RUN:-0}"
[ "${RUN:-0}" -eq 0 ] && ok "训练已全部结束" || no "还有 ${RUN} 个训练进程在跑"

echo
echo "=============================================================="
echo "验收汇总：PASS $PASS / FAIL $FAIL"
[ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "还有 $FAIL 项未通过"
