#!/usr/bin/env bash
# 打印冻结归因 / Genre 协同有效性的关键结论（训练结束后用）
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PD=/home/wl/.venv-feddieg/bin/python

echo "################ 训练批次输出（尾部）################"
tail -18 results/dg_run_supplement.out 2>/dev/null || echo "(没有 dg_run_supplement.out)"
echo
echo "################ 完成 run 数 ################"
echo "  $(ls -1 results/raw/*/*/seed_*/dg_summary.csv 2>/dev/null | wc -l) 个（25 = 15 旧 + 10 新；30 = 全部）"
echo
echo "################ 冻结归因结论 ################"
if [ -f results/freeze/tables/dg_freeze_verdict.csv ]; then
  $PD - <<'PYEOF'
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 400)
v = pd.read_csv("results/freeze/tables/dg_freeze_verdict.csv")
for _, r in v.iterrows():
    print(f"[{r['verdict_id']}] {r['question']}")
    if isinstance(r['evidence'], str) and r['evidence'].strip():
        print(f"     {r['evidence']}")
    if isinstance(r['judgement'], str) and r['judgement'].strip():
        print(f"     -> {r['judgement']}")
print()
c = pd.read_csv("results/freeze/tables/dg_freeze_cells.csv")
print(c[["cell","n_seeds","turnover_from_ref","frozen_share","ecr","added","n_edges","neighbor_lift","ndcg_win_mean"]].round(4).to_string(index=False))
PYEOF
else
  echo "  (还没有冻结归因结果)"
fi
echo
echo "################ Genre 协同有效性结论 ################"
if [ -f results/genre/tables/dg_genre_verdict.csv ]; then
  $PD - <<'PYEOF'
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 400)
v = pd.read_csv("results/genre/tables/dg_genre_verdict.csv")
for _, r in v.iterrows():
    print(f"[{r['verdict_id']}] {r['question']}")
    if isinstance(r['evidence'], str) and r['evidence'].strip():
        print(f"     {r['evidence']}")
    if isinstance(r['judgement'], str) and r['judgement'].strip():
        print(f"     -> {r['judgement']}")
print()
d = pd.read_csv("results/genre/tables/dg_genre_descriptives.csv")
print(d[["metric","n","mean","ci95_lo","ci95_hi","pos_rate"]].round(4).to_string(index=False))
print()
cr = pd.read_csv("results/genre/tables/dg_genre_correlations.csv")
print(cr[["metric","target","n","pearson","spearman","spearman_p"]].round(4).to_string(index=False))
PYEOF
else
  echo "  (还没有 Genre 协同有效性结果)"
fi
echo
echo "################ 产出文件 ################"
ls -1 results/freeze/tables/ results/freeze/figures/ results/genre/tables/ results/genre/figures/ 2>/dev/null | head -40
