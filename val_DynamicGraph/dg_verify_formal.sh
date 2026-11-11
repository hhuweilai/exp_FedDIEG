#!/usr/bin/env bash
# 阶段 5 验收（正式复跑版）：核验 128 个 formal run、正式聚合表、新臂行为、报告完整性。
#
# 与 dg_verify.sh（legacy，检查 results/raw + results/aggregated）分开：
# 正式表在 results/{raw,agg}_formal 等隔离目录，legacy 目录保持不动以便「修前/修后」对照。
#
#   bash val_DynamicGraph/dg_verify_formal.sh
set -u
VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$VAL_DIR" || exit 1
PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
RAW="${RAW_ROOT:-results/raw_formal}"
AGG="${AGG_ROOT:-results/agg_formal}"
REPORT="${REPORT:-实验验证报告_动态图_2026-11-10.md}"
PASS=0; FAIL=0
ok () { echo "  [PASS] $1"; PASS=$((PASS+1)); }
no () { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "================ 1) 训练 run 数（formal 批次）================"
ML=$(ls -1 "$RAW"/movielens-1m/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
AZ=$(ls -1 "$RAW"/amazon-digital_music/*/seed_*/dg_summary.csv 2>/dev/null | wc -l)
echo "  ML: $ML 个 | Amazon: $AZ 个 | 合计 $((ML+AZ))（期望 78 + 50 = 128）"
[ "$ML" -ge 78 ] && ok "MovieLens-1M 78 个 run（6 主臂×5 + static 变体 + periodic 变体 + genre + 冻结网格 + 锚点）" \
    || no "MovieLens-1M 只完成 $ML/78"
[ "$AZ" -ge 50 ] && ok "Amazon 50 个 run（6 主臂×5 + 冻结网格）" || no "Amazon 只完成 $AZ/50"

echo
echo "================ 2) 正式聚合表 ================"
for f in "$AGG/tables/dg_arm_summary.csv" "$AGG/tables/dg_contrasts.csv" \
         "$AGG/tables/dg_contrasts_by_seed.csv" "$AGG/tables/dg_contrasts_window.csv" \
         "$AGG/tables/dg_arm_window.csv" "$AGG/tables/dg_verdict.csv" \
         "$AGG/tables/dg_diagnostics.csv" "$AGG/tables/dg_hygiene.csv" \
         "$AGG/figures/dg_fig_arms_movielens-1m.png" \
         "$AGG/figures/dg_fig_arms_amazon-digital_music.png"; do
    [ -f "$f" ] && ok "$f" || no "缺少 $f"
done

echo
echo "================ 3) 诊断表（冻结 / 邻居质量 / Genre）==============="
for f in results/freeze_formal/movielens-1m/tables/dg_freeze_cells.csv \
         results/freeze_formal/amazon-digital_music/tables/dg_freeze_cells.csv \
         results/quality_formal/movielens-1m/tables/dg_quality_verdict.csv \
         results/quality_formal/amazon-digital_music/tables/dg_quality_verdict.csv \
         results/genre_formal/tables/dg_genre_verdict.csv; do
    [ -f "$f" ] && ok "$f" || no "缺少 $f"
done

echo
echo "================ 4) 数据级严格性（Python）================"
PY_TMP=$(mktemp)
RAW="$RAW" AGG="$AGG" "$PY" -X utf8 - >"$PY_TMP" 2>&1 <<'PYEOF'
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

raw = Path(os.environ["RAW"]); agg = Path(os.environ["AGG"])
P = F = 0
def ok(m):
    global P; P += 1; print(f"  [PASS] {m}")
def no(m):
    global F; F += 1; print(f"  [FAIL] {m}")

files = sorted(raw.rglob("dg_summary.csv"))
if not files:
    no("raw_formal 下没有 dg_summary.csv"); sys.exit(0)
s = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

# 4.1 修复标记：ECR 定义 + core 版本在所有 run 上一致
def uniq(col):
    v = sorted(set(str(x) for x in s[col].dropna().unique()))
    return v
ecr = uniq("ecr_def"); ver = uniq("core_version")
ok(f"ecr_def 全为 v2：{ecr}") if ecr == ["v2_empty_to_full"] else no(f"ecr_def 不一致：{ecr}")
ok(f"core_version 唯一：{ver}") if len(ver) == 1 else no(f"core_version 有 {len(ver)} 种：{ver}")

# 4.2 主臂 5 种子齐全
#   注意：`dataset` 列存的是传给 --dataset 的原始名（Amazon 里带空格），
#   而结果目录名是 slug（下划线），两套名字都要认。
RAW_DS = {
    "movielens-1m": ["movielens-1m"],
    "amazon-digital_music": ["amazon-digital music", "amazon-digital_music"],
}
main = ["nograph", "static", "periodic", "dynamic", "onboarding", "fresh_only"]
for ds, names in RAW_DS.items():
    for arm in main:
        sub = s[s["dataset"].isin(names) & (s["arm_dir"] == arm)]
        n = sub["seed"].nunique()
        ok(f"{ds}/{arm}: {n} 个种子") if n >= 5 else no(f"{ds}/{arm} 只有 {n}/5 个种子")

# 4.3 static_build120：第 120 轮**一次性建图**后冻结（W1 前 119 轮无图，第 120 轮起边数恒定）
rows = []
for p in files:
    d = p.parent
    if "static_build120" not in str(d):
        continue
    f = d / "dg_graph_round_metrics.csv"
    if not f.exists():
        continue
    g = pd.read_csv(f).sort_values("global_round")
    pos = g[g["n_edges"] > 0]
    first = int(pos["global_round"].min()) if not pos.empty else -1
    later = g[g["global_round"] > 120]["n_edges"]
    const = bool(later.nunique() == 1) if not later.empty else False
    rows.append((str(d.parent.parent.name), int(g["seed"].iloc[0]), first, const))
if rows:
    bad1 = [r for r in rows if r[2] != 120]
    ok(f"static_build120：{len(rows)} 个 run 的首次有图轮次都是第 120 轮") if not bad1 \
        else no(f"static_build120 建图轮次异常：{bad1[:3]}")
    bad2 = [r for r in rows if not r[3]]
    ok(f"static_build120：{len(rows)} 个 run 在第 120 轮之后边数恒定（确实冻结）") if not bad2 \
        else no(f"static_build120 建图后仍在变边：{bad2[:3]}")
else:
    no("找不到 static_build120 的图轮指标")

# 4.4 冻结臂：冻结点之后边不再变化
for arm in ("static_at120", "static_build120"):
    sub = s[s["arm_dir"] == arm]
    if sub.empty:
        no(f"没有 {arm} 的 run"); continue
    bad = sub[sub["edge_changed_after_freeze"] != 0]
    ok(f"{arm}：{len(sub)} 个 run 冻结后边变化为 0") if bad.empty else no(f"{arm} 有 {len(bad)} 个 run 冻结后仍变边")
    fr = sorted(set(int(x) for x in sub["freeze_round"].dropna()))
    ok(f"{arm}：freeze_round 全为 120（{fr}）") if fr == [120] else no(f"{arm} freeze_round 异常：{fr}")
    bo = sorted(set(int(x) for x in sub["build_only"].dropna()))
    want = 1 if arm == "static_build120" else 0
    ok(f"{arm}：build_only={want}") if bo == [want] else no(f"{arm} build_only 异常：{bo}")

# 4.5 onboarding 覆盖：每个用户最终都有邻居（一有摘要就上图）
#     注意 active_coverage 是「逐轮在当窗评估用户上取比例」，早轮还没轮到上传的用户当然没边，
#     所以判定用「首次覆盖人数 / 更新次数」，「全轮平均覆盖」只作为参考值打印。
onb = s[s["arm_dir"] == "onboarding"]
if onb.empty:
    no("没有 onboarding 的 run")
else:
    nu = onb["n_users"].astype(float)
    fc = onb["n_first_covered"].astype(float)
    fu = onb["n_user_updates"].astype(float)
    ok(f"onboarding：{len(onb)} 个 run 的首次覆盖人数 == 用户总数（{int(nu.min())}）") if (fc == nu).all() \
        else no(f"onboarding 首次覆盖人数不足：{list(zip(fc.astype(int), nu.astype(int)))[:3]}")
    ok(f"onboarding：每个用户恰好建边一次（更新 {int(fu.min())}~{int(fu.max())} 次 == 用户数）") if (fu == nu).all() \
        else no(f"onboarding 更新次数不等于用户数：{list(zip(fu.astype(int), nu.astype(int)))[:3]}")
    cov = onb["mean_active_coverage"].astype(float)
    ok(f"onboarding 全轮平均覆盖 {cov.mean():.4f}（参考值：含早轮尚未上传的用户）") if cov.min() >= 0.85 \
        else no(f"onboarding 平均覆盖偏低：min={cov.min():.4f}")

# 4.6 fresh_only 与 onboarding 的触发条件确实不同
#     onboarding：只给「还没有邻居的人」建边 ⇒ 每人恰好 1 次（n_user_updates == n_users）
#     fresh_only：给「自上次更新后有新上传」的人**重估** ⇒ 同一用户被多次更新（n_user_updates ≫ n_users）
fo = s[s["arm_dir"] == "fresh_only"]
if fo.empty or onb.empty:
    no("缺少 fresh_only 或 onboarding 的 run")
else:
    nu = fo["n_users"].astype(float)
    fu = fo["n_user_updates"].astype(float)
    ok(f"fresh_only：累计更新 {int(fu.min())}~{int(fu.max())} 次 ≫ 用户数 {int(nu.min())}（重复重估，触发条件与 onboarding 不同）") \
        if (fu > nu).all() else no(f"fresh_only 的更新次数没有超过用户数：{list(zip(fu.astype(int), nu.astype(int)))[:3]}")
    cfo, con = fo["mean_active_coverage"].astype(float), onb["mean_active_coverage"].astype(float)
    ok(f"fresh_only 覆盖 {cfo.mean():.4f} vs onboarding {con.mean():.4f}（覆盖面同量级，差别在更新对象）") \
        if abs(cfo.mean() - con.mean()) < 0.15 else no(f"两条臂覆盖率差异过大：{cfo.mean():.4f} vs {con.mean():.4f}")

# 4.7 periodic_boundary：每个窗口边界刷新一次 = 5 次
pb = s[s["arm_dir"] == "periodic_boundary"]
if pb.empty:
    no("没有 periodic_boundary 的 run")
else:
    bad = []
    for p in files:
        if "periodic_boundary" not in str(p.parent):
            continue
        g = pd.read_csv(p.parent / "dg_graph_round_metrics.csv")
        n_upd = int(g["is_update"].sum()); n_ref = int(g["is_refresh"].sum())
        if n_upd != 5 or n_ref != 5:
            bad.append((str(p.parent.parent.name), int(g["seed"].iloc[0]), n_upd, n_ref))
    ok(f"periodic_boundary：{len(pb)} 个 run 均为 5 次刷新（窗口边界）") if not bad \
        else no(f"periodic_boundary 刷新次数异常：{bad[:3]}")

# 4.8 dynamic 的图确实在动（turnover > 0）
dyn = s[(s["arm_dir"] == "dynamic")]
if dyn.empty:
    no("没有 dynamic 的 run")
else:
    tv = dyn["mean_graph_turnover"].astype(float)
    ok(f"dynamic 平均 graph_turnover={tv.mean():.4f}（>0.05：图在演化）") if tv.min() > 0.05 \
        else no(f"dynamic turnover 偏低：min={tv.min():.4f}")

# 4.9 turnover 语义自查：它是「本轮被更新用户的邻居集合变化」的 Jaccard 距离。
#     onboarding 每次都是 ∅→K 的首次建边 ⇒ 必然 1.0；dynamic 是 EMA 增量 ⇒ 远小于 1。
#     （所以两者的 turnover 不能直接比大小，报告里必须写清口径）
if not onb.empty:
    tv = onb["mean_graph_turnover"].astype(float)
    ok("onboarding 的 graph_turnover ≡ 1.0（首次建边 ∅→K，符合定义）") if (abs(tv - 1.0) < 1e-6).all() \
        else no(f"onboarding turnover 不等于 1.0：{sorted(set(tv.round(4)))[:5]}")
if not dyn.empty and not onb.empty:
    ok(f"dynamic 的 turnover 明显小于 onboarding（{dyn['mean_graph_turnover'].mean():.3f} vs "
       f"{onb['mean_graph_turnover'].mean():.3f}：增量更新 vs 首次建边）") \
        if dyn["mean_graph_turnover"].mean() < 0.6 else no("dynamic 的 turnover 接近 1（不像增量更新）")

# 4.10 hygiene 全通过
h = pd.read_csv(agg / "tables" / "dg_hygiene.csv")
bad = h[h["ok"].astype(int) == 0]
if bad.empty:
    ok(f"hygiene {len(h)} 项全通过")
else:
    no(f"hygiene 有 {len(bad)} 项未通过：{bad['check'].tolist()[:5]}")

# 4.11 种子级检验列存在（B5 口径已落地）
c = pd.read_csv(agg / "tables" / "dg_contrasts.csv")
need = {"delta_seed_mean", "ci95_seed_lo", "ci95_seed_hi", "p_seed_t", "seeds_positive", "seeds_total"}
miss = need - set(c.columns)
ok("dg_contrasts.csv 含种子级统计列（B5）") if not miss else no(f"dg_contrasts.csv 缺列：{sorted(miss)}")
print(f"  -- 本节小结：PASS {P} / FAIL {F}")
sys.exit(0 if F == 0 else 1)
PYEOF
PY_RC=$?
cat "$PY_TMP"; rm -f "$PY_TMP"
if [ "$PY_RC" -eq 0 ]; then
    ok "第 4 节数据级检查全部通过"
else
    no "第 4 节数据级检查有失败项（明细见上）"
fi

echo
echo "================ 5) 报告完整性 ================"
if [ ! -f "$REPORT" ]; then
    no "报告文件不存在：$REPORT"
else
    LINES=$(wc -l < "$REPORT"); BYTES=$(stat -c %s "$REPORT")
    echo "  $REPORT：$LINES 行 / $BYTES 字节"
    for sec in "## 一、实验设置与严格性" "## 二、问题一" "## 三、问题二" "## 四、问题三" \
               "## 五、对研究问题的含义" "## 六、局限" "## 七、复现与产出清单"; do
        grep -q "^$sec" "$REPORT" && ok "章节存在：$sec" || no "缺少章节：$sec"
    done
    N_EMPTY=$(grep -c "（无数据）" "$REPORT" || true)
    [ "${N_EMPTY:-0}" -eq 0 ] && ok "没有「（无数据）」占位符" || no "有 $N_EMPTY 处「（无数据）」占位符"
    MISS=0
    for img in $(grep -oE '!\[[^]]*\]\(([^)]+)\)' "$REPORT" | sed -E 's/.*\(([^)]+)\)/\1/'); do
        [ -f "$img" ] || { echo "         缺图：$img"; MISS=$((MISS+1)); }
    done
    [ "$MISS" -eq 0 ] && ok "所有引用的图都存在" || no "$MISS 张图缺失"
    grep -q "DynamicGraph 相对 NoGraph" "$REPORT" && ok "报告含问题一的核心对比句" || no "报告缺少问题一核心对比句"
fi

echo
echo "================ 6) 是否还有训练进程 ================"
RUN=$(pgrep -cf "[v]al_DynamicGraph/dg_run.py" || true)
echo "  在跑 dg_run.py: ${RUN:-0}"
[ "${RUN:-0}" -eq 0 ] && ok "训练已全部结束" || no "还有 ${RUN} 个训练进程在跑"

echo
echo "=============================================================="
echo "验收汇总：PASS $PASS / FAIL $FAIL"
[ "$FAIL" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "还有 $FAIL 项未通过"
