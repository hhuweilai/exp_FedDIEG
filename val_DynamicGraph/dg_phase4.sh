#!/usr/bin/env bash
# 阶段 4：把 formal 批次的 128 个 run 聚合成「正式表」，并重出报告与三类诊断。
#
# 为什么单独一个脚本
# ------------------
# 正式复跑（F7 摘要上传时机 + F8 时间切分泄漏 修复后）的 run 全部落在
# `results/raw_formal/`，**不能**写回 `results/raw/` 或 `results/aggregated/`：
# 那些目录是 legacy（修复前）结果，阶段 5 要拿两者做「修前 / 修后」对照。
# 所以这里用隔离目录：
#   results/agg_formal/            主对照（臂汇总 / 对照 / 判定 / hygiene）
#   results/freeze_formal/         冻结 2x2 网格（ρ=0.8）
#   results/quality_formal/        无标签版问题三
#   results/genre_formal/          ML 真实 Genre（问题三）
#   实验验证报告_动态图_2026-11-10.md
#
# 用法：
#   bash val_DynamicGraph/dg_phase4.sh            # 全量（等训练全部结束后再跑）
#   bash val_DynamicGraph/dg_phase4.sh --agg-only # 只做主对照（阶段 3 冒烟检查也用这个）
#   DG_AGG_ONLY=1 bash val_DynamicGraph/dg_phase4.sh
set -uo pipefail

VAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$VAL_DIR/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/wl/.venv-feddieg/bin/python}"
RAW="${RAW_ROOT:-$VAL_DIR/results/raw_formal}"
AGG="${AGG_ROOT:-$VAL_DIR/results/agg_formal}"
FZ="${FREEZE_ROOT:-$VAL_DIR/results/freeze_formal}"
QL="${QUALITY_ROOT:-$VAL_DIR/results/quality_formal}"
GR="${GENRE_ROOT:-$VAL_DIR/results/genre_formal}"
REPORT="${REPORT:-$VAL_DIR/实验验证报告_动态图_2026-11-10.md}"

MODE="all"
for a in "$@"; do
    case "$a" in
        --agg-only) MODE="agg" ;;
        --all) MODE="all" ;;
        *) echo "未知参数：$a（用法：--agg-only | --all）" >&2; exit 2 ;;
    esac
done
[ "${DG_AGG_ONLY:-0}" = "1" ] && MODE="agg"

mkdir -p "$AGG"
N=$(find "$RAW" -name dg_summary.csv 2>/dev/null | wc -l)
echo "=============================================================="
echo "阶段 4 聚合 | raw=$RAW（$N 个已完成 run）"
echo "  模式   : $MODE"
echo "  主对照 : $AGG"
echo "=============================================================="
if [ "$N" -eq 0 ]; then
    echo "还没跑完任何 run。" >&2
    exit 1
fi

echo ""
echo "---- 1/5 主对照聚合 ----"
"$PY" -X utf8 "$VAL_DIR/dg_aggregate.py" --root "$RAW" --out "$AGG" || {
    echo "[FAIL] 主对照聚合失败" >&2; exit 1; }
echo "  -> $AGG/tables/dg_verdict.csv"

if [ "$MODE" = "agg" ]; then
    echo ""
    echo "（--agg-only：跳过诊断与报告）"
    exit 0
fi

echo ""
echo "---- 2/5 冻结归因（ρ=0.8 的 2x2 网格）----"
mkdir -p "$FZ"
"$PY" -X utf8 "$VAL_DIR/dg_freeze.py" --root "$RAW" --out "$FZ" \
    || echo "[warn] 冻结归因失败（可能网格 run 还没齐）"

echo ""
echo "---- 3/5 无标签版问题三（邻居质量）----"
mkdir -p "$QL"
"$PY" -X utf8 "$VAL_DIR/dg_neighbor_quality.py" --root "$RAW" --out "$QL" \
    --datasets "movielens-1m,amazon-digital music" \
    || echo "[warn] 邻居质量诊断失败"

echo ""
echo "---- 4/5 ML 真实 Genre 诊断 ----"
mkdir -p "$GR"
"$PY" -X utf8 "$VAL_DIR/dg_genre.py" --root "$RAW" --out "$GR" \
    || echo "[warn] Genre 诊断失败（需要 dynamic_genre 的 dg_genre_metrics.csv）"

echo ""
echo "---- 5/5 生成正式报告 ----"
"$PY" -X utf8 "$VAL_DIR/dg_report.py" \
    --agg-root "$AGG" --freeze-root "$FZ" --genre-root "$GR" \
    --quality-root "$QL" --raw-root "$RAW" --out "$REPORT" \
    || { echo "[FAIL] 报告生成失败" >&2; exit 1; }

echo ""
echo "=============================================================="
echo "阶段 4 完成："
echo "  主表   : $AGG/tables/"
echo "  报告   : $REPORT"
echo "  诊断   : $FZ  $QL  $GR"
echo "=============================================================="
