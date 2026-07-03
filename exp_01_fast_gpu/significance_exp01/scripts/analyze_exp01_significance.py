from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Sequence, Tuple


DATASETS = [
    "movielens-1m",
    "amazon-digital_music",
    "amazon-video_games",
    "amazon-arts_crafts_and_sewing",
]
DATASET_LABELS = {
    "movielens-1m": "MovieLens-1M",
    "amazon-digital_music": "Digital Music",
    "amazon-video_games": "Video Games",
    "amazon-arts_crafts_and_sewing": "Arts Crafts and Sewing",
}
METHODS = ["FedMF", "FedSASRec", "GPFedRec", "FCUCR", "TarDGR", "F3CRec", "DGFedRS", "FedDIEG"]
BASELINES = [method for method in METHODS if method != "FedDIEG"]
METRICS = ["best_test_ndcg", "best_test_hr", "best_test_recall"]
PRIMARY_METRIC = "best_test_ndcg"
SEEDS = [42, 2024, 3407, 1234, 777]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def normal_cdf(x: float) -> float:
    return NormalDist().cdf(x)


def paired_ttest(diff: Sequence[float]) -> Tuple[float, float]:
    clean = [float(value) for value in diff if math.isfinite(float(value))]
    n = len(clean)
    if n < 2:
        return float("nan"), float("nan")
    mu = mean(clean)
    sd = std(clean)
    if sd == 0:
        return math.inf if mu > 0 else (-math.inf if mu < 0 else 0.0), 0.0 if mu != 0 else 1.0
    t_stat = mu / (sd / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.ttest_1samp(clean, popmean=0.0, alternative="greater").pvalue)
    except Exception:
        # Normal approximation fallback. With n=5 this is conservative enough
        # for screening, while scipy should be used for final reporting.
        p_value = 1.0 - normal_cdf(t_stat)
    return float(t_stat), float(p_value)


def wilcoxon_signed_rank(diff: Sequence[float]) -> Tuple[float, float]:
    clean = [float(value) for value in diff if math.isfinite(float(value)) and abs(float(value)) > 1e-12]
    n = len(clean)
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy import stats  # type: ignore

        result = stats.wilcoxon(clean, alternative="greater", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        # Exact sign-flip permutation on paired differences.
        observed = sum(clean)
        total = 0
        ge = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            signed_sum = sum(sign * value for sign, value in zip(signs, clean))
            if signed_sum >= observed - 1e-12:
                ge += 1
        return observed, ge / total


def holm_adjust(rows: List[Dict[str, object]], p_key: str, out_key: str) -> None:
    indexed = [(idx, float(row[p_key])) for idx, row in enumerate(rows) if math.isfinite(float(row[p_key]))]
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running = 0.0
    adjusted = {}
    for rank, (idx, p_value) in enumerate(indexed):
        value = min((m - rank) * p_value, 1.0)
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    for idx, row in enumerate(rows):
        row[out_key] = adjusted.get(idx, "")


def significance_marker(p_value: object) -> str:
    try:
        p = float(p_value)
    except Exception:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_mean_std(values: Sequence[float], marker: str = "") -> str:
    return f"{mean(values):.4f} ± {std(values):.4f}{marker}"


def read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def collect_results(results_root: Path, datasets: Sequence[str], seeds: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            summary_path = results_root / dataset / f"seed_{seed}" / "summary.csv"
            if not summary_path.exists():
                print(f"[WARN] missing summary: {summary_path}")
                continue
            for row in read_summary(summary_path):
                method = row.get("method", "")
                if method not in METHODS:
                    continue
                record: Dict[str, object] = {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "seed": int(row.get("seed", seed)),
                    "method": method,
                }
                for metric in METRICS:
                    record[metric] = float(row[metric])
                rows.append(record)
    return rows


def index_values(rows: Sequence[Dict[str, object]]) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    values: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["method"]))
        values[key] = {metric: float(row[metric]) for metric in METRICS}
    return values


def build_multiseed_summary(rows: Sequence[Dict[str, object]], datasets: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for dataset in datasets:
        for method in METHODS:
            sub = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not sub:
                continue
            result: Dict[str, object] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "method": method,
                "n": len(sub),
            }
            for metric in METRICS:
                vals = [float(row[metric]) for row in sub]
                result[f"{metric}_mean"] = mean(vals)
                result[f"{metric}_std"] = std(vals)
                result[f"{metric}_mean_std"] = fmt_mean_std(vals)
            out.append(result)
    return out


def best_baseline_for_dataset(summary: Sequence[Dict[str, object]], dataset: str, metric: str) -> str:
    candidates = [row for row in summary if row["dataset"] == dataset and row["method"] in BASELINES]
    if not candidates:
        return ""
    return str(max(candidates, key=lambda row: float(row[f"{metric}_mean"]))["method"])


def build_tests(rows: Sequence[Dict[str, object]], summary: Sequence[Dict[str, object]], datasets: Sequence[str]) -> List[Dict[str, object]]:
    values = index_values(rows)
    tests: List[Dict[str, object]] = []
    for dataset in datasets:
        best_baseline = best_baseline_for_dataset(summary, dataset, PRIMARY_METRIC)
        for metric in METRICS:
            for baseline in BASELINES:
                diffs = []
                paired_seeds = []
                for seed in SEEDS:
                    ours = values.get((dataset, seed, "FedDIEG"))
                    base = values.get((dataset, seed, baseline))
                    if ours is None or base is None:
                        continue
                    diffs.append(ours[metric] - base[metric])
                    paired_seeds.append(seed)
                if len(diffs) < 2:
                    continue
                t_stat, p_t = paired_ttest(diffs)
                w_stat, p_w = wilcoxon_signed_rank(diffs)
                tests.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "metric": metric,
                        "comparison": f"FedDIEG vs {baseline}",
                        "baseline": baseline,
                        "is_strongest_baseline": baseline == best_baseline if metric == PRIMARY_METRIC else "",
                        "n_pairs": len(diffs),
                        "seeds": " ".join(str(seed) for seed in paired_seeds),
                        "mean_diff": mean(diffs),
                        "std_diff": std(diffs),
                        "relative_gain_pct": 100.0 * mean(diffs) / max(mean([values[(dataset, seed, baseline)][metric] for seed in paired_seeds]), 1e-12),
                        "t_stat": t_stat,
                        "p_value_ttest": p_t,
                        "wilcoxon_stat": w_stat,
                        "p_value_wilcoxon": p_w,
                    }
                )
    for metric in METRICS:
        metric_rows = [row for row in tests if row["metric"] == metric]
        holm_adjust(metric_rows, "p_value_ttest", "p_value_ttest_holm")
        holm_adjust(metric_rows, "p_value_wilcoxon", "p_value_wilcoxon_holm")
    return tests


def build_metric_table(
    rows: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
    datasets: Sequence[str],
    metric: str,
) -> List[Dict[str, object]]:
    summary = build_multiseed_summary(rows, datasets)
    marker_by_dataset = {}
    for dataset in datasets:
        best_baseline = best_baseline_for_dataset(summary, dataset, metric)
        test = next(
            (
                row
                for row in tests
                if row["dataset"] == dataset
                and row["metric"] == metric
                and row["baseline"] == best_baseline
            ),
            None,
        )
        marker_by_dataset[dataset] = significance_marker(test.get("p_value_ttest_holm", "") if test else "")

    table: List[Dict[str, object]] = []
    for method in METHODS:
        row: Dict[str, object] = {"method": method}
        for dataset in datasets:
            entry = next((item for item in summary if item["dataset"] == dataset and item["method"] == method), None)
            if entry is None:
                row[DATASET_LABELS.get(dataset, dataset)] = ""
                continue
            vals = [float(r[metric]) for r in rows if r["dataset"] == dataset and r["method"] == method]
            marker = marker_by_dataset.get(dataset, "") if method == "FedDIEG" else ""
            row[DATASET_LABELS.get(dataset, dataset)] = fmt_mean_std(vals, marker)
        table.append(row)
    return table


def build_paper_table(rows: Sequence[Dict[str, object]], tests: Sequence[Dict[str, object]], datasets: Sequence[str]) -> List[Dict[str, object]]:
    return build_metric_table(rows, tests, datasets, PRIMARY_METRIC)


def write_dataset_level_outputs(
    results_root: Path,
    datasets: Sequence[str],
    summary: Sequence[Dict[str, object]],
    tests: Sequence[Dict[str, object]],
) -> None:
    summary_fields = [
        "dataset",
        "dataset_label",
        "method",
        "n",
        "best_test_ndcg_mean",
        "best_test_ndcg_std",
        "best_test_ndcg_mean_std",
        "best_test_hr_mean",
        "best_test_hr_std",
        "best_test_hr_mean_std",
        "best_test_recall_mean",
        "best_test_recall_std",
        "best_test_recall_mean_std",
    ]
    test_fields = [
        "dataset",
        "dataset_label",
        "metric",
        "comparison",
        "baseline",
        "is_strongest_baseline",
        "n_pairs",
        "seeds",
        "mean_diff",
        "std_diff",
        "relative_gain_pct",
        "t_stat",
        "p_value_ttest",
        "p_value_ttest_holm",
        "wilcoxon_stat",
        "p_value_wilcoxon",
        "p_value_wilcoxon_holm",
    ]
    for dataset in datasets:
        write_csv(
            results_root / dataset / "multiseed_summary.csv",
            [row for row in summary if row["dataset"] == dataset],
            summary_fields,
        )
        write_csv(
            results_root / dataset / "significance_tests.csv",
            [row for row in tests if row["dataset"] == dataset],
            test_fields,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 multi-seed significance.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_root = root / "exp_01_fast_gpu" / "significance_exp01" / "results"
    tables_root = root / "exp_01_fast_gpu" / "significance_exp01" / "tables"
    datasets = list(args.datasets)
    global SEEDS
    SEEDS = list(args.seeds)

    rows = collect_results(results_root, datasets, SEEDS)
    if not rows:
        raise SystemExit(f"No summary rows found under {results_root}")

    summary = build_multiseed_summary(rows, datasets)
    tests = build_tests(rows, summary, datasets)
    paper_table = build_paper_table(rows, tests, datasets)
    ndcg_table = build_metric_table(rows, tests, datasets, "best_test_ndcg")
    hr_table = build_metric_table(rows, tests, datasets, "best_test_hr")
    recall_table = build_metric_table(rows, tests, datasets, "best_test_recall")

    summary_fields = [
        "dataset",
        "dataset_label",
        "method",
        "n",
        "best_test_ndcg_mean",
        "best_test_ndcg_std",
        "best_test_ndcg_mean_std",
        "best_test_hr_mean",
        "best_test_hr_std",
        "best_test_hr_mean_std",
        "best_test_recall_mean",
        "best_test_recall_std",
        "best_test_recall_mean_std",
    ]
    test_fields = [
        "dataset",
        "dataset_label",
        "metric",
        "comparison",
        "baseline",
        "is_strongest_baseline",
        "n_pairs",
        "seeds",
        "mean_diff",
        "std_diff",
        "relative_gain_pct",
        "t_stat",
        "p_value_ttest",
        "p_value_ttest_holm",
        "wilcoxon_stat",
        "p_value_wilcoxon",
        "p_value_wilcoxon_holm",
    ]
    paper_fields = ["method"] + [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]

    write_csv(tables_root / "exp01_multiseed_summary.csv", summary, summary_fields)
    write_csv(tables_root / "exp01_paired_tests.csv", tests, test_fields)
    write_csv(tables_root / "exp01_paper_table.csv", paper_table, paper_fields)
    write_csv(tables_root / "exp01_ndcg_mean_std.csv", ndcg_table, paper_fields)
    write_csv(tables_root / "exp01_hr_mean_std.csv", hr_table, paper_fields)
    write_csv(tables_root / "exp01_recall_mean_std.csv", recall_table, paper_fields)
    write_dataset_level_outputs(results_root, datasets, summary, tests)
    print(f"Wrote tables to {tables_root}")


if __name__ == "__main__":
    main()
