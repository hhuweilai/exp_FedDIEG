from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence


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
VARIANTS = ["Ours-Plain", "Ours-NoTemp", "Ours-NoStruct", "Ours-Full"]
ABLATIONS = [variant for variant in VARIANTS if variant != "Ours-Full"]
DISPLAY_VARIANTS = {
    "Ours-Full": "Full",
    "Ours-Plain": "Plain",
    "Ours-NoTemp": "w/o Temp",
    "Ours-NoStruct": "w/o Struct",
}
METRICS = [
    "avg_window_ndcg",
    "best_test_ndcg",
    "final_test_ndcg",
    "max_drop",
    "boundary_max_drop",
    "avg_boundary_drop",
    "window_variance",
    "loss_oscillation",
    "avg_ecr",
    "stability_score",
]
METRIC_LABELS = {
    "avg_window_ndcg": "Avg Window NDCG@10",
    "best_test_ndcg": "Best NDCG@10",
    "final_test_ndcg": "Final NDCG@10",
    "max_drop": "Max Drop",
    "boundary_max_drop": "Boundary Max Drop",
    "avg_boundary_drop": "Avg Boundary Drop",
    "window_variance": "Window Variance",
    "loss_oscillation": "Loss Oscillation",
    "avg_ecr": "Avg ECR",
    "stability_score": "Stability Score",
}
LOWER_IS_BETTER = {
    "max_drop",
    "boundary_max_drop",
    "avg_boundary_drop",
    "window_variance",
    "loss_oscillation",
    "avg_ecr",
}
PRIMARY_METRICS = ["avg_window_ndcg", "boundary_max_drop", "loss_oscillation", "avg_ecr", "stability_score"]
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


def paired_ttest(diff: Sequence[float]) -> tuple[float, float]:
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
        p_value = 1.0 - normal_cdf(t_stat)
    return float(t_stat), float(p_value)


def wilcoxon_signed_rank(diff: Sequence[float]) -> tuple[float, float]:
    clean = [float(value) for value in diff if math.isfinite(float(value)) and abs(float(value)) > 1e-12]
    n = len(clean)
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy import stats  # type: ignore

        result = stats.wilcoxon(clean, alternative="greater", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        observed = sum(clean)
        total = 0
        ge = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            signed_sum = sum(sign * value for sign, value in zip(signs, clean))
            if signed_sum >= observed - 1e-12:
                ge += 1
        return observed, ge / total


def holm_adjust(rows: list[dict[str, object]], p_key: str, out_key: str) -> None:
    indexed = [(idx, float(row[p_key])) for idx, row in enumerate(rows) if math.isfinite(float(row[p_key]))]
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running = 0.0
    adjusted: dict[int, float] = {}
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
    return f"{mean(values):.4f} +- {std(values):.4f}{marker}"


def display_variant(variant: str) -> str:
    return DISPLAY_VARIANTS.get(variant, variant)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def collect_results(results_root: Path, datasets: Sequence[str], seeds: Sequence[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            summary_path = results_root / dataset / f"seed_{seed}" / "exp03_summary.csv"
            if not summary_path.exists():
                print(f"[WARN] missing summary: {summary_path}")
                continue
            for row in read_summary(summary_path):
                variant = row.get("variant", "")
                if variant not in VARIANTS:
                    continue
                record: dict[str, object] = {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "seed": int(seed),
                    "variant": variant,
                    "display_variant": display_variant(variant),
                }
                for metric in METRICS:
                    if metric in row and row[metric] != "":
                        record[metric] = float(row[metric])
                    else:
                        record[metric] = float("nan")
                rows.append(record)
    return rows


def index_values(rows: Sequence[dict[str, object]]) -> dict[tuple[str, int, str], dict[str, float]]:
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["variant"]))
        values[key] = {metric: float(row[metric]) for metric in METRICS}
    return values


def build_multiseed_summary(rows: Sequence[dict[str, object]], datasets: Sequence[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for dataset in datasets:
        for variant in VARIANTS:
            sub = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            if not sub:
                continue
            result: dict[str, object] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "variant": variant,
                "display_variant": display_variant(variant),
                "n": len(sub),
            }
            for metric in METRICS:
                vals = [float(row[metric]) for row in sub if math.isfinite(float(row[metric]))]
                result[f"{metric}_mean"] = mean(vals)
                result[f"{metric}_std"] = std(vals)
                result[f"{metric}_mean_std"] = fmt_mean_std(vals)
            out.append(result)
    return out


def best_ablation_for_dataset(summary: Sequence[dict[str, object]], dataset: str, metric: str) -> str:
    candidates = [row for row in summary if row["dataset"] == dataset and row["variant"] in ABLATIONS]
    if not candidates:
        return ""
    key = f"{metric}_mean"
    if metric in LOWER_IS_BETTER:
        return str(min(candidates, key=lambda row: float(row[key]))["variant"])
    return str(max(candidates, key=lambda row: float(row[key]))["variant"])


def build_tests(
    rows: Sequence[dict[str, object]],
    summary: Sequence[dict[str, object]],
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    values = index_values(rows)
    tests: list[dict[str, object]] = []
    for dataset in datasets:
        strongest_by_metric = {metric: best_ablation_for_dataset(summary, dataset, metric) for metric in METRICS}
        for metric in METRICS:
            for ablation in ABLATIONS:
                diffs = []
                paired_seeds = []
                for seed in seeds:
                    full = values.get((dataset, seed, "Ours-Full"))
                    base = values.get((dataset, seed, ablation))
                    if full is None or base is None:
                        continue
                    raw_diff = full[metric] - base[metric]
                    diff = -raw_diff if metric in LOWER_IS_BETTER else raw_diff
                    diffs.append(diff)
                    paired_seeds.append(seed)
                if len(diffs) < 2:
                    continue
                t_stat, p_t = paired_ttest(diffs)
                w_stat, p_w = wilcoxon_signed_rank(diffs)
                ablation_mean = mean([values[(dataset, seed, ablation)][metric] for seed in paired_seeds])
                full_mean = mean([values[(dataset, seed, "Ours-Full")][metric] for seed in paired_seeds])
                signed_mean_diff = full_mean - ablation_mean
                rel_denom = max(abs(ablation_mean), 1e-12)
                tests.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "comparison": f"Ours-Full vs {ablation}",
                        "ablation": ablation,
                        "ablation_display": display_variant(ablation),
                        "is_strongest_ablation": ablation == strongest_by_metric[metric],
                        "higher_is_better_for_test": metric not in LOWER_IS_BETTER,
                        "n_pairs": len(diffs),
                        "seeds": " ".join(str(seed) for seed in paired_seeds),
                        "full_mean": full_mean,
                        "ablation_mean": ablation_mean,
                        "signed_mean_diff": signed_mean_diff,
                        "test_mean_diff": mean(diffs),
                        "test_std_diff": std(diffs),
                        "relative_change_pct": 100.0 * signed_mean_diff / rel_denom,
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
    rows: Sequence[dict[str, object]],
    tests: Sequence[dict[str, object]],
    datasets: Sequence[str],
    metric: str,
) -> list[dict[str, object]]:
    summary = build_multiseed_summary(rows, datasets)
    marker_by_dataset: dict[str, str] = {}
    for dataset in datasets:
        best_ablation = best_ablation_for_dataset(summary, dataset, metric)
        test = next(
            (
                row
                for row in tests
                if row["dataset"] == dataset and row["metric"] == metric and row["ablation"] == best_ablation
            ),
            None,
        )
        marker_by_dataset[dataset] = significance_marker(test["p_value_ttest"]) if test else ""

    table: list[dict[str, object]] = []
    for variant in VARIANTS:
        row: dict[str, object] = {"variant": variant, "display_variant": display_variant(variant)}
        for dataset in datasets:
            vals = [
                float(result[metric])
                for result in rows
                if result["dataset"] == dataset and result["variant"] == variant
            ]
            marker = marker_by_dataset.get(dataset, "") if variant == "Ours-Full" else ""
            row[DATASET_LABELS.get(dataset, dataset)] = fmt_mean_std(vals, marker) if vals else ""
        table.append(row)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp03 multi-seed ablation significance.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    base_dir = args.root / "exp_03_fast_gpu" / "significance_exp03"
    results_root = base_dir / "results"
    tables_dir = base_dir / "tables"
    datasets = list(args.datasets)
    seeds = list(args.seeds)

    rows = collect_results(results_root, datasets, seeds)
    if not rows:
        raise SystemExit("No Exp03 summary rows found.")

    summary = build_multiseed_summary(rows, datasets)
    tests = build_tests(rows, summary, datasets, seeds)

    for dataset in datasets:
        dataset_summary = [row for row in summary if row["dataset"] == dataset]
        dataset_tests = [row for row in tests if row["dataset"] == dataset]
        write_csv(results_root / dataset / "multiseed_summary.csv", dataset_summary, list(dataset_summary[0].keys()))
        if dataset_tests:
            write_csv(results_root / dataset / "significance_tests.csv", dataset_tests, list(dataset_tests[0].keys()))

    write_csv(tables_dir / "exp03_multiseed_summary.csv", summary, list(summary[0].keys()))
    if tests:
        write_csv(tables_dir / "exp03_paired_tests.csv", tests, list(tests[0].keys()))
    for metric in METRICS:
        metric_table = build_metric_table(rows, tests, datasets, metric)
        write_csv(tables_dir / f"exp03_{metric}_mean_std.csv", metric_table, list(metric_table[0].keys()))

    primary_tests = [row for row in tests if row["metric"] in PRIMARY_METRICS and row["is_strongest_ablation"]]
    if primary_tests:
        write_csv(tables_dir / "exp03_primary_paired_tests.csv", primary_tests, list(primary_tests[0].keys()))

    print(f"Wrote Exp03 significance tables to {tables_dir}")


if __name__ == "__main__":
    main()
