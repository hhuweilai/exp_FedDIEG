"""Experiment 5: FedDIEG key hyperparameter sensitivity.

This file is a thin orchestration layer around
``exp_03_fast_gpu/run_exp03_fast_gpu.py``.  It keeps the Experiment 3
chronological-drift protocol and sweeps the four paper-level hyperparameters:

- K: feddieg_neighbors
- rho: edge_ema
- lambda_1: lambda_temp
- lambda_2: lambda_struct

The runner can either print/write the commands or execute them sequentially.
After jobs finish, it collects the four report metrics requested for Exp5:
avg_window_ndcg, max_drop, avg_ecr, and stability_score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXP3_RUNNER = ROOT / "exp_03_fast_gpu" / "run_exp03_fast_gpu.py"

DEFAULT_DATASETS = (
    "movielens-1m",
    "amazon-digital music",
    "amazon-video games",
    "amazon-arts crafts and sewing",
)

REPORT_METRICS = ("avg_window_ndcg", "max_drop", "avg_ecr", "stability_score")


@dataclass(frozen=True)
class SweepSpec:
    key: str
    symbol: str
    code_param: str
    cli_arg: str
    default: float
    values: Sequence[float]


SWEEPS = (
    SweepSpec("K", "K", "feddieg_neighbors", "--feddieg-neighbors", 10, (5, 10, 20, 40)),
    SweepSpec("rho", "rho", "edge_ema", "--edge-ema", 0.80, (0.0, 0.3, 0.5, 0.8, 0.9)),
    SweepSpec("lambda1", "lambda_1", "lambda_temp", "--lambda-temp", 0.01, (0.0, 0.005, 0.01, 0.05, 0.1)),
    SweepSpec("lambda2", "lambda_2", "lambda_struct", "--lambda-struct", 0.20, (0.0, 0.05, 0.1, 0.2, 0.4)),
)

BASE_FEDDIEG = {
    "feddieg_neighbors": 10,
    "edge_ema": 0.80,
    "lambda_temp": 0.01,
    "lambda_struct": 0.20,
    "feddieg_graph_mix": 0.05,
    "feddieg_neighbor_temp": 0.07,
    "feddieg_min_sim": 0.20,
    "feddieg_tail_power": 4.0,
    "feddieg_memory_momentum": 0.80,
    "feddieg_short_score_mix": 0.05,
    "feddieg_confidence_power": 0.5,
    "feddieg_agreement_power": 0.5,
    "feddieg_score_local_mix": 0.2,
    "feddieg_boundary_refresh_rate": 0.30,
    "feddieg_boundary_refresh_momentum": 0.20,
}

PARAM_TO_CLI = {
    "feddieg_neighbors": "--feddieg-neighbors",
    "edge_ema": "--edge-ema",
    "lambda_temp": "--lambda-temp",
    "lambda_struct": "--lambda-struct",
    "feddieg_graph_mix": "--feddieg-graph-mix",
    "feddieg_neighbor_temp": "--feddieg-neighbor-temp",
    "feddieg_min_sim": "--feddieg-min-sim",
    "feddieg_tail_power": "--feddieg-tail-power",
    "feddieg_memory_momentum": "--feddieg-memory-momentum",
    "feddieg_short_score_mix": "--feddieg-short-score-mix",
    "feddieg_confidence_power": "--feddieg-confidence-power",
    "feddieg_agreement_power": "--feddieg-agreement-power",
    "feddieg_score_local_mix": "--feddieg-score-local-mix",
    "feddieg_boundary_refresh_rate": "--feddieg-boundary-refresh-rate",
    "feddieg_boundary_refresh_momentum": "--feddieg-boundary-refresh-momentum",
}


@dataclass
class Job:
    dataset: str
    dataset_slug: str
    job_id: str
    out_dir: Path
    values: Dict[str, float]
    sweep_key: str = "default"
    symbol: str = "default"
    code_param: str = "default"
    value: float = math.nan
    is_default: bool = False


def dataset_slug(dataset: str) -> str:
    return dataset.strip().lower().replace(" ", "_").replace("-", "_")


def value_text(value: float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def value_slug(value: float) -> str:
    return value_text(value).replace("-", "m").replace(".", "p")


def as_float(value: object) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def is_same_value(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def ensure_relative_to_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize FedDIEG Exp5 sensitivity sweeps.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--base-out", type=Path, default=ROOT / "exp_05_sensitivity" / "final")
    parser.add_argument("--fig-out", type=Path, default=ROOT / "figures" / "exp05_sensitivity")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--execute", action="store_true", help="Actually run missing jobs. Without this, commands are only written.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only collect finished jobs and rebuild tables/figures.")
    parser.add_argument("--force", action="store_true", help="Rerun jobs even when exp03_summary.csv already exists.")
    parser.add_argument("--commands-file", type=Path, default=None)
    parser.add_argument("--keep-individual-plots", action="store_true")

    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--rounds-per-window", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=99)
    parser.add_argument("--window-mode", choices=["slice", "cumulative"], default="slice")
    parser.add_argument("--ecr-update-scope", choices=["sampled", "active"], default="sampled")
    parser.add_argument("--oscillation-window", type=int, default=10)
    return parser.parse_args()


def build_jobs(datasets: Sequence[str], base_out: Path) -> List[Job]:
    jobs: List[Job] = []
    for dataset in datasets:
        slug = dataset_slug(dataset)
        default_values = dict(BASE_FEDDIEG)
        jobs.append(
            Job(
                dataset=dataset,
                dataset_slug=slug,
                job_id="default",
                out_dir=base_out / slug / "default",
                values=default_values,
                value=math.nan,
                is_default=True,
            )
        )
        for sweep in SWEEPS:
            for value in sweep.values:
                if is_same_value(value, sweep.default):
                    continue
                values = dict(BASE_FEDDIEG)
                values[sweep.code_param] = value
                jobs.append(
                    Job(
                        dataset=dataset,
                        dataset_slug=slug,
                        job_id=f"{sweep.key}_{value_slug(value)}",
                        out_dir=base_out / slug / sweep.key / f"value_{value_slug(value)}",
                        values=values,
                        sweep_key=sweep.key,
                        symbol=sweep.symbol,
                        code_param=sweep.code_param,
                        value=float(value),
                    )
                )
    return jobs


def build_command(job: Job, args: argparse.Namespace) -> List[str]:
    command = [
        args.python_bin,
        str(EXP3_RUNNER),
        "--dataset",
        job.dataset,
        "--variants",
        "Ours-Full",
        "--num-windows",
        str(args.num_windows),
        "--rounds-per-window",
        str(args.rounds_per_window),
        "--window-mode",
        args.window_mode,
        "--eval-every",
        str(args.eval_every),
        "--participation-rate",
        str(args.participation_rate),
        "--microbatch-size",
        str(args.microbatch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--local-epochs",
        str(args.local_epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--num-threads",
        str(args.num_threads),
        "--candidate-count",
        str(args.candidate_count),
        "--ecr-update-scope",
        args.ecr_update_scope,
        "--oscillation-window",
        str(args.oscillation_window),
        "--out-dir",
        str(job.out_dir),
    ]
    if args.clients_per_round is not None:
        command.extend(["--clients-per-round", str(args.clients_per_round)])
    if args.max_eval_users is not None:
        command.extend(["--max-eval-users", str(args.max_eval_users)])
    if not args.keep_individual_plots:
        command.append("--no-plot")

    for code_param, value in job.values.items():
        command.extend([PARAM_TO_CLI[code_param], value_text(value)])
    return command


def command_line(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def summary_path(job: Job) -> Path:
    return job.out_dir / "exp03_summary.csv"


def read_ours_full_summary(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("variant") == "Ours-Full":
                return row
    return None


def job_is_complete(job: Job) -> bool:
    row = read_ours_full_summary(summary_path(job))
    if row is None:
        return False
    try:
        for metric in REPORT_METRICS:
            as_float(row.get(metric))
    except (TypeError, ValueError):
        return False
    return True


def write_commands(jobs: Sequence[Job], args: argparse.Namespace) -> Path:
    commands_file = args.commands_file or ensure_relative_to_root(args.base_out) / "exp05_commands.sh"
    commands_file = ensure_relative_to_root(commands_file)
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "",
    ]
    for job in jobs:
        line = command_line(build_command(job, args))
        if job_is_complete(job) and not args.force:
            lines.append(f"# already complete: {job.dataset_slug}/{job.job_id}")
            lines.append(f"# {line}")
        else:
            lines.append(line)
        lines.append("")
    commands_file.write_text("\n".join(lines), encoding="utf-8")
    return commands_file


def run_jobs(jobs: Sequence[Job], args: argparse.Namespace) -> None:
    for index, job in enumerate(jobs, start=1):
        if job_is_complete(job) and not args.force:
            print(f"[{index}/{len(jobs)}] skip complete: {job.dataset_slug}/{job.job_id}")
            continue
        print(f"[{index}/{len(jobs)}] run: {job.dataset_slug}/{job.job_id}")
        print(command_line(build_command(job, args)))
        job.out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_command(job, args), cwd=str(ROOT), check=True)


def collect_raw_rows(jobs: Sequence[Job]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for job in jobs:
        summary = read_ours_full_summary(summary_path(job))
        status = "complete" if summary is not None else "missing"
        row: Dict[str, object] = {
            "status": status,
            "dataset": job.dataset,
            "dataset_slug": job.dataset_slug,
            "job_id": job.job_id,
            "sweep_key": job.sweep_key,
            "symbol": job.symbol,
            "code_param": job.code_param,
            "value": "" if job.is_default else value_text(job.value),
            "is_default_job": int(job.is_default),
            "out_dir": str(job.out_dir),
        }
        if summary is not None:
            row.update({metric: as_float(summary.get(metric)) for metric in REPORT_METRICS})
            row["best_test_ndcg"] = as_float(summary.get("best_test_ndcg"))
            row["final_test_ndcg"] = as_float(summary.get("final_test_ndcg"))
        rows.append(row)
    return rows


def row_metric(row: Dict[str, object], metric: str) -> float:
    return as_float(row.get(metric))


def build_sensitivity_rows(raw_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    complete = [row for row in raw_rows if row.get("status") == "complete"]
    by_dataset_job = {(str(row["dataset"]), str(row["job_id"])): row for row in complete}
    rows: List[Dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in complete}):
        baseline = by_dataset_job.get((dataset, "default"))
        if baseline is None:
            continue
        for sweep in SWEEPS:
            for value in sweep.values:
                if is_same_value(value, sweep.default):
                    source = baseline
                    job_id = "default"
                    is_default = 1
                else:
                    job_id = f"{sweep.key}_{value_slug(value)}"
                    source = by_dataset_job.get((dataset, job_id))
                    is_default = 0
                if source is None:
                    continue
                row: Dict[str, object] = {
                    "dataset": dataset,
                    "dataset_slug": source.get("dataset_slug", dataset_slug(dataset)),
                    "sweep_key": sweep.key,
                    "symbol": sweep.symbol,
                    "code_param": sweep.code_param,
                    "value": value_text(value),
                    "job_id": job_id,
                    "is_default": is_default,
                }
                for metric in REPORT_METRICS:
                    current = row_metric(source, metric)
                    base = row_metric(baseline, metric)
                    row[metric] = current
                    row[f"delta_{metric}"] = current - base
                    if abs(base) > 1e-12:
                        row[f"relative_{metric}_pct"] = (current - base) / base * 100.0
                    else:
                        row[f"relative_{metric}_pct"] = ""
                rows.append(row)
    return rows


def mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else math.nan


def std(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) <= 1:
        return 0.0 if clean else math.nan
    avg = mean(clean)
    return math.sqrt(sum((value - avg) ** 2 for value in clean) / len(clean))


def build_mean_rows(sensitivity_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in sensitivity_rows:
        grouped.setdefault((str(row["sweep_key"]), str(row["value"])), []).append(row)

    rows: List[Dict[str, object]] = []
    sweep_order = {sweep.key: idx for idx, sweep in enumerate(SWEEPS)}
    value_order = {
        (sweep.key, value_text(value)): idx
        for sweep in SWEEPS
        for idx, value in enumerate(sweep.values)
    }
    for (sweep_key, value), group in grouped.items():
        first = group[0]
        row: Dict[str, object] = {
            "sweep_key": sweep_key,
            "symbol": first["symbol"],
            "code_param": first["code_param"],
            "value": value,
            "num_datasets": len(group),
        }
        for metric in REPORT_METRICS:
            vals = [row_metric(item, metric) for item in group]
            row[f"mean_{metric}"] = mean(vals)
            row[f"std_{metric}"] = std(vals)
            delta_vals = [row_metric(item, f"delta_{metric}") for item in group]
            row[f"mean_delta_{metric}"] = mean(delta_vals)
        rows.append(row)

    rows.sort(key=lambda row: (sweep_order[str(row["sweep_key"])], value_order[(str(row["sweep_key"]), str(row["value"]))]))
    return rows


def build_rank_rows(sensitivity_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sweep in SWEEPS:
        group = [row for row in sensitivity_rows if row["sweep_key"] == sweep.key]
        datasets = sorted({str(row["dataset"]) for row in group})
        dataset_ranges = []
        best_values = []
        for dataset in datasets:
            subset = [row for row in group if row["dataset"] == dataset]
            if not subset:
                continue
            ndcgs = [row_metric(row, "avg_window_ndcg") for row in subset]
            stabilities = [row_metric(row, "stability_score") for row in subset]
            default = next((row for row in subset if int(row["is_default"]) == 1), None)
            baseline = row_metric(default, "avg_window_ndcg") if default is not None else math.nan
            ndcg_range = max(ndcgs) - min(ndcgs)
            dataset_ranges.append((ndcg_range / baseline * 100.0) if math.isfinite(baseline) and baseline > 0 else math.nan)
            best = max(subset, key=lambda row: row_metric(row, "avg_window_ndcg"))
            best_values.append(str(best["value"]))
            rows.append(
                {
                    "level": "dataset",
                    "dataset": dataset,
                    "sweep_key": sweep.key,
                    "symbol": sweep.symbol,
                    "code_param": sweep.code_param,
                    "best_value_by_ndcg": best["value"],
                    "ndcg_range_pct": dataset_ranges[-1],
                    "stability_range": max(stabilities) - min(stabilities),
                }
            )
        rows.append(
            {
                "level": "overall",
                "dataset": "ALL",
                "sweep_key": sweep.key,
                "symbol": sweep.symbol,
                "code_param": sweep.code_param,
                "best_value_by_ndcg": ",".join(best_values),
                "ndcg_range_pct": mean(dataset_ranges),
                "stability_range": "",
            }
        )
    return rows


def make_metric_plot(rows: Sequence[Dict[str, object]], metric: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(DEFAULT_DATASETS)
    present_datasets = sorted({str(row["dataset"]) for row in rows})
    datasets = [dataset for dataset in datasets if dataset in present_datasets] + [
        dataset for dataset in present_datasets if dataset not in DEFAULT_DATASETS
    ]
    colors = {
        "movielens-1m": "#4c78a8",
        "amazon-digital music": "#f58518",
        "amazon-video games": "#54a24b",
        "amazon-arts crafts and sewing": "#b279a2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    axes_flat = list(axes.flat)
    for ax, sweep in zip(axes_flat, SWEEPS):
        labels = [value_text(value) for value in sweep.values]
        xs = list(range(len(labels)))
        for dataset in datasets:
            dataset_rows = [
                row
                for row in rows
                if row["dataset"] == dataset and row["sweep_key"] == sweep.key
            ]
            by_value = {str(row["value"]): row_metric(row, metric) for row in dataset_rows}
            ys = [by_value.get(label, math.nan) for label in labels]
            if all(not math.isfinite(value) for value in ys):
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=dataset, color=colors.get(dataset))
        mean_ys = []
        for label in labels:
            vals = [
                row_metric(row, metric)
                for row in rows
                if row["sweep_key"] == sweep.key and str(row["value"]) == label
            ]
            mean_ys.append(mean(vals))
        if any(math.isfinite(value) for value in mean_ys):
            ax.plot(xs, mean_ys, marker="s", linewidth=2.4, linestyle="--", color="#222222", label="mean")
        ax.axvline(labels.index(value_text(sweep.default)), color="#888888", linestyle=":", linewidth=1.2)
        ax.set_title(f"{sweep.symbol} ({sweep.code_param})")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel(sweep.symbol)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.28)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def make_plots(rows: Sequence[Dict[str, object]], fig_out: Path) -> None:
    try:
        make_metric_plot(rows, "avg_window_ndcg", "Avg Window NDCG@10", fig_out / "fig8_exp05_avg_window_ndcg.png")
        make_metric_plot(rows, "stability_score", "Stability Score", fig_out / "fig9_exp05_stability_score.png")
        make_metric_plot(rows, "max_drop", "Max Drop", fig_out / "fig10_exp05_max_drop.png")
        make_metric_plot(rows, "avg_ecr", "Average ECR", fig_out / "fig11_exp05_avg_ecr.png")
    except Exception as exc:  # pragma: no cover - plotting should not block aggregation.
        print(f"[warn] plotting skipped: {exc}")


def aggregate(jobs: Sequence[Job], base_out: Path, fig_out: Path) -> None:
    raw_rows = collect_raw_rows(jobs)
    raw_fields = [
        "status",
        "dataset",
        "dataset_slug",
        "job_id",
        "sweep_key",
        "symbol",
        "code_param",
        "value",
        "is_default_job",
        *REPORT_METRICS,
        "best_test_ndcg",
        "final_test_ndcg",
        "out_dir",
    ]
    write_csv(base_out / "exp05_raw_summary.csv", raw_rows, raw_fields)

    missing_rows = [row for row in raw_rows if row["status"] != "complete"]
    write_csv(
        base_out / "exp05_missing_jobs.csv",
        missing_rows,
        ["dataset", "dataset_slug", "job_id", "sweep_key", "code_param", "value", "out_dir"],
    )

    sensitivity_rows = build_sensitivity_rows(raw_rows)
    sensitivity_fields = [
        "dataset",
        "dataset_slug",
        "sweep_key",
        "symbol",
        "code_param",
        "value",
        "job_id",
        "is_default",
    ]
    for metric in REPORT_METRICS:
        sensitivity_fields.extend([metric, f"delta_{metric}", f"relative_{metric}_pct"])
    write_csv(base_out / "exp05_sensitivity_summary.csv", sensitivity_rows, sensitivity_fields)

    mean_rows = build_mean_rows(sensitivity_rows)
    mean_fields = ["sweep_key", "symbol", "code_param", "value", "num_datasets"]
    for metric in REPORT_METRICS:
        mean_fields.extend([f"mean_{metric}", f"std_{metric}", f"mean_delta_{metric}"])
    write_csv(base_out / "exp05_mean_by_value.csv", mean_rows, mean_fields)

    rank_rows = build_rank_rows(sensitivity_rows)
    write_csv(
        base_out / "exp05_sensitivity_rank.csv",
        rank_rows,
        ["level", "dataset", "sweep_key", "symbol", "code_param", "best_value_by_ndcg", "ndcg_range_pct", "stability_range"],
    )

    table_payload = {
        "metrics": list(REPORT_METRICS),
        "default": {sweep.key: {"symbol": sweep.symbol, "code_param": sweep.code_param, "value": sweep.default} for sweep in SWEEPS},
        "rows": sensitivity_rows,
        "mean_by_value": mean_rows,
    }
    fig_out.mkdir(parents=True, exist_ok=True)
    (fig_out / "table7_exp05_sensitivity_values.json").write_text(
        json.dumps(table_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_plots(sensitivity_rows, fig_out)

    complete = sum(1 for row in raw_rows if row["status"] == "complete")
    print(f"Aggregated {complete}/{len(raw_rows)} completed Exp5 jobs.")
    print(f"Tables: {base_out}")
    print(f"Figures: {fig_out}")


def main() -> None:
    args = parse_args()
    args.base_out = ensure_relative_to_root(args.base_out)
    args.fig_out = ensure_relative_to_root(args.fig_out)
    jobs = build_jobs(args.datasets, args.base_out)

    if not args.aggregate_only:
        commands_file = write_commands(jobs, args)
        print(f"Command file written: {commands_file}")
        if args.execute:
            run_jobs(jobs, args)
        else:
            print("Dry run only. Add --execute to run these jobs.")
            for job in jobs[:5]:
                print(command_line(build_command(job, args)))
            if len(jobs) > 5:
                print(f"... {len(jobs) - 5} more commands in {commands_file}")

    if args.execute or args.aggregate_only:
        aggregate(jobs, args.base_out, args.fig_out)


if __name__ == "__main__":
    main()
