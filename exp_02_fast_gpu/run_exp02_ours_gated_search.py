from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUNNER = ROOT / "exp_02_fast_gpu" / "run_exp02_fast_gpu.py"


DATASETS = {
    "digital_music": {
        "name": "amazon-digital music",
        "base": ["--lr", "0.0012", "--feddieg-neighbors", "10", "--feddieg-warmup-rounds", "80", "--feddieg-neighbor-temp", "0.07", "--feddieg-min-sim", "0.20", "--feddieg-tail-power", "4.0", "--feddieg-memory-momentum", "0.80"],
        "graph_mix": ["0.02", "0.03", "0.05"],
        "final_loss": ["0.0", "0.05"],
        "refresh": ["0.0", "0.1", "0.2"],
    },
    "movielens_1m": {
        "name": "movielens-1m",
        "base": ["--lr", "0.0012", "--feddieg-neighbors", "10", "--feddieg-warmup-rounds", "100", "--feddieg-neighbor-temp", "0.07", "--feddieg-min-sim", "0.20", "--feddieg-tail-power", "4.0", "--feddieg-memory-momentum", "0.80"],
        "graph_mix": ["0.02", "0.03", "0.05"],
        "final_loss": ["0.0"],
        "refresh": ["0.0", "0.1", "0.2"],
    },
    "video_games": {
        "name": "amazon-video games",
        "base": ["--lr", "0.0012", "--feddieg-neighbors", "10", "--feddieg-warmup-rounds", "80", "--feddieg-neighbor-temp", "0.07", "--feddieg-min-sim", "0.20", "--feddieg-tail-power", "4.0", "--feddieg-memory-momentum", "0.80"],
        "graph_mix": ["0.0", "0.01", "0.02", "0.03"],
        "final_loss": ["0.0"],
        "refresh": ["0.0", "0.1", "0.2"],
    },
    "arts_crafts_sewing": {
        "name": "amazon-arts crafts and sewing",
        "base": ["--lr", "0.0012", "--feddieg-neighbors", "8", "--feddieg-warmup-rounds", "100", "--feddieg-neighbor-temp", "0.08", "--feddieg-min-sim", "0.30", "--feddieg-tail-power", "4.0", "--feddieg-memory-momentum", "0.85"],
        "graph_mix": ["0.01", "0.02", "0.03", "0.05"],
        "final_loss": ["0.0"],
        "refresh": ["0.0", "0.1", "0.2"],
    },
}


def read_one_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))


def read_baselines(dataset_key: str) -> dict[str, float]:
    path = ROOT / "exp_02_fast_gpu" / "final" / dataset_key / "exp02_summary.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    baselines = [row for row in rows if row["method"] != "FedDIEG"]
    return {
        "avg_window_ndcg": max(float(row["avg_window_ndcg"]) for row in baselines),
        "average_adaptation_gain": max(float(row["average_adaptation_gain"]) for row in baselines),
        "average_boundary_drop": min(float(row["average_boundary_drop"]) for row in baselines),
    }


def compare_to_baseline(avg: float, adapt: float, drop: float, baseline: dict[str, float]) -> tuple[bool, float]:
    avg_gap = avg - baseline["avg_window_ndcg"]
    adapt_gap = adapt - baseline["average_adaptation_gain"]
    drop_gap = baseline["average_boundary_drop"] - drop
    pass_all = avg_gap > 0 and adapt_gap > 0 and drop_gap > 0
    # Boundary drop is usually smaller in scale; weight it enough to surface robust candidates.
    score = avg_gap + adapt_gap + 1.5 * drop_gap
    return pass_all, score


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def has_invalid_numbers(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(token in text for token in ("loss nan", ",nan", " nan", "inf"))


def candidate_grid(spec: dict[str, object]) -> list[tuple[str, list[str]]]:
    grids = []
    confidence_values = ["0.0", "0.5", "1.0"]
    agreement_values = ["0.0", "0.5"]
    score_values = ["0.0"]
    local_score_values = ["0.0", "0.2"]
    refresh_values = spec.get("refresh", ["0.0"])
    short_score_values = ["0.0", "0.05"]
    for graph_mix, final_loss, conf, agree, residual, local_score, refresh, short_score in itertools.product(
        spec["graph_mix"],
        spec["final_loss"],
        confidence_values,
        agreement_values,
        score_values,
        local_score_values,
        refresh_values,
        short_score_values,
    ):
        name = f"g{graph_mix}_f{final_loss}_c{conf}_a{agree}_l{local_score}_r{refresh}_s{short_score}".replace(".", "p")
        args = [
            *spec["base"],
            "--feddieg-graph-mix", graph_mix,
            "--feddieg-final-loss-weight", final_loss,
            "--feddieg-confidence-power", conf,
            "--feddieg-agreement-power", agree,
            "--feddieg-score-residual-mix", residual,
            "--feddieg-score-local-mix", local_score,
            "--feddieg-boundary-refresh-rate", refresh,
            "--feddieg-boundary-refresh-momentum", "0.80",
            "--feddieg-short-score-mix", short_score,
        ]
        grids.append((name, args))
    return grids


def run_search(dataset_key: str, rounds_per_window: int, eval_every: int, max_configs: int | None) -> list[dict[str, object]]:
    spec = DATASETS[dataset_key]
    out_root = ROOT / "exp_02_fast_gpu" / "gated_search" / dataset_key
    rows = []
    candidates = candidate_grid(spec)
    if max_configs is not None:
        candidates = candidates[:max_configs]
    baseline = read_baselines(dataset_key)
    for idx, (name, extra_args) in enumerate(candidates, start=1):
        out_dir = out_root / name
        cmd = [
            PYTHON,
            str(RUNNER),
            "--dataset",
            spec["name"],
            "--methods",
            "FedDIEG",
            "--num-windows",
            "5",
            "--rounds-per-window",
            str(rounds_per_window),
            "--window-mode",
            "slice",
            "--eval-every",
            str(eval_every),
            "--participation-rate",
            "0.1",
            "--microbatch-size",
            "512",
            "--eval-batch-size",
            "2048",
            "--local-epochs",
            "1",
            "--candidate-count",
            "99",
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
            "--no-plot",
            *extra_args,
        ]
        print(f"\n[{dataset_key}] {idx}/{len(candidates)} {name}", flush=True)
        subprocess.run(cmd, check=True)
        summary = read_one_summary(out_dir / "exp02_summary.csv")
        avg = float(summary["avg_window_ndcg"])
        adapt = float(summary["average_adaptation_gain"])
        drop = float(summary["average_boundary_drop"])
        valid = all(math.isfinite(value) for value in (avg, adapt, drop)) and not has_invalid_numbers(out_dir / "exp02_fast_gpu.log")
        pass_all, score = compare_to_baseline(avg, adapt, drop, baseline)
        row = {
            "dataset": dataset_key,
            "config": name,
            "valid": valid,
            "pass_all": bool(valid and pass_all),
            "search_score": score if valid else -999.0,
            "avg_window_ndcg": avg,
            "avg_ndcg_gap": avg - baseline["avg_window_ndcg"],
            "average_adaptation_gain": adapt,
            "adapt_gap": adapt - baseline["average_adaptation_gain"],
            "average_boundary_drop": drop,
            "boundary_drop_gap": baseline["average_boundary_drop"] - drop,
            "final_window_ndcg": float(summary["final_window_ndcg"]),
            "config_args": " ".join(extra_args),
        }
        rows.append(row)
        write_csv(out_root / "gated_search_summary.csv", sorted(rows, key=lambda r: (r["pass_all"], r["valid"], r["search_score"]), reverse=True))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--rounds-per-window", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()

    all_rows = []
    for dataset_key in args.datasets:
        all_rows.extend(run_search(dataset_key, args.rounds_per_window, args.eval_every, args.max_configs))
    write_csv(ROOT / "exp_02_fast_gpu" / "gated_search" / "all_gated_search_summary.csv", all_rows)


if __name__ == "__main__":
    main()
