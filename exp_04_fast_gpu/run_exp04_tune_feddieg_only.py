from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from run_exp04_fast_gpu import (
    ROOT,
    Exp4Config,
    SUMMARY_FIELDS,
    TRAJECTORY_FIELDS,
    Tee,
    load_cold_start_data,
    make_plot,
    run_method,
    set_seed,
    write_csv,
)


DATASETS = (
    "movielens-1m",
    "amazon-digital music",
    "amazon-video games",
    "amazon-arts crafts and sewing",
)
DATASET_FOLDERS = {
    "movielens-1m": "movielens-1m",
    "amazon-digital music": "amazon-digital_music",
    "amazon-video games": "amazon-video_games",
    "amazon-arts crafts and sewing": "amazon-arts_crafts_and_sewing",
}
DATASET_LABELS = {
    "movielens-1m": "MovieLens-1M",
    "amazon-digital_music": "Digital Music",
    "amazon-video_games": "Video Games",
    "amazon-arts_crafts_and_sewing": "Arts Crafts & Sewing",
}
AGGREGATE_FIELDS = [
    "folder",
    "method",
    "initial_ndcg",
    "growth_rate",
    "early_growth_rate",
    "auc_ndcg",
    "best_ndcg",
    "best_round",
    "final_ndcg",
    "initial_hr",
    "final_hr",
]
ANCHOR_FIELDS = ["dataset", "folder", "method", "initial_ndcg", "growth_rate", "auc_ndcg"]


FED_CONFIG_CANDIDATES: Dict[str, Dict[str, float | int]] = {
    "balanced": {
        "feddieg_neighbors": 20,
        "feddieg_graph_mix": 0.30,
        "feddieg_neighbor_temp": 0.05,
        "feddieg_min_sim": 0.05,
        "feddieg_final_loss_weight": 0.15,
        "feddieg_memory_momentum": 0.45,
        "feddieg_short_score_mix": 0.20,
        "feddieg_proto_topk": 8,
        "feddieg_proto_state_mix": 0.35,
        "feddieg_proto_memory_mix": 0.75,
        "feddieg_proto_edge_mix": 0.45,
        "feddieg_proto_score_mix": 0.55,
    },
    "proto_strong_light_graph": {
        "feddieg_neighbors": 24,
        "feddieg_graph_mix": 0.18,
        "feddieg_neighbor_temp": 0.04,
        "feddieg_min_sim": 0.03,
        "feddieg_final_loss_weight": 0.12,
        "feddieg_memory_momentum": 0.50,
        "feddieg_short_score_mix": 0.25,
        "feddieg_proto_topk": 10,
        "feddieg_proto_state_mix": 0.45,
        "feddieg_proto_memory_mix": 0.85,
        "feddieg_proto_edge_mix": 0.35,
        "feddieg_proto_score_mix": 0.75,
    },
    "proto_edge_strong": {
        "feddieg_neighbors": 30,
        "feddieg_graph_mix": 0.25,
        "feddieg_neighbor_temp": 0.04,
        "feddieg_min_sim": 0.03,
        "feddieg_final_loss_weight": 0.15,
        "feddieg_memory_momentum": 0.50,
        "feddieg_short_score_mix": 0.20,
        "feddieg_proto_topk": 10,
        "feddieg_proto_state_mix": 0.40,
        "feddieg_proto_memory_mix": 0.80,
        "feddieg_proto_edge_mix": 0.60,
        "feddieg_proto_score_mix": 0.70,
    },
    "stable_conservative": {
        "feddieg_neighbors": 16,
        "feddieg_graph_mix": 0.15,
        "feddieg_neighbor_temp": 0.06,
        "feddieg_min_sim": 0.05,
        "feddieg_final_loss_weight": 0.10,
        "feddieg_memory_momentum": 0.55,
        "feddieg_short_score_mix": 0.10,
        "feddieg_proto_topk": 8,
        "feddieg_proto_state_mix": 0.30,
        "feddieg_proto_memory_mix": 0.70,
        "feddieg_proto_edge_mix": 0.25,
        "feddieg_proto_score_mix": 0.65,
    },
}


def dataset_folder(dataset: str) -> str:
    return DATASET_FOLDERS[dataset]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def backup_file(path: Path, stamp: str, enabled: bool) -> None:
    if enabled and path.exists():
        shutil.copy2(path, Path(str(path) + f".bak_feddieg_tune_{stamp}"))


def replace_method_rows(
    path: Path,
    rows: List[Dict[str, object]],
    fields: Sequence[str],
    method: str,
    stamp: str,
    backup: bool,
) -> None:
    existing = read_csv_rows(path)
    merged = [row for row in existing if row.get("method") != method] + rows
    backup_file(path, stamp, backup)
    write_csv(path, merged, list(fields))


def aggregate_row(folder: str, summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "folder": folder,
        "method": summary["method"],
        "initial_ndcg": summary["initial_ndcg"],
        "growth_rate": summary["growth_rate"],
        "early_growth_rate": summary["early_growth_rate"],
        "auc_ndcg": summary["auc_ndcg"],
        "best_ndcg": summary["best_ndcg"],
        "best_round": summary["best_round"],
        "final_ndcg": summary["final_ndcg"],
        "initial_hr": summary["initial_hr"],
        "final_hr": summary["final_hr"],
    }


def rebuild_aggregates(base_out: Path) -> None:
    aggregate_rows: List[Dict[str, object]] = []
    anchor_rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        folder = dataset_folder(dataset)
        summary_path = base_out / folder / "summary.csv"
        if not summary_path.exists():
            continue
        summaries = pd.read_csv(summary_path)
        for _, row in summaries.iterrows():
            aggregate_rows.append(
                {
                    "folder": folder,
                    "method": row["method"],
                    "initial_ndcg": row["initial_ndcg"],
                    "growth_rate": row["growth_rate"],
                    "early_growth_rate": row["early_growth_rate"],
                    "auc_ndcg": row["auc_ndcg"],
                    "best_ndcg": row["best_ndcg"],
                    "best_round": row["best_round"],
                    "final_ndcg": row["final_ndcg"],
                    "initial_hr": row["initial_hr"],
                    "final_hr": row["final_hr"],
                }
            )
            anchor_rows.append(
                {
                    "dataset": DATASET_LABELS[folder],
                    "folder": folder,
                    "method": row["method"],
                    "initial_ndcg": row["initial_ndcg"],
                    "growth_rate": row["growth_rate"],
                    "auc_ndcg": row["auc_ndcg"],
                }
            )
    write_csv(base_out / "exp04_all_models_main_metrics.csv", aggregate_rows, AGGREGATE_FIELDS)
    write_csv(base_out / "exp04_main_metrics_with_auc.csv", aggregate_rows, AGGREGATE_FIELDS)
    write_csv(base_out / "exp04_new_client_anchor_main_metrics.csv", anchor_rows, ANCHOR_FIELDS)


def build_base_config(dataset: str, args: argparse.Namespace) -> Exp4Config:
    return Exp4Config(
        dataset=dataset,
        methods=("FedDIEG",),
        seed=args.seed,
        pretrain_rounds=args.pretrain_rounds,
        adapt_rounds=args.adapt_rounds,
        new_client_ratio=args.new_client_ratio,
        new_user_strategy=args.new_user_strategy,
        min_new_user_history=args.min_new_user_history,
        d_init=args.d_init,
        new_interactions_per_round=args.new_interactions_per_round,
        participation_rate=args.participation_rate,
        clients_per_round=args.clients_per_round,
        adapt_clients_per_round=args.adapt_clients_per_round,
        local_epochs=args.local_epochs,
        adapt_epochs=args.adapt_epochs,
        lr=args.lr,
        adapt_lr=args.adapt_lr,
        microbatch_size=args.microbatch_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_users=args.max_eval_users,
        num_threads=args.num_threads,
    )


def apply_candidate(cfg: Exp4Config, candidate: Dict[str, float | int]) -> Exp4Config:
    return replace(cfg, **candidate)


def select_best_candidate(
    candidate_summaries: Dict[str, Dict[str, Dict[str, object]]],
    metric: str,
) -> Tuple[str, pd.DataFrame]:
    rows = []
    for name, by_dataset in candidate_summaries.items():
        metric_values = [float(summary[metric]) for summary in by_dataset.values()]
        initial_values = [float(summary["initial_ndcg"]) for summary in by_dataset.values()]
        rows.append(
            {
                "candidate": name,
                f"mean_{metric}": float(np.mean(metric_values)),
                "mean_initial_ndcg": float(np.mean(initial_values)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        [f"mean_{metric}", "mean_initial_ndcg", "candidate"],
        ascending=[False, False, True],
    )
    return str(scores.iloc[0]["candidate"]), scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune FedDIEG only, then overwrite FedDIEG rows in Experiment 4 final outputs."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--candidate-names",
        nargs="+",
        default=list(FED_CONFIG_CANDIDATES),
        choices=list(FED_CONFIG_CANDIDATES),
    )
    parser.add_argument("--base-out", type=Path, default=ROOT / "exp_04_fast_gpu" / "final")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "exp_04_fast_gpu" / "tune_feddieg_only")
    parser.add_argument("--selection-metric", choices=["auc_ndcg", "initial_ndcg"], default="auc_ndcg")
    parser.add_argument("--pretrain-rounds", type=int, default=50)
    parser.add_argument("--adapt-rounds", type=int, default=10)
    parser.add_argument("--new-client-ratio", type=float, default=0.25)
    parser.add_argument("--new-user-strategy", choices=["random", "low_history", "high_history"], default="low_history")
    parser.add_argument("--min-new-user-history", type=int, default=5)
    parser.add_argument("--d-init", type=int, default=3)
    parser.add_argument("--new-interactions-per-round", type=int, default=2)
    parser.add_argument("--participation-rate", type=float, default=0.1)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--adapt-clients-per-round", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--adapt-lr", type=float, default=0.0002)
    parser.add_argument("--microbatch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_out = args.base_out.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")

    log_handle = (args.work_dir / f"feddieg_tune_{stamp}.log").open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Candidates: {', '.join(args.candidate_names)}")
    print(f"Base output to update: {args.base_out}")

    candidate_rows: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    candidate_summaries: Dict[str, Dict[str, Dict[str, object]]] = {}

    for dataset in args.datasets:
        folder = dataset_folder(dataset)
        base_cfg = build_base_config(dataset, args)
        print(f"\nLoading data for {dataset}")
        data = load_cold_start_data(base_cfg).to(device)
        for candidate_name in args.candidate_names:
            candidate = FED_CONFIG_CANDIDATES[candidate_name]
            cfg = apply_candidate(base_cfg, candidate)
            print(f"\n[tune] dataset={dataset} candidate={candidate_name}")
            print(json.dumps(candidate, ensure_ascii=False, indent=2))
            rows, summary, _ = run_method("FedDIEG", data, cfg, device, variant="full")
            candidate_rows.setdefault(candidate_name, {})[folder] = rows
            candidate_summaries.setdefault(candidate_name, {})[folder] = summary

            cand_dir = args.work_dir / candidate_name / folder
            cand_dir.mkdir(parents=True, exist_ok=True)
            write_csv(cand_dir / "trajectory.csv", rows, TRAJECTORY_FIELDS)
            write_csv(cand_dir / "summary.csv", [summary], SUMMARY_FIELDS)
            with (cand_dir / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)
            print(
                f"[tune] {candidate_name}/{folder}: initial={summary['initial_ndcg']:.6f}, "
                f"growth={summary['growth_rate']:.6f}, auc={summary['auc_ndcg']:.6f}"
            )

    best_candidate, score_table = select_best_candidate(candidate_summaries, args.selection_metric)
    score_path = args.work_dir / "candidate_scores.csv"
    score_table.to_csv(score_path, index=False)
    selected_path = args.work_dir / "selected_feddieg_config.json"
    with selected_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_candidate": best_candidate,
                "selection_metric": args.selection_metric,
                "candidate_config": FED_CONFIG_CANDIDATES[best_candidate],
                "scores": score_table.to_dict(orient="records"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSelected candidate: {best_candidate}")
    print(score_table.to_string(index=False))

    if args.no_overwrite:
        print("Skip overwrite because --no-overwrite was set.")
        return

    for dataset in args.datasets:
        folder = dataset_folder(dataset)
        out_dir = args.base_out / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = candidate_rows[best_candidate][folder]
        summary = candidate_summaries[best_candidate][folder]
        replace_method_rows(out_dir / "trajectory.csv", rows, TRAJECTORY_FIELDS, "FedDIEG", stamp, args.backup)
        replace_method_rows(out_dir / "summary.csv", [summary], SUMMARY_FIELDS, "FedDIEG", stamp, args.backup)
        if not args.skip_plot:
            make_plot(
                out_dir,
                out_dir / "trajectory.csv",
                f"Experiment 4 Prototype Anchoring - {dataset}",
                "ndcg_trajectory.png",
            )
        print(f"[overwrite] replaced FedDIEG rows in {out_dir}")

    rebuild_aggregates(args.base_out)
    print(f"[overwrite] rebuilt aggregate CSV files under {args.base_out}")
    print("Done. FedDIEG-only tuning completed and best global candidate has overwritten prior FedDIEG rows.")


if __name__ == "__main__":
    main()
