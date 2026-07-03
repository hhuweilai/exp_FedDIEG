from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

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


APPEND_BASELINES = ("FedMF", "FCUCR", "F3CRec", "DGFedRS")
DEFAULT_DATASETS = (
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


def dataset_folder(dataset: str) -> str:
    return DATASET_FOLDERS.get(dataset, dataset.replace(" ", "_"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def backup_file(path: Path, stamp: str, enabled: bool) -> None:
    if enabled and path.exists():
        shutil.copy2(path, Path(str(path) + f".bak_append_baselines_{stamp}"))


def merge_rows(
    existing_rows: List[Dict[str, object]],
    new_rows: List[Dict[str, object]],
    methods: Sequence[str],
    replace_existing: bool,
) -> List[Dict[str, object]]:
    method_set = set(methods)
    if not replace_existing:
        return existing_rows + new_rows
    return [row for row in existing_rows if row.get("method") not in method_set] + new_rows


def write_append_config(out_dir: Path, cfg: Exp4Config) -> None:
    with (out_dir / "append_baselines_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)


def merge_dataset_outputs(
    out_dir: Path,
    rows: List[Dict[str, object]],
    summaries: List[Dict[str, object]],
    methods: Sequence[str],
    replace_existing: bool,
    backup: bool,
    stamp: str,
) -> None:
    trajectory_path = out_dir / "trajectory.csv"
    summary_path = out_dir / "summary.csv"
    backup_file(trajectory_path, stamp, backup)
    backup_file(summary_path, stamp, backup)

    merged_trajectory = merge_rows(read_csv_rows(trajectory_path), rows, methods, replace_existing)
    merged_summary = merge_rows(read_csv_rows(summary_path), summaries, methods, replace_existing)

    write_csv(trajectory_path, merged_trajectory, TRAJECTORY_FIELDS)
    write_csv(summary_path, merged_summary, SUMMARY_FIELDS)


def summary_to_aggregate(folder: str, summary: Dict[str, object]) -> Dict[str, object]:
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


def merge_aggregate_rows(
    aggregate_rows: List[Dict[str, object]],
    folder: str,
    summaries: List[Dict[str, object]],
    methods: Sequence[str],
    replace_existing: bool,
) -> List[Dict[str, object]]:
    method_set = set(methods)
    if replace_existing:
        aggregate_rows = [
            row
            for row in aggregate_rows
            if not (row.get("folder") == folder and row.get("method") in method_set)
        ]
    return aggregate_rows + [summary_to_aggregate(folder, summary) for summary in summaries]


def build_config(dataset: str, args: argparse.Namespace) -> Exp4Config:
    return Exp4Config(
        dataset=dataset,
        methods=tuple(args.methods),
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


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    stamp: str,
) -> List[Dict[str, object]]:
    cfg = build_config(dataset, args)
    folder = dataset_folder(dataset)
    out_dir = (args.base_out / folder).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "exp04_append_baselines.log"
    log_handle = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    try:
        print(f"\n[append-baselines] dataset={dataset} out_dir={out_dir}")
        print(f"[append-baselines] methods={', '.join(args.methods)}")
        print(f"[append-baselines] device={device}")
        write_append_config(out_dir, cfg)

        set_seed(cfg.seed)
        data = load_cold_start_data(cfg).to(device)
        print(
            f"users={data.num_users}, items={data.num_items}, old={data.old_users.numel()}, "
            f"new={data.new_users.numel()}, eval_new={data.eval_users.numel()}, d_init={cfg.d_init}"
        )

        rows: List[Dict[str, object]] = []
        summaries: List[Dict[str, object]] = []
        for method in cfg.methods:
            method_rows, summary, _ = run_method(method, data, cfg, device, variant="full")
            rows.extend(method_rows)
            summaries.append(summary)

        merge_dataset_outputs(
            out_dir=out_dir,
            rows=rows,
            summaries=summaries,
            methods=cfg.methods,
            replace_existing=args.replace_existing,
            backup=args.backup,
            stamp=stamp,
        )
        if not args.skip_plot:
            make_plot(
                out_dir,
                out_dir / "trajectory.csv",
                f"Experiment 4 Prototype Anchoring - {dataset}",
                "ndcg_trajectory.png",
            )
        print(f"[append-baselines] merged summary.csv and trajectory.csv into {out_dir}")
        return summaries
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the new Experiment 4 baselines and append them to existing result folders."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(APPEND_BASELINES), choices=list(APPEND_BASELINES))
    parser.add_argument("--base-out", type=Path, default=ROOT / "exp_04_fast_gpu" / "final")
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
    parser.add_argument("--replace-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_out = args.base_out.resolve()
    args.base_out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = time.strftime("%Y%m%d_%H%M%S")

    aggregate_path = args.base_out / "exp04_main_metrics_with_auc.csv"
    backup_file(aggregate_path, stamp, args.backup)
    aggregate_rows: List[Dict[str, object]] = read_csv_rows(aggregate_path)

    for dataset in args.datasets:
        summaries = run_dataset(dataset, args, device, stamp)
        folder = dataset_folder(dataset)
        aggregate_rows = merge_aggregate_rows(
            aggregate_rows,
            folder,
            summaries,
            args.methods,
            args.replace_existing,
        )
        write_csv(aggregate_path, aggregate_rows, AGGREGATE_FIELDS)
        print(f"[append-baselines] updated aggregate: {aggregate_path}")

    print("\nDone. Appended new baseline results without rerunning FedDIEG/FedSASRec/GPFedRec.")


if __name__ == "__main__":
    main()
