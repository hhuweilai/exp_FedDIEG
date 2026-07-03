from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable, Sequence


VARIANTS = ["Ours-Plain", "Ours-NoTemp", "Ours-NoStruct", "Ours-Full"]
FILES = ["exp03_summary.csv", "exp03_window_metrics.csv", "exp03_round_metrics.csv"]
COMPAT_NAMES = {
    "exp03_summary.csv": "summary.csv",
    "exp03_window_metrics.csv": "window_metrics.csv",
    "exp03_round_metrics.csv": "round_metrics.csv",
}
REPORT_FIELDS = [
    "variant",
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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def variant_key(row: dict[str, str]) -> tuple[int, int, int]:
    variant = row.get("variant", "")
    order = VARIANTS.index(variant) if variant in VARIANTS else len(VARIANTS)
    window = int(float(row.get("window", 0) or 0))
    global_round = int(float(row.get("global_round", 0) or row.get("last_global_round", 0) or 0))
    return order, window, global_round


def merge_file(seed_dir: Path, file_name: str) -> None:
    sources = [seed_dir / "ablations" / file_name, seed_dir / "full_refresh" / file_name]
    fieldnames: list[str] = []
    merged_rows: list[dict[str, str]] = []
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Missing expected file: {source}")
        source_fields, rows = read_rows(source)
        if not fieldnames:
            fieldnames = source_fields
        elif source_fields != fieldnames:
            raise ValueError(f"Column mismatch in {source}: {source_fields} != {fieldnames}")
        merged_rows.extend(rows)
    merged_rows.sort(key=variant_key)
    write_rows(seed_dir / file_name, fieldnames, merged_rows)
    compat_name = COMPAT_NAMES.get(file_name)
    if compat_name:
        write_rows(seed_dir / compat_name, fieldnames, merged_rows)


def validate_summary(seed_dir: Path) -> list[dict[str, str]]:
    fields, rows = read_rows(seed_dir / "exp03_summary.csv")
    variants = [row.get("variant", "") for row in rows]
    missing = [variant for variant in VARIANTS if variant not in variants]
    extra = [variant for variant in variants if variant not in VARIANTS]
    if missing:
        raise ValueError(f"Missing variants in merged summary: {missing}")
    if extra:
        raise ValueError(f"Unexpected variants in merged summary: {extra}")
    if len(rows) != len(VARIANTS):
        raise ValueError(f"Expected {len(VARIANTS)} summary rows, got {len(rows)}")
    if "variant" not in fields:
        raise ValueError("Merged summary does not contain variant column.")
    return rows


def copy_configs(seed_dir: Path) -> None:
    copies = [
        (seed_dir / "ablations" / "config.json", seed_dir / "ablation_base_config.json"),
        (seed_dir / "full_refresh" / "config.json", seed_dir / "full_refresh_config.json"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def write_report(seed_dir: Path, summary_rows: Sequence[dict[str, str]]) -> None:
    fields = [field for field in REPORT_FIELDS if any(field in row for row in summary_rows)]
    write_rows(seed_dir / "exp03_report_table.csv", fields, summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Exp03 ablation and full-refresh outputs for one seed.")
    parser.add_argument("--seed-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_dir = args.seed_dir.resolve()
    for file_name in FILES:
        merge_file(seed_dir, file_name)
    summary_rows = validate_summary(seed_dir)
    copy_configs(seed_dir)
    write_report(seed_dir, summary_rows)
    print(f"Merged Exp03 seed outputs: {seed_dir}")


if __name__ == "__main__":
    main()
