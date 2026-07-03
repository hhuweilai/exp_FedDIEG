from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_METHODS = ["FedMF", "FedSASRec", "GPFedRec", "FCUCR", "F3CRec", "DGFedRS", "FedDIEG"]
FILES = ["exp02_summary.csv", "exp02_window_metrics.csv", "exp02_round_metrics.csv"]
COMPAT_NAMES = {
    "exp02_summary.csv": "summary.csv",
    "exp02_window_metrics.csv": "window_metrics.csv",
    "exp02_round_metrics.csv": "round_metrics.csv",
}


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


def merge_file(seed_dir: Path, file_name: str) -> None:
    sources = [seed_dir / "baselines" / file_name, seed_dir / "feddieg" / file_name]
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
    write_rows(seed_dir / file_name, fieldnames, merged_rows)
    compat_name = COMPAT_NAMES.get(file_name)
    if compat_name:
        write_rows(seed_dir / compat_name, fieldnames, merged_rows)


def validate_summary(seed_dir: Path) -> None:
    _, rows = read_rows(seed_dir / "exp02_summary.csv")
    methods = [row.get("method", "") for row in rows]
    missing = [method for method in EXPECTED_METHODS if method not in methods]
    extra = [method for method in methods if method not in EXPECTED_METHODS]
    if missing:
        raise ValueError(f"Missing methods in merged summary: {missing}")
    if extra:
        raise ValueError(f"Unexpected methods in merged summary: {extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Exp02 baseline and FedDIEG outputs for one seed.")
    parser.add_argument("--seed-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_dir = args.seed_dir.resolve()
    for file_name in FILES:
        merge_file(seed_dir, file_name)
    validate_summary(seed_dir)
    print(f"Merged Exp02 seed outputs: {seed_dir}")


if __name__ == "__main__":
    main()
