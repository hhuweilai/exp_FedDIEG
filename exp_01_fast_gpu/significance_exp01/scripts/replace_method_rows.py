from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise SystemExit(f"No header found in {path}")
    return fields, rows


def write_csv_atomic(path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp_path, path)


def replace_method_rows(target: Path, source: Path, method: str, method_col: str) -> None:
    target_fields, target_rows = read_csv(target)
    source_fields, source_rows = read_csv(source)
    if method_col not in target_fields:
        raise SystemExit(f"Target {target} has no column: {method_col}")
    if method_col not in source_fields:
        raise SystemExit(f"Source {source} has no column: {method_col}")

    replacement_rows = [row for row in source_rows if row.get(method_col) == method]
    if not replacement_rows:
        raise SystemExit(f"No rows for method={method} in {source}")

    output_rows: List[Dict[str, str]] = []
    insert_at = None
    removed = 0
    for row in target_rows:
        if row.get(method_col) == method:
            removed += 1
            if insert_at is None:
                insert_at = len(output_rows)
            continue
        output_rows.append(row)

    if insert_at is None:
        insert_at = len(output_rows)

    normalized = [{field: row.get(field, "") for field in target_fields} for row in replacement_rows]
    output_rows[insert_at:insert_at] = normalized
    write_csv_atomic(target, target_fields, output_rows)
    print(f"{target}: removed={removed}, inserted={len(normalized)}, method={method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace all rows for one method in a CSV file.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--method-col", default="method")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replace_method_rows(args.target, args.source, args.method, args.method_col)


if __name__ == "__main__":
    main()
