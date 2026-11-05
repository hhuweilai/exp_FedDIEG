"""通用小工具：日志分流、随机种子、设备选择、CSV 写出。"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


class Tee:
    """把 stdout/stderr 同时写进日志文件（沿用旧脚本的实现）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message: str) -> None:
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:  # tqdm 等库会查询
        return False


def open_log(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / name).open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, handle)  # type: ignore[assignment]
    return handle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def safe_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def masked_mean(values: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if values.size == 0 or mask is None or mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def warn_if_legacy_results(out_dir: Path) -> None:
    """如果目标目录里是"旧实现"的结果，提醒会被本次运行覆盖。

    旧实现（2026-11-05 之前的 val01/02/03）写出的 config.json 里没有 ``core_version``
    字段，据此可以判定。
    """
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    if "core_version" in payload:
        return
    print(
        "\n" + "!" * 78
        + f"\n注意：{out_dir} 里已有的是【旧实现】的结果"
        "\n（config.json 没有 core_version 字段 = 旧版单模型仿真 + 有缺陷的动态图）。"
        "\n本次运行会覆盖它。若想保留旧结果，请先手动改名/备份，例如："
        f"\n  Rename-Item '{out_dir}' '{out_dir.name}_legacy'"
        "\n" + "!" * 78
    )

