"""读写：LeRobot v3（唯一磁盘形态）。"""

from __future__ import annotations

from pathlib import Path

from phi_rds.core.dataset import RobotDataSet
from phi_rds.io.lerobot import (
    is_lerobot_root,
    open_lerobot_dataset,
    read_lerobot,
    write_lerobot,
)
from phi_rds.schema.layout import Layout


def detect_format(root: str | Path) -> str | None:
    return "lerobot" if is_lerobot_root(root) else None


def read_dataset(
    root: str | Path,
    *,
    format: str | None = None,
    layout: Layout | None = None,
    **kwargs,
) -> RobotDataSet:
    root = Path(root)
    kind = format if format not in (None, "auto") else detect_format(root)
    if kind != "lerobot":
        raise ValueError(f"expected LeRobot v3 dataset under {root}")
    return read_lerobot(root, layout=layout, **kwargs)


def write_dataset(
    dataset: RobotDataSet,
    root: str | Path,
    *,
    format: str = "lerobot",
    layout: Layout | None = None,
    overwrite: bool = True,
    copy_videos: bool = True,
    **kwargs,
) -> Path:
    if format != "lerobot":
        raise ValueError("only format='lerobot' is supported")
    if layout is None:
        layout = dataset.schema.layout if dataset.schema else None
    if layout is None:
        raise ValueError("write_dataset requires layout= or schema.layout")
    return write_lerobot(
        dataset, root, layout, overwrite=overwrite, copy_videos=copy_videos, **kwargs
    )


__all__ = [
    "read_dataset",
    "write_dataset",
    "read_lerobot",
    "write_lerobot",
    "open_lerobot_dataset",
    "detect_format",
    "is_lerobot_root",
]
