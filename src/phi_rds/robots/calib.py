"""校准 JSON：部位 → {xyz_m, rpy_deg} → 4×4。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def _transform(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = SciR.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def load_align(path: Path | None) -> dict[str, np.ndarray]:
    """读 calib JSON → {part: (4,4)}。缺文件返回空 dict。"""
    if path is None or not Path(path).is_file():
        return {}
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    for part, record in (blob.get("align") or {}).items():
        xyz = tuple(record.get("xyz_m") or (0.0, 0.0, 0.0))
        rpy = tuple(record.get("rpy_deg") or (0.0, 0.0, 0.0))
        out[str(part)] = _transform(xyz, *rpy)  # type: ignore[arg-type]
    return out
