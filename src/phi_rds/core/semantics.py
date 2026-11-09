"""参考系 / 单位 / 整集语义（保持轻量）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Units:
    length: str = "m"
    angle: str = "rad"
    gripper: str = "normalized_01"
    time: str = "s"


@dataclass(frozen=True)
class FrameRef:
    expressed_in: str
    body_of: str | None = None
    axes: str = "FLU"


@dataclass(frozen=True)
class FrameEdge:
    child: str
    parent: str
    kind: Literal["extrinsic", "fk"] = "extrinsic"
    matrix_flat16: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FrameGraph:
    frames: tuple[str, ...] = ()
    edges: tuple[FrameEdge, ...] = ()
    root: str | None = None


@dataclass(frozen=True)
class FieldSemantics:
    kind: str = "other"
    role: str = "observation"
    part: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSemantics:
    semantic_level: str = "raw"
    robot_type: str = ""
    layout_id: str = ""
    units: Units = field(default_factory=Units)
    default_frames: FrameGraph = field(default_factory=FrameGraph)
    extras: dict[str, Any] = field(default_factory=dict)
