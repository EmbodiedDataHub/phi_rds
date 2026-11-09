"""数据集 / Episode 元信息。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phi_rds.schema.spec import DatasetSchema
from phi_rds.core.semantics import EpisodeSemantics
from phi_rds.core.stats import DatasetStats, EpisodeStats
from phi_rds.types import DataLevel, EpisodeStatus


@dataclass(frozen=True)
class ViewInfo:
    parent_uri: str | None = None
    selector: str | None = None
    is_view: bool = False


@dataclass(frozen=True)
class DatasetMeta:
    """身份 = name + version + robot；布局见 ``DatasetSchema``。"""

    name: str
    version: str
    robot: str
    uri: str
    fps: float = 30.0
    schema_id: str = ""
    level: DataLevel | None = None
    standard_version: str | None = None
    semantics: EpisodeSemantics = field(default_factory=EpisodeSemantics)
    stats: DatasetStats = field(default_factory=DatasetStats)
    view: ViewInfo = field(default_factory=ViewInfo)
    extras: dict[str, Any] = field(default_factory=dict)

    def bind_schema(self, schema: DatasetSchema) -> DatasetMeta:
        return DatasetMeta(
            name=self.name,
            version=self.version,
            robot=self.robot or schema.robot_type,
            uri=self.uri,
            fps=schema.fps or self.fps,
            schema_id=schema.id,
            level=schema.level if schema.level is not None else self.level,
            standard_version=schema.standard_version or self.standard_version,
            semantics=self.semantics,
            stats=self.stats,
            view=self.view,
            extras=dict(self.extras),
        )


@dataclass(frozen=True)
class VideoSpan:
    path: str
    from_timestamp: float = 0.0
    to_timestamp: float | None = None


@dataclass(frozen=True)
class EpisodeSpan:
    videos: dict[str, VideoSpan] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeMeta:
    uid: str
    num_frames: int
    fps: float = 30.0
    task: str = ""
    status: EpisodeStatus = "active"
    episode_index: int | None = None
    span: EpisodeSpan = field(default_factory=EpisodeSpan)
    stats: EpisodeStats = field(default_factory=EpisodeStats)
    extras: dict[str, Any] = field(default_factory=dict)
