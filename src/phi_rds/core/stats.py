"""数据集 / Episode 统计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from phi_rds.types import StatsScope

DEFAULT_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("percentile() needs at least one value")
    return float(np.percentile(array, np.clip(q, 0, 100), method="linear"))


def percentiles(
    values: Sequence[float] | np.ndarray,
    qs: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for q in qs:
        key = f"p{int(q)}" if float(q).is_integer() else f"p{q}"
        out[key] = float("nan") if array.size == 0 else percentile(array, float(q))
    return out


@dataclass(frozen=True)
class FeatureStats:
    name: str
    count: int = 0
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    min: tuple[float, ...] = ()
    max: tuple[float, ...] = ()
    quantiles: dict[str, tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class LengthDistribution:
    count: int = 0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None

    def quantile_map(self) -> dict[str, float | None]:
        return {f"p{k}": getattr(self, f"p{k}") for k in (1, 5, 10, 25, 50, 75, 90, 95, 99)}

    @classmethod
    def from_values(cls, values: Sequence[float] | np.ndarray) -> LengthDistribution:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return cls()
        qs = percentiles(array)
        return cls(
            count=int(array.size),
            min=float(array.min()),
            max=float(array.max()),
            mean=float(array.mean()),
            std=float(array.std(ddof=0)),
            **{k: qs.get(k) for k in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99")},
        )


@dataclass(frozen=True)
class EpisodeStats:
    num_frames: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetStats:
    num_episodes: int = 0
    num_frames: int = 0
    feature_list: tuple[str, ...] = ()
    status_counts: dict[str, int] = field(default_factory=dict)
    frame_count: LengthDistribution | None = None
    features: dict[str, FeatureStats] = field(default_factory=dict)
    scope: StatsScope = "self"
    extras: dict[str, Any] = field(default_factory=dict)

    def with_counts(
        self,
        *,
        num_episodes: int | None = None,
        num_frames: int | None = None,
        status_counts: dict[str, int] | None = None,
        scope: StatsScope | None = None,
        frame_count: LengthDistribution | None = None,
    ) -> DatasetStats:
        return DatasetStats(
            num_episodes=self.num_episodes if num_episodes is None else num_episodes,
            num_frames=self.num_frames if num_frames is None else num_frames,
            feature_list=self.feature_list,
            status_counts=dict(self.status_counts if status_counts is None else status_counts),
            frame_count=self.frame_count if frame_count is None else frame_count,
            features=dict(self.features),
            scope=self.scope if scope is None else scope,
            extras=dict(self.extras),
        )
