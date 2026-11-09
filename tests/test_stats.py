"""分位数完整性测试。"""

from __future__ import annotations

from phi_rds import (
    DEFAULT_PERCENTILES,
    DatasetMeta,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    LengthDistribution,
    LazyHandle,
    Pose,
    RobotBody,
    RobotDataSet,
    percentile,
    percentiles,
)
import numpy as np


def test_default_percentiles_complete():
    assert DEFAULT_PERCENTILES == (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)


def test_length_distribution_from_values():
    values = list(range(1, 101))  # 1..100
    dist = LengthDistribution.from_values(values)
    assert dist.count == 100
    assert dist.min == 1.0
    assert dist.max == 100.0
    assert dist.p50 == percentile(values, 50)
    assert dist.p1 is not None and dist.p99 is not None
    qmap = dist.quantile_map()
    for key in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
        assert key in qmap
        assert qmap[key] is not None


def test_percentiles_helper():
    out = percentiles([0, 10, 20, 30, 40], qs=(0, 50, 100))
    assert out["p0"] == 0.0
    assert out["p50"] == 20.0
    assert out["p100"] == 40.0


def test_dataset_frame_count_quantiles():
    def ep(uid: str, n: int) -> Episode:
        T = np.tile(np.eye(4), (n, 1, 1))
        return Episode(
            meta=EpisodeMeta(
                uid=uid, num_frames=n, fps=30.0,
                stats=EpisodeStats(num_frames=n),
            ),
            state=RobotBody(base=Pose(LazyHandle.from_memory(T))),
        )

    meta = DatasetMeta(
        name="d", version="v1", robot="r", uri="memory://d", fps=30.0,
    )
    ds = RobotDataSet.from_episodes(
        meta, [ep("a", 10), ep("b", 20), ep("c", 30), ep("d", 40)],
    )
    dist = ds.meta.stats.frame_count
    assert dist is not None
    assert dist.count == 4
    assert dist.p50 is not None
    assert dist.p95 is not None
    assert set(dist.quantile_map()) >= {
        "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
    }
