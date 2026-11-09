"""RobotDataSet 基本行为冒烟测试。"""

from __future__ import annotations

import numpy as np

from phi_rds import (
    DatasetMeta,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    LazyHandle,
    Pose,
    RobotBody,
    RobotDataSet,
)


def _episode(uid: str, n: int = 4, status: str = "active") -> Episode:
    matrices = np.tile(np.eye(4), (n, 1, 1))
    body = RobotBody(base=Pose(LazyHandle.from_memory(matrices)))
    return Episode(
        meta=EpisodeMeta(
            uid=uid,
            num_frames=n,
            fps=30.0,
            task="demo",
            status=status,  # type: ignore[arg-type]
            stats=EpisodeStats(num_frames=n),
        ),
        state=body,
    )


def test_open_slice_and_lazy_meta():
    meta = DatasetMeta(
        name="demo",
        version="v1",
        robot="g1",
        uri="memory://demo",
        fps=30.0,
    )
    ds = RobotDataSet.from_episodes(
        meta,
        [_episode("a", 3), _episode("b", 5, status="deleted"), _episode("c", 7)],
    )
    assert len(ds) == 3
    assert ds.meta.stats.num_episodes == 3
    assert ds.meta.stats.num_frames == 15
    assert ds.meta.stats.status_counts["deleted"] == 1

    sub = ds.range(0, 2)
    assert isinstance(sub, RobotDataSet)
    assert len(sub) == 2
    assert sub.meta.view.is_view
    assert sub.uids() == ("a", "b")

    active = ds.filter(lambda m: m.status == "active")
    assert active.uids() == ("a", "c")

    ep = ds["c"]
    assert ep.uid == "c"
    assert ep.num_frames == 7
    # Pose 懒句柄可 materialize
    assert ep.state.base is not None
    assert ep.state.base.numpy().shape == (7, 4, 4)
