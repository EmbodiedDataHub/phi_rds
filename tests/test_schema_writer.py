"""Schema 先行 + DatasetWriter 灌数。"""

from __future__ import annotations

import numpy as np

from phi_rds import (
    Arm,
    DatasetMeta,
    DatasetSchema,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    Feature,
    JointGroup,
    Layout,
    LazyHandle,
    Pose,
    RobotBody,
    RobotDataSet,
    Slice,
    keys,
)


def _ep(uid: str, n: int = 4) -> Episode:
    T = np.tile(np.eye(4), (n, 1, 1))
    joints = np.zeros((n, 7))
    body = RobotBody(
        arm_single=Arm(
            ee=Pose(LazyHandle.from_memory(T)),
            joints=JointGroup(
                position=LazyHandle.from_memory(joints),
                names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
            ),
        ),
    )
    return Episode(
        meta=EpisodeMeta(uid=uid, num_frames=n, fps=30.0, stats=EpisodeStats(num_frames=n)),
        state=body,
        action=body,
    )


def test_create_add_finalize_roundtrip(tmp_path):
    layout = Layout(
        id="umi",
        robot_type="umi",
        frame=keys.BASE,
        features=(
            Feature(
                keys.STATE_ARM_TCP,
                (
                    Slice(
                        kind="pose9",
                        from_="state",
                        part=keys.ARM_SINGLE,
                        names=keys.pose9_names(keys.ARM_SINGLE),
                    ),
                ),
            ),
        ),
    )
    schema = DatasetSchema.from_layout(layout, fps=30.0, profile="standard")
    meta = DatasetMeta(name="w", version="v1", robot="umi", uri="")
    writer = RobotDataSet.create(tmp_path / "ds", schema, meta)
    writer.add_episode(_ep("a", 3))
    writer.add_episode(_ep("b", 5))
    root = writer.finalize()

    ds = RobotDataSet.load(root, layout=layout)
    assert ds.schema.id == "umi"
    assert len(ds) == 2
    assert ds.schema.feature_names[0] == keys.STATE_ARM_TCP
    assert ds[0].state.arm_single is not None
    assert ds[0].state.arm_single.ee.numpy().shape == (3, 4, 4)
