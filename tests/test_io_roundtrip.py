"""LeRobot v3 往返。"""

from __future__ import annotations

import numpy as np

from phi_rds import (
    Arm,
    DatasetMeta,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    Feature,
    Gripper,
    JointGroup,
    Layout,
    LazyHandle,
    Pose,
    RobotBody,
    RobotDataSet,
    Slice,
    keys,
    read_dataset,
    write_dataset,
)


def _episode(uid: str, n: int = 4) -> Episode:
    matrices = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    matrices[:, 0, 3] = np.linspace(0, 1, n)
    joints = np.zeros((n, 7), dtype=np.float64)
    joints[:, 0] = np.linspace(0, 0.5, n)
    body = RobotBody(
        base=Pose(LazyHandle.from_memory(matrices.copy())),
        arm_single=Arm(
            ee=Pose(LazyHandle.from_memory(matrices.copy())),
            gripper=Gripper(open=LazyHandle.from_memory(np.ones((n, 1)))),
            joints=JointGroup(
                position=LazyHandle.from_memory(joints),
                names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
            ),
        ),
    )
    return Episode(
        meta=EpisodeMeta(
            uid=uid,
            num_frames=n,
            fps=30.0,
            task="pick",
            stats=EpisodeStats(num_frames=n),
        ),
        state=body,
        action=body,
    )


def test_lerobot_official_roundtrip(tmp_path):
    meta = DatasetMeta(
        name="demo",
        version="v1",
        robot="umi",
        uri="memory://demo",
        fps=30.0,
        schema_id="umi",
    )
    ds = RobotDataSet.from_episodes(meta, [_episode("x", 4)])
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
            Feature(
                keys.STATE_ARM_JOINT,
                (
                    Slice(
                        kind="joint",
                        from_="state",
                        part=keys.ARM_SINGLE,
                        names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
                        fk_names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
                        src_width=7,
                    ),
                ),
            ),
            Feature(
                keys.ACTION_ARM_TCP,
                (
                    Slice(
                        kind="pose9",
                        from_="action",
                        part=keys.ARM_SINGLE,
                        names=keys.pose9_names(keys.ARM_SINGLE),
                    ),
                ),
            ),
        ),
    )
    out = tmp_path / "v3"
    write_dataset(ds, out, layout=layout)
    loaded = read_dataset(out, layout=layout)
    assert len(loaded) == 1
    ep = loaded[0]
    assert ep.state.arm_single is not None
    assert ep.state.arm_single.ee is not None
    assert ep.state.arm_single.ee.numpy().shape == (4, 4, 4)
