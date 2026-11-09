"""L1/L2 标识 + convert 工具。"""

from __future__ import annotations

import json

import numpy as np

from phi_rds import (
    LATEST_STANDARD,
    DatasetMeta,
    DatasetSchema,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    Feature,
    FrameEdge,
    FrameGraph,
    Layout,
    LazyHandle,
    Pose,
    RobotBody,
    RobotDataSet,
    Slice,
    apply_T,
    change_frame,
    convert_axes,
    create_dataset,
    encode_pose,
    invert_T,
    keys,
    lookup_T,
    open_dataset,
    resolve_standard,
)


def _ep(uid: str, n: int = 4) -> Episode:
    T = np.tile(np.eye(4), (n, 1, 1))
    return Episode(
        meta=EpisodeMeta(uid=uid, num_frames=n, fps=30.0, stats=EpisodeStats(num_frames=n)),
        state=RobotBody(base=Pose(LazyHandle.from_memory(T))),
        action=RobotBody(base=Pose(LazyHandle.from_memory(T))),
    )


def _layout() -> Layout:
    return Layout(
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


def test_schema_l1_l2_defaults():
    layout = _layout()
    s1 = DatasetSchema.l1(layout)
    assert s1.level == "L1"
    assert s1.standard_version is None
    assert s1.is_l1

    s2 = DatasetSchema.l2(layout)
    assert s2.level == "L2"
    assert s2.standard_version == LATEST_STANDARD
    assert s2.is_l2
    assert resolve_standard(None).version == LATEST_STANDARD


def test_l2_roundtrip_stamps_info(tmp_path):
    layout = _layout()
    # need arm_single for encode
    n = 3
    joints = np.zeros((n, 7))
    from phi_rds import Arm, JointGroup

    body = RobotBody(
        arm_single=Arm(
            ee=Pose(LazyHandle.from_memory(np.tile(np.eye(4), (n, 1, 1)))),
            joints=JointGroup(
                position=LazyHandle.from_memory(joints),
                names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
            ),
        ),
    )
    ep = Episode(
        meta=EpisodeMeta(uid="a", num_frames=n, fps=30.0, stats=EpisodeStats(num_frames=n)),
        state=body,
        action=body,
    )
    schema = DatasetSchema.l2(layout, fps=30.0)
    meta = DatasetMeta(name="w", version="v1", robot="umi", uri="")
    root = tmp_path / "l2"
    w = create_dataset(root, schema, meta)
    w.add_episode(ep)
    w.finalize()

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["phi_level"] == "L2"
    assert info["phi_standard_version"] == LATEST_STANDARD

    ds = open_dataset(root, layout=layout)
    assert ds.is_l2
    assert ds.standard_version == LATEST_STANDARD
    assert ds.level == "L2"


def test_open_path_alias(tmp_path):
    layout = _layout()
    n = 2
    from phi_rds import Arm, JointGroup

    body = RobotBody(
        arm_single=Arm(
            ee=Pose(LazyHandle.from_memory(np.tile(np.eye(4), (n, 1, 1)))),
            joints=JointGroup(
                position=LazyHandle.from_memory(np.zeros((n, 7))),
                names=keys.arm_joint_names(keys.ARM_SINGLE, 7),
            ),
        ),
    )
    ep = Episode(
        meta=EpisodeMeta(uid="a", num_frames=n, fps=30.0, stats=EpisodeStats(num_frames=n)),
        state=body,
        action=body,
    )
    schema = DatasetSchema.l1(layout)
    root = tmp_path / "l1"
    w = RobotDataSet.create(root, schema, DatasetMeta(name="x", version="v1", robot="umi", uri=""))
    w.add_episode(ep)
    w.finalize()

    ds = RobotDataSet.open(root, layout=layout)
    assert ds.is_l1
    assert open_dataset(root, layout=layout).is_l1


def test_convert_basics():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(invert_T(T) @ T, np.eye(4))

    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, :3, 3] = np.arange(5)[:, None]
    shifted = change_frame(poses, T)
    assert shifted.shape == (5, 4, 4)
    assert np.allclose(shifted[0, :3, 3], [1, 2, 3])

    left = apply_T(poses, T, side="left")
    assert np.allclose(left, shifted)

    vec = encode_pose(poses, "pose9")
    assert vec.shape == (5, 9)

    # axes roundtrip FLU → RDF → FLU
    back = convert_axes(convert_axes(poses, "FLU", "RDF"), "RDF", "FLU")
    assert np.allclose(back, poses, atol=1e-9)


def test_lookup_T_frame_graph():
    T_base_wrist = np.eye(4)
    T_base_wrist[:3, 3] = [0.5, 0.0, 0.3]
    graph = FrameGraph(
        frames=("base", "wrist"),
        edges=(
            FrameEdge(
                child="wrist",
                parent="base",
                matrix_flat16=tuple(T_base_wrist.reshape(-1).tolist()),
            ),
        ),
        root="base",
    )
    T = lookup_T(graph, "wrist", "base")
    assert np.allclose(T, T_base_wrist)
    T_inv = lookup_T(graph, "base", "wrist")
    assert np.allclose(T_inv @ T_base_wrist, np.eye(4), atol=1e-9)
