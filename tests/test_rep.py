"""位姿多种表示互转。"""

from __future__ import annotations

import numpy as np

from phi_rds import (
    Feature,
    Layout,
    Pose,
    Slice,
    convert_pose,
    decode,
    encode,
    encode_pose,
    decode_pose,
    keys,
)
from phi_rds.core.episode import Episode
from phi_rds.core.body import Arm, RobotBody
from phi_rds.core.lazy import LazyHandle
from phi_rds.core.meta import EpisodeMeta
from phi_rds.core.stats import EpisodeStats


def _random_poses(n: int = 8) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SciR

    mats = np.tile(np.eye(4), (n, 1, 1))
    mats[:, :3, :3] = SciR.random(n, random_state=0).as_matrix()
    mats[:, :3, 3] = np.linspace(0, 1, n)[:, None] * np.array([0.1, 0.2, 0.3])
    return mats


def test_roundtrip_common_reps():
    mats = _random_poses()
    for name in ("pose9", "quat_wxyz", "quat_xyzw", "euler_xyz", "euler_xyz_deg", "rot6d"):
        vec = encode_pose(mats, name)
        back = decode_pose(vec, name)
        assert back.shape == mats.shape
        assert np.allclose(back[:, :3, 3], mats[:, :3, 3], atol=1e-6)
        # 旋转：比较旋转矩阵（允许符号歧义用矩阵差）
        assert np.allclose(back[:, :3, :3], mats[:, :3, :3], atol=1e-5), name


def test_convert_pose_quat_to_euler():
    mats = _random_poses(4)
    quat = encode_pose(mats, "quat_wxyz")
    euler = convert_pose(quat, src="quat_wxyz", dst="euler_xyz")
    back = decode_pose(euler, "euler_xyz")
    assert np.allclose(back, mats, atol=1e-5)


def test_pose_helpers():
    mats = _random_poses(3)
    pose = Pose.from_matrix(mats)
    q = pose.as_quat(order="wxyz")
    assert q.shape == (3, 7)
    e = pose.as_euler("xyz", degrees=True)
    assert e.shape == (3, 6)
    rebuilt = Pose.from_quat(q[:, :3], q[:, 3:], order="wxyz")
    assert rebuilt.rep == "quat_wxyz"
    assert rebuilt.numpy().shape == (3, 7)
    assert np.allclose(rebuilt.as_matrix(), mats, atol=1e-5)


def test_layout_encode_quat():
    mats = _random_poses(5)
    body = RobotBody(
        arm_single=Arm(ee=Pose.from_matrix(mats)),
    )
    ep = Episode(
        meta=EpisodeMeta(uid="u", num_frames=5, fps=30.0, stats=EpisodeStats(num_frames=5)),
        state=body,
    )
    layout = Layout(
        id="t",
        robot_type="t",
        frame=keys.BASE,
        features=(
            Feature(
                "observation.state.arm.tcp_pose",
                (
                    Slice(
                        kind="pose",
                        from_="state",
                        part=keys.ARM_SINGLE,
                        names=("x", "y", "z", "qw", "qx", "qy", "qz"),
                        pose_rep="quat_wxyz",
                    ),
                ),
            ),
        ),
    )
    cols = encode(ep, layout)
    assert cols["observation.state.arm.tcp_pose"].shape == (5, 7)
    decoded = decode(cols, layout)
    assert np.allclose(decoded.poses[("state", keys.ARM_SINGLE)], mats, atol=1e-5)
