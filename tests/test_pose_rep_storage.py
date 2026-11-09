"""Pose 可按任意 rep 原样存储。"""

from __future__ import annotations

import numpy as np

from phi_rds import MATRIX_REP, Pose, encode_pose
from phi_rds.core.body import RobotBody, Arm
from phi_rds import keys


def test_pose_stores_quat_without_forcing_matrix():
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, 0, 3] = [0.0, 0.1, 0.2]
    quat7 = encode_pose(T, "quat_wxyz")

    pose = Pose.from_rep(quat7, "quat_wxyz")
    assert pose.rep == "quat_wxyz"
    assert pose.numpy().shape == (3, 7)
    assert np.allclose(pose.as_matrix()[:, 0, 3], [0.0, 0.1, 0.2], atol=1e-6)
    assert np.allclose(pose.as_rep("quat_wxyz"), quat7)


def test_pose_matrix_default_and_ee_matrix():
    T = np.tile(np.eye(4), (2, 1, 1))
    pose = Pose.from_matrix(T)
    assert pose.rep == MATRIX_REP
    assert pose.as_matrix().shape == (2, 4, 4)

    body = RobotBody(arm_single=Arm(ee=pose))
    assert body.ee_matrix(keys.ARM_SINGLE).shape == (2, 4, 4)
