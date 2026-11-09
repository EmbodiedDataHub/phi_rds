#!/usr/bin/env python3
"""AgiBot 完整示例：拼 Episode → 写出 L2 → 打开读取。

AgiBot 按双臂（left/right）处理。运行：

  cd phi_rds && pip install -e ".[lerobot]" && python examples/agibot_write_load.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phi_rds import (
    Arm,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    Gripper,
    JointGroup,
    Labels,
    LazyHandle,
    Pose,
    RobotBody,
    create_l2,
    encode_pose,
    keys,
    open_dataset,
)
from phi_rds.convert import fill_delta_chunks


def _poses(n: int, *, drift: float = 0.01, y0: float = 0.0) -> np.ndarray:
    """(N,4,4)：单位旋转 + 沿 x 平移。"""
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(n) * drift
    T[:, 1, 3] = y0
    return T


def _arm(ee: np.ndarray, joints: np.ndarray, grip: np.ndarray, part: str) -> Arm:
    return Arm(
        ee=Pose.from_matrix(ee),
        joints=JointGroup(
            position=LazyHandle.from_memory(joints),
            names=keys.arm_joint_names(part, joints.shape[1]),
        ),
        gripper=Gripper(open=LazyHandle.from_memory(grip.reshape(-1, 1))),
    )


def make_agibot_episode(uid: str, n: int = 8, *, task: str = "pick_place") -> Episode:
    """双臂 AgiBot 一条假轨迹（state ≈ action，仅演示读写）。"""
    jl = np.zeros((n, 7), dtype=np.float64)
    jr = np.zeros((n, 7), dtype=np.float64)
    jl[:, 0] = np.linspace(0, 0.3, n)
    jr[:, 0] = np.linspace(0, -0.3, n)
    gl = np.linspace(0.2, 0.9, n)
    gr = np.linspace(0.9, 0.2, n)

    eye = np.tile(np.eye(4), (n, 1, 1))
    body = RobotBody(
        torso=Pose.from_matrix(eye),
        head=Pose.from_matrix(eye),
        arm_left=_arm(_poses(n, y0=0.15), jl, gl, keys.ARM_LEFT),
        arm_right=_arm(_poses(n, drift=0.012, y0=-0.15), jr, gr, keys.ARM_RIGHT),
    )
    return Episode(
        meta=EpisodeMeta(
            uid=uid,
            num_frames=n,
            fps=30.0,
            task=task,
            stats=EpisodeStats(num_frames=n),
        ),
        state=body,
        action=body,
        labels=Labels(task=task),
    )


def main() -> None:
    out = Path(__file__).resolve().parent / "_demo_agibot_l2"

    # ------------------------------------------------------------------
    # 1) 写出 L2：AgiBot 双臂预设
    # ------------------------------------------------------------------
    w = create_l2(
        out,
        robot="agibot",
        name="agibot_demo",
        version="v1",
        arms="bimanual",
        fps=30.0,
    )
    layout = w.schema.layout
    w.add_episode(fill_delta_chunks(make_agibot_episode("ep-000"), layout))
    w.add_episode(fill_delta_chunks(make_agibot_episode("ep-001", task="handover"), layout))
    root = w.finalize()

    print("[write]", root)
    print("       level=", w.schema.level, "standard=", w.schema.standard_version)
    print("       layout=", w.schema.layout.id if w.schema.layout else None)
    print("       features=")
    for name in w.schema.feature_names:
        print("        ", name)

    # ------------------------------------------------------------------
    # 2) 打开：按 robot_type=agibot 推断双臂 layout
    # ------------------------------------------------------------------
    ds = open_dataset(root)
    print("[read] episodes=", len(ds), "level=", ds.level, "standard=", ds.standard_version)

    ep = ds[0]
    print("       uid=", ep.uid, "frames=", ep.num_frames, "task=", ep.labels.task)
    assert ep.state.arm_left is not None and ep.state.arm_right is not None
    Tl = ep.state.arm_left.ee.as_matrix()
    Tr = ep.state.arm_right.ee.as_matrix()
    print("       left  ee", Tl.shape, "x0..x-1", f"{Tl[0, 0, 3]:.4f}..{Tl[-1, 0, 3]:.4f}")
    print("       right ee", Tr.shape, "y0", f"{Tr[0, 1, 3]:.4f}")

    # ------------------------------------------------------------------
    # 3) 转换 + 训练旁路
    # ------------------------------------------------------------------
    print("[convert] left pose9", encode_pose(Tl, "pose9").shape)
    official = ds.as_lerobot()
    print("[train]  LeRobotDataset frames=", official.meta.total_frames, "fps=", official.meta.fps)
    print("OK")


if __name__ == "__main__":
    main()
