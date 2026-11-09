#!/usr/bin/env python3
"""Phi RDS 端到端示例：组装 Episode → 写出 L2 → 打开读取 → 转换工具。

运行：
  cd phi_rds && pip install -e ".[lerobot]" && python examples/end_to_end.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phi_rds import (
    Episode,
    change_frame,
    create_l2,
    encode_pose,
    open_dataset,
)
from phi_rds.convert import fill_delta_chunks


def _fake_poses(n: int, *, drift: float = 0.01) -> np.ndarray:
    """生成 (N,4,4) 位姿：单位旋转 + 沿 x 缓慢平移。"""
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(n) * drift
    return T


def main() -> None:
    out = Path(__file__).resolve().parent / "_demo_l2"
    n = 8

    # ------------------------------------------------------------------
    # 1) 用数组快速组装 Episode（不必手搓 LazyHandle / Layout）
    # ------------------------------------------------------------------
    ee = _fake_poses(n)
    ee_action = _fake_poses(n, drift=0.012)
    joints = np.zeros((n, 7), dtype=np.float64)
    joints[:, 0] = np.linspace(0, 0.4, n)
    gripper = np.linspace(0.2, 0.9, n)

    ep0 = Episode.from_arrays(
        "ep-000",
        fps=30.0,
        task="pick_cup",
        ee=ee,
        ee_action=ee_action,
        joints=joints,
        joints_action=joints.copy(),
        gripper=gripper,
        gripper_action=gripper.copy(),
    )
    ep1 = Episode.from_arrays(
        "ep-001",
        fps=30.0,
        task="pick_cup",
        ee=_fake_poses(n, drift=0.02),
        ee_action=_fake_poses(n, drift=0.02),
        joints=joints + 0.05,
        joints_action=joints + 0.05,
        gripper=1.0 - gripper,
        gripper_action=1.0 - gripper,
    )

    # ------------------------------------------------------------------
    # 2) 转换（delta_chunk）→ 写出 L2（encode 只序列化）
    # ------------------------------------------------------------------
    writer = create_l2(out, robot="umi", name="demo_pick", version="v1")
    layout = writer.schema.layout
    writer.add_episode(fill_delta_chunks(ep0, layout))
    writer.add_episode(fill_delta_chunks(ep1, layout))
    root = writer.finalize()
    print(f"[write] {root}")
    print(f"        level={writer.schema.level} standard={writer.schema.standard_version}")
    print(f"        features={list(writer.schema.feature_names)}")

    # ------------------------------------------------------------------
    # 3) 打开：任何人一行搞定
    # ------------------------------------------------------------------
    ds = open_dataset(root)  # 按 robot_type 自动套机型 layout
    print(f"[read]  episodes={len(ds)} level={ds.level} standard={ds.standard_version}")
    ep = ds[0]
    print(f"        ep0 uid={ep.uid} frames={ep.num_frames} task={ep.labels.task}")
    assert ep.state.arm_single is not None and ep.state.arm_single.ee is not None
    T = ep.state.arm_single.ee.numpy()
    print(f"        ee shape={T.shape}  x0..x-1={T[0, 0, 3]:.4f}..{T[-1, 0, 3]:.4f}")

    # ------------------------------------------------------------------
    # 4) 转换工具：位姿表示 / 换系
    # ------------------------------------------------------------------
    pose9 = encode_pose(T, "pose9")
    print(f"[convert] pose9 shape={pose9.shape}")  # (N, 9)

    # 假设要把位姿从 A 系改写到 B 系：左乘 T_B_A
    T_B_A = np.eye(4)
    T_B_A[:3, 3] = [0.1, 0.0, 0.05]
    T_in_B = change_frame(T, T_B_A)
    print(f"[convert] change_frame x0={T_in_B[0, 0, 3]:.4f} (was {T[0, 0, 3]:.4f})")

    # ------------------------------------------------------------------
    # 5) 训练直连：官方 LeRobotDataset（可选）
    # ------------------------------------------------------------------
    official = ds.as_lerobot()
    print(f"[train]  LeRobotDataset frames={official.meta.total_frames} fps={official.meta.fps}")

    print("OK")


if __name__ == "__main__":
    main()
