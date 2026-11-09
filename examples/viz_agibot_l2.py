#!/usr/bin/env python3
"""可视化验收 L2：写出的 head/torso/tcp pose vs phi_rds FK 重算。

不依赖 phi_data / LeRobotDataset（直接读 parquet），避免 viewer 在 2D delta_chunk 上崩。

  cd phi_rds && python examples/viz_agibot_l2.py
  # → data/output/agibot_v3_sem/demo/g1/viz_l2_poses.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from phi_rds import encode_pose, keys
from phi_rds.robots.agibot import (
    JOINTS_ARM_L,
    JOINTS_ARM_R,
    JOINTS_HEAD,
    JOINTS_WAIST,
    AgibotG1,
)

REPO = Path(__file__).resolve().parents[2]
L2 = REPO / "data/output/agibot_v3_sem/demo/g1"
OUT = L2 / "viz_l2_poses.png"


def _ep0_table(root: Path):
    meta = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    table = pq.read_table(
        root / "data" / f"chunk-{int(meta['data/chunk_index']):03d}"
        / f"file-{int(meta['data/file_index']):03d}.parquet"
    )
    mask = np.asarray(table.column("episode_index").to_pylist()) == 0
    return table, mask


def _col(table, mask, name: str) -> np.ndarray:
    vals = np.asarray(table.column(name).to_pylist())[mask]
    return np.stack(vals) if getattr(vals, "dtype", None) == object else np.asarray(vals)


def main() -> int:
    if not (L2 / "meta/info.json").is_file():
        print("missing L2:", L2)
        return 1

    table, mask = _ep0_table(L2)
    head_w = _col(table, mask, "observation.state.head.pose")
    torso_w = _col(table, mask, "observation.state.torso.pose")
    tcp_w = _col(table, mask, "observation.state.arm.tcp_pose")
    tj = _col(table, mask, "observation.state.torso.joint")
    hj = _col(table, mask, "observation.state.head.joint")
    aj = _col(table, mask, "observation.state.arm.joint")

    # std torso [yaw,pitch,roll,lift] → URDF [lift,pitch]
    waist = np.stack([tj[:, 3], tj[:, 1]], axis=1)
    angles: dict[str, np.ndarray] = {}
    for i, n in enumerate(JOINTS_WAIST):
        angles[n] = waist[:, i]
    for i, n in enumerate(JOINTS_HEAD):
        angles[n] = hj[:, i]
    for i, n in enumerate(JOINTS_ARM_L):
        angles[n] = aj[:, i]
    for i, n in enumerate(JOINTS_ARM_R):
        angles[n] = aj[:, 7 + i]

    poses = AgibotG1().poses(angles)
    head_fk = encode_pose(poses[keys.HEAD], "pose9")
    torso_fk = encode_pose(poses[keys.TORSO], "pose9")

    err_h = float(np.max(np.abs(head_w - head_fk)))
    err_t = float(np.max(np.abs(torso_w - torso_fk)))
    print(f"head.pose  max|Δ| vs FK = {err_h:.3e}")
    print(f"torso.pose max|Δ| vs FK = {err_t:.3e}")

    info = json.loads((L2 / "meta/info.json").read_text())
    vids = sorted(p.name for p in (L2 / "videos").iterdir() if p.is_dir())
    print("cameras", vids)
    print("base.pose pad", info["features"]["observation.state.base.pose"].get("pad"))

    # subsample for plot
    step = max(1, len(head_w) // 200)
    idx = np.arange(0, len(head_w), step)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    def traj(ax, pose9, color, label):
        ax.plot(pose9[idx, 0], pose9[idx, 1], pose9[idx, 2], color=color, lw=1.2, label=label)

    traj(ax1, head_w, "#1f77b4", "head written")
    traj(ax1, head_fk, "#ff7f0e", "head FK")
    traj(ax1, torso_w, "#2ca02c", "torso written")
    traj(ax1, torso_fk, "#d62728", "torso FK")
    # TCP left xyz
    ax1.plot(tcp_w[idx, 0], tcp_w[idx, 1], tcp_w[idx, 2], color="#9467bd", lw=1.0, label="tcp L")
    ax1.plot(tcp_w[idx, 9], tcp_w[idx, 10], tcp_w[idx, 11], color="#8c564b", lw=1.0, label="tcp R")
    ax1.set_title("L2 poses (m) · written vs FK")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.plot(np.max(np.abs(head_w - head_fk), axis=1), label="head |Δ|∞", color="#1f77b4")
    ax2.plot(np.max(np.abs(torso_w - torso_fk), axis=1), label="torso |Δ|∞", color="#2ca02c")
    ax2.set_yscale("log")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("max abs error")
    ax2.set_title(f"FK closed-loop  head={err_h:.1e}  torso={err_t:.1e}")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"phi_rds L2  ·  cams={', '.join(v.split('.')[-1] for v in vids)}  ·  no phi_data",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)
    ok = err_h < 1e-5 and err_t < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
