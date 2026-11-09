"""入门 API：from_arrays → fill_delta_chunks → create_l2 → open_dataset。"""

from __future__ import annotations

import numpy as np

from phi_rds import Episode, Segment, create_l2, fill_delta_chunks, open_dataset, schema_for


def test_schema_for_l2():
    s = schema_for("umi", level="L2")
    assert s.is_l2
    assert s.standard_version
    assert "observation.state.arm.tcp_pose" in s.feature_names


def test_from_arrays_create_open(tmp_path):
    n = 4
    ee = np.tile(np.eye(4), (n, 1, 1))
    ee[:, 0, 3] = np.linspace(0, 1, n)
    joints = np.zeros((n, 7))
    ep = Episode.from_arrays(
        "a",
        fps=30.0,
        task="demo",
        subtasks=[Segment(0, 2, "reach"), Segment(2, 4, "grasp")],
        ee=ee,
        ee_action=ee.copy(),
        joints=joints,
        joints_action=joints.copy(),
        gripper=np.ones(n),
        gripper_action=np.ones(n),
    )
    root = tmp_path / "ds"
    w = create_l2(root, robot="umi", name="t")
    w.add_episode(fill_delta_chunks(ep, w.schema.layout))
    w.finalize()

    assert (root / "meta" / "subtasks.parquet").is_file()
    assert (root / "meta" / "tasks.parquet").is_file()

    ds = open_dataset(root)
    assert ds.is_l2
    assert len(ds) == 1
    got = ds[0]
    assert got.uid == "a"
    assert got.state.arm_single is not None
    assert got.state.arm_single.ee is not None
    assert got.state.arm_single.ee.numpy().shape == (n, 4, 4)
    assert abs(float(got.state.arm_single.ee.numpy()[-1, 0, 3]) - 1.0) < 1e-5
    assert got.labels.task == "demo"
    assert [(s.start, s.end, s.text) for s in got.labels.subtasks] == [
        (0, 2, "reach"),
        (2, 4, "grasp"),
    ]
