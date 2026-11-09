#!/usr/bin/env python3
"""AgiBot → L1（无损 v3）→ L2（语义统一）。仅依赖 phi_rds（+kinpy FK）。

  L1: 源数据原样进 LeRobot v3，仅换标准字段名
  L2 convert: 读 L1 → open01 / pose9 / FK+align / fill_delta_chunks
  L2 write:   encode 只序列化（不算 delta）
  Ray:        --ray → map(convert) / reduce(write)

  python examples/agibot_l0_l1_l2.py --max-episodes 1
  python examples/agibot_l0_l1_l2.py --max-episodes 4 --ray
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from phi_rds import (
    Arm,
    CameraStream,
    DatasetMeta,
    DatasetSchema,
    Episode,
    EpisodeMeta,
    EpisodeStats,
    FeatureSpec,
    Gripper,
    JointGroup,
    Labels,
    LazyHandle,
    Pose,
    RobotBody,
    create_dataset,
    create_l2,
    encode_pose,
    keys,
    open_dataset,
    schema_for,
)
from phi_rds.convert import fill_delta_chunks
from phi_rds.core.episode import Field, FieldKind, FieldMap
from phi_rds.core.meta import EpisodeSpan
from phi_rds.core.semantics import FieldSemantics
from phi_rds.exec import (
    LocalExecutor,
    RayExecutor,
    check_results,
    detect_num_workers,
    load_shard_episodes,
    make_rows,
    process_row,
)
from phi_rds.io.lerobot import open_lerobot_dataset
from phi_rds.robots.agibot import (
    JOINTS_ARM_L as _URDF_ARM_L,
    JOINTS_ARM_R as _URDF_ARM_R,
    JOINTS_HEAD as _URDF_HEAD,
    JOINTS_WAIST as _URDF_WAIST,
    agibot_frame_poses,
)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/agibot_demo/sample_dataset"
OUT_L1 = REPO / "data/output/agibot_v3_raw/demo/g1"
OUT_L2 = REPO / "data/output/agibot_v3_sem/demo/g1"
FPS = 30.0

# L1：源相机全量入库（含鱼眼）；L2：仅头部中间 RGB + 左右腕 RGB
CAM_STD = {
    "head": keys.CAM_HEAD,  # head_color.mp4 = 头部中间 RGB
    "hand_left": keys.CAM_WRIST_LEFT,
    "hand_right": keys.CAM_WRIST_RIGHT,
    "head_center_fisheye": keys.CAM_HEAD_FISHEYE,
    "head_left_fisheye": keys.CAM_HEAD_LEFT_FISHEYE,
    "head_right_fisheye": keys.CAM_HEAD_RIGHT_FISHEYE,
    "back_left_fisheye": keys.CAM_BACK_LEFT_FISHEYE,
    "back_right_fisheye": keys.CAM_BACK_RIGHT_FISHEYE,
}
# L2 只保留这三路；不写 cameras 内外参列、不挂 depth/fisheye
CAM_L2 = (keys.CAM_HEAD, keys.CAM_WRIST_LEFT, keys.CAM_WRIST_RIGHT)

EE_NAMES = (
    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw",
    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw",
)
JOINT_NAMES = tuple(
    [f"left_joint{i}" for i in range(1, 8)] + [f"right_joint{i}" for i in range(1, 8)]
)

# 写出到 meta/field_semantics.json；names 同时进 info.json features
FIELD_SEMANTICS: dict[str, dict] = {
    "observation.state.arm.ee_pose": {
        "description": "双臂末端：left/right 各 xyz(m)+quat(xyzw)。来自 h5 state/end/{position,orientation}，仅拼接。",
        "layout": "left[x,y,z,qx,qy,qz,qw] + right[...]",
        "names": list(EE_NAMES),
        "units": {"xyz": "m", "quat": "unit_quaternion"},
        "quaternion_order": "xyzw",
        "source": ["state/end/position", "state/end/orientation"],
    },
    "action.arm.ee_pose": {
        "description": "双臂末端目标，布局同 state.ee_pose。来自 action/end/{position,orientation}。",
        "layout": "left[x,y,z,qx,qy,qz,qw] + right[...]",
        "names": list(EE_NAMES),
        "units": {"xyz": "m", "quat": "unit_quaternion"},
        "quaternion_order": "xyzw",
        "source": ["action/end/position", "action/end/orientation"],
    },
    "observation.state.arm.joint": {
        "description": "双臂关节角，左7+右7。来自 state/joint/position。",
        "names": list(JOINT_NAMES),
        "units": "rad",
        "source": ["state/joint/position"],
    },
    "action.arm.joint": {
        "description": "双臂关节目标，左7+右7。来自 action/joint/position。",
        "names": list(JOINT_NAMES),
        "units": "rad",
        "source": ["action/joint/position"],
    },
    "observation.state.arm.joint_current": {
        "description": "双臂关节电流 current_value（非 effort）。",
        "names": list(JOINT_NAMES),
        "units": "vendor_raw",
        "source": ["state/joint/current_value"],
    },
    "observation.state.arm.gripper": {
        "description": "夹爪 effector 行程 mm（越大越合，非开口宽度）。来自 state/effector/position。",
        "names": ["left_gripper_travel_mm", "right_gripper_travel_mm"],
        "units": "mm",
        "source": ["state/effector/position"],
    },
    "action.arm.gripper": {
        "description": "夹爪闭合指令 closedness∈[0,1]，1=合。来自 action/effector/position。",
        "names": ["left_gripper_closedness", "right_gripper_closedness"],
        "units": "closedness_01",
        "source": ["action/effector/position"],
    },
    "observation.state.torso.joint": {
        "description": "腰 [pitch, lift]，源列序不变。来自 state/waist/position。",
        "names": ["waist_pitch", "waist_lift"],
        "units": {"waist_pitch": "rad", "waist_lift": "m"},
        "source": ["state/waist/position"],
    },
    "action.torso.joint": {
        "description": "腰目标 [pitch, lift]。来自 action/waist/position。",
        "names": ["waist_pitch", "waist_lift"],
        "units": {"waist_pitch": "rad", "waist_lift": "m"},
        "source": ["action/waist/position"],
    },
    "observation.state.head.joint": {
        "description": "头 [yaw, pitch]。来自 state/head/position。",
        "names": ["head_yaw", "head_pitch"],
        "units": "rad",
        "source": ["state/head/position"],
    },
    "action.head.joint": {
        "description": "头目标 [yaw, pitch]。来自 action/head/position。",
        "names": ["head_yaw", "head_pitch"],
        "units": "rad",
        "source": ["action/head/position"],
    },
    "action.base.velocity": {
        "description": "底盘速度 (2,)，轴含义同厂商。来自 action/robot/velocity。",
        "names": ["base_vel_0", "base_vel_1"],
        "units": "vendor_raw",
        "source": ["action/robot/velocity"],
    },
    "observation.timestamp_ns": {
        "description": "采集时间戳 ns。异于 LeRobot 自动 timestamp（秒）。来自 /timestamp。",
        "names": ["timestamp_ns"],
        "units": "ns",
        "source": ["timestamp"],
    },
}


def list_refs(source: Path) -> list[tuple[int, int]]:
    refs = []
    for td in sorted((source / "proprio_stats").iterdir()):
        if not td.is_dir() or not td.name.isdigit():
            continue
        for ed in sorted(td.iterdir()):
            if (ed / "proprio_stats.h5").is_file():
                refs.append((int(td.name), int(ed.name)))
    return refs


def _h5(group, *path) -> np.ndarray | None:
    node = group
    for k in path:
        if node is None or k not in node:
            return None
        node = node[k]
    arr = np.asarray(node)
    return None if arr.size == 0 else arr


def load_raw(source: Path, task_id: int, episode_id: int) -> dict:
    h5 = source / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
    with h5py.File(h5, "r", locking=False) as f:
        st, act = f["state"], f.get("action")
        n = len(f["timestamp"])
        data = {
            "n": n,
            "timestamp_ns": np.asarray(f["timestamp"], dtype=np.int64),
            "state_joint": _h5(st, "joint", "position"),
            "state_joint_current": _h5(st, "joint", "current_value"),
            "state_waist": _h5(st, "waist", "position"),
            "state_head": _h5(st, "head", "position"),
            "state_grip": _h5(st, "effector", "position"),
            "state_ee_pos": _h5(st, "end", "position"),
            "state_ee_ori": _h5(st, "end", "orientation"),
            "action_joint": _h5(act, "joint", "position"),
            "action_waist": _h5(act, "waist", "position"),
            "action_head": _h5(act, "head", "position"),
            "action_grip": _h5(act, "effector", "position"),
            "action_ee_pos": _h5(act, "end", "position"),
            "action_ee_ori": _h5(act, "end", "orientation"),
            "action_base_vel": _h5(act, "robot", "velocity"),
        }

    task = f"task_{task_id}"
    task_rec = None
    tj = source / "task_info" / f"task_{task_id}.json"
    if tj.is_file():
        records = json.loads(tj.read_text())
        for rec in (records if isinstance(records, list) else [records]):
            if rec.get("episode_id") == episode_id:
                task = str(rec.get("task_name") or task)
                task_rec = rec
                break

    cams: dict[str, str] = {}
    vdir = source / "observations" / str(task_id) / str(episode_id) / "videos"
    if vdir.is_dir():
        for p in sorted(vdir.glob("*.mp4")):
            stem = p.stem[:-6] if p.stem.endswith("_color") else p.stem
            slot = CAM_STD.get(stem)
            if slot is None:
                continue
            # 头部中间 RGB 优先 head_color；勿被其它同槽覆盖
            if slot in cams and slot == keys.CAM_HEAD and stem != "head":
                continue
            cams[slot] = str(p.resolve())

    return {
        "uid": f"task{task_id}/ep{episode_id}",
        "task": task,
        "task_rec": task_rec,
        "cams": cams,
        "task_id": task_id,
        "episode_id": episode_id,
        **data,
    }


def pack_ee_pose(pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """(N,2,3)+(N,2,4) → (N,14)：每臂 xyz||quat(xyzw)，左|右。"""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2, 3)
    ori = np.asarray(ori, dtype=np.float64).reshape(-1, 2, 4)
    return np.concatenate([pos[:, 0], ori[:, 0], pos[:, 1], ori[:, 1]], axis=1)


def _spec(name: str, array: np.ndarray) -> FeatureSpec:
    arr = np.asarray(array)
    shape = tuple(int(x) for x in arr.shape[1:]) if arr.ndim > 1 else (1,)
    meta = FIELD_SEMANTICS[name]
    return FeatureSpec(
        name=name,
        dtype="int64" if np.issubdtype(arr.dtype, np.integer) else "float64",
        shape=shape or (1,),
        names=tuple(meta["names"]),
        extras={"phi_semantics": {k: v for k, v in meta.items() if k != "names"}},
    )


def _field(name: str, array: np.ndarray) -> Field:
    meta = FIELD_SEMANTICS[name]
    return Field(
        id=name,
        kind=FieldKind.VECTOR,
        semantics=FieldSemantics(
            kind="vector",
            role="action" if name.startswith("action.") else "observation",
            extras={k: v for k, v in meta.items() if k != "names"},
        ),
        payload=LazyHandle.from_memory(np.asarray(array)),
    )


def build_l1_episode(raw: dict) -> Episode:
    n = raw["n"]
    cols: dict[str, np.ndarray] = {}
    if raw["state_joint"] is not None:
        cols["observation.state.arm.joint"] = np.asarray(raw["state_joint"], dtype=np.float64)
    if raw["state_joint_current"] is not None:
        cols["observation.state.arm.joint_current"] = np.asarray(raw["state_joint_current"], dtype=np.float64)
    if raw["state_waist"] is not None:
        cols["observation.state.torso.joint"] = np.asarray(raw["state_waist"], dtype=np.float64)
    if raw["state_head"] is not None:
        cols["observation.state.head.joint"] = np.asarray(raw["state_head"], dtype=np.float64)
    if raw["state_grip"] is not None:
        cols["observation.state.arm.gripper"] = np.asarray(raw["state_grip"], dtype=np.float64)
    if raw["state_ee_pos"] is not None and raw["state_ee_ori"] is not None:
        cols["observation.state.arm.ee_pose"] = pack_ee_pose(raw["state_ee_pos"], raw["state_ee_ori"])
    if raw["action_joint"] is not None:
        cols["action.arm.joint"] = np.asarray(raw["action_joint"], dtype=np.float64)
    if raw["action_waist"] is not None:
        cols["action.torso.joint"] = np.asarray(raw["action_waist"], dtype=np.float64)
    if raw["action_head"] is not None:
        cols["action.head.joint"] = np.asarray(raw["action_head"], dtype=np.float64)
    if raw["action_grip"] is not None:
        cols["action.arm.gripper"] = np.asarray(raw["action_grip"], dtype=np.float64)
    if raw["action_ee_pos"] is not None and raw["action_ee_ori"] is not None:
        cols["action.arm.ee_pose"] = pack_ee_pose(raw["action_ee_pos"], raw["action_ee_ori"])
    if raw["action_base_vel"] is not None:
        cols["action.base.velocity"] = np.asarray(raw["action_base_vel"], dtype=np.float64)
    cols["observation.timestamp_ns"] = np.asarray(raw["timestamp_ns"], dtype=np.int64).reshape(-1, 1)

    missing = set(cols) - set(FIELD_SEMANTICS)
    if missing:
        raise RuntimeError(f"FIELD_SEMANTICS missing {sorted(missing)}")

    fmap = FieldMap()
    for name, arr in cols.items():
        if arr.shape[0] != n:
            raise ValueError(f"{name}: {arr.shape[0]} != {n}")
        fmap.put(_field(name, arr))

    extras = {
        "task_id": raw["task_id"],
        "episode_id": raw["episode_id"],
    }
    if raw["task_rec"] is not None:
        extras["task_rec"] = raw["task_rec"]

    return Episode(
        meta=EpisodeMeta(
            uid=raw["uid"], num_frames=n, fps=FPS, task=raw["task"],
            stats=EpisodeStats(num_frames=n),
            span=EpisodeSpan(extras=extras),
        ),
        state=RobotBody(),
        cameras=tuple(
            CameraStream(name=n, rgb=LazyHandle.from_video(p))
            for n, p in sorted(raw["cams"].items())
        ),
        labels=Labels(task=raw["task"]),
        fields=fmap,
        source=str(SRC),
    )


def write_l1(episodes: list[Episode], root: Path) -> Path:
    specs = [
        _spec(n, np.asarray(episodes[0].fields[n].payload.load()))
        for n in sorted(episodes[0].fields.ids())
    ]
    schema = DatasetSchema.free(
        specs, fps=FPS, robot_type="agibot", id="agibot_g1_l1", level="L1",
    )
    w = create_dataset(
        root, schema,
        DatasetMeta(name="agibot_g1_l1", version="v1", robot="agibot", uri=""),
        overwrite=True,
    )
    for ep in episodes:
        w.add_episode(ep)
    out = w.finalize()

    # L1/L2 均不入库相机内外参 / depth；extras 仅记 task 元信息
    for i, ep in enumerate(episodes):
        dest = out / "meta" / "source_assets" / f"episode-{i:06d}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "extras.json").write_text(
            json.dumps(ep.meta.span.extras, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    info = json.loads((out / "meta" / "info.json").read_text())
    feats = {}
    for name, meta in FIELD_SEMANTICS.items():
        if name in info["features"]:
            feats[name] = dict(meta)
    for name, blob in info["features"].items():
        if blob.get("dtype") != "video":
            continue
        cam = keys.camera_from_key(name) if keys.is_image_key(name) else name
        feats[name] = {
            "description": f"RGB mp4，标准键 {name}，相机槽 {cam}；不解码不缩放。",
            "dtype": "video",
            "camera_slot": cam,
        }
    (out / "meta" / "field_semantics.json").write_text(
        json.dumps({"layer": "L1", "features": feats}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def _col(table, name: str) -> np.ndarray:
    vals = table.column(name).to_pylist()
    arr = np.asarray(vals)
    return np.stack(vals) if getattr(arr, "dtype", None) == object else arr


def ee_pose_to_matrices(ee: np.ndarray) -> np.ndarray:
    """(N,14) xyz+quat(xyzw) → (N,2,4,4)。"""
    from scipy.spatial.transform import Rotation as R

    ee = np.asarray(ee, dtype=np.float64).reshape(-1, 14)
    out = np.tile(np.eye(4), (len(ee), 2, 1, 1))
    for arm, sl in enumerate((slice(0, 7), slice(7, 14))):
        b = ee[:, sl]
        out[:, arm, :3, 3] = b[:, :3]
        out[:, arm, :3, :3] = R.from_quat(b[:, 3:7]).as_matrix()
    return out


def _arm(ee, joints, grip, part, *, urdf_names: tuple[str, ...]) -> Arm:
    return Arm(
        ee=Pose.from_matrix(ee),
        joints=JointGroup(
            position=LazyHandle.from_memory(np.asarray(joints, dtype=np.float64)),
            names=urdf_names,
        ),
        gripper=Gripper(open=LazyHandle.from_memory(np.asarray(grip, dtype=np.float64).reshape(-1, 1))),
    )


def _fill_poses(body: RobotBody) -> RobotBody:
    """关节 → URDF FK + calib align，写 torso/head/arm 位姿。

    约定与黄金 ``phi_data.datasets.agibot.Convert.poses`` / ``agibot_pad_demo`` 一致：
    torso 为根下绝对；其余部位为相对 torso（viewer 验证用 ``torso @ pose`` 还原）。
    """
    angles = body.joint_angles()
    if not angles:
        return body
    return body.with_ee(agibot_frame_poses(angles))


def episode_from_l1(root: Path, ep_index: int, uid: str) -> Episode:
    lr = open_lerobot_dataset(root)
    if getattr(lr, "hf_dataset", None) is None and hasattr(lr, "_ensure_hf_dataset_loaded"):
        lr._ensure_hf_dataset_loaded()
    row = lr.meta.episodes[ep_index]
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    table = lr.hf_dataset.data.slice(start, stop - start)
    n = stop - start
    task = ""
    if row.get("tasks"):
        t = row["tasks"]
        task = t[0] if isinstance(t, list) else str(t)

    T = ee_pose_to_matrices(_col(table, "observation.state.arm.ee_pose"))
    # L1 gripper 仍是 mm；官方无端点 → 本 episode 按列 min/max → open01（1=开）
    g_mm = _col(table, "observation.state.arm.gripper")
    g_lo = g_mm.min(axis=0)
    g_hi = g_mm.max(axis=0)
    g_open = np.clip((g_hi - g_mm) / np.maximum(g_hi - g_lo, 1e-6), 0.0, 1.0)
    j = _col(table, "observation.state.arm.joint")
    # a_t = s_{t+1}（末帧重复）；夹爪仍用 L1 action 指令 open01
    j_act = np.concatenate([j[1:], j[-1:]], axis=0)
    T_act = np.concatenate([T[1:], T[-1:]], axis=0)
    if "action.arm.gripper" in table.column_names:
        g_act = np.clip(1.0 - _col(table, "action.arm.gripper"), 0.0, 1.0)
    else:
        g_act = np.concatenate([g_open[1:], g_open[-1:]], axis=0)

    def _shift_jg(jg: JointGroup | None) -> JointGroup | None:
        if jg is None:
            return None
        pos = np.asarray(jg.position[:], dtype=np.float64)  # materialize lazy
        nxt = np.concatenate([pos[1:], pos[-1:]], axis=0)
        return JointGroup(position=LazyHandle.from_memory(nxt), names=jg.names)

    # L1 torso.joint 源序 [pitch, lift] → FK/URDF 组序 [lift, pitch]
    waist = head_j = None
    if "observation.state.torso.joint" in table.column_names:
        w = np.asarray(_col(table, "observation.state.torso.joint"), dtype=np.float64)
        waist = JointGroup(
            position=LazyHandle.from_memory(np.stack([w[:, 1], w[:, 0]], axis=1)),
            names=_URDF_WAIST,
        )
    if "observation.state.head.joint" in table.column_names:
        h = np.asarray(_col(table, "observation.state.head.joint"), dtype=np.float64)
        head_j = JointGroup(
            position=LazyHandle.from_memory(h),
            names=_URDF_HEAD,
        )

    action = _fill_poses(RobotBody(
        arm_left=_arm(T_act[:, 0], j_act[:, :7], g_act[:, 0], keys.ARM_LEFT, urdf_names=_URDF_ARM_L),
        arm_right=_arm(T_act[:, 1], j_act[:, 7:14], g_act[:, 1], keys.ARM_RIGHT, urdf_names=_URDF_ARM_R),
        waist=_shift_jg(waist),
        head_joints=_shift_jg(head_j),
    ))

    # L2 相机：只要 head + wrist_left/right；顺序固定；不附内外参
    by_cam: dict[str, str] = {}
    for vk in lr.meta.video_keys:
        cam = keys.camera_from_key(vk) if keys.is_image_key(vk) else vk
        if cam not in CAM_L2:
            continue
        try:
            path = root / lr.meta.get_video_file_path(ep_index, vk)
        except Exception:  # noqa: BLE001
            continue
        if path.is_file():
            by_cam[cam] = str(path)
    cams = [
        CameraStream(name=cam, rgb=LazyHandle.from_video(by_cam[cam]))
        for cam in CAM_L2
        if cam in by_cam
    ]

    state = _fill_poses(RobotBody(
        arm_left=_arm(T[:, 0], j[:, :7], g_open[:, 0], keys.ARM_LEFT, urdf_names=_URDF_ARM_L),
        arm_right=_arm(T[:, 1], j[:, 7:14], g_open[:, 1], keys.ARM_RIGHT, urdf_names=_URDF_ARM_R),
        waist=waist,
        head_joints=head_j,
    ))

    return Episode(
        meta=EpisodeMeta(
            uid=uid, num_frames=n, fps=FPS, task=task,
            stats=EpisodeStats(num_frames=n),
        ),
        state=state,
        action=action,
        cameras=tuple(cams),
        labels=Labels(task=task),
        source=str(root),
    )


def _l2_layout():
    return schema_for("agibot", level="L2", fps=FPS, arms="bimanual").layout


def convert_l2_episode(root: Path, ep_index: int, uid: str) -> Episode:
    """转换阶段：L1→几何统一（FK）→ delta_chunk 进 fields。写出前必须完成。"""
    ep = episode_from_l1(root, ep_index, uid)
    return fill_delta_chunks(ep, _l2_layout())


def convert_l2_from_row(row: dict) -> list[Episode]:
    """Ray/Local map 入口（模块级，可 pickle）。``row['items']`` = [(ep_index, uid), …]。"""
    l1_root = Path(row["l1_root"])
    layout = _l2_layout()
    out: list[Episode] = []
    for ep_index, uid in row["items"]:
        ep = episode_from_l1(l1_root, int(ep_index), str(uid))
        out.append(fill_delta_chunks(ep, layout))
    return out


def write_l2(episodes: list[Episode], root: Path) -> Path:
    """只写出：要求 Episode.fields 已含 delta_chunk（见 convert_l2_episode）。"""
    w = create_l2(
        root, robot="agibot", name="agibot_g1_l2", version="v1",
        arms="bimanual", fps=FPS, overwrite=True,
    )
    for ep in episodes:
        w.add_episode(ep)
    return w.finalize()


def write_l2_via_exec(
    l1_root: Path,
    uids: list[str],
    out: Path,
    *,
    ray: bool = False,
    episodes_per_shard: int | None = None,
    num_workers: int | str | None = None,
) -> Path:
    """convert(map) → driver reduce 写出。

    分块二选一：``episodes_per_shard``（每 worker 固定集数）或 ``num_workers``/``\"auto\"``（均分）。
    """
    import sys

    examples_dir = str(Path(__file__).resolve().parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    items = [(i, uids[i]) for i in range(len(uids))]
    rows = make_rows(
        convert="agibot_l0_l1_l2:convert_l2_from_row",
        items=items,
        output=out,
        episodes_per_shard=episodes_per_shard,
        num_workers=num_workers,  # type: ignore[arg-type]
        extra={"l1_root": str(l1_root.resolve())},
    )
    sizes = [int(r["n_items"]) for r in rows]
    print(f"  shards={len(rows)} sizes={sizes}"
          + (f" (auto workers≈{detect_num_workers()})" if num_workers == "auto" else ""))
    executor = RayExecutor() if ray else LocalExecutor()
    results = executor.run(rows, process_row, check_results)
    episodes: list[Episode] = []
    for result in results:
        episodes.extend(load_shard_episodes(result["shard_path"]))
    root = write_l2(episodes, out)
    shard_dir = out / "_shards"
    if shard_dir.is_dir():
        import shutil

        shutil.rmtree(shard_dir, ignore_errors=True)
    return root


def check_l1(raw: dict, root: Path) -> None:
    import pyarrow.parquet as pq

    m = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    table = pq.read_table(
        root / "data" / f"chunk-{int(m['data/chunk_index']):03d}"
        / f"file-{int(m['data/file_index']):03d}.parquet"
    )
    mask = np.asarray(table.column("episode_index").to_pylist()) == 0

    def take(name):
        col = np.asarray(table.column(name).to_pylist())[mask]
        return np.stack(col) if getattr(col, "dtype", None) == object else np.asarray(col)

    assert float(np.max(np.abs(take("observation.state.arm.joint") - raw["state_joint"]))) == 0.0
    assert float(np.max(np.abs(
        take("observation.state.arm.ee_pose") - pack_ee_pose(raw["state_ee_pos"], raw["state_ee_ori"])
    ))) == 0.0
    assert float(np.max(np.abs(take("observation.state.arm.gripper") - raw["state_grip"]))) == 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--max-episodes", type=int, default=1)
    ap.add_argument("--ray", action="store_true", help="L1→L2 转换用 RayExecutor.map")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--episodes-per-shard", type=int, default=None,
        help="每 worker 固定处理的 episode 数（优先）",
    )
    g.add_argument(
        "--num-workers", default=None,
        help="worker 数均分；传 auto 则探测可用 CPU/Ray 资源",
    )
    args = ap.parse_args()
    if not args.source.is_dir():
        print("missing source:", args.source)
        return 1

    print("=== source → L1 ===")
    raws, eps = [], []
    for tid, eid in list_refs(args.source)[: args.max_episodes]:
        raw = load_raw(args.source, tid, eid)
        ep = build_l1_episode(raw)
        raws.append(raw)
        eps.append(ep)
        print(f"  {ep.uid} frames={ep.num_frames} cams={[c.name for c in ep.cameras]}")
    root1 = write_l1(eps, OUT_L1)
    check_l1(raws[0], root1)
    info1 = json.loads((root1 / "meta/info.json").read_text())
    print(f"  {root1}  v={info1.get('codebase_version')} level={info1.get('phi_level')}")

    print("=== L1 → L2 convert → write ===")
    from phi_rds.io.lerobot import load_episode_uids
    uids = load_episode_uids(root1)
    num_workers: int | str | None = None
    if args.num_workers is not None:
        num_workers = "auto" if args.num_workers == "auto" else int(args.num_workers)
    use_exec = bool(args.ray or args.max_episodes > 1 or args.episodes_per_shard or num_workers)
    if use_exec:
        print(f"  exec={'ray' if args.ray else 'local'}")
        root2 = write_l2_via_exec(
            root1, uids, OUT_L2,
            ray=args.ray,
            episodes_per_shard=args.episodes_per_shard,
            num_workers=num_workers,
        )
    else:
        l2_eps = [convert_l2_episode(root1, i, uids[i]) for i in range(len(uids))]
        root2 = write_l2(l2_eps, OUT_L2)
    info2 = json.loads((root2 / "meta/info.json").read_text())
    ep0 = open_dataset(root2)[0]
    print(f"  {root2}  v={info2.get('codebase_version')} level={info2.get('phi_level')} "
          f"std={info2.get('phi_standard_version')}")
    print(f"  ee={ep0.state.arm_left.ee.as_matrix().shape} "
          f"pose9={encode_pose(ep0.state.arm_left.ee.as_matrix(), 'pose9').shape}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
