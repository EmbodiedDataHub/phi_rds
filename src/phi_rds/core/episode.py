"""Episode：单条轨迹 = EpisodeMeta + 语义 + 几何/相机/标签（payload 懒加载）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from phi_rds import keys
from phi_rds.core.body import Arm, Gripper, JointGroup, Pose, RobotBody
from phi_rds.core.lazy import LazyHandle
from phi_rds.core.meta import EpisodeMeta
from phi_rds.core.semantics import EpisodeSemantics, FieldSemantics
from phi_rds.core.stats import EpisodeStats
from phi_rds.types import FieldId


class FieldKind(str, Enum):
    POSE = "pose"
    JOINTS = "joints"
    GRIPPER = "gripper"
    IMAGE = "image"
    DEPTH = "depth"
    LABEL = "label"
    SCALAR = "scalar"
    VECTOR = "vector"
    DELTA_CHUNK = "delta_chunk"
    OTHER = "other"


@dataclass
class Field:
    id: FieldId
    kind: FieldKind
    semantics: FieldSemantics
    payload: LazyHandle[Any]

    def meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "semantics": self.semantics,
            "shape": self.payload.shape,
            "dtype": self.payload.dtype,
        }


@dataclass
class FieldMap:
    """FieldId → Field；支持按前缀筛选。"""

    _items: dict[FieldId, Field] = field(default_factory=dict)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __getitem__(self, key: FieldId) -> Field:
        return self._items[key]

    def __iter__(self) -> Iterator[FieldId]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: FieldId, default: Field | None = None) -> Field | None:
        return self._items.get(key, default)

    def ids(self) -> tuple[FieldId, ...]:
        return tuple(self._items.keys())

    def values(self) -> tuple[Field, ...]:
        return tuple(self._items.values())

    def put(self, item: Field) -> None:
        self._items[item.id] = item

    def update(self, items: Mapping[FieldId, Field] | list[Field]) -> None:
        if isinstance(items, Mapping):
            self._items.update(items)
        else:
            for item in items:
                self.put(item)

    def prefix(self, prefix: str) -> FieldMap:
        out = FieldMap()
        for key, item in self._items.items():
            if key.startswith(prefix):
                out.put(item)
        return out

    def as_dict(self) -> dict[FieldId, Field]:
        return dict(self._items)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Labels:
    task: str = ""
    subtasks: tuple[Segment, ...] = ()


@dataclass
class CameraStream:
    """一路相机（对齐 LeRobot video key）。默认只挂路径，不解码像素。"""

    name: str
    rgb: LazyHandle[Any] | None = None
    depth: LazyHandle[Any] | None = None
    intrinsic: Any = None
    extrinsic: Any = None

    def calib(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.intrinsic is not None:
            out["intrinsic"] = self.intrinsic
        if self.extrinsic is not None:
            out["extrinsic"] = self.extrinsic
        return out


@dataclass
class Episode:
    """一条轨迹。

    - ``meta``：episode 级元信息（uid / 帧数 / status / span …）
    - ``semantics``：整集语义（可被 meta.semantics_override 覆盖意图时由调用方合并）
    - ``state`` / ``action`` / ``cameras`` / ``labels``：便利视图
    - ``fields``：可选权威字段表（语义 + LazyHandle）；为空时以几何树为准
    """

    meta: EpisodeMeta
    state: RobotBody
    action: RobotBody | None = None
    cameras: tuple[CameraStream, ...] = ()
    labels: Labels = field(default_factory=Labels)
    semantics: EpisodeSemantics = field(default_factory=EpisodeSemantics)
    fields: FieldMap = field(default_factory=FieldMap)
    source: str | None = None

    # --- 常用快捷属性（兼容旧心智模型）---
    @property
    def uid(self) -> str:
        return self.meta.uid

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return self.meta.num_frames

    @property
    def stats(self) -> EpisodeStats:
        return self.meta.stats

    def camera_calib(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for camera in self.cameras:
            blob = camera.calib()
            if blob:
                out[camera.name] = blob
        return out

    def camera(self, name: str) -> CameraStream:
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"camera {name!r} not in episode {self.uid}")

    def with_meta(self, **updates: Any) -> Episode:
        from dataclasses import replace

        return replace(self, meta=replace(self.meta, **updates))

    @staticmethod
    def from_arrays(
        uid: str,
        *,
        fps: float = 30.0,
        task: str = "",
        subtasks: Sequence[Segment] | None = None,
        ee: np.ndarray | None = None,
        ee_action: np.ndarray | None = None,
        joints: np.ndarray | None = None,
        joints_action: np.ndarray | None = None,
        gripper: np.ndarray | None = None,
        gripper_action: np.ndarray | None = None,
        part: str = keys.ARM_SINGLE,
        cameras: Mapping[str, str] | None = None,
    ) -> Episode:
        """用 numpy / 视频路径快速组装一条 Episode（入门推荐）。

        - ``ee`` / ``ee_action``: ``(N,4,4)`` 齐次位姿
        - ``joints`` / ``joints_action``: ``(N,J)``
        - ``gripper`` / ``gripper_action``: ``(N,)`` 或 ``(N,1)``，0=闭 1=开
        - ``subtasks``: ``Segment(start, end, text)`` 列表，半开区间 ``[start, end)``
        - ``cameras``: ``{相机名: mp4路径}``，只挂路径（LeRobot v3 同款）；像素用 ``as_lerobot()``
        """

        def _n_frames() -> int:
            for arr in (ee, ee_action, joints, joints_action, gripper, gripper_action):
                if arr is not None:
                    return int(np.asarray(arr).shape[0])
            raise ValueError("from_arrays needs at least one array")

        def _arm(
            ee_arr: np.ndarray | None,
            joint_arr: np.ndarray | None,
            grip_arr: np.ndarray | None,
        ) -> Arm | None:
            if ee_arr is None and joint_arr is None and grip_arr is None:
                return None
            ee_pose = Pose.from_matrix(np.asarray(ee_arr)) if ee_arr is not None else None
            jg = None
            if joint_arr is not None:
                joint_arr = np.asarray(joint_arr)
                jg = JointGroup(
                    position=LazyHandle.from_memory(joint_arr),
                    names=keys.arm_joint_names(part, joint_arr.shape[1]),
                )
            grip = None
            if grip_arr is not None:
                g = np.asarray(grip_arr, dtype=np.float64)
                if g.ndim == 1:
                    g = g.reshape(-1, 1)
                grip = Gripper(open=LazyHandle.from_memory(g))
            return Arm(ee=ee_pose, joints=jg, gripper=grip)

        n = _n_frames()
        state_arm = _arm(ee, joints, gripper)
        action_arm = _arm(ee_action, joints_action, gripper_action)

        def _put(body: RobotBody, arm: Arm | None) -> RobotBody:
            if arm is None:
                return body
            if part == keys.ARM_LEFT:
                return RobotBody(arm_left=arm, base=body.base)
            if part == keys.ARM_RIGHT:
                return RobotBody(arm_right=arm, base=body.base)
            return RobotBody(arm_single=arm, base=body.base)

        state = _put(RobotBody(), state_arm)
        action = _put(RobotBody(), action_arm) if action_arm is not None else None

        cam_streams = tuple(
            CameraStream(name=name, rgb=LazyHandle.from_video(path))
            for name, path in (cameras or {}).items()
        )
        return Episode(
            meta=EpisodeMeta(
                uid=uid,
                num_frames=n,
                fps=fps,
                task=task,
                stats=EpisodeStats(num_frames=n),
            ),
            state=state,
            action=action,
            cameras=cam_streams,
            labels=Labels(task=task, subtasks=tuple(subtasks or ())),
        )
