"""列布局 + Episode↔列编解码：Layout / Feature / Slice / encode / decode。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

import numpy as np

from phi_rds.convert.pose import decode_pose, encode_pose, resolve_rep

if TYPE_CHECKING:
    from phi_rds.core.body import RobotBody
    from phi_rds.core.episode import Episode


@dataclass(frozen=True)
class Slice:
    """Feature 内一段。kind 决定从 RobotBody 取值方式。

    位姿相关：
      - ``kind="pose9"``：兼容旧写法，等价 ``pose`` + ``pose_rep="pose9"``
      - ``kind="pose"``：按 ``pose_rep``（默认 pose9）编码 / 解码
    """

    kind: str
    from_: str
    part: str
    names: tuple[str, ...]
    delta_from: str | None = None
    fk_names: tuple[str, ...] | None = None
    src_width: int | None = None
    src_slots: tuple[int, ...] | None = None
    optional: bool = False
    parts: tuple[str, ...] = ()
    chunk_size: int = 0
    direction: str = ""
    pose_rep: str | None = None
    """位姿向量写法，见 ``phi_rds.convert.PRESETS``（quat_wxyz / euler_xyz / …）。"""

    @property
    def real_width(self) -> int:
        return len(self.names) if self.src_width is None else self.src_width

    @property
    def resolved_pose_rep(self) -> str:
        if self.pose_rep:
            return self.pose_rep
        if self.kind == "pose9":
            return "pose9"
        if self.kind == "pose":
            return "pose9"
        raise ValueError(f"slice kind {self.kind!r} is not a pose kind")



@dataclass(frozen=True)
class Feature:
    name: str
    slices: tuple[Slice, ...]
    dtype: str = "float32"
    shape: tuple[int, ...] | None = None

    @property
    def names(self) -> list[str]:
        out: list[str] = []
        for piece in self.slices:
            out.extend(piece.names)
        return out


@dataclass(frozen=True)
class Layout:
    id: str
    robot_type: str
    frame: str
    features: tuple[Feature, ...]


def slice_width(piece: Slice) -> int:
    if piece.kind == "delta_chunk":
        return piece.chunk_size * len(piece.names)
    if piece.kind in ("pose", "pose9"):
        from phi_rds.convert.pose import resolve_rep

        # 以 PoseRep.dim 为准；names 用于列标注，长度应匹配
        return resolve_rep(piece.resolved_pose_rep).dim
    return len(piece.names)


def pad_slots(feature: Feature) -> list[int]:
    """feature 内补 0 槽的全局下标；写入 info.json 的 feature[\"pad\"]。

    joint：实有列由 src_slots（或前 real_width）占，其余为 pad；
    pose/gripper：optional（部位缺失）则整段 pad；delta_chunk 不标 pad。
    """
    out: list[int] = []
    start = 0
    for piece in feature.slices:
        width = slice_width(piece)
        if piece.kind == "delta_chunk":
            # 2D 特征 (chunk, D)：pad 按最后一维标（与文档一致时通常不标）
            start += width
            continue
        if piece.kind == "joint":
            real = set(piece.src_slots) if piece.src_slots else set(range(piece.real_width))
            out.extend(start + i for i in range(width) if i not in real)
        elif piece.optional:
            out.extend(range(start, start + width))
        start += width
    return out


# ---------------------------------------------------------------------------
# Episode ↔ 列
# ---------------------------------------------------------------------------

def pose9(matrix: np.ndarray) -> np.ndarray:
    return encode_pose(matrix, "pose9")


def pose9_to_matrix(pose9_values: np.ndarray) -> np.ndarray:
    return decode_pose(pose9_values, "pose9")

def _pad_to(array: np.ndarray, width: int) -> np.ndarray:
    if array.shape[1] == width:
        return array
    pad = np.zeros((array.shape[0], width - array.shape[1]), dtype=array.dtype)
    return np.concatenate([array, pad], axis=1)


def _is_pose_kind(kind: str) -> bool:
    return kind in ("pose", "pose9")


def _raw_slice(piece: Slice, source: "RobotBody") -> np.ndarray:
    if _is_pose_kind(piece.kind):
        matrix = source.ee_matrix(piece.part)
        if piece.delta_from is not None:
            matrix = np.linalg.inv(source.ee_matrix(piece.delta_from)) @ matrix
        return encode_pose(matrix, piece.resolved_pose_rep)
    if piece.kind == "gripper":
        return np.asarray(source.gripper_of(piece.part), dtype=np.float64).reshape(-1, 1)
    if piece.kind == "joint":
        position = np.asarray(source.joint_group(piece.part).numpy(), dtype=np.float64)
        return position[:, : piece.real_width]
    raise ValueError(f"unknown slice kind {piece.kind!r}")


def _slice_array(piece: Slice, episode: "Episode", n_frames: int) -> np.ndarray:
    if _is_pose_kind(piece.kind):
        width = resolve_rep(piece.resolved_pose_rep).dim
    else:
        width = len(piece.names)
    source = episode.state if piece.from_ == "state" else episode.action
    if source is None:
        if piece.optional:
            return np.zeros((n_frames, width), dtype=np.float64)
        raise ValueError(f"slice {piece.part}/{piece.kind} needs missing '{piece.from_}'")
    try:
        raw = _raw_slice(piece, source)
    except KeyError:
        if piece.optional:
            return np.zeros((n_frames, width), dtype=np.float64)
        raise
    if piece.src_slots:
        out = np.zeros((n_frames, width), dtype=np.float64)
        for col, slot in enumerate(piece.src_slots):
            out[:, slot] = raw[:, col]
        return out
    return _pad_to(raw, width)


def encode(episode: "Episode", layout: Layout) -> dict[str, np.ndarray]:
    """Episode → 列。只做序列化，不算 FK / delta_chunk。

    ``delta_chunk`` 必须已由转换阶段写入 ``episode.fields``
   （见 ``phi_rds.convert.fill_delta_chunks``）。
    """
    from phi_rds import keys

    n = episode.num_frames
    data: dict[str, np.ndarray] = {
        keys.TIMESTAMP: (np.arange(n, dtype=np.float32) / episode.fps),
    }
    for feature in layout.features:
        if not feature.slices:
            continue
        needs_action = any(piece.from_ == "action" for piece in feature.slices)
        if needs_action and episode.action is None:
            continue
        if feature.slices[0].kind == "delta_chunk":
            field = episode.fields.get(feature.name)
            if field is None:
                raise ValueError(
                    f"delta_chunk feature {feature.name!r} missing on episode.fields; "
                    "call phi_rds.convert.fill_delta_chunks(episode, layout) in the "
                    "convert stage before write/encode"
                )
            payload = field.payload
            arr = np.asarray(payload.load() if hasattr(payload, "load") else payload)
            data[feature.name] = arr.astype(feature.dtype, copy=False)
            continue
        parts = [_slice_array(piece, episode, n) for piece in feature.slices]
        data[feature.name] = np.concatenate(parts, axis=1).astype(feature.dtype)
    return data


@dataclass
class Decoded:
    poses: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    deltas: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    delta_refs: dict[tuple[str, str], str] = field(default_factory=dict)
    joints: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    joint_names: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    grippers: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def angles(self, from_: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for (stream, part), values in self.joints.items():
            if stream != from_:
                continue
            for column, name in enumerate(self.joint_names[(stream, part)]):
                out[name] = values[:, column]
        return out


def decode(columns: Mapping[str, np.ndarray], layout: Layout) -> Decoded:
    out = Decoded()
    for feature in layout.features:
        if feature.name not in columns:
            continue
        if feature.slices and feature.slices[0].kind == "delta_chunk":
            continue
        values = np.asarray(columns[feature.name], dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError(f"feature {feature.name!r} must be 2-D, got {values.shape}")
        expected = sum(slice_width(piece) for piece in feature.slices)
        if values.shape[1] != expected:
            raise ValueError(f"feature {feature.name!r} width {values.shape[1]} != layout {expected}")
        start = 0
        for piece in feature.slices:
            stop = start + slice_width(piece)
            key = (piece.from_, piece.part)
            chunk = values[:, start:stop]
            if _is_pose_kind(piece.kind):
                rep = piece.resolved_pose_rep
                dim = resolve_rep(rep).dim
                matrix = decode_pose(chunk[:, :dim], rep)
                if piece.delta_from is None:
                    out.poses[key] = matrix
                else:
                    out.deltas[key] = matrix
                    out.delta_refs[key] = piece.delta_from
            elif piece.kind == "gripper":
                out.grippers[key] = chunk
            elif piece.kind == "joint":
                real = piece.real_width
                out.joints[key] = chunk[:, list(piece.src_slots)] if piece.src_slots else chunk[:, :real]
                out.joint_names[key] = piece.fk_names or tuple(piece.names[:real])
            else:
                raise ValueError(f"unknown slice kind {piece.kind!r}")
            start = stop
    return out
