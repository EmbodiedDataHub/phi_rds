"""位姿表示互转：以 (N,4,4) 为枢纽；``Pose`` 本身可存任意 ``rep``。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

AngleUnit = Literal["rad", "deg"]
TranslationLayout = Literal["xyz", "none"]
RotationLayout = Literal[
    "mat_cols2",   # 旋转矩阵前两列 (6,)；pose9 的旋转部分
    "mat9",        # 旋转矩阵行主序展平 (9,)
    "quat_wxyz",
    "quat_xyzw",
    "euler",       # 需 euler_seq
    "rot6d",       # 同 mat_cols2，Zhou et al. 连续 6D
    "axis_angle",  # 轴角向量 (3,)，|v|=θ
]


@dataclass(frozen=True)
class PoseRep:
    """一种位姿向量写法的完整语义。"""

    name: str
    rotation: RotationLayout
    translation: TranslationLayout = "xyz"
    euler_seq: str = "xyz"
    """scipy / URDF 习惯的 intrinsic 序列，如 xyz、zyx。"""

    angle_unit: AngleUnit = "rad"
    """仅对 euler / axis_angle 有意义；四元数无单位。"""

    @property
    def rotation_dim(self) -> int:
        return {
            "mat_cols2": 6,
            "rot6d": 6,
            "mat9": 9,
            "quat_wxyz": 4,
            "quat_xyzw": 4,
            "euler": 3,
            "axis_angle": 3,
        }[self.rotation]

    @property
    def dim(self) -> int:
        t = 3 if self.translation == "xyz" else 0
        return t + self.rotation_dim

    @property
    def names(self) -> tuple[str, ...]:
        """默认列名（无部位前缀）。"""
        parts: list[str] = []
        if self.translation == "xyz":
            parts.extend(("x", "y", "z"))
        if self.rotation in ("mat_cols2", "rot6d"):
            parts.extend(("r00", "r10", "r20", "r01", "r11", "r21"))
        elif self.rotation == "mat9":
            parts.extend(f"r{i}{j}" for i in range(3) for j in range(3))
        elif self.rotation == "quat_wxyz":
            parts.extend(("qw", "qx", "qy", "qz"))
        elif self.rotation == "quat_xyzw":
            parts.extend(("qx", "qy", "qz", "qw"))
        elif self.rotation == "euler":
            parts.extend(tuple(f"e{ax}" for ax in self.euler_seq.lower()))
        elif self.rotation == "axis_angle":
            parts.extend(("ax", "ay", "az"))
        return tuple(parts)


# --- 常用预设 ---
POSE9 = PoseRep(name="pose9", rotation="mat_cols2", translation="xyz")
QUAT_WXYZ_XYZ = PoseRep(name="quat_wxyz", rotation="quat_wxyz", translation="xyz")
QUAT_XYZW_XYZ = PoseRep(name="quat_xyzw", rotation="quat_xyzw", translation="xyz")
EULER_XYZ_RAD = PoseRep(name="euler_xyz", rotation="euler", euler_seq="xyz", angle_unit="rad")
EULER_XYZ_DEG = PoseRep(name="euler_xyz_deg", rotation="euler", euler_seq="xyz", angle_unit="deg")
ROT6D_XYZ = PoseRep(name="rot6d", rotation="rot6d", translation="xyz")

PRESETS: dict[str, PoseRep] = {
    p.name: p
    for p in (POSE9, QUAT_WXYZ_XYZ, QUAT_XYZW_XYZ, EULER_XYZ_RAD, EULER_XYZ_DEG, ROT6D_XYZ)
}


def resolve_rep(rep: PoseRep | str) -> PoseRep:
    if isinstance(rep, PoseRep):
        return rep
    if rep not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown pose representation {rep!r}; known: {known}")
    return PRESETS[rep]


def _as_batch_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.reshape(1, 4, 4)
    if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4) or (4,4), got {matrix.shape}")
    return matrix


def _rot_from_matrix(matrix: np.ndarray):
    from scipy.spatial.transform import Rotation as SciR

    return SciR.from_matrix(matrix[:, :3, :3])


def _rotation_to_vecs(matrix: np.ndarray, rep: PoseRep) -> np.ndarray:
    n = matrix.shape[0]
    R = matrix[:, :3, :3]
    if rep.rotation in ("mat_cols2", "rot6d"):
        return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)
    if rep.rotation == "mat9":
        return R.reshape(n, 9)
    rot = _rot_from_matrix(matrix)
    if rep.rotation == "quat_wxyz":
        # scipy as_quat → xyzw
        xyzw = rot.as_quat()
        return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    if rep.rotation == "quat_xyzw":
        return rot.as_quat()
    if rep.rotation == "euler":
        return rot.as_euler(rep.euler_seq, degrees=(rep.angle_unit == "deg"))
    if rep.rotation == "axis_angle":
        # as_rotvec 单位为 rad
        vec = rot.as_rotvec()
        if rep.angle_unit == "deg":
            vec = np.degrees(vec)
        return vec
    raise ValueError(f"unsupported rotation layout {rep.rotation!r}")


def _rotation_from_vecs(rot_vecs: np.ndarray, rep: PoseRep, n: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SciR

    rot_vecs = np.asarray(rot_vecs, dtype=np.float64)
    if rot_vecs.ndim == 1:
        rot_vecs = rot_vecs.reshape(1, -1)
    if rep.rotation in ("mat_cols2", "rot6d"):
        if rot_vecs.shape[1] != 6:
            raise ValueError(f"{rep.rotation} needs 6 cols, got {rot_vecs.shape}")
        col0 = rot_vecs[:, 0:3]
        col1 = rot_vecs[:, 3:6]
        # Gram-Schmidt（rot6d / pose9 回读）
        b1 = col0 / (np.linalg.norm(col0, axis=1, keepdims=True) + 1e-12)
        dot = np.sum(b1 * col1, axis=1, keepdims=True)
        b2 = col1 - dot * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-12)
        b3 = np.cross(b1, b2)
        R = np.stack([b1, b2, b3], axis=2)  # (N,3,3) columns
        return R
    if rep.rotation == "mat9":
        return rot_vecs.reshape(n, 3, 3)
    if rep.rotation == "quat_wxyz":
        wxyz = rot_vecs
        xyzw = np.concatenate([wxyz[:, 1:4], wxyz[:, 0:1]], axis=1)
        return SciR.from_quat(xyzw).as_matrix()
    if rep.rotation == "quat_xyzw":
        return SciR.from_quat(rot_vecs).as_matrix()
    if rep.rotation == "euler":
        return SciR.from_euler(
            rep.euler_seq, rot_vecs, degrees=(rep.angle_unit == "deg"),
        ).as_matrix()
    if rep.rotation == "axis_angle":
        vec = rot_vecs
        if rep.angle_unit == "deg":
            vec = np.radians(vec)
        return SciR.from_rotvec(vec).as_matrix()
    raise ValueError(f"unsupported rotation layout {rep.rotation!r}")


def encode_pose(matrix: np.ndarray, rep: PoseRep | str) -> np.ndarray:
    """(N,4,4)|(4,4) → (N, dim)。"""
    rep = resolve_rep(rep)
    if rep.name == "homogeneous_4x4":
        batch = _as_batch_matrix(matrix)
        return batch.reshape(batch.shape[0], 16)
    batch = _as_batch_matrix(matrix)
    rot = _rotation_to_vecs(batch, rep)
    if rep.translation == "xyz":
        return np.concatenate([batch[:, :3, 3], rot], axis=1)
    return rot


def decode_pose(values: np.ndarray, rep: PoseRep | str) -> np.ndarray:
    """(N, dim)|(dim,) → (N,4,4)。"""
    rep = resolve_rep(rep)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if rep.name == "homogeneous_4x4":
        if values.shape[1] != 16:
            raise ValueError(f"homogeneous_4x4 needs 16 cols, got {values.shape}")
        return values.reshape(-1, 4, 4)
    expected = rep.dim
    if values.shape[1] != expected:
        raise ValueError(f"{rep.name} needs {expected} cols, got {values.shape}")
    n = values.shape[0]
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    if rep.translation == "xyz":
        out[:, :3, 3] = values[:, :3]
        rot_vecs = values[:, 3:]
    else:
        rot_vecs = values
    out[:, :3, :3] = _rotation_from_vecs(rot_vecs, rep, n)
    return out


def convert_pose(
    values: np.ndarray,
    *,
    src: PoseRep | str,
    dst: PoseRep | str,
) -> np.ndarray:
    """任意两种 PoseRep 互转（经 4x4）。"""
    return encode_pose(decode_pose(values, src), dst)


POSE_PRESETS = PRESETS
