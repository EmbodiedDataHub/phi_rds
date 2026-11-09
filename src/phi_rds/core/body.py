"""几何树：Pose / JointGroup / Arm / RobotBody（payload 可为懒句柄）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from phi_rds import keys
from phi_rds.core.lazy import LazyHandle

# Pose.rep 特殊值：存 (N,4,4) 齐次矩阵（不在 convert.PRESETS 里）
MATRIX_REP = "matrix"


def _as_array(value: np.ndarray | LazyHandle[Any]) -> np.ndarray:
    if isinstance(value, LazyHandle):
        loaded = value.load()
        return np.asarray(loaded)
    return np.asarray(value)


def _is_matrix_rep(rep: str) -> bool:
    return rep in (MATRIX_REP, "homo", "homogeneous", "T", "mat4")


@dataclass
class Pose:
    """位姿序列：可存**任意**表示，用 ``rep`` 标明写法。

    - ``rep="matrix"``（默认）：``values`` 为 ``(N,4,4)`` 齐次矩阵
    - 其它：``phi_rds.convert.PRESETS`` 中的名字（如 ``pose9`` / ``quat_wxyz``），
      ``values`` 为 ``(N, dim)``

    需要齐次矩阵时用 ``as_matrix()``；需要别的向量写法用 ``as_rep(...)``。
    互转枢纽仍是 4×4（见 ``convert.pose``），但**不必**在内存里始终存矩阵。
    """

    values: np.ndarray | LazyHandle[np.ndarray]
    rep: str = MATRIX_REP

    def numpy(self) -> np.ndarray:
        """原始存储数组（形状随 ``rep`` 而定）。"""
        return np.asarray(_as_array(self.values), dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """规范成 ``(N,4,4)``；若已是 matrix 则校验形状后返回。"""
        array = self.numpy()
        if _is_matrix_rep(self.rep):
            if array.ndim != 3 or array.shape[1:] != (4, 4):
                raise ValueError(
                    f"Pose(rep={self.rep!r}) expects (N,4,4), got {array.shape}"
                )
            return array
        from phi_rds.convert import decode_pose

        return decode_pose(array, self.rep)

    def as_rep(self, rep: str = "pose9") -> np.ndarray:
        """导出为指定写法（与当前 ``self.rep`` 相同时零拷贝逻辑上的同数组）。"""
        if rep == self.rep or (_is_matrix_rep(rep) and _is_matrix_rep(self.rep)):
            if _is_matrix_rep(rep):
                return self.as_matrix()
            return self.numpy()
        if _is_matrix_rep(rep):
            return self.as_matrix()
        from phi_rds.convert import encode_pose

        return encode_pose(self.as_matrix(), rep)

    def as_quat(self, *, order: str = "wxyz") -> np.ndarray:
        """(N,7)=xyz+quat。"""
        name = "quat_wxyz" if order == "wxyz" else "quat_xyzw"
        return self.as_rep(name)

    def as_euler(self, seq: str = "xyz", *, degrees: bool = False) -> np.ndarray:
        """(N,6)=xyz+euler。"""
        name = f"euler_{seq.lower()}"
        if degrees:
            name = f"{name}_deg"
        return self.as_rep(name)

    def as_pose9(self) -> np.ndarray:
        return self.as_rep("pose9")

    @classmethod
    def from_matrix(cls, matrix: np.ndarray | LazyHandle[np.ndarray]) -> Pose:
        """显式按齐次矩阵构造。"""
        if not isinstance(matrix, LazyHandle):
            matrix = LazyHandle.from_memory(np.asarray(matrix))
        return cls(values=matrix, rep=MATRIX_REP)

    @classmethod
    def from_rep(cls, values: np.ndarray, rep: str = "pose9") -> Pose:
        """按给定表示**原样存储**（不先转成矩阵）。``rep="matrix"`` 时 values 应为 (N,4,4)。"""
        if _is_matrix_rep(rep):
            return cls.from_matrix(values)
        return cls(values=LazyHandle.from_memory(np.asarray(values)), rep=rep)

    @classmethod
    def from_quat(
        cls,
        xyz: np.ndarray,
        quat: np.ndarray,
        *,
        order: str = "wxyz",
    ) -> Pose:
        xyz = np.asarray(xyz, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        if quat.ndim == 1:
            quat = quat.reshape(1, -1)
        values = np.concatenate([xyz, quat], axis=1)
        name = "quat_wxyz" if order == "wxyz" else "quat_xyzw"
        return cls.from_rep(values, name)

    @classmethod
    def from_euler(
        cls,
        xyz: np.ndarray,
        euler: np.ndarray,
        *,
        seq: str = "xyz",
        degrees: bool = False,
    ) -> Pose:
        xyz = np.asarray(xyz, dtype=np.float64)
        euler = np.asarray(euler, dtype=np.float64)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        if euler.ndim == 1:
            euler = euler.reshape(1, -1)
        values = np.concatenate([xyz, euler], axis=1)
        name = f"euler_{seq.lower()}"
        if degrees:
            name = f"{name}_deg"
        return cls.from_rep(values, name)


@dataclass
class JointGroup:
    position: np.ndarray | LazyHandle[np.ndarray]
    names: tuple[str, ...]

    def numpy(self) -> np.ndarray:
        array = _as_array(self.position)
        if array.ndim != 2 or array.shape[1] != len(self.names):
            raise ValueError(
                f"JointGroup: position {array.shape} vs {len(self.names)} names"
            )
        return array


@dataclass
class Gripper:
    """平行夹爪开口量；约定 0=闭 1=开（normalized_01）。"""

    open: np.ndarray | LazyHandle[np.ndarray]

    def numpy(self) -> np.ndarray:
        return np.asarray(_as_array(self.open), dtype=np.float64)


@dataclass
class Arm:
    ee: Pose | None = None
    gripper: Gripper | None = None
    joints: JointGroup | None = None


@dataclass
class RobotBody:
    """观测或动作几何。缺的部位留 None。"""

    base: Pose | None = None
    torso: Pose | None = None
    head: Pose | None = None
    arm_left: Arm | None = None
    arm_right: Arm | None = None
    arm_single: Arm | None = None
    waist: JointGroup | None = None
    head_joints: JointGroup | None = None

    def _arm(self, part: str) -> Arm | None:
        return {
            keys.ARM_LEFT: self.arm_left,
            keys.ARM_RIGHT: self.arm_right,
            keys.ARM_SINGLE: self.arm_single,
        }.get(part)

    def ee_matrix(self, part: str) -> np.ndarray:
        """该部位位姿的 ``(N,4,4)``（必要时从 ``Pose.rep`` 转换）。"""
        body = self._arm(part)
        if body is not None and body.ee is not None:
            return body.ee.as_matrix()
        pose = {
            keys.BASE: self.base,
            keys.TORSO: self.torso,
            keys.HEAD: self.head,
        }.get(part)
        if pose is not None:
            return pose.as_matrix()
        raise KeyError(f"no pose for {part!r}")

    def gripper_of(self, part: str) -> np.ndarray:
        body = self._arm(part)
        if body is None or body.gripper is None:
            raise KeyError(f"no gripper for {part!r}")
        return body.gripper.numpy()

    def joint_group(self, part: str) -> JointGroup:
        if part == keys.WAIST and self.waist is not None:
            return self.waist
        if part == keys.HEAD and self.head_joints is not None:
            return self.head_joints
        body = self._arm(part)
        if body is not None and body.joints is not None:
            return body.joints
        raise KeyError(f"no joints for {part!r}")

    def joint_angles(self) -> dict[str, np.ndarray]:
        groups = [self.waist, self.head_joints]
        groups += [
            body.joints
            for body in (self.arm_left, self.arm_right, self.arm_single)
            if body is not None
        ]
        out: dict[str, np.ndarray] = {}
        for group in groups:
            if group is None:
                continue
            position = group.numpy()
            for col, name in enumerate(group.names):
                out[name] = position[:, col]
        return out

    def with_ee(self, poses: dict[str, np.ndarray]) -> RobotBody:
        """``poses[part]`` 须为 ``(N,4,4)`` 齐次矩阵。"""

        def fill_arm(body: Arm | None, part: str) -> Arm | None:
            if body is None or part not in poses:
                return body
            return replace(body, ee=Pose.from_matrix(poses[part]))

        return replace(
            self,
            base=Pose.from_matrix(poses[keys.BASE]) if keys.BASE in poses else self.base,
            torso=Pose.from_matrix(poses[keys.TORSO]) if keys.TORSO in poses else self.torso,
            head=Pose.from_matrix(poses[keys.HEAD]) if keys.HEAD in poses else self.head,
            arm_left=fill_arm(self.arm_left, keys.ARM_LEFT),
            arm_right=fill_arm(self.arm_right, keys.ARM_RIGHT),
            arm_single=fill_arm(self.arm_single, keys.ARM_SINGLE),
        )
