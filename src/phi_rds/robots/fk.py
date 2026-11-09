"""基于 URDF 的正运动学（纯 FK，不含 align）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class RobotFK:
    """``solve`` 给出 T_base_target；align / TCP 由调用方在 FK 之后施加。"""

    def __init__(self, urdf_text: str) -> None:
        try:
            import kinpy as kp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "phi_rds.robots.fk requires kinpy; pip install kinpy"
            ) from exc
        self.chain = kp.build_chain_from_urdf(urdf_text)

    def solve(
        self,
        joint_angles: dict[str, np.ndarray],
        targets: Sequence[str],
        base: str | None = None,
    ) -> dict[str, np.ndarray]:
        """``joint_angles``: {urdf_joint: (N,)} → {target_link: (N,4,4)}。"""
        names = list(joint_angles)
        n_frames = len(joint_angles[names[0]])
        out: dict[str, list[np.ndarray]] = {target: [] for target in targets}
        for frame_index in range(n_frames):
            angles = {name: float(joint_angles[name][frame_index]) for name in names}
            fk = self.chain.forward_kinematics(angles)
            inv_base = (
                np.linalg.inv(np.array(fk[base].matrix(), dtype=np.float64))
                if base is not None
                else np.eye(4)
            )
            for target in targets:
                out[target].append(inv_base @ np.array(fk[target].matrix(), dtype=np.float64))
        return {target: np.stack(stack) for target, stack in out.items()}

    def link_names(self) -> list[str]:
        return list(self.chain.forward_kinematics({}))

    def joint_names(self) -> list[str]:
        return list(self.chain.get_joint_parameter_names())
