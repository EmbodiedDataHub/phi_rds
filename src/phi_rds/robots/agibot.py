"""AgiBot G1：URDF FK + calib align → torso/head（及臂）位姿。

路径约定（相对仓库根，不依赖 phi_data 包）：
  data/urdfs/g1_official/G1_120s.urdf
  calib/agibot.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phi_rds import keys
from phi_rds.robots.calib import load_align
from phi_rds.robots.fk import RobotFK

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "calib/agibot.json").is_file() and (
            parent / "data/urdfs/g1_official/G1_120s.urdf"
        ).is_file():
            return parent
    return here.parents[4]


_REPO = _repo_root()
URDF_PATH = _REPO / "data/urdfs/g1_official/G1_120s.urdf"
URDF_URL = (
    "https://raw.githubusercontent.com/AgibotTech/genie_sim/main/"
    "source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg/G1_120s/G1_120s.urdf"
)
CALIB_PATH = _REPO / "calib/agibot.json"

# URDF 关节名（与厂商 URDF / 旧 Convert 一致）
JOINTS_WAIST = ("idx01_body_joint1", "idx02_body_joint2")  # lift, pitch
JOINTS_HEAD = ("idx11_head_joint1", "idx12_head_joint2")  # yaw, pitch
JOINTS_ARM_L = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
JOINTS_ARM_R = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))

LINKS = {
    keys.TORSO: "body_link2",
    keys.HEAD: "head_link2",
    keys.ARM_LEFT: "gripper_l_base_link",
    keys.ARM_RIGHT: "gripper_r_base_link",
}


@dataclass
class AgibotG1:
    """G1 机型：加载 URDF，按参考系 torso 做相对 FK，再乘 align。"""

    urdf_path: Path = URDF_PATH
    calib_path: Path = CALIB_PATH
    reference_part: str = keys.TORSO
    links: dict[str, str] = field(default_factory=lambda: dict(LINKS))
    _fk: RobotFK | None = field(default=None, repr=False)

    def fk(self) -> RobotFK:
        if self._fk is None:
            path = Path(self.urdf_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"AgiBot URDF missing: {path} (optional download URL: {URDF_URL})"
                )
            self._fk = RobotFK(path.read_text(encoding="utf-8"))
        return self._fk

    def align(self) -> dict[str, np.ndarray]:
        matrices = load_align(self.calib_path)
        for part in self.links:
            matrices.setdefault(part, np.eye(4))
        return matrices

    def poses(self, angles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """``angles``: {urdf_joint_name: (N,)} → {part: (N,4,4)}。"""
        if not angles:
            return {}
        links = self.links
        ref = self.reference_part
        fk = self.fk()
        align_m = self.align()
        ref_link = links[ref]
        others = [part for part in links if part != ref]
        ref_pose = fk.solve(angles, [ref_link])[ref_link]
        relative = fk.solve(angles, [links[part] for part in others], base=ref_link)
        inv_ref_align = np.linalg.inv(align_m[ref])
        out = {ref: ref_pose @ align_m[ref]}
        for part in others:
            out[part] = inv_ref_align @ relative[links[part]] @ align_m[part]
        return out


_default: AgibotG1 | None = None


def agibot_frame_poses(angles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """便捷：G1 默认 URDF/calib → torso/head/arm 位姿。"""
    global _default
    if _default is None:
        _default = AgibotG1()
    return _default.poses(angles)
