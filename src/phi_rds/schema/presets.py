"""机型预设 Layout / Schema：按《统一格式字段及命名标准(LeRobot v3)》。

L2 含：tcp / delta_chunk / gripper / joint。
AgiBot 另含 base/torso/head.pose + torso/head.joint（无源部位标 pad）；
无躯干/头/底盘的机型（如 YAM）不写入这些槽。
"""

from __future__ import annotations

from typing import Literal

from phi_rds import keys
from phi_rds.schema.layout import Feature, Layout, Slice
from phi_rds.schema.spec import DatasetSchema
from phi_rds.types import DataLevel

ArmMode = Literal["single", "bimanual"]

# AgiBot JointGroup 列序（与 URDF / phi_data parse 一致）：waist=[lift,pitch]
# → 标准槽；缺失 yaw/roll 写 0，info.json 标 pad
_AGIBOT_FRAME_JOINT_STD: dict[str, tuple[str, ...]] = {
    keys.WAIST: ("waist_lift", "waist_pitch"),
    keys.HEAD: ("head_yaw", "head_pitch"),
}
# G1：torso/head 有 FK；base 无里程计 → 仅 base 全 pad
_AGIBOT_FRAME_POSE_OPTIONAL: dict[str, bool] = {
    keys.BASE: True,
    keys.TORSO: False,
    keys.HEAD: False,
}


def _arm_parts(mode: ArmMode) -> tuple[str, ...]:
    if mode == "bimanual":
        return (keys.ARM_LEFT, keys.ARM_RIGHT)
    return (keys.ARM_SINGLE,)


def _tcp(from_: str, parts: tuple[str, ...]) -> Feature:
    name = keys.STATE_ARM_TCP if from_ == "state" else keys.ACTION_ARM_TCP
    return Feature(
        name,
        tuple(
            Slice(kind="pose9", from_=from_, part=p, names=keys.pose9_names(p))
            for p in parts
        ),
    )


def _tcp_names(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(n for p in parts for n in keys.pose9_names(p))


def _delta(from_: str, parts: tuple[str, ...], chunk: int, direction: str) -> Feature:
    name = keys.STATE_ARM_TCP_DELTA if from_ == "state" else keys.ACTION_ARM_TCP_DELTA
    per = _tcp_names(parts)
    return Feature(
        name,
        (
            Slice(
                kind="delta_chunk",
                from_=from_,
                part="",
                names=per,
                parts=parts,
                chunk_size=chunk,
                direction=direction,
            ),
        ),
        shape=(chunk, len(per)),
    )


def _gripper(from_: str, parts: tuple[str, ...]) -> Feature:
    name = keys.STATE_ARM_GRIPPER if from_ == "state" else keys.ACTION_ARM_GRIPPER
    return Feature(
        name,
        tuple(
            Slice(kind="gripper", from_=from_, part=p, names=(keys.gripper_name(p),))
            for p in parts
        ),
    )


def _joints(from_: str, parts: tuple[str, ...], count: int = 7) -> Feature:
    """每臂固定 7 槽。实有关节少于 7 时尾部补 0，并在 info.json 标 pad。"""
    std = keys.ARM_JOINT_COUNT
    real = max(1, min(int(count), std))
    name = keys.STATE_ARM_JOINT if from_ == "state" else keys.ACTION_ARM_JOINT
    return Feature(
        name,
        tuple(
            Slice(
                kind="joint",
                from_=from_,
                part=p,
                names=keys.arm_joint_names(p, std),
                fk_names=keys.arm_joint_names(p, real),
                src_width=real,
            )
            for p in parts
        ),
    )


def _frame_pose(part: str, *, optional: bool = True) -> Feature:
    return Feature(
        keys.frame_pose_key(part),
        (
            Slice(
                kind="pose9",
                from_="state",
                part=part,
                names=keys.POSE9,
                optional=optional,
            ),
        ),
    )


def _frame_joint(
    out_name: str,
    body_part: str,
    std_names: tuple[str, ...],
    std_of_real: tuple[str, ...] | None,
) -> Feature:
    """标准槽关节；实有列由 src_slots 填入，其余 0 + pad。"""
    if std_of_real:
        real = min(len(std_of_real), len(std_names))
        src_slots = tuple(std_names.index(n) for n in std_of_real[:real])
    else:
        real = 0
        src_slots = None
    return Feature(
        out_name,
        (
            Slice(
                kind="joint",
                from_="state",
                part=body_part,
                names=std_names,
                src_width=real or None,
                src_slots=src_slots,
                optional=True,
            ),
        ),
    )


def preset_layout(
    robot: str,
    *,
    arms: ArmMode = "single",
    joint_count: int = 7,
    include_gripper: bool = True,
    full_standard: bool = True,
) -> Layout:
    """按机型名给统一命名 Layout。

    ``full_standard=True``（L2 默认）：含 delta_chunk；
    AgiBot 再加 base/torso/head（无源部位 pad），其它机型不加空槽。
    """
    parts = _arm_parts(arms)
    is_agibot = robot.lower().startswith("agibot")
    features: list[Feature] = [_tcp("state", parts)]
    if full_standard:
        features.append(_delta("state", parts, keys.STATE_CHUNK, "past"))
    if include_gripper:
        features.append(_gripper("state", parts))
    features.append(_joints("state", parts, joint_count))

    if full_standard and is_agibot:
        for part in keys.FRAME_PARTS:
            features.append(
                _frame_pose(part, optional=_AGIBOT_FRAME_POSE_OPTIONAL.get(part, True))
            )
        features.append(
            _frame_joint(
                keys.frame_joint_key(keys.TORSO),
                keys.WAIST,
                keys.TORSO_JOINT_NAMES,
                _AGIBOT_FRAME_JOINT_STD.get(keys.WAIST),
            )
        )
        features.append(
            _frame_joint(
                keys.frame_joint_key(keys.HEAD),
                keys.HEAD,
                keys.HEAD_JOINT_NAMES,
                _AGIBOT_FRAME_JOINT_STD.get(keys.HEAD),
            )
        )

    features.append(_tcp("action", parts))
    if full_standard:
        features.append(_delta("action", parts, keys.ACTION_CHUNK, "future"))
    if include_gripper:
        features.append(_gripper("action", parts))
    features.append(_joints("action", parts, joint_count))
    return Layout(
        id=f"{robot}_{arms}",
        robot_type=robot,
        frame=keys.TORSO if is_agibot else keys.BASE,
        features=tuple(features),
    )


def schema_for(
    robot: str,
    *,
    level: DataLevel = "L2",
    fps: float = 30.0,
    standard: str | None = None,
    arms: ArmMode = "single",
    joint_count: int = 7,
) -> DatasetSchema:
    """机型 → Schema。``level`` 默认 L2（带最新标准版本号）。"""
    layout = preset_layout(
        robot,
        arms=arms,
        joint_count=joint_count,
        full_standard=(level == "L2"),
    )
    if level == "L1":
        return DatasetSchema.l1(layout, fps=fps)
    return DatasetSchema.l2(layout, fps=fps, standard=standard)
