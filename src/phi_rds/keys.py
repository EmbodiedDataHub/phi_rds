"""跨数据集稳定命名：部位、相机槽、LeRobot v3 特征名。"""

from __future__ import annotations

# --- 部位 ---
ARM_LEFT = "arm_left"
ARM_RIGHT = "arm_right"
ARM_SINGLE = "arm_single"
TORSO = "torso"
HEAD = "head"
BASE = "base"
WAIST = "waist"

ARM_PARTS = (ARM_LEFT, ARM_RIGHT, ARM_SINGLE)
FRAME_PARTS = (TORSO, HEAD, BASE)

_COLUMN_SIDE = {ARM_LEFT: "left", ARM_RIGHT: "right", ARM_SINGLE: ""}

# --- 相机槽 ---
CAM_HEAD = "head"
CAM_HEAD_LEFT = "head_left"
CAM_HEAD_RIGHT = "head_right"
CAM_HEAD_FISHEYE = "head_fisheye"
CAM_WRIST_LEFT = "wrist_left"
CAM_WRIST_RIGHT = "wrist_right"
CAM_WRIST = "wrist"
CAM_FRONT = "front"
CAM_BACK_LEFT = "back_left"
CAM_BACK_RIGHT = "back_right"
CAM_BACK_LEFT_FISHEYE = "back_left_fisheye"
CAM_BACK_RIGHT_FISHEYE = "back_right_fisheye"
CAM_HEAD_LEFT_FISHEYE = "head_left_fisheye"
CAM_HEAD_RIGHT_FISHEYE = "head_right_fisheye"

# --- 标准关节槽 ---
TORSO_JOINT_NAMES = ("waist_yaw", "waist_pitch", "waist_roll", "waist_lift")
HEAD_JOINT_NAMES = ("head_yaw", "head_pitch", "head_roll")
ARM_JOINT_COUNT = 7

POSE9 = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21")

# --- LeRobot v3 特征名 ---
OBS_STATE = "observation.state"
ACTION = "action"
OBS_IMAGES = "observation.images"
STATE_CHUNK = 10
ACTION_CHUNK = 64
DELTA_CHUNK = "delta_chunk"
ARM = "arm"

STATE_ARM_TCP = f"{OBS_STATE}.arm.tcp_pose"
STATE_ARM_TCP_DELTA = f"{STATE_ARM_TCP}.{DELTA_CHUNK}"
STATE_ARM_GRIPPER = f"{OBS_STATE}.arm.gripper"
STATE_ARM_JOINT = f"{OBS_STATE}.arm.joint"
ACTION_ARM_TCP = f"{ACTION}.arm.tcp_pose"
ACTION_ARM_TCP_DELTA = f"{ACTION_ARM_TCP}.{DELTA_CHUNK}"
ACTION_ARM_GRIPPER = f"{ACTION}.arm.gripper"
ACTION_ARM_JOINT = f"{ACTION}.arm.joint"

EPISODE_INDEX = "episode_index"
FRAME_INDEX = "frame_index"
INDEX = "index"
TASK_INDEX = "task_index"
TIMESTAMP = "timestamp"


def field_id(role: str, *parts: str) -> str:
    return ".".join((role, *parts))


def column_side(part: str) -> str:
    if part not in _COLUMN_SIDE:
        raise KeyError(f"not an arm part: {part!r}")
    return _COLUMN_SIDE[part]


def pose9_names(part: str = ARM_SINGLE, *, delta: bool = False) -> tuple[str, ...]:
    prefix = "delta_" if delta else ""
    side = column_side(part)
    if side:
        prefix = f"{prefix}{side}_"
    return tuple(prefix + axis for axis in POSE9)


def gripper_name(part: str = ARM_SINGLE) -> str:
    side = column_side(part)
    return f"{side}_gripper" if side else "gripper"


def arm_joint_names(part: str, count: int = ARM_JOINT_COUNT) -> tuple[str, ...]:
    side = column_side(part)
    prefix = f"{side}_" if side else ""
    return tuple(f"{prefix}joint{i}" for i in range(1, count + 1))


def frame_pose_key(part: str) -> str:
    return f"{OBS_STATE}.{part}.pose"


def frame_joint_key(part: str) -> str:
    return f"{OBS_STATE}.{part}.joint"


def image_key(camera: str) -> str:
    return f"{OBS_IMAGES}.{camera}"


def camera_from_key(feature: str) -> str:
    prefix = OBS_IMAGES + "."
    if not feature.startswith(prefix):
        raise ValueError(f"not an image feature: {feature!r}")
    return feature[len(prefix) :]


def is_image_key(feature: str) -> bool:
    return feature.startswith(OBS_IMAGES + ".")
