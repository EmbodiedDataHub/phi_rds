"""转换工具子包：位姿表示 + 坐标系变换 + delta_chunk。

用法：
  from phi_rds.convert import encode_pose, change_frame, fill_delta_chunks
"""

from phi_rds.convert.delta import (
    delta_chunk,
    delta_chunk_from_bodies,
    fill_delta_chunks,
    shift_series,
)
from phi_rds.convert.frame import (
    apply_T,
    axes_rotation,
    change_frame,
    compose_T,
    convert_axes,
    invert_T,
    lookup_T,
    reexpress_via_graph,
    translate,
)
from phi_rds.convert.pose import (
    PRESETS,
    PRESETS as POSE_PRESETS,
    PoseRep,
    convert_pose,
    decode_pose,
    encode_pose,
    resolve_rep,
)

__all__ = [
    "PoseRep",
    "PRESETS",
    "POSE_PRESETS",
    "resolve_rep",
    "encode_pose",
    "decode_pose",
    "convert_pose",
    "apply_T",
    "invert_T",
    "compose_T",
    "change_frame",
    "translate",
    "axes_rotation",
    "convert_axes",
    "lookup_T",
    "reexpress_via_graph",
    "shift_series",
    "delta_chunk",
    "delta_chunk_from_bodies",
    "fill_delta_chunks",
]
