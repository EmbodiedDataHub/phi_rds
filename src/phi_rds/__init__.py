"""Phi RDS — 简洁的机器人数据集 SDK（LeRobot v3）。

日常只需记：
  open_dataset / create_l1|create_l2 / Episode.from_arrays / convert
"""

from __future__ import annotations

from pathlib import Path

from phi_rds import keys
from phi_rds.convert.delta import fill_delta_chunks
from phi_rds.convert.frame import apply_T, change_frame, compose_T, convert_axes, invert_T, lookup_T
from phi_rds.convert.pose import (
    PRESETS as POSE_REP_PRESETS,
    PoseRep,
    convert_pose,
    decode_pose,
    encode_pose,
    resolve_rep,
)
from phi_rds.core.body import MATRIX_REP, Arm, Gripper, JointGroup, Pose, RobotBody
from phi_rds.core.dataset import DatasetWriter, RobotDataSet
from phi_rds.core.episode import CameraStream, Episode, Labels, Segment
from phi_rds.core.lazy import LazyHandle, VideoLocator
from phi_rds.core.meta import DatasetMeta, EpisodeMeta
from phi_rds.core.semantics import FrameEdge, FrameGraph
from phi_rds.core.stats import (
    DEFAULT_PERCENTILES,
    DatasetStats,
    EpisodeStats,
    LengthDistribution,
    percentile,
    percentiles,
)
from phi_rds.io import open_lerobot_dataset, read_dataset, write_dataset
from phi_rds.schema.layout import Feature, Layout, Slice, decode, encode
from phi_rds.schema.presets import preset_layout, schema_for
from phi_rds.schema.spec import LATEST_STANDARD, DatasetSchema, FeatureSpec, resolve_standard
from phi_rds.types import DataLevel

__version__ = "0.1.0"


def open_dataset(path, **kwargs) -> RobotDataSet:
    """打开任意 LeRobot v3 / Phi RDS 目录。"""
    return RobotDataSet.open(path, **kwargs)


def create_dataset(path, schema: DatasetSchema, meta: DatasetMeta, **kwargs) -> DatasetWriter:
    """高级写出：自行提供 Schema + Meta。"""
    return RobotDataSet.create(path, schema, meta, **kwargs)


def create_l1(
    path: str | Path,
    *,
    robot: str,
    name: str,
    version: str = "v1",
    fps: float = 30.0,
    arms: str = "single",
    overwrite: bool = True,
) -> DatasetWriter:
    """L1 写出：命名统一、布局按机型预设。"""
    schema = schema_for(robot, level="L1", fps=fps, arms=arms)  # type: ignore[arg-type]
    meta = DatasetMeta(name=name, version=version, robot=robot, uri="")
    return RobotDataSet.create(path, schema, meta, overwrite=overwrite)


def create_l2(
    path: str | Path,
    *,
    robot: str,
    name: str,
    version: str = "v1",
    fps: float = 30.0,
    standard: str | None = None,
    arms: str = "single",
    overwrite: bool = True,
) -> DatasetWriter:
    """L2 写出：完整统一标准 + 标准版本号（默认最新）。"""
    schema = schema_for(robot, level="L2", fps=fps, standard=standard, arms=arms)  # type: ignore[arg-type]
    meta = DatasetMeta(name=name, version=version, robot=robot, uri="")
    return RobotDataSet.create(path, schema, meta, overwrite=overwrite)


__all__ = [
    "keys",
    "open_dataset",
    "create_l1",
    "create_l2",
    "create_dataset",
    "schema_for",
    "preset_layout",
    "RobotDataSet",
    "DatasetWriter",
    "DatasetSchema",
    "FeatureSpec",
    "DataLevel",
    "LATEST_STANDARD",
    "resolve_standard",
    "Episode",
    "CameraStream",
    "Labels",
    "Segment",
    "Pose",
    "MATRIX_REP",
    "JointGroup",
    "Gripper",
    "Arm",
    "RobotBody",
    "LazyHandle",
    "VideoLocator",
    "DatasetMeta",
    "EpisodeMeta",
    "EpisodeStats",
    "DatasetStats",
    "LengthDistribution",
    "DEFAULT_PERCENTILES",
    "percentile",
    "percentiles",
    "FrameEdge",
    "FrameGraph",
    "PoseRep",
    "POSE_REP_PRESETS",
    "encode_pose",
    "decode_pose",
    "convert_pose",
    "resolve_rep",
    "apply_T",
    "invert_T",
    "compose_T",
    "change_frame",
    "convert_axes",
    "lookup_T",
    "fill_delta_chunks",
    "Layout",
    "Feature",
    "Slice",
    "encode",
    "decode",
    "read_dataset",
    "write_dataset",
    "open_lerobot_dataset",
    "__version__",
]
