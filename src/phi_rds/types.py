"""共享字面量。"""

from __future__ import annotations

from typing import Literal, TypeAlias

StatsScope: TypeAlias = Literal["self", "parent", "stale"]  # 统计范围; self: 当前数据集; parent: 父级数据集; stale: 过时数据集
EpisodeStatus: TypeAlias = Literal["active", "deleted", "failed", "unknown"]  # 任务状态; active: 活跃; deleted: 已删除; failed: 失败; unknown: 未知
DataLevel: TypeAlias = Literal["L1", "L2"]  # 数据级别; L1: 第一级; L2: 第二级
FieldId: TypeAlias = str    # 字段权威 id，如 "state.arm_left.ee" / "camera.head.rgb"
