"""执行引擎：Local / Ray 共用 map 协议。

流程（与写出解耦）::

    load → convert（FK / 换系 / fill_delta_chunks …）→ encode（只序列化）→ shard

map 函数必须是**模块级**可 pickle（Ray 要求）。
"""

from phi_rds.exec.local import LocalExecutor
from phi_rds.exec.map import (
    check_results,
    detect_num_workers,
    load_shard_episodes,
    make_rows,
    plan_shard_slices,
    process_row,
)
from phi_rds.exec.ray import RayExecutor

__all__ = [
    "LocalExecutor",
    "RayExecutor",
    "make_rows",
    "plan_shard_slices",
    "detect_num_workers",
    "process_row",
    "check_results",
    "load_shard_episodes",
]
