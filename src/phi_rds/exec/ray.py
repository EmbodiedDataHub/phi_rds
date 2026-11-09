"""Ray 适配：与 LocalExecutor 同一协议；map 必须是模块级可 pickle 函数。

加入方式（推荐）::

    from phi_rds.exec import RayExecutor, make_rows, process_row, check_results

    rows = make_rows(
        convert="my_pkg.pipeline:convert_l2_from_row",
        items=refs, output=out, layout_robot="agibot", layout_arms="bimanual",
        episodes_per_shard=4, extra={"l1_root": str(l1)},
    )
    results = RayExecutor().run(rows, process_row, check_results)
    # driver 再按 shard pickle 合并进 DatasetWriter / finalize

``convert_l2_from_row(row)`` 内完成：读 L1 → FK/换系 → ``fill_delta_chunks`` → 返回 Episode 列表。
"""

from __future__ import annotations

import os
from typing import Callable, Sequence, TypeVar

R = TypeVar("R")
A = TypeVar("A")
T = TypeVar("T")


class RayExecutor:
    """elastic map；reduce 在 driver 上跑。"""

    def run(
        self,
        rows: Sequence[R],
        map_fn: Callable[[R], A],
        reduce_fn: Callable[[list[A]], T],
        on_item: Callable[[A], None] | None = None,
    ) -> T:
        import ray

        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        conc = os.environ.get("PHI_RAY_MAP_CONCURRENCY")
        # staged：phi_rds 包与 examples/ 都在 working_dir（cwd）下
        cwd = os.getcwd()
        py_bits = [
            cwd,
            os.path.join(cwd, "examples"),
            os.path.join(cwd, "phi_rds", "src"),  # 本地未 flatten 时
            os.path.join(cwd, "phi_rds", "examples"),
            os.environ.get("PYTHONPATH", ""),
        ]
        py_path = os.pathsep.join(p for p in py_bits if p)
        os.environ["PYTHONPATH"] = py_path
        runtime_env: dict = {"env_vars": {"PYTHONPATH": py_path}}
        # 新集群镜像没有 mcap。装到 driver 的 runtime env 上，worker 任务会继承。
        pip = [p.strip() for p in os.environ.get("PHI_RAY_PIP", "").split(",") if p.strip()]
        if pip:
            runtime_env["pip"] = pip
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
        ctx = ray.data.DataContext.get_current()
        ctx.execution_no_progress_timeout_s = -1
        # 节点是 4 vCPU。一片占满一台，三路 x264 才不会被别的 shard 抢走。
        map_kwargs: dict = {"num_cpus": 4}
        if conc:
            map_kwargs["concurrency"] = int(conc)
        items = list(rows)
        # 默认按数据量并成约 200 个 block，657 片会挤在 200 个 task 里。
        ds = ray.data.from_items(items, override_num_blocks=len(items)).map(map_fn, **map_kwargs)
        results: list[A] = []
        for row in ds.iter_rows():
            results.append(row)
            if on_item is not None:
                on_item(row)
        return reduce_fn(results)
