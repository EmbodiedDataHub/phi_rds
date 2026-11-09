"""本地串行 Executor：与 RayExecutor 同一 ``run(rows, map_fn, reduce_fn)`` 协议。"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

R = TypeVar("R")
A = TypeVar("A")
T = TypeVar("T")


class LocalExecutor:
    def run(
        self,
        rows: Sequence[R],
        map_fn: Callable[[R], A],
        reduce_fn: Callable[[list[A]], T],
        on_item: Callable[[A], None] | None = None,
    ) -> T:
        results: list[A] = []
        for row in rows:
            item = map_fn(row)
            results.append(item)
            if on_item is not None:
                on_item(item)
        return reduce_fn(results)
