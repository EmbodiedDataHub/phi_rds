"""Local / Ray 共用的 map 行协议。

分块（二选一，互斥优先序）::

  1. ``episodes_per_shard=K``  — 每 worker 固定处理 K 集（最后一块可能不足 K）
  2. ``num_workers=W``         — 按 W 个 worker **均分** N 集（前 ``N%W`` 块多 1 集）
  3. 都未给               — 等价 ``episodes_per_shard=1``
  4. ``num_workers="auto"``  — 探测可用并行度（Ray cluster CPU 或 ``os.cpu_count``）再均分

每行一个 shard：convert →（后续应写出官方 LeRobot 子集；当前仍 pickle Episode 过渡）。
``convert`` 规格：``\"module.path:callable\"``，模块级（Ray 要 pickle）。
"""

from __future__ import annotations

import importlib
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal


def detect_num_workers() -> int:
    """可用 worker 数：优先 Ray cluster CPU，否则本机 ``cpu_count``，至少 1。"""
    try:
        import ray

        if ray.is_initialized():
            res = ray.cluster_resources()
            cpus = float(res.get("CPU", 0) or 0)
            if cpus >= 1:
                return max(1, int(cpus))
    except Exception:  # noqa: BLE001
        pass
    return max(1, int(os.cpu_count() or 1))


def plan_shard_slices(
    n_items: int,
    *,
    episodes_per_shard: int | None = None,
    num_workers: int | Literal["auto"] | None = None,
) -> list[tuple[int, int]]:
    """把 ``[0, n_items)`` 切成连续半开区间 ``(start, stop)``。

    - 指定 ``episodes_per_shard`` → 固定块长（优先）
    - 否则指定 ``num_workers`` / ``\"auto\"`` → 均分
    - 都未指定 → 每集一块
    """
    n = max(0, int(n_items))
    if n == 0:
        return []

    if episodes_per_shard is not None:
        if num_workers is not None:
            raise ValueError("pass only one of episodes_per_shard / num_workers")
        k = max(1, int(episodes_per_shard))
        return [(s, min(s + k, n)) for s in range(0, n, k)]

    if num_workers is not None:
        if num_workers == "auto":
            w = detect_num_workers()
        else:
            w = max(1, int(num_workers))
        w = min(w, n)  # worker 多于集数时每集一块即可
        base, rem = divmod(n, w)
        # 前 rem 个 shard 长度为 base+1，其余 base
        slices: list[tuple[int, int]] = []
        start = 0
        for i in range(w):
            size = base + (1 if i < rem else 0)
            if size <= 0:
                continue
            stop = start + size
            slices.append((start, stop))
            start = stop
        return slices

    return [(i, i + 1) for i in range(n)]


def make_rows(
    *,
    convert: str,
    items: list[Any],
    output: str | Path,
    episodes_per_shard: int | None = None,
    num_workers: int | Literal["auto"] | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict]:
    """按 ``plan_shard_slices`` 切 ``items``，每块一行 map 任务。"""
    slices = plan_shard_slices(
        len(items),
        episodes_per_shard=episodes_per_shard,
        num_workers=num_workers,
    )
    base = dict(extra or {})
    rows: list[dict] = []
    for shard_index, (start, stop) in enumerate(slices):
        chunk = items[start:stop]
        rows.append({
            **base,
            "convert": convert,
            "output": str(output),
            "shard_index": shard_index,
            "item_start": start,
            "item_stop": stop,
            "n_items": stop - start,
            "items_hex": [pickle.dumps(item, protocol=5).hex() for item in chunk],
        })
    return rows


def _resolve(spec: str) -> Callable[[dict], list]:
    if ":" not in spec:
        raise ValueError(f"convert spec must be 'module:callable', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"cannot resolve convert {spec!r}")
    return fn


def process_row(row: dict) -> dict:
    """map：只跑转换，写出 pickle shard。不做 encode。"""
    shard_index = int(row["shard_index"])
    items = [pickle.loads(bytes.fromhex(h)) for h in row["items_hex"]]
    row = {**row, "items": items}
    try:
        convert_fn = _resolve(row["convert"])
        episodes = convert_fn(row)
        if not isinstance(episodes, list):
            episodes = [episodes]
        out = Path(row["output"])
        shard_dir = out / "_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard-{shard_index:05d}.pkl"
        shard_path.write_bytes(pickle.dumps({"episodes": episodes}, protocol=5))
        return {
            "shard_index": shard_index,
            "status": "OK",
            "shard_path": str(shard_path),
            "n_episodes": len(episodes),
            "uids": [ep.uid for ep in episodes],
        }
    except Exception as error:  # noqa: BLE001
        return {
            "shard_index": shard_index,
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
        }


def check_results(results: list[dict], expected_shards: int | None = None) -> list[dict]:
    failed = [r for r in results if r.get("status") == "FAILED"]
    if failed:
        detail = "; ".join(
            f"shard={r.get('shard_index')}: {r.get('error')}" for r in failed[:5]
        )
        raise RuntimeError(f"map failed ({len(failed)} shards): {detail}")
    if expected_shards is not None and len(results) != expected_shards:
        raise RuntimeError(f"map returned {len(results)} results, expected {expected_shards}")
    counts = Counter(r.get("status") for r in results)
    if counts.get("OK", 0) != len(results):
        raise RuntimeError(f"unexpected statuses: {dict(counts)}")
    return sorted(results, key=lambda r: int(r["shard_index"]))


def load_shard_episodes(shard_path: str | Path) -> list:
    blob = pickle.loads(Path(shard_path).read_bytes())
    return list(blob["episodes"])
