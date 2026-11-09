"""delta_chunk 转换工具：相对当前 state 末端的时序增量块。

与 FK / change_frame 同类——在**转换阶段**算完，挂到 ``Episode.fields``，
写出时 ``encode`` 只序列化，不再计算。

公式（每个部位、每个 horizon ``k``）::

    Δ(k) = inv(T_state(t)) @ T_tgt(t ± k)   → pose9
    past  → tgt = state；future → tgt = action
    越界补 0；输出形状 ``(N, chunk, 9 * n_parts)``
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from phi_rds.convert.pose import encode_pose
from phi_rds.core.episode import Field, FieldKind, FieldMap
from phi_rds.core.lazy import LazyHandle
from phi_rds.core.semantics import FieldSemantics

if TYPE_CHECKING:
    from phi_rds.core.body import RobotBody
    from phi_rds.core.episode import Episode
    from phi_rds.schema.layout import Layout


def shift_series(
    seq: np.ndarray,
    k: int,
    direction: str,
    n: int | None = None,
) -> np.ndarray:
    """沿时间轴平移；越界填 0。``direction`` ∈ {``past``, ``future``}。"""
    if direction not in ("past", "future"):
        raise ValueError(f"direction must be 'past' or 'future', got {direction!r}")
    n = int(seq.shape[0] if n is None else n)
    out = np.zeros_like(seq)
    if k >= n or k < 0:
        return out
    if direction == "future":
        out[: n - k] = seq[k:]
    else:
        out[k:] = seq[: n - k]
    return out


def delta_chunk(
    ref_poses: Mapping[str, np.ndarray],
    tgt_poses: Mapping[str, np.ndarray],
    *,
    parts: Iterable[str],
    chunk_size: int,
    direction: str,
    pose_rep: str = "pose9",
) -> np.ndarray:
    """纯数组工具：``ref/tgt[part]`` 为 ``(N,4,4)`` → ``(N, chunk, 9*|parts|)``。"""
    parts = tuple(parts)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not parts:
        raise ValueError("parts must be non-empty")
    n = int(next(iter(ref_poses.values())).shape[0])
    inv_ref = {
        p: np.linalg.inv(np.asarray(ref_poses[p], dtype=np.float64)) for p in parts
    }
    tgt = {p: np.asarray(tgt_poses[p], dtype=np.float64) for p in parts}
    blocks = []
    for k in range(int(chunk_size)):
        cols = [
            encode_pose(
                inv_ref[p] @ shift_series(tgt[p], k, direction, n),
                pose_rep,
            )
            for p in parts
        ]
        blocks.append(np.concatenate(cols, axis=1))
    return np.stack(blocks, axis=1)


def delta_chunk_from_bodies(
    state: "RobotBody",
    target: "RobotBody",
    *,
    parts: Iterable[str],
    chunk_size: int,
    direction: str,
    pose_rep: str = "pose9",
) -> np.ndarray:
    """从 RobotBody 取 ee 矩阵再算 delta_chunk。"""
    parts = tuple(parts)
    ref = {p: np.asarray(state.ee_matrix(p), dtype=np.float64) for p in parts}
    tgt = {p: np.asarray(target.ee_matrix(p), dtype=np.float64) for p in parts}
    return delta_chunk(
        ref, tgt, parts=parts, chunk_size=chunk_size, direction=direction, pose_rep=pose_rep
    )


def fill_delta_chunks(episode: "Episode", layout: "Layout") -> "Episode":
    """计算 Layout 中全部 ``delta_chunk``，写入 ``episode.fields`` 后返回新 Episode。

    这是 SDK 提供的标准计算方法，**不会**在 ``add_episode`` / ``write_lerobot`` 里自动执行。
    写 L2 前请显式调用::

        ep = fill_delta_chunks(ep, writer.schema.layout)
        writer.add_episode(ep)

    公式：相对当前 state 末端的 SE(3) 增量，再编码为 pose9。
    state 侧 direction=past、chunk=10；action 侧 direction=future、chunk=64。
    未调用就写出时，``encode`` 会报错提示本函数。
    """
    fields = FieldMap()
    fields.update(episode.fields.as_dict())
    for feature in layout.features:
        if not feature.slices or feature.slices[0].kind != "delta_chunk":
            continue
        piece = feature.slices[0]
        if episode.state is None:
            raise ValueError(f"delta_chunk {feature.name!r} needs episode.state")
        tgt_body = episode.state if piece.direction == "past" else episode.action
        if tgt_body is None:
            raise ValueError(
                f"delta_chunk {feature.name!r} direction={piece.direction!r} "
                "needs episode.action"
            )
        arr = delta_chunk_from_bodies(
            episode.state,
            tgt_body,
            parts=piece.parts,
            chunk_size=piece.chunk_size,
            direction=piece.direction,
        ).astype(feature.dtype, copy=False)
        fields.put(
            Field(
                id=feature.name,
                kind=FieldKind.DELTA_CHUNK,
                semantics=FieldSemantics(
                    kind="delta_chunk",
                    role="action" if piece.direction == "future" else "observation",
                    extras={
                        "direction": piece.direction,
                        "chunk_size": piece.chunk_size,
                        "parts": list(piece.parts),
                    },
                ),
                payload=LazyHandle.from_memory(arr),
            )
        )
    return replace(episode, fields=fields)
