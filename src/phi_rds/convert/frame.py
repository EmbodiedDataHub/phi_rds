"""齐次变换 / 轴向约定 / FrameGraph 换系。"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from phi_rds.core.semantics import FrameEdge, FrameGraph

AxesName = str


# ---------------------------------------------------------------------------
# 齐次变换基础
# ---------------------------------------------------------------------------

def as_T(value: np.ndarray) -> np.ndarray:
    """规范到 ``(..., 4, 4)``。"""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[-2:] != (4, 4):
        raise ValueError(f"expected (...,4,4), got {arr.shape}")
    return arr


def invert_T(T: np.ndarray) -> np.ndarray:
    """刚体逆：``R^T``, ``-R^T t``。支持 batch。"""
    T = as_T(T)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def compose_T(*transforms: np.ndarray) -> np.ndarray:
    """从左到右复合：``T0 @ T1 @ ...``。"""
    if not transforms:
        raise ValueError("compose_T requires at least one transform")
    acc = as_T(transforms[0])
    for T in transforms[1:]:
        acc = acc @ as_T(T)
    return acc


def apply_T(
    poses: np.ndarray,
    T: np.ndarray,
    *,
    side: Literal["left", "right"] = "left",
) -> np.ndarray:
    """对位姿序列施加固定刚体变换。

    - ``side="left"``：``T @ pose``（改世界系 / 外参左乘）
    - ``side="right"``：``pose @ T``（改体坐标系 / 工具偏置）
    """
    poses = as_T(poses)
    T = as_T(T)
    if T.ndim == 2:
        return (T @ poses) if side == "left" else (poses @ T)
    if poses.ndim == 2:
        poses = poses[None, ...]
    return (T @ poses) if side == "left" else (poses @ T)


def change_frame(
    poses_in_A: np.ndarray,
    T_B_A: np.ndarray,
) -> np.ndarray:
    """把表达在 A 系的位姿改写到 B 系：``T_B_P = T_B_A @ T_A_P``。"""
    return apply_T(poses_in_A, T_B_A, side="left")


def translate(T: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    """在变换平移上加偏移（世界系）。"""
    out = as_T(T).copy()
    out[..., :3, 3] = out[..., :3, 3] + np.asarray(xyz, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# 轴向约定（FLU / RDF / RFU）
# ---------------------------------------------------------------------------

# 列 = 目标系基向量在 FLU 下的坐标；P_flu = R @ P_src
_AXES_FROM_FLU: dict[str, np.ndarray] = {
    # identity
    "FLU": np.eye(3),
    # OpenCV 相机：X右 Y下 Z前  ←  FLU(X前 Y左 Z上)
    "RDF": np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    ),
    # X右 Y前 Z上
    "RFU": np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}


def axes_rotation(src: AxesName, dst: AxesName) -> np.ndarray:
    """``(3,3)``：把 ``src`` 轴向向量变到 ``dst`` 轴向。"""
    src_s, dst_s = str(src), str(dst)
    if src_s not in _AXES_FROM_FLU or dst_s not in _AXES_FROM_FLU:
        raise ValueError(f"unsupported axes {src_s!r}→{dst_s!r}; known={sorted(_AXES_FROM_FLU)}")
    R_src = _AXES_FROM_FLU[src_s]  # basis of src in FLU
    R_dst = _AXES_FROM_FLU[dst_s]
    # v_flu = R_src @ v_src = R_dst @ v_dst  →  v_dst = R_dst.T @ R_src @ v_src
    return R_dst.T @ R_src


def convert_axes(poses: np.ndarray, src: AxesName, dst: AxesName) -> np.ndarray:
    """旋转部分按轴向约定重写；平移同步变到新轴向。"""
    if src == dst:
        return as_T(poses).copy()
    R = axes_rotation(src, dst)
    T = np.eye(4)
    T[:3, :3] = R
    # P_dst = R @ P_src @ R^T  （相似变换，保持相对几何）
    return apply_T(apply_T(poses, T, side="left"), invert_T(T), side="right")


# ---------------------------------------------------------------------------
# FrameGraph 查询
# ---------------------------------------------------------------------------

def _edge_matrix(edge: FrameEdge) -> np.ndarray:
    if edge.kind == "fk":
        raise ValueError(f"cannot resolve dynamic FK edge {edge.child!r} without robot model")
    if edge.matrix_flat16 is None:
        raise ValueError(f"edge {edge.child!r}←{edge.parent!r} has no matrix")
    return np.asarray(edge.matrix_flat16, dtype=np.float64).reshape(4, 4)


def lookup_T(
    graph: FrameGraph,
    source: str,
    target: str,
) -> np.ndarray:
    """求 ``T_target_source``（``p_target = T @ p_source``）。

    图上边语义：``child`` 相对 ``parent`` 的外参 = ``T_parent_child``。
    """
    if source == target:
        return np.eye(4)

    parent_of: dict[str, FrameEdge] = {edge.child: edge for edge in graph.edges}

    def path_to_root(frame: str) -> list[str]:
        chain = [frame]
        seen = {frame}
        while frame in parent_of:
            frame = parent_of[frame].parent
            if frame in seen:
                raise ValueError(f"cycle in FrameGraph at {frame!r}")
            chain.append(frame)
            seen.add(frame)
        return chain

    src_path = path_to_root(source)
    tgt_path = path_to_root(target)
    src_set = set(src_path)
    ancestor = next((f for f in tgt_path if f in src_set), None)
    if ancestor is None:
        raise KeyError(f"no common ancestor between {source!r} and {target!r}")

    # source → ancestor：沿边上升，每条边是 T_parent_child
    T_anc_src = np.eye(4)
    for frame in src_path:
        if frame == ancestor:
            break
        T_anc_src = compose_T(_edge_matrix(parent_of[frame]), T_anc_src)

    # target → ancestor，再取逆 → T_target_ancestor
    T_anc_tgt = np.eye(4)
    for frame in tgt_path:
        if frame == ancestor:
            break
        T_anc_tgt = compose_T(_edge_matrix(parent_of[frame]), T_anc_tgt)

    return compose_T(invert_T(T_anc_tgt), T_anc_src)


def reexpress_via_graph(
    poses: np.ndarray,
    graph: FrameGraph,
    *,
    from_frame: str,
    to_frame: str,
) -> np.ndarray:
    """用 ``FrameGraph`` 把位姿从 ``from_frame`` 改写到 ``to_frame``。"""
    return change_frame(poses, lookup_T(graph, from_frame, to_frame))

