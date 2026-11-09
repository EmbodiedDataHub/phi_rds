"""懒句柄：默认只持引用，``.load()`` 才物化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")


class StorageKind(str, Enum):
    MEMORY = "memory"   # 内存
    VIDEO_MP4 = "video_mp4" # 视频
    OTHER = "other" # 其他


@dataclass(frozen=True)
class VideoLocator:
    path: str # 视频路径
    from_timestamp: float = 0.0 # 开始时间
    to_timestamp: float | None = None # 结束时间
    fps: float | None = None # 帧率


@dataclass(frozen=True)
class StorageRef:   # 
    kind: StorageKind # 存储类型
    uri: str | None = None # 存储路径
    video: VideoLocator | None = None # 视频定位


@dataclass
class LazyHandle(Generic[T]): 
    """懒句柄：默认只持引用，``.load()`` 才物化。"""
    storage: StorageRef # 存储引用
    shape: tuple[int, ...] | None = None # 形状
    dtype: str | None = None # 数据类型
    _cache: Any = field(default=None, repr=False, compare=False) # 缓存
    _loader: Any = field(default=None, repr=False, compare=False) # 加载器

    def load(self) -> T: # 加载数据
        if self._cache is not None:
            return self._cache
        if self._loader is None:
            raise RuntimeError(
                f"LazyHandle(kind={self.storage.kind}) has no data; use from_memory() or bind_loader()"
            )
        self._cache = self._loader(self)
        return self._cache

    def bind_loader(self, loader: Any) -> LazyHandle[T]:
        self._loader = loader
        return self

    @staticmethod
    def from_memory(value: T) -> LazyHandle[T]:
        shape = tuple(value.shape) if isinstance(value, np.ndarray) else None
        dtype = str(value.dtype) if isinstance(value, np.ndarray) else None
        handle: LazyHandle[T] = LazyHandle(
            storage=StorageRef(kind=StorageKind.MEMORY),
            shape=shape,
            dtype=dtype,
        )
        handle._cache = value
        return handle

    @staticmethod
    def from_video(
        path: str,
        *,
        from_timestamp: float = 0.0,
        to_timestamp: float | None = None,
        fps: float | None = None,
    ) -> LazyHandle[Any]:
        return LazyHandle(
            storage=StorageRef(
                kind=StorageKind.VIDEO_MP4,
                uri=path,
                video=VideoLocator(
                    path=path,
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp,
                    fps=fps,
                ),
            ),
            dtype="uint8",
        )
