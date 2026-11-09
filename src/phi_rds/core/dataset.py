"""RobotDataSet：Schema 先行 + 懒 Episode 索引 + 高效灌数写出。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, runtime_checkable

from phi_rds.core.episode import Episode
from phi_rds.core.meta import DatasetMeta, EpisodeMeta, ViewInfo
from phi_rds.schema.spec import DatasetSchema
from phi_rds.core.stats import DatasetStats, EpisodeStats, LengthDistribution


@runtime_checkable
class EpisodeStore(Protocol):
    """底层存储适配器（内存 / LeRobot v3 等）"""

    def list_metas(self) -> Iterable[EpisodeMeta]: ...
    def open(self, uid: str) -> Episode: ...
    def exists(self, uid: str) -> bool: ...


@dataclass(frozen=True)
class EpisodeIndexRow:
    uid: str
    num_frames: int
    status: str = "active"
    task: str = ""
    episode_index: int | None = None


@dataclass
class EpisodeIndex:
    rows: tuple[EpisodeIndexRow, ...] = ()
    _by_uid: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_uid = {row.uid: i for i, row in enumerate(self.rows)}

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[EpisodeIndexRow]:
        return iter(self.rows)

    def uids(self) -> tuple[str, ...]:
        return tuple(row.uid for row in self.rows)

    def row(self, key: int | str) -> EpisodeIndexRow:
        if isinstance(key, int):
            return self.rows[key]
        idx = self._by_uid.get(key)
        if idx is None:
            raise KeyError(f"episode uid {key!r} not in index")
        return self.rows[idx]

    def position(self, uid: str) -> int:
        if uid not in self._by_uid:
            raise KeyError(f"episode uid {uid!r} not in index")
        return self._by_uid[uid]

    def select(self, uids: Sequence[str]) -> EpisodeIndex:
        unknown = [uid for uid in uids if uid not in self._by_uid]
        if unknown:
            raise KeyError(f"unknown uids: {unknown[:5]}{'…' if len(unknown) > 5 else ''}")
        rows = [
            EpisodeIndexRow(
                uid=self.row(uid).uid,
                num_frames=self.row(uid).num_frames,
                status=self.row(uid).status,
                task=self.row(uid).task,
                episode_index=new_i,
            )
            for new_i, uid in enumerate(uids)
        ]
        return EpisodeIndex(rows=tuple(rows))

    def range(self, start: int, stop: int) -> EpisodeIndex:
        start = max(0, start)
        stop = min(len(self.rows), stop)
        rows = tuple(
            EpisodeIndexRow(
                uid=row.uid,
                num_frames=row.num_frames,
                status=row.status,
                task=row.task,
                episode_index=i,
            )
            for i, row in enumerate(self.rows[start:stop])
        )
        return EpisodeIndex(rows=rows)

    @staticmethod
    def from_metas(metas: Iterable[EpisodeMeta]) -> EpisodeIndex:
        rows = [
            EpisodeIndexRow(
                uid=meta.uid,
                num_frames=meta.num_frames,
                status=meta.status,
                task=meta.task,
                episode_index=meta.episode_index if meta.episode_index is not None else i,
            )
            for i, meta in enumerate(metas)
        ]
        return EpisodeIndex(rows=tuple(rows))

    def recount_stats(self) -> DatasetStats:
        status_counts: dict[str, int] = {}
        num_frames = 0
        lengths: list[int] = []
        for row in self.rows:
            status_counts[row.status] = status_counts.get(row.status, 0) + 1
            num_frames += row.num_frames
            lengths.append(row.num_frames)
        return DatasetStats(
            num_episodes=len(self.rows),
            num_frames=num_frames,
            status_counts=status_counts,
            frame_count=LengthDistribution.from_values(lengths) if lengths else None,
            scope="self",
        )


@dataclass
class InMemoryEpisodeStore:
    _episodes: dict[str, Episode] = field(default_factory=dict)
    _metas: dict[str, EpisodeMeta] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def add(self, episode: Episode) -> None:
        uid = episode.uid
        if uid not in self._episodes:
            self._order.append(uid)
        self._episodes[uid] = episode
        self._metas[uid] = episode.meta

    def list_metas(self) -> Iterable[EpisodeMeta]:
        for uid in self._order:
            yield self._metas[uid]

    def open(self, uid: str) -> Episode:
        return self._episodes[uid]

    def exists(self, uid: str) -> bool:
        return uid in self._episodes


@dataclass
class DatasetWriter:
    """高效灌数：Schema 已固定，只追加 Episode，最后 finalize。

    内部一次 ``LeRobotDataset.create``，之后每集 ``encode → add_frame* → save_episode``。
    视频：数值走官方 API（``use_videos=False``），finalize 时按 v3 规范挂载 mp4 + episode 段 meta。
    """

    root: Path
    schema: DatasetSchema
    meta: DatasetMeta
    _ds: Any = field(default=None, repr=False)
    _uids: list[str] = field(default_factory=list, repr=False)
    _episodes: list[Episode] = field(default_factory=list, repr=False)
    _finalized: bool = False
    copy_videos: bool = True
    release_sources: bool = False

    @classmethod
    def create(
        cls,
        root: str | Path,
        schema: DatasetSchema,
        meta: DatasetMeta,
        *,
        overwrite: bool = True,
        release_sources: bool = False,
    ) -> DatasetWriter:
        from phi_rds.io.lerobot import _create_kwargs, _require_lerobot

        LeRobotDataset = _require_lerobot()
        root = Path(root)
        if root.exists():
            if not overwrite:
                raise FileExistsError(root)
            import shutil

            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)

        features = schema.to_lerobot_features()
        if not features:
            raise ValueError("schema has no features; refuse to create empty dataset")

        meta = meta.bind_schema(schema)
        meta = replace(meta, uri=str(root.resolve()), fps=schema.fps)
        ds = LeRobotDataset.create(
            **_create_kwargs(
                int(round(schema.fps)),
                features,
                root,
                schema.robot_type or meta.robot,
                metadata_buffer_size=10,
            )
        )
        return cls(
            root=root,
            schema=schema,
            meta=meta,
            _ds=ds,
            release_sources=release_sources,
        )

    def add_episode(self, episode: Episode) -> None:
        if self._finalized:
            raise RuntimeError("writer already finalized")
        if self._ds is None:
            raise RuntimeError("writer has no underlying dataset")

        import numpy as np

        from phi_rds import keys
        from phi_rds.schema.layout import encode

        feature_names = set(self.schema.to_lerobot_features())
        # lerobot 0.4.x validate_frame 用 list 形状与 ndarray.shape(tuple) 做 !=，会误报；写成 tuple
        for _name, feat in getattr(self._ds, "features", {}).items():
            shape = feat.get("shape")
            if isinstance(shape, list):
                feat["shape"] = tuple(int(x) for x in shape)

        if self.schema.layout is not None:
            # L2 的 delta_chunk 不在这里自动算；写出前请显式调用 fill_delta_chunks(episode, layout)
            columns = encode(episode, self.schema.layout)
        else:
            # free：直接按 Schema 列名从 episode.fields 取数
            columns = {}
            for name in feature_names:
                if name not in episode.fields:
                    continue
                handle = episode.fields[name].payload
                columns[name] = np.asarray(handle.load() if hasattr(handle, "load") else handle)
            missing = feature_names - set(columns) - {n for n, f in self.schema.feature_map().items() if f.is_video}
            if missing:
                raise ValueError(
                    f"episode missing schema columns {sorted(missing)}; "
                    "provide schema.layout or fill episode.fields"
                )

        task = episode.labels.task or ""
        feat_specs = getattr(self._ds, "features", {}) or {}
        for frame_index in range(episode.num_frames):
            frame: dict[str, Any] = {"task": task}
            for name, array in columns.items():
                if name == keys.TIMESTAMP or name not in feature_names:
                    continue
                value = np.asarray(array[frame_index])
                spec = self.schema.get(name)
                shape = tuple(spec.shape) if spec and spec.shape else value.shape
                value = value.reshape(shape) if len(shape) > 1 else value.reshape(-1)
                want = (feat_specs.get(name) or {}).get("dtype")
                if want and str(value.dtype) != want and want.startswith("float"):
                    value = value.astype(want, copy=False)
                frame[name] = value
            self._ds.add_frame(frame)
        self._ds.save_episode()
        self._uids.append(episode.uid)
        self._episodes.append(episode)

    def _rewrite_tasks_parquet(self) -> None:
        """按 LeRobot 官方格式重写 ``meta/tasks.parquet``。

        任务句子是行索引（index name = ``task``），唯一数据列是 ``task_index``。
        ``LeRobotDataset.__getitem__`` 用 ``tasks.iloc[task_index].name`` 取句子，
        ``get_task_index`` 用句子做 ``loc``。两列扁平表读出来的 ``item["task"]`` 是整数。

        编号仍按 episode 首次出现的顺序，和 ``save_episode`` 写进数据表的 ``task_index`` 一致。
        先写本地临时文件再替换，避免 OSS-FUSE 上 ``to_parquet`` 落成 0 字节。
        """
        import os
        import shutil
        import tempfile

        import pandas as pd

        names: list[str] = []
        for ep in self._episodes:
            name = (ep.labels.task if ep.labels else None) or (ep.meta.task if ep.meta else None) or ""
            name = str(name).strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return
        frame = pd.DataFrame(
            {"task_index": list(range(len(names)))},
            index=pd.Index(names, name="task"),
        )
        dest = self.root / "meta" / "tasks.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        tmp = Path(tmp_name)
        partial = dest.with_name(dest.name + ".partial")
        try:
            frame.to_parquet(tmp)
            shutil.copyfile(tmp, partial)
            os.replace(partial, dest)
            with open(dest, "rb") as handle:
                os.fsync(handle.fileno())
        finally:
            tmp.unlink(missing_ok=True)
            if partial.exists():
                partial.unlink(missing_ok=True)

    def _rewrite_subtasks_parquet(self) -> None:
        """写出 ``meta/subtasks.parquet``（统一格式：index=文本，列=subtask_index）。"""
        from phi_rds.io.lerobot import write_subtasks_parquet

        write_subtasks_parquet(self.root, self._episodes)

    def finalize(self) -> Path:
        if self._finalized:
            return self.root
        if self._ds is None:
            raise RuntimeError("writer has no underlying dataset")
        self._ds.finalize()
        self._finalized = True

        # OSS-FUSE 上官方 save_episode_tasks 的早期 to_parquet 偶发落成 0 字节；
        # finalize 末尾用 episode.labels.task 再写一次（原子 replace）。
        self._rewrite_tasks_parquet()
        self._rewrite_subtasks_parquet()

        from phi_rds.io.serialize import dump_json

        dump_json(self.root / "meta" / "dataset_meta.json", self.meta)
        dump_json(
            self.root / "meta" / "schema.json",
            {
                "id": self.schema.id,
                "fps": self.schema.fps,
                "robot_type": self.schema.robot_type,
                "level": self.schema.level,
                "standard_version": self.schema.standard_version,
                "profile": self.schema.profile,
                "features": [f.name for f in self.schema.features],
            },
        )
        # 回写 info：任何人打开都能立刻看到 L1/L2 + 标准版本
        info_path = self.root / "meta" / "info.json"
        if info_path.is_file():
            import json

            info = json.loads(info_path.read_text(encoding="utf-8"))
            info.update(self.schema.to_stamp())
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if self.copy_videos and self._episodes:
            from phi_rds.io.lerobot import attach_videos

            attach_videos(
                self.root,
                self._episodes,
                fps=float(self.schema.fps),
                release_sources=self.release_sources,
            )
        if self._episodes:
            from phi_rds.io.lerobot import attach_episode_meta

            attach_episode_meta(self.root, self._episodes)
        return self.root


@dataclass
class RobotDataSet:
    """数据集视图：持有固定 ``schema`` + 懒 Episode 索引。

    日常用法只记两个入口：
      - ``RobotDataSet.open(path)`` / ``open_dataset(path)``  — 打开即读
      - ``RobotDataSet.create(path, schema, meta)``           — 先布局再灌数
    """

    meta: DatasetMeta
    schema: DatasetSchema
    index: EpisodeIndex
    store: EpisodeStore
    _meta_cache: dict[str, EpisodeMeta] = field(default_factory=dict, repr=False)
    _episode_cache: dict[str, Episode] = field(default_factory=dict, repr=False)
    _episode_cache_limit: int = 32

    @property
    def level(self):
        return self.schema.level or self.meta.level

    @property
    def standard_version(self) -> str | None:
        return self.schema.standard_version or self.meta.standard_version

    @property
    def is_l1(self) -> bool:
        return self.level == "L1"

    @property
    def is_l2(self) -> bool:
        return self.level == "L2"

    @classmethod
    def open(
        cls,
        meta: DatasetMeta | str | Path | None = None,
        store: EpisodeStore | None = None,
        *,
        schema: DatasetSchema | None = None,
        **load_kwargs: Any,
    ) -> RobotDataSet:
        """打开数据集。

        - ``RobotDataSet.open(path)``：从目录加载（最常用）
        - ``RobotDataSet.open(meta, store, schema=...)``：内部组装
        """
        if isinstance(meta, (str, Path)) and store is None:
            return cls.load(meta, **load_kwargs)
        if meta is None or store is None:
            raise TypeError("open(path) or open(meta, store, schema=...) required")
        return cls._from_store(meta, store, schema=schema)

    @classmethod
    def _from_store(
        cls,
        meta: DatasetMeta,
        store: EpisodeStore,
        *,
        schema: DatasetSchema | None = None,
    ) -> RobotDataSet:
        index = EpisodeIndex.from_metas(store.list_metas())
        stats = index.recount_stats()
        merged = replace(
            meta,
            stats=replace(
                meta.stats,
                num_episodes=stats.num_episodes,
                num_frames=stats.num_frames,
                status_counts=stats.status_counts,
                frame_count=stats.frame_count,
                scope="self",
                feature_list=schema.feature_names if schema else meta.stats.feature_list,
            ),
        )
        if schema is None:
            schema = DatasetSchema(
                id=merged.schema_id or "unknown",
                fps=merged.fps,
                robot_type=merged.robot,
                level=merged.level,
                standard_version=merged.standard_version,
                profile="free",
            )
        else:
            merged = merged.bind_schema(schema)
        return cls(meta=merged, schema=schema, index=index, store=store)

    @classmethod
    def from_episodes(
        cls,
        meta: DatasetMeta,
        episodes: Sequence[Episode],
        *,
        schema: DatasetSchema | None = None,
    ) -> RobotDataSet:
        store = InMemoryEpisodeStore()
        for episode in episodes:
            store.add(episode)
        return cls._from_store(meta, store, schema=schema)

    @classmethod
    def create(
        cls,
        root: str | Path,
        schema: DatasetSchema,
        meta: DatasetMeta,
        *,
        overwrite: bool = True,
        release_sources: bool = False,
    ) -> DatasetWriter:
        """先定布局，再 ``add_episode`` 灌数，最后 ``finalize``。"""
        return DatasetWriter.create(
            root, schema, meta, overwrite=overwrite, release_sources=release_sources
        )

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[Episode]:
        for row in self.index:
            yield self[row.uid]

    def __getitem__(self, key: int | str) -> Episode:
        row = self.index.row(key)
        return self._open(row.uid, episode_index=row.episode_index)

    def uids(self) -> tuple[str, ...]:
        return self.index.uids()

    def meta_of(self, key: int | str) -> EpisodeMeta:
        row = self.index.row(key)
        cached = self._meta_cache.get(row.uid)
        if cached is not None:
            return replace(cached, episode_index=row.episode_index)
        if self.store.exists(row.uid):
            return self._open(row.uid, episode_index=row.episode_index).meta
        return EpisodeMeta(
            uid=row.uid,
            num_frames=row.num_frames,
            fps=self.meta.fps,
            task=row.task,
            status=row.status,  # type: ignore[arg-type]
            episode_index=row.episode_index,
            stats=EpisodeStats(num_frames=row.num_frames),
        )

    def _open(self, uid: str, *, episode_index: int | None) -> Episode:
        cached = self._episode_cache.get(uid)
        if cached is not None:
            if cached.meta.episode_index != episode_index:
                return cached.with_meta(episode_index=episode_index)
            return cached
        episode = self.store.open(uid)
        if episode.meta.episode_index != episode_index:
            episode = episode.with_meta(episode_index=episode_index)
        self._meta_cache[uid] = episode.meta
        self._remember(uid, episode)
        return episode

    def _remember(self, uid: str, episode: Episode) -> None:
        self._episode_cache[uid] = episode
        if len(self._episode_cache) <= self._episode_cache_limit:
            return
        oldest = next(iter(self._episode_cache))
        if oldest != uid:
            self._episode_cache.pop(oldest, None)

    def select(self, uids: Sequence[str]) -> RobotDataSet:
        return self._view(self.index.select(uids), selector=f"select({len(uids)} uids)")

    def range(self, start: int, stop: int) -> RobotDataSet:
        return self._view(self.index.range(start, stop), selector=f"range({start},{stop})")

    def take(self, n: int) -> RobotDataSet:
        return self.range(0, n)

    def skip(self, n: int) -> RobotDataSet:
        return self.range(n, len(self))

    def filter(
        self,
        pred: Callable[[EpisodeMeta], bool],
        *,
        use_full_meta: bool = False,
    ) -> RobotDataSet:
        kept: list[str] = []
        for row in self.index:
            meta = self.meta_of(row.uid) if use_full_meta else EpisodeMeta(
                uid=row.uid,
                num_frames=row.num_frames,
                fps=self.meta.fps,
                task=row.task,
                status=row.status,  # type: ignore[arg-type]
                episode_index=row.episode_index,
            )
            if pred(meta):
                kept.append(row.uid)
        return self.select(kept)

    def _view(self, new_index: EpisodeIndex, *, selector: str) -> RobotDataSet:
        count_stats = new_index.recount_stats()
        parent_features = dict(self.meta.stats.features)
        new_stats = DatasetStats(
            num_episodes=count_stats.num_episodes,
            num_frames=count_stats.num_frames,
            feature_list=self.schema.feature_names,
            status_counts=count_stats.status_counts,
            frame_count=count_stats.frame_count,
            features=parent_features,
            scope="parent" if parent_features else "self",
            extras=dict(self.meta.stats.extras),
        )
        new_meta = replace(
            self.meta,
            stats=new_stats,
            view=ViewInfo(parent_uri=self.meta.uri, selector=selector, is_view=True),
        )
        return RobotDataSet(
            meta=new_meta,
            schema=self.schema,
            index=new_index,
            store=self.store,
            _episode_cache_limit=self._episode_cache_limit,
        )

    def refresh_count_stats(self) -> RobotDataSet:
        stats = self.index.recount_stats()
        new_meta = replace(
            self.meta,
            stats=replace(
                self.meta.stats.with_counts(
                    num_episodes=stats.num_episodes,
                    num_frames=stats.num_frames,
                    status_counts=stats.status_counts,
                    scope="self",
                ),
                frame_count=stats.frame_count,
            ),
        )
        return RobotDataSet(
            meta=new_meta,
            schema=self.schema,
            index=self.index,
            store=self.store,
            _meta_cache=dict(self._meta_cache),
            _episode_cache=dict(self._episode_cache),
            _episode_cache_limit=self._episode_cache_limit,
        )

    def save(self, root: str | Path, **kwargs: Any) -> Path:
        from phi_rds.io import write_dataset

        layout = kwargs.pop("layout", None) or (self.schema.layout if self.schema else None)
        return write_dataset(self, root, layout=layout, **kwargs)

    def as_lerobot(self):
        """底层官方 ``LeRobotDataset``（训练 / DataLoader 直连）。"""
        from phi_rds.io.lerobot import open_lerobot_dataset

        if not self.meta.uri:
            raise ValueError("dataset has no uri; open from disk first")
        return open_lerobot_dataset(self.meta.uri)

    @classmethod
    def load(cls, root: str | Path, **kwargs) -> RobotDataSet:
        from phi_rds.io import read_dataset

        return read_dataset(root, **kwargs)
