"""LeRobot v3 读写：默认走官方 ``LeRobotDataset`` API。

写：``create`` → ``add_frame`` → ``save_episode`` → ``finalize``
读：``LeRobotDataset(..., download_videos=False)`` + ``meta.episodes`` / ``hf_dataset``

视频（v3）：一个 mp4 可含多集。路径由 ``chunk_index``/``file_index`` 决定；
每集用 ``from_timestamp``/``to_timestamp`` 切段。官方取帧 =
``decode(path, from_timestamp + episode_relative_ts)``。我们只挂句柄，不解码。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from phi_rds import keys
from phi_rds.schema.layout import decode, encode
from phi_rds.core.dataset import RobotDataSet
from phi_rds.core.episode import CameraStream, Episode, Labels, Segment
from phi_rds.core.body import Arm, Gripper, JointGroup, Pose, RobotBody
from phi_rds.io.serialize import dump_json, from_dict, load_json
from phi_rds.core.lazy import LazyHandle, StorageKind, StorageRef, VideoLocator
from phi_rds.schema.layout import Layout
from phi_rds.core.meta import DatasetMeta, EpisodeMeta, EpisodeSpan, VideoSpan
from phi_rds.core.semantics import EpisodeSemantics
from phi_rds.core.stats import DatasetStats, EpisodeStats, FeatureStats

_PHI_META = "meta/dataset_meta.json"
_PHI_SEMANTICS = "meta/semantics.json"
_UID_FILE = "meta/episode_uids.json"  # legacy only; prefer episodes parquet `source.uid`
_EP_SOURCE_UID_COL = "source.uid"
_EP_SOURCE_COL = "source"  # legacy absolute path; prefer source.path
# episodes 列名即 ``cameras``；JSON 值为各相机 intrinsic/extrinsic（对齐飞书相机标准）
_EP_CAMERAS_COL = "cameras"
_EP_SUBTASKS_COL = "subtasks"
# Episode 溯源列（见 docs/DATASET_META.md）
# isnull → bool；images.*.shape → list<int> [H,W,C]；其余 string
_EP_SOURCE_META_COLS = (
    "source.uid",
    "source.path",
    "source.gripper_model",
    "source.images.head.topic",
    "source.observation.state.arm.tcp_pose.isnull",
    "source.action.arm.tcp_pose.isnull",
    "source.observation.images.head.shape",
    "source.observation.images.wrist_left.shape",
    "source.observation.images.wrist_right.shape",
)
_EP_SOURCE_BOOL_COLS = frozenset({
    "source.observation.state.arm.tcp_pose.isnull",
    "source.action.arm.tcp_pose.isnull",
})
_EP_SOURCE_SHAPE_COLS = frozenset({
    "source.observation.images.head.shape",
    "source.observation.images.wrist_left.shape",
    "source.observation.images.wrist_right.shape",
})
_EP_LEGACY_DROP = ("episode_uid", "source", "source.ee", "tcp_source", "camera_calib")


def _cameras_json_payload(raw: Any) -> dict[str, Any] | None:
    """归一成标准 cameras 对象：{cam: {intrinsic, extrinsic}}。"""
    if not isinstance(raw, dict) or not raw:
        return None
    # 兼容旧包装 {"reference":..., "cameras": {...}}
    if "cameras" in raw and isinstance(raw["cameras"], dict):
        sample = next(iter(raw["cameras"].values()), None)
        if isinstance(sample, dict) and "intrinsic" in sample:
            raw = raw["cameras"]
    out: dict[str, Any] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "intrinsic" in entry and "extrinsic" in entry:
            out[str(name)] = {
                "intrinsic": entry["intrinsic"],
                "extrinsic": entry["extrinsic"],
            }
        elif "fx" in entry or "K" in entry:
            # 极旧平铺内参：无法可靠补外参，跳过（由转换器重写）
            continue
    return out or None


def _normalize_source_field(key: str, val: Any) -> Any:
    """把 source_fields 归一成 parquet 原生类型：bool / list[int] / str。"""
    if key in _EP_SOURCE_BOOL_COLS:
        if val is None or val == "":
            return None
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")
        return bool(val)
    if key in _EP_SOURCE_SHAPE_COLS:
        if val is None or val == "":
            return None
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"{key} must be list[int], got {type(val).__name__}")
        return [int(x) for x in val]
    if val is None:
        return ""
    return str(val)


def _pa_string_array(values: list[Any]):
    import pyarrow as pa

    return pa.array(["" if v is None else str(v) for v in values], type=pa.string())


def _source_field_array(key: str, values: list[Any]):
    import pyarrow as pa

    if key in _EP_SOURCE_BOOL_COLS:
        return pa.array(values, type=pa.bool_())
    if key in _EP_SOURCE_SHAPE_COLS:
        return pa.array(values, type=pa.list_(pa.int32()))
    return _pa_string_array(values)


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise ImportError(
            "LeRobot I/O requires the official `lerobot` package. "
            "Install with: pip install 'phi-rds[lerobot]'"
        ) from error
    return LeRobotDataset


def is_lerobot_root(root: str | Path) -> bool:
    return (Path(root) / "meta" / "info.json").is_file()


def open_lerobot_dataset(root: str | Path, *, repo_id: str | None = None):
    """官方打开本地 v3 目录（``download_videos=False``）。"""
    LeRobotDataset = _require_lerobot()
    root = Path(root)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return LeRobotDataset(
        repo_id or f"local/{root.name}",
        root=root,
        download_videos=False,
    )


def _numeric_features(columns: dict[str, np.ndarray], layout: Layout) -> dict[str, Any]:
    from phi_rds.schema.layout import pad_slots

    features: dict[str, Any] = {}
    for feature in layout.features:
        if feature.name not in columns:
            continue
        array = np.asarray(columns[feature.name])
        if feature.shape is not None:
            shape = tuple(int(dim) for dim in feature.shape)
        elif feature.slices and feature.slices[0].kind == "delta_chunk":
            piece = feature.slices[0]
            shape = (int(piece.chunk_size), len(piece.names))
        else:
            shape = (int(array.shape[1]),) if array.ndim == 2 else (1,)
        dtype = "int64" if np.issubdtype(array.dtype, np.integer) else (
            "float32" if array.dtype == np.float32 else "float64"
        )
        entry: dict[str, Any] = {
            "dtype": dtype,
            "shape": list(shape),
            "names": list(feature.names) if feature.names else None,
        }
        pad = pad_slots(feature)
        if pad:
            entry["pad"] = pad
        features[feature.name] = entry
    return features


def _create_kwargs(
    fps: int,
    features: dict,
    root: Path,
    robot_type: str,
    *,
    metadata_buffer_size: int = 10,
) -> dict[str, Any]:
    return {
        "repo_id": f"local/{root.name}",
        "fps": int(fps),
        "features": features,
        "root": root,
        "robot_type": robot_type,
        "use_videos": False,  # 与 phi_data 一致：数值官方写，视频另行挂载
        "metadata_buffer_size": metadata_buffer_size,
    }


def write_lerobot(
    dataset: RobotDataSet,
    root: str | Path,
    layout: Layout,
    *,
    overwrite: bool = True,
    copy_videos: bool = True,
    chunks_size: int = 1000,
    data_files_size_in_mb: int = 100,
    video_files_size_in_mb: int = 200,
    **_: Any,
) -> Path:
    """用官方 API 写出 LeRobot v3 目录。"""
    LeRobotDataset = _require_lerobot()
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(root)
        shutil.rmtree(root)
    # create() 自己 mkdir，且 exist_ok=False
    root.parent.mkdir(parents=True, exist_ok=True)

    uids = list(dataset.uids())
    if not uids:
        raise ValueError("write_lerobot: empty dataset")

    first = dataset[uids[0]]
    columns0 = encode(first, layout)
    features = _numeric_features(columns0, layout)
    fps = int(round(first.fps or dataset.meta.fps or 30.0))

    ds = LeRobotDataset.create(
        **_create_kwargs(
            fps,
            features,
            root,
            layout.robot_type or dataset.meta.robot,
            metadata_buffer_size=max(10, len(uids)),
        )
    )
    if hasattr(ds.meta, "update_chunk_settings"):
        ds.meta.update_chunk_settings(
            chunks_size=chunks_size,
            data_files_size_in_mb=data_files_size_in_mb,
            video_files_size_in_mb=video_files_size_in_mb,
        )

    episodes: list[Episode] = []
    for uid in uids:
        episode = dataset[uid]
        episodes.append(episode)
        columns = encode(episode, layout)
        task = episode.labels.task or ""
        for frame_index in range(episode.num_frames):
            frame: dict[str, Any] = {"task": task}
            for name, array in columns.items():
                if name == keys.TIMESTAMP or name not in features:
                    continue
                value = np.asarray(array[frame_index])
                shape = tuple(features[name]["shape"])
                frame[name] = value.reshape(shape) if len(shape) > 1 else value.reshape(-1)
            ds.add_frame(frame)
        ds.save_episode()

    ds.finalize()
    # 官方 aggregate_stats 把每集分位数按集长加权平均，那不是全集分位数。
    _rewrite_pooled_quantiles(root)

    meta = replace(
        dataset.meta,
        uri=str(root.resolve()),
        schema_id=layout.id,
        robot=dataset.meta.robot or layout.robot_type,
    )
    dump_json(root / _PHI_META, meta)
    dump_json(root / _PHI_SEMANTICS, meta.semantics)

    if copy_videos:
        attach_videos(root, episodes, fps=float(fps))
    write_subtasks_parquet(root, episodes)
    attach_episode_meta(root, episodes)

    return root


def _subtasks_vocab(episodes: list[Episode]) -> list[str]:
    names: list[str] = []
    for episode in episodes:
        for segment in episode.labels.subtasks:
            text = str(segment.text or "")
            if text not in names:
                names.append(text)
    return names


def _subtasks_json(episode: Episode, text_to_id: dict[str, int]) -> str:
    """episode 级 subtasks：Labels.Segment → JSON（[start,end) + subtask_index）。"""
    n_frames = episode.num_frames
    segments: list[dict[str, int]] = []
    for segment in episode.labels.subtasks:
        start = max(0, min(n_frames, int(segment.start)))
        end = max(start, min(n_frames, int(segment.end)))
        if end <= start:
            continue
        text = str(segment.text or "")
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "subtask_index": int(text_to_id[text]),
            }
        )
    return json.dumps({"segments": segments}, sort_keys=True)


def write_subtasks_parquet(root: Path, episodes: list[Episode]) -> dict[str, int]:
    """数据集级词表：``meta/subtasks.parquet``，index=文本，列=subtask_index。"""
    import pandas as pd

    root = Path(root)
    vocab = _subtasks_vocab(episodes)
    frame = pd.DataFrame(
        {"subtask_index": np.arange(len(vocab), dtype=np.int64)},
        index=pd.Index(list(vocab), name="subtask"),
    )
    dest = root / "meta" / "subtasks.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest)
    return {text: index for index, text in enumerate(vocab)}


def load_subtask_vocab(root: Path) -> dict[int, str]:
    """``subtask_index → 文本``；文件不存在时返回空。"""
    import pandas as pd

    path = Path(root) / "meta" / "subtasks.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    if "subtask_index" not in frame.columns:
        return {}
    return {
        int(index): str(text)
        for text, index in zip(frame.index.astype(str), frame["subtask_index"].tolist())
    }


def parse_subtasks_json(raw: Any, vocab: dict[int, str]) -> tuple[Segment, ...]:
    """episodes.subtasks JSON → ``Labels.subtasks``。"""
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        try:
            blob = json.loads(raw)
        except json.JSONDecodeError:
            return ()
    elif isinstance(raw, dict):
        blob = raw
    else:
        return ()
    segments: list[Segment] = []
    for item in blob.get("segments") or []:
        try:
            start = int(item["start_frame"])
            end = int(item["end_frame"])
            index = int(item["subtask_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        segments.append(Segment(start=start, end=end, text=vocab.get(index, "")))
    return tuple(segments)


_POOLED_QUANTILES = (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
)


def _feature_matrix(column) -> np.ndarray:
    """Parquet 数值列 → (frames, dim)。"""
    import pyarrow as pa

    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if pa.types.is_fixed_size_list(array.type):
        width = int(array.type.list_size)
        flat = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        return flat.reshape(-1, width)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        rows = array.to_pylist()
        return np.asarray(rows, dtype=np.float64)
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    return values.reshape(-1, 1)


def _rewrite_pooled_quantiles(root: Path) -> None:
    """用全部帧重算 meta/stats.json 的 q01/q10/q50/q90/q99。min/max/mean/std 保持官方值。"""
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        return
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        return
    import pyarrow.parquet as pq

    stats = json.loads(stats_path.read_text())
    names = [name for name, feat in stats.items() if isinstance(feat, dict) and "q01" in feat]
    if not names:
        return
    schema_names = set(pq.ParquetFile(files[0]).schema_arrow.names)
    names = [name for name in names if name in schema_names]
    table = pq.read_table(files, columns=names)
    replaced = 0
    for name in names:
        matrix = _feature_matrix(table.column(name))
        if matrix.size == 0:
            continue
        quantiles = np.quantile(matrix, [q for _, q in _POOLED_QUANTILES], axis=0)
        feat = stats[name]
        for index, (key, _) in enumerate(_POOLED_QUANTILES):
            feat[key] = [float(v) for v in np.asarray(quantiles[index]).reshape(-1)]
        replaced += 1
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")
    print(f"  pooled quantiles rewritten features={replaced} frames={table.num_rows}", flush=True)


def _rgb_source(camera: CameraStream) -> Path | None:
    if camera.rgb is None:
        return None
    storage = camera.rgb.storage
    src = storage.video.path if storage.video else storage.uri
    if not src:
        return None
    path = Path(src)
    return path if path.is_file() else None


def _install_mp4(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _mp4_has_moov(path: Path) -> bool:
    """Incomplete ffmpeg writes (esp. on OSS-FUSE) leave ftyp+mdat and no moov."""
    try:
        with path.open("rb") as handle:
            # moov is usually near the end; also check head for faststart layouts
            head = handle.read(1 << 20)
            if b"moov" in head:
                return True
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - (1 << 20)))
            return b"moov" in handle.read()
    except OSError:
        return False


def _local_tmp_parent() -> Path:
    """Prefer real local disk for ffmpeg mux — FUSE often drops the moov trailer."""
    tmp_parent = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    tmp_parent.mkdir(parents=True, exist_ok=True)
    return tmp_parent


def _ffmpeg_concat(sources: list[Path], dest: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    """Concat to a local tempfile, then install. Never mux straight onto FUSE/OSS."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    tmp_parent = _local_tmp_parent()
    fd, concat_list = tempfile.mkstemp(suffix=".ffconcat", text=True, dir=str(tmp_parent))
    tmp_out: Path | None = None
    try:
        with os.fdopen(fd, "w") as handle:
            handle.writelines(f"file '{Path(path).resolve()}'\n" for path in sources)
        fd_out, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_parent))
        os.close(fd_out)
        tmp_out = Path(tmp_name)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list, *extra, str(tmp_out),
            ],
            capture_output=True,
            text=True,
        )
        if (
            result.returncode == 0
            and tmp_out.is_file()
            and tmp_out.stat().st_size > 0
            and _mp4_has_moov(tmp_out)
        ):
            _install_mp4(tmp_out, dest)
        elif result.returncode == 0 and tmp_out.is_file() and tmp_out.stat().st_size > 0:
            # Treat missing moov as failure so caller can fall back / raise
            result = subprocess.CompletedProcess(
                result.args,
                returncode=1,
                stdout=result.stdout,
                stderr=(result.stderr or "") + "\nmissing moov atom after ffmpeg concat\n",
            )
        return result
    finally:
        Path(concat_list).unlink(missing_ok=True)
        if tmp_out is not None:
            tmp_out.unlink(missing_ok=True)


def _concat_mp4(sources: list[Path], dest: Path) -> None:
    """多集 mp4 → 单文件（优先 stream copy，失败再重编码）。"""
    if len(sources) == 1:
        src = Path(sources[0])
        if not _mp4_has_moov(src):
            raise RuntimeError(f"source mp4 missing moov (unplayable): {src}")
        _install_mp4(src, dest)
        return
    result = _ffmpeg_concat(sources, dest, ["-c", "copy"])
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0 or not _mp4_has_moov(dest):
        result = _ffmpeg_concat(
            sources, dest, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high"],
        )
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0 or not _mp4_has_moov(dest):
        raise RuntimeError(f"concat failed {dest}: {(result.stderr or '')[-800:]}")


def _release_packed_sources(sources: list[Path], dest: Path) -> None:
    """成片已经落盘后删掉原始 mp4。硬链接只去掉一个名字，字节留在成片上。"""
    dest_path = dest.resolve() if dest.exists() else None
    for src in sources:
        try:
            if not src.exists():
                continue
            if dest_path is not None and src.resolve() == dest_path:
                continue
            src.unlink()
        except OSError:
            continue


def pack_mp4s(
    root: Path,
    video_key: str,
    items: list[tuple[int, Path, int]],
    *,
    fps: float,
    limit_mb: int = 200,
    chunks_size: int = 1000,
    release_sources: bool = False,
) -> dict[int, dict[str, float | int]]:
    """按 ``video_files_size_in_mb`` 把多集打进同一 mp4（官方 v3 packing）。

    ``items``: ``(episode_index, src_mp4, n_frames)``，按 episode 顺序。
    返回每集的 ``chunk_index/file_index/from_timestamp/to_timestamp``。
    """
    limit = max(1, int(limit_mb)) * 1024 * 1024
    chunks_size = max(1, int(chunks_size))
    ranges: dict[int, dict[str, float | int]] = {}
    batch: list[tuple[int, Path, int]] = []
    batch_bytes = 0
    chunk_index = 0
    file_index = 0

    def flush() -> None:
        nonlocal batch, batch_bytes, chunk_index, file_index
        if not batch:
            return
        dest = root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        _concat_mp4([path for _, path, _ in batch], dest)
        if release_sources:
            _release_packed_sources([path for _, path, _ in batch], dest)
        timestamp = 0.0
        for episode_index, _, n_frames in batch:
            duration = float(n_frames) / float(fps or 30.0)
            ranges[int(episode_index)] = {
                "chunk_index": chunk_index,
                "file_index": file_index,
                "from_timestamp": timestamp,
                "to_timestamp": timestamp + duration,
            }
            timestamp += duration
        file_index += 1
        if file_index >= chunks_size:
            chunk_index, file_index = chunk_index + 1, 0
        batch, batch_bytes = [], 0

    for episode_index, source, n_frames in items:
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(src)
        size = src.stat().st_size
        if batch and batch_bytes + size > limit:
            flush()
        batch.append((int(episode_index), src, int(n_frames)))
        batch_bytes += size
    flush()
    return ranges


def attach_videos(
    root: Path,
    episodes: list[Episode],
    *,
    fps: float,
    release_sources: bool = False,
    packed_rows: list[dict[str, Any]] | None = None,
) -> None:
    """官方 v3 视频挂载：按大小把多集 concat 进同一 mp4，写 from/to_timestamp。

    布局：
      videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
      meta/episodes … videos/{video_key}/{chunk_index,file_index,from_timestamp,to_timestamp}
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tmpl = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["video_path"] = tmpl
    limit_mb = int(info.get("video_files_size_in_mb") or 200)
    chunks_size = int(info.get("chunks_size") or 1000)

    # 清掉可能残留的旧一集一文件布局。视频已写到别处时不要动本目录。
    if packed_rows is None:
        videos_root = root / "videos"
        if videos_root.is_dir():
            shutil.rmtree(videos_root)

    per_ep: list[dict[str, Any]] = [{} for _ in episodes]
    video_keys: list[str] = []

    if packed_rows is not None:
        if len(packed_rows) != len(episodes):
            raise ValueError(f"packed video rows {len(packed_rows)} != episodes {len(episodes)}")
        per_ep = [dict(row) for row in packed_rows]
        seen: list[str] = []
        for row in per_ep:
            for key in row:
                marker = "/chunk_index"
                if key.startswith("videos/") and key.endswith(marker):
                    video_key = key[len("videos/") : -len(marker)]
                    if video_key not in seen:
                        seen.append(video_key)
        video_keys = seen
        for video_key in video_keys:
            info.setdefault("features", {})[video_key] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": float(fps)},
            }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_video_columns(root, per_ep)
        return

    cam_names = sorted(
        {cam.name for ep in episodes for cam in ep.cameras if _rgb_source(cam) is not None}
    )
    for camera_name in cam_names:
        video_key = keys.image_key(camera_name)
        video_keys.append(video_key)
        items: list[tuple[int, Path, int]] = []
        for ep_index, episode in enumerate(episodes):
            camera = next((c for c in episode.cameras if c.name == camera_name), None)
            if camera is None:
                continue
            src = _rgb_source(camera)
            if src is None:
                continue
            items.append((ep_index, src, int(episode.num_frames)))
        ranges = pack_mp4s(
            root,
            video_key,
            items,
            fps=float(fps),
            limit_mb=limit_mb,
            chunks_size=chunks_size,
            release_sources=release_sources,
        )
        for ep_index, meta in ranges.items():
            per_ep[ep_index].update(
                {
                    f"videos/{video_key}/chunk_index": int(meta["chunk_index"]),
                    f"videos/{video_key}/file_index": int(meta["file_index"]),
                    f"videos/{video_key}/from_timestamp": float(meta["from_timestamp"]),
                    f"videos/{video_key}/to_timestamp": float(meta["to_timestamp"]),
                }
            )
        sample = next((root / "videos" / video_key).rglob("*.mp4"), None)
        shape = [480, 640, 3]
        if sample is not None:
            try:
                import av

                with av.open(str(sample)) as container:
                    stream = container.streams.video[0]
                    shape = [int(stream.height), int(stream.width), 3]
            except Exception:  # noqa: BLE001
                pass
        info.setdefault("features", {})[video_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(fps)},
        }

    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_video_columns(root, per_ep)


def _write_video_columns(root: Path, per_ep: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not any(per_ep):
        return
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir():
        return
    for path in sorted(ep_dir.rglob("*.parquet")):
        table = pq.read_table(path)
        indices = [int(np.asarray(v).reshape(-1)[0]) for v in table.column("episode_index").to_pylist()]
        names = set(table.column_names)
        keys_union = sorted({k for row in per_ep for k in row})
        for key in keys_union:
            values = [per_ep[i].get(key, 0 if "timestamp" not in key else 0.0) for i in indices]
            if "timestamp" in key:
                col = pa.array([float(v) for v in values], type=pa.float64())
            else:
                col = pa.array([int(v) for v in values], type=pa.int64())
            if key in names:
                table = table.set_column(table.schema.get_field_index(key), key, col)
            else:
                table = table.append_column(key, col)
                names.add(key)
        pq.write_table(table, path)


def attach_episode_meta(root: Path, episodes: list[Episode]) -> None:
    """把 episode 级元信息写进 ``meta/episodes/*.parquet``。

    - ``cameras``：JSON string，值为 ``{cam: {intrinsic, extrinsic}}``（飞书相机标准）
    - ``subtasks``：JSON string，值为 ``{segments:[{start_frame,end_frame,subtask_index}]}``
    - ``source.*``：溯源列；身份只用 ``source.uid``
    """
    import pyarrow.parquet as pq

    root = Path(root)
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir() or not episodes:
        return

    text_to_id = write_subtasks_parquet(root, episodes)
    per_ep: list[dict[str, Any]] = []
    for ep in episodes:
        extras = dict(ep.meta.span.extras or {})
        cams = extras.get("cameras")
        if cams is None:
            cams = extras.get("calib")
        if cams is None and ep.cameras:
            cams = {
                cam.name: cam.intrinsic
                for cam in ep.cameras
                if cam.intrinsic
            } or None
        cams = _cameras_json_payload(cams) if cams is not None else None
        src_fields = dict(extras.get("source_fields") or {})
        row: dict[str, Any] = {
            _EP_CAMERAS_COL: json.dumps(cams, ensure_ascii=False) if cams is not None else "",
            _EP_SUBTASKS_COL: _subtasks_json(ep, text_to_id),
        }
        for key in _EP_SOURCE_META_COLS:
            val = src_fields.get(key)
            if val is None and key == "source.path" and ep.source:
                val = str(ep.source)
            if val is None and key == _EP_SOURCE_UID_COL and ep.uid:
                uid = str(ep.uid)
                val = uid[len("episode_"):] if uid.startswith("episode_") else uid
            row[key] = _normalize_source_field(key, val)
        for key in ("task", "operator_id"):
            if key in extras and key not in row:
                row[key] = extras[key]
        per_ep.append(row)

    write_keys = (_EP_CAMERAS_COL, _EP_SUBTASKS_COL, *_EP_SOURCE_META_COLS)
    for path in sorted(ep_dir.rglob("*.parquet")):
        table = pq.read_table(path)
        indices = [int(np.asarray(v).reshape(-1)[0]) for v in table.column("episode_index").to_pylist()]
        names = set(table.column_names)
        drop = [c for c in _EP_LEGACY_DROP if c in names]
        if drop:
            table = table.drop(drop)
            names -= set(drop)
        for key in write_keys:
            default: Any = None if key in (_EP_SOURCE_BOOL_COLS | _EP_SOURCE_SHAPE_COLS) else ""
            values = [per_ep[i].get(key, default) if 0 <= i < len(per_ep) else default for i in indices]
            col = (
                _source_field_array(key, values)
                if key in _EP_SOURCE_META_COLS
                else _pa_string_array(values)
            )
            if key in names:
                table = table.set_column(table.schema.get_field_index(key), key, col)
            else:
                table = table.append_column(key, col)
                names.add(key)
        pq.write_table(table, path)


def load_episode_uids(root: Path) -> list[str]:
    """从 episodes parquet 读 ``source.uid``；兼容旧 ``episode_uid`` / ``episode_uids.json``。"""
    import pyarrow.parquet as pq

    root = Path(root)
    ep_dir = root / "meta" / "episodes"
    by_index: dict[int, str] = {}
    if ep_dir.is_dir():
        for path in sorted(ep_dir.rglob("*.parquet")):
            table = pq.read_table(path)
            names = table.column_names
            uid_col = (
                _EP_SOURCE_UID_COL if _EP_SOURCE_UID_COL in names
                else ("episode_uid" if "episode_uid" in names else None)
            )
            if uid_col is None:
                continue
            idxs = table.column("episode_index").to_pylist()
            uids = table.column(uid_col).to_pylist()
            for idx, uid in zip(idxs, uids):
                if uid:
                    by_index[int(np.asarray(idx).reshape(-1)[0])] = str(uid)
    if by_index:
        return [by_index[i] for i in sorted(by_index)]

    uid_path = root / _UID_FILE
    if uid_path.is_file():
        return list(load_json(uid_path))

    info_path = root / "meta" / "info.json"
    total = 0
    if info_path.is_file():
        total = int(json.loads(info_path.read_text(encoding="utf-8")).get("total_episodes") or 0)
    return [f"ep-{i:06d}" for i in range(total)]


def _attach_videos_official(
    root: Path,
    dataset: RobotDataSet,
    uids: list[str],
    fps: int,
) -> None:
    """兼容旧入口：从 dataset 取 Episode 再挂视频。"""
    episodes = [dataset[uid] for uid in uids]
    attach_videos(root, episodes, fps=float(fps))
    attach_episode_meta(root, episodes)


class LeRobotEpisodeStore:
    """基于官方 ``LeRobotDataset`` 的懒 EpisodeStore。"""

    def __init__(
        self,
        root: Path,
        *,
        layout: Layout | None = None,
        uids: list[str] | None = None,
        dataset_meta: DatasetMeta | None = None,
    ) -> None:
        self.root = Path(root)
        self.layout = layout
        self._ds = open_lerobot_dataset(self.root)
        self._uids = uids or self._default_uids()
        self._dataset_meta = dataset_meta
        self._subtask_vocab = load_subtask_vocab(self.root)
        self._metas = self._build_metas()

    def _default_uids(self) -> list[str]:
        return load_episode_uids(self.root)

    def _build_metas(self) -> dict[str, EpisodeMeta]:
        fps = float(self._ds.meta.fps or 30.0)
        out: dict[str, EpisodeMeta] = {}
        for i, uid in enumerate(self._uids):
            row = self._ds.meta.episodes[i]
            length = int(row.get("length") or 0)
            if not length:
                length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
            task = ""
            tasks = row.get("tasks")
            if tasks:
                task = tasks[0] if isinstance(tasks, list) else str(tasks)
            videos: dict[str, VideoSpan] = {}
            for video_key in self._ds.meta.video_keys:
                try:
                    rel = self._ds.meta.get_video_file_path(i, video_key)
                    path = self.root / rel
                except Exception:  # noqa: BLE001
                    path = Path("")
                cam = keys.camera_from_key(video_key) if keys.is_image_key(video_key) else video_key
                videos[cam] = VideoSpan(
                    path=str(path),
                    from_timestamp=float(row.get(f"videos/{video_key}/from_timestamp", 0.0) or 0.0),
                    to_timestamp=row.get(f"videos/{video_key}/to_timestamp"),
                )
            extras: dict[str, Any] = {
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
            }
            src_fields: dict[str, Any] = {}
            for key in _EP_SOURCE_META_COLS:
                if row.get(key) not in (None, ""):
                    src_fields[key] = row[key]
            if src_fields:
                extras["source_fields"] = src_fields
            # Episode.source prefers relative source.path, else legacy absolute source
            src_path = src_fields.get("source.path") or row.get(_EP_SOURCE_COL) or ""
            if src_path:
                extras["source"] = str(src_path)
            cams_raw = row.get(_EP_CAMERAS_COL) or row.get("camera_calib") or ""
            if isinstance(cams_raw, str) and cams_raw.strip():
                try:
                    parsed = json.loads(cams_raw)
                    cams = _cameras_json_payload(parsed) or parsed
                    extras["cameras"] = cams
                except json.JSONDecodeError:
                    extras["cameras_raw"] = cams_raw
            subtasks_raw = row.get(_EP_SUBTASKS_COL)
            if subtasks_raw not in (None, ""):
                extras[_EP_SUBTASKS_COL] = subtasks_raw
            out[uid] = EpisodeMeta(
                uid=uid,
                num_frames=length,
                fps=fps,
                task=task,
                episode_index=i,
                span=EpisodeSpan(videos=videos, extras=extras),
                stats=EpisodeStats(num_frames=length),
            )
        return out

    def list_metas(self) -> Iterable[EpisodeMeta]:
        for uid in self._uids:
            yield self._metas[uid]

    def exists(self, uid: str) -> bool:
        return uid in self._metas

    def open(self, uid: str) -> Episode:
        meta = self._metas[uid]
        ep_index = int(meta.episode_index or 0)
        columns = self._load_columns(ep_index)
        cameras = self._load_cameras(ep_index, meta)

        state = RobotBody()
        action = None
        if self.layout is not None and columns:
            decoded = decode(columns, self.layout)
            state = _body_from_decoded(decoded, "state")
            action_body = _body_from_decoded(decoded, "action")
            if any(
                [
                    action_body.arm_left,
                    action_body.arm_right,
                    action_body.arm_single,
                    action_body.waist,
                    action_body.head_joints,
                ]
            ):
                action = action_body

        source = str((meta.span.extras or {}).get("source") or self.root)
        subtasks = parse_subtasks_json(
            (meta.span.extras or {}).get(_EP_SUBTASKS_COL),
            self._subtask_vocab,
        )
        return Episode(
            meta=meta,
            state=state,
            action=action,
            cameras=tuple(cameras),
            labels=Labels(task=meta.task, subtasks=subtasks),
            semantics=(
                self._dataset_meta.semantics
                if self._dataset_meta is not None
                else EpisodeSemantics()
            ),
            source=source,
        )

    def _load_columns(self, episode_index: int) -> dict[str, np.ndarray]:
        """官方路径：episodes 行 → hf_dataset.data.slice。"""
        row = self._ds.meta.episodes[int(episode_index)]
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        # 触发懒加载（若需要）
        if getattr(self._ds, "hf_dataset", None) is None and hasattr(self._ds, "_ensure_hf_dataset_loaded"):
            self._ds._ensure_hf_dataset_loaded()
        table = self._ds.hf_dataset.data.slice(start, stop - start)
        skip = set(self._ds.meta.video_keys)
        out: dict[str, np.ndarray] = {}
        for name in table.column_names:
            if name in skip:
                continue
            values = table.column(name).to_pylist()
            try:
                out[name] = np.asarray(values)
            except ValueError:
                out[name] = np.asarray(values, dtype=object)
        return out

    def _load_cameras(self, episode_index: int, meta: EpisodeMeta) -> list[CameraStream]:
        """挂 LeRobot v3 视频句柄：同一 mp4 可含多集，用 from/to_timestamp 切段。

        路径 = ``meta.get_video_file_path(ep, key)``（由 chunk/file_index 决定）。
        官方取帧：``decode(path, from_timestamp + episode_relative_ts)``。
        此处不解码像素。
        """
        cameras: list[CameraStream] = []
        fps = meta.fps
        for video_key in self._ds.meta.video_keys:
            cam_name = keys.camera_from_key(video_key) if keys.is_image_key(video_key) else video_key
            try:
                rel = self._ds.meta.get_video_file_path(int(episode_index), video_key)
                path = self.root / rel
            except Exception:  # noqa: BLE001
                continue
            row = self._ds.meta.episodes[int(episode_index)]
            from_ts = float(row.get(f"videos/{video_key}/from_timestamp", 0.0) or 0.0)
            to_ts = row.get(f"videos/{video_key}/to_timestamp")
            handle = LazyHandle(
                storage=StorageRef(
                    kind=StorageKind.VIDEO_MP4,
                    uri=str(path),
                    video=VideoLocator(
                        path=str(path),
                        from_timestamp=from_ts,
                        to_timestamp=float(to_ts) if to_ts is not None else None,
                        fps=fps,
                    ),
                ),
                dtype="uint8",
            )
            cameras.append(CameraStream(name=cam_name, rgb=handle))
        return cameras


def _body_from_decoded(decoded, stream: str) -> RobotBody:
    def pose(part: str) -> Pose | None:
        mat = decoded.poses.get((stream, part))
        return Pose.from_matrix(mat) if mat is not None else None

    def joints(part: str) -> JointGroup | None:
        arr = decoded.joints.get((stream, part))
        if arr is None:
            return None
        names = decoded.joint_names.get((stream, part), tuple(f"j{i}" for i in range(arr.shape[1])))
        return JointGroup(position=LazyHandle.from_memory(arr), names=names)

    def arm(part: str) -> Arm | None:
        ee = pose(part)
        j = joints(part)
        g = decoded.grippers.get((stream, part))
        gripper = Gripper(open=LazyHandle.from_memory(g)) if g is not None else None
        if ee is None and j is None and gripper is None:
            return None
        return Arm(ee=ee, gripper=gripper, joints=j)

    return RobotBody(
        base=pose(keys.BASE),
        torso=pose(keys.TORSO),
        head=pose(keys.HEAD),
        arm_left=arm(keys.ARM_LEFT),
        arm_right=arm(keys.ARM_RIGHT),
        arm_single=arm(keys.ARM_SINGLE),
        waist=joints(keys.WAIST),
        head_joints=joints(keys.HEAD),
    )


def read_lerobot(
    root: str | Path,
    *,
    layout: Layout | None = None,
    name: str | None = None,
    version: str = "v1",
    robot: str | None = None,
    **_: Any,
) -> RobotDataSet:
    """用官方 ``LeRobotDataset`` 打开目录并包装为 ``RobotDataSet``。

    第一步从 ``info.json`` 还原 ``DatasetSchema``（整集布局）。
    """
    from phi_rds.schema.spec import DatasetSchema

    root = Path(root)
    if not is_lerobot_root(root):
        raise ValueError(f"not a LeRobot dataset root: {root}")

    official = open_lerobot_dataset(root)
    fps = float(official.meta.fps or 30.0)
    if layout is None:
        layout = _infer_preset_layout(official.meta.info)
    schema = DatasetSchema.from_lerobot_info(official.meta.info, layout=layout)
    uids = load_episode_uids(root)

    stats = DatasetStats(
        num_episodes=int(official.meta.total_episodes or 0),
        num_frames=int(official.meta.total_frames or 0),
        feature_list=schema.feature_names,
        features=_stats_from_official(official),
        scope="self",
    )

    if (root / _PHI_META).is_file():
        meta = from_dict(DatasetMeta, load_json(root / _PHI_META))
        meta = replace(meta, uri=str(root.resolve()), stats=stats, fps=fps)
        if (root / _PHI_SEMANTICS).is_file():
            meta = replace(meta, semantics=from_dict(EpisodeSemantics, load_json(root / _PHI_SEMANTICS)))
    else:
        if schema.level == "L2":
            sem_level = "frame_unified"
        elif schema.level == "L1":
            sem_level = "name_unified"
        else:
            sem_level = "name_unified"
        meta = DatasetMeta(
            name=name or root.name,
            version=version,
            robot=robot or str(official.meta.info.get("robot_type") or "unknown"),
            uri=str(root.resolve()),
            fps=fps,
            schema_id=schema.id,
            level=schema.level,
            standard_version=schema.standard_version,
            semantics=EpisodeSemantics(
                robot_type=str(official.meta.info.get("robot_type") or ""),
                layout_id=layout.id if layout else schema.id,
                semantic_level=sem_level,  # type: ignore[arg-type]
            ),
            stats=stats,
        )

    store = LeRobotEpisodeStore(
        root,
        layout=layout or schema.layout,
        uids=uids,
        dataset_meta=meta,
    )
    return RobotDataSet._from_store(meta, store, schema=schema)


def _infer_preset_layout(info: dict[str, Any]) -> Layout | None:
    """打开时若未传 layout，按 phi_schema_id / robot_type / 列维名套机型预设。"""
    try:
        from phi_rds.schema.presets import preset_layout
    except Exception:  # noqa: BLE001
        return None

    schema_id = str(info.get("phi_schema_id") or "").strip()
    robot = str(info.get("robot_type") or "").strip()
    arms: str | None = None

    # 优先 schema id：agibot_bimanual / umi_single
    tag = schema_id or robot
    if tag.endswith("_bimanual"):
        robot, arms = tag[: -len("_bimanual")], "bimanual"
    elif tag.endswith("_single"):
        robot, arms = tag[: -len("_single")], "single"
    elif not robot:
        robot = tag

    if not robot:
        return None

    features = info.get("features") or {}
    if arms is None:
        # feature key 本身通常不含 left_；看 names / pad 维名
        name_tokens: list[str] = []
        for blob in features.values():
            if not isinstance(blob, dict):
                continue
            names = blob.get("names")
            if isinstance(names, (list, tuple)):
                name_tokens.extend(str(x) for x in names)
        arms = (
            "bimanual"
            if any("left_" in n or "right_" in n for n in name_tokens)
            else "single"
        )

    joint_count = 7
    joint_feat = features.get(keys.STATE_ARM_JOINT) or features.get(keys.ACTION_ARM_JOINT)
    if isinstance(joint_feat, dict):
        shape = joint_feat.get("shape") or ()
        width = int(shape[0]) if shape else 0
        if width > 0:
            n_arms = 2 if arms == "bimanual" else 1
            if width % n_arms == 0:
                joint_count = width // n_arms

    try:
        return preset_layout(robot, arms=arms, joint_count=joint_count)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _stats_from_official(official) -> dict[str, FeatureStats]:
    raw = getattr(official.meta, "stats", None) or {}
    if not raw and (Path(official.root) / "meta" / "stats.json").is_file():
        raw = load_json(Path(official.root) / "meta" / "stats.json")
    out: dict[str, FeatureStats] = {}
    for name, blob in (raw or {}).items():
        if not isinstance(blob, dict):
            continue
        out[name] = FeatureStats(
            name=name,
            count=_scalar_int(blob.get("count")),
            mean=_as_tuple(blob.get("mean")),
            std=_as_tuple(blob.get("std")),
            min=_as_tuple(blob.get("min")),
            max=_as_tuple(blob.get("max")),
        )
    return out


def _scalar_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0]) if value.size else 0
    if isinstance(value, (list, tuple)) and value:
        return _scalar_int(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_tuple(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, np.ndarray):
        return tuple(float(x) for x in value.reshape(-1).tolist())
    try:
        return tuple(float(x) for x in value)
    except TypeError:
        return ()
