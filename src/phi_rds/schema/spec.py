"""数据集布局 Schema：读/写的第一步——先定整集结构，再灌 Episode。

数据标识（不是平台流水线）：
  - **L1**：原始格式 → LeRobot v3；命名统一；布局与信息无损保留
  - **L2**：统一标准布局；必须带 ``standard_version``（默认最新）

高效要点：Schema 在 create/load 钉死一次；写出只追加 Episode。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from phi_rds.schema.layout import Layout
from phi_rds.types import DataLevel


Profile = Literal["standard", "free"]


# 日历版本：YYYY.MM；发版时追加到 REGISTRY，并把 LATEST 指向最新。
LATEST = "2026.04"
LATEST_STANDARD = LATEST

REGISTRY: dict[str, dict] = {
    "2026.04": {
        "doc": "https://zcnce50wan15.feishu.cn/wiki/Jitbwmov2iJMQpkpy6Zc0TPMnJf",
        "notes": "统一命名 + pose9 默认列约定",
    },
}


@dataclass(frozen=True)
class StandardSpec:
    """一份已发布的数据标准。"""

    version: str
    doc: str = ""
    notes: str = ""

    @property
    def is_latest(self) -> bool:
        return self.version == LATEST


def resolve_standard(version: str | None = None) -> StandardSpec:
    """``None`` / ``\"latest\"`` → 当前最新标准。"""
    ver = LATEST if version in (None, "", "latest") else str(version)
    info = REGISTRY.get(ver)
    if info is None:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise ValueError(f"unknown standard version {ver!r}; known: {known}")
    return StandardSpec(version=ver, doc=str(info.get("doc") or ""), notes=str(info.get("notes") or ""))


def list_standards() -> tuple[StandardSpec, ...]:
    return tuple(resolve_standard(v) for v in sorted(REGISTRY))




@dataclass(frozen=True)
class FeatureSpec:
    """单个 LeRobot feature 的布局声明。"""

    name: str
    dtype: str = "float32"
    shape: tuple[int, ...] = ()
    names: tuple[str, ...] | None = None
    pad: tuple[int, ...] = ()
    is_video: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_lerobot(self) -> dict[str, Any]:
        if self.is_video:
            entry: dict[str, Any] = {
                "dtype": "video",
                "shape": list(self.shape) if self.shape else [480, 640, 3],
                "names": list(self.names) if self.names else ["height", "width", "channels"],
            }
            entry.update(self.extras)
            return entry
        entry = {
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape else [1],
            "names": list(self.names) if self.names else None,
        }
        if self.pad:
            entry["pad"] = list(self.pad)
        entry.update(self.extras)
        return entry

    @staticmethod
    def from_lerobot(name: str, blob: dict[str, Any]) -> FeatureSpec:
        dtype = str(blob.get("dtype") or "float32")
        shape = tuple(int(x) for x in (blob.get("shape") or []))
        names = blob.get("names")
        names_t = tuple(names) if isinstance(names, (list, tuple)) else None
        pad = blob.get("pad") or ()
        pad_t = tuple(int(x) for x in pad) if pad else ()
        extras = {
            k: v for k, v in blob.items()
            if k not in {"dtype", "shape", "names", "pad"}
        }
        return FeatureSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            names=names_t,
            pad=pad_t,
            is_video=(dtype == "video"),
            extras=extras,
        )


@dataclass(frozen=True)
class DatasetSchema:
    """整集布局。创建 / 打开数据集的第一步。

    - ``level="L1"``：命名统一的 LeRobot v3；布局可按源数据保留
    - ``level="L2"``：完整统一标准；``standard_version`` 必填（默认最新）
    - ``layout``：可选，把 ``Episode.state/action`` 编到 features
    """

    id: str
    fps: float
    robot_type: str = ""
    level: DataLevel | None = None
    standard_version: str | None = None
    profile: Profile = "free"
    features: tuple[FeatureSpec, ...] = ()
    layout: Layout | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate feature names in schema {self.id!r}")
        if self.level == "L2" and not self.standard_version:
            object.__setattr__(self, "standard_version", LATEST)
        if self.level == "L2" and self.standard_version:
            resolve_standard(self.standard_version)  # validate

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def is_l1(self) -> bool:
        return self.level == "L1"

    @property
    def is_l2(self) -> bool:
        return self.level == "L2"

    def feature_map(self) -> dict[str, FeatureSpec]:
        return {f.name: f for f in self.features}

    def get(self, name: str) -> FeatureSpec:
        for feat in self.features:
            if feat.name == name:
                return feat
        raise KeyError(name)

    def to_lerobot_features(self) -> dict[str, dict[str, Any]]:
        """供官方 ``LeRobotDataset.create(features=...)`` 一次用完。"""
        return {f.name: f.to_lerobot() for f in self.features if not f.name.startswith("_")}

    def numeric_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(f for f in self.features if not f.is_video)

    def video_features(self) -> tuple[FeatureSpec, ...]:
        return tuple(f for f in self.features if f.is_video)

    def to_stamp(self) -> dict[str, Any]:
        """写入 ``info.json`` 的轻量标记，便于任何人快速识别。"""
        stamp: dict[str, Any] = {"phi_schema_id": self.id}
        if self.level:
            stamp["phi_level"] = self.level
        if self.standard_version:
            stamp["phi_standard_version"] = self.standard_version
        return stamp

    # ----- 构造：大家只用这几个 -----
    @staticmethod
    def l1(
        layout: Layout,
        *,
        fps: float = 30.0,
        videos: Sequence[FeatureSpec] = (),
        id: str | None = None,
    ) -> DatasetSchema:
        """L1：LeRobot v3 + 命名统一；布局按 ``layout``（通常贴近源数据）。"""
        return DatasetSchema.from_layout(
            layout,
            fps=fps,
            level="L1",
            profile="free",
            videos=videos,
            id=id,
        )

    @staticmethod
    def l2(
        layout: Layout,
        *,
        fps: float = 30.0,
        standard: str | None = None,
        videos: Sequence[FeatureSpec] = (),
        id: str | None = None,
    ) -> DatasetSchema:
        """L2：完整统一标准布局；``standard`` 默认最新。"""
        spec = resolve_standard(standard)
        return DatasetSchema.from_layout(
            layout,
            fps=fps,
            level="L2",
            standard_version=spec.version,
            profile="standard",
            videos=videos,
            id=id,
        )

    @staticmethod
    def free(
        features: Sequence[FeatureSpec],
        *,
        fps: float = 30.0,
        robot_type: str = "",
        id: str = "free",
        layout: Layout | None = None,
        level: DataLevel | None = None,
        standard_version: str | None = None,
    ) -> DatasetSchema:
        return DatasetSchema(
            id=id,
            fps=fps,
            robot_type=robot_type,
            level=level,
            standard_version=standard_version,
            profile="free",
            features=tuple(features),
            layout=layout,
        )

    @staticmethod
    def from_layout(
        layout: Layout,
        *,
        fps: float = 30.0,
        level: DataLevel | None = None,
        standard_version: str | None = None,
        profile: Profile | None = None,
        videos: Sequence[FeatureSpec] = (),
        id: str | None = None,
    ) -> DatasetSchema:
        """由 ``Layout``（Episode↔列绑定）生成 Schema；可选附加视频列。"""
        from phi_rds.schema.layout import pad_slots, slice_width

        specs: list[FeatureSpec] = []
        for feature in layout.features:
            if feature.shape is not None:
                shape = tuple(int(x) for x in feature.shape)
            elif feature.slices and feature.slices[0].kind == "delta_chunk":
                piece = feature.slices[0]
                shape = (int(piece.chunk_size), len(piece.names))
            else:
                shape = (sum(slice_width(s) for s in feature.slices) or 1,)
            pad = tuple(pad_slots(feature))
            specs.append(
                FeatureSpec(
                    name=feature.name,
                    dtype=feature.dtype,
                    shape=shape,
                    names=tuple(feature.names) if feature.names else None,
                    pad=pad,
                )
            )
        specs.extend(videos)
        if profile is None:
            profile = "standard" if level == "L2" else "free"
        if level == "L2" and standard_version is None:
            standard_version = LATEST
        return DatasetSchema(
            id=id or layout.id,
            fps=fps,
            robot_type=layout.robot_type,
            level=level,
            standard_version=standard_version,
            profile=profile,
            features=tuple(specs),
            layout=layout,
        )

    @staticmethod
    def from_lerobot_info(info: dict[str, Any], *, layout: Layout | None = None) -> DatasetSchema:
        """从已有 v3 ``info.json`` 还原布局（读路径；含 L1/L2 标记）。"""
        raw = info.get("features") or {}
        skip = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
        specs = [
            FeatureSpec.from_lerobot(name, blob)
            for name, blob in raw.items()
            if name not in skip and isinstance(blob, dict)
        ]
        level = info.get("phi_level")
        if level not in ("L1", "L2", None):
            level = None
        std_ver = info.get("phi_standard_version")
        if std_ver is not None:
            std_ver = str(std_ver)
        profile: Profile
        if level == "L2":
            profile = "standard"
        elif layout is not None:
            profile = "standard"
        else:
            profile = "free"
        return DatasetSchema(
            id=str(info.get("phi_schema_id") or info.get("robot_type") or "lerobot"),
            fps=float(info.get("fps") or 30.0),
            robot_type=str(info.get("robot_type") or ""),
            level=level,  # type: ignore[arg-type]
            standard_version=std_ver,
            profile=profile,
            features=tuple(specs),
            layout=layout,
            extras={"codebase_version": info.get("codebase_version")},
        )

    def with_features(self, features: Iterable[FeatureSpec]) -> DatasetSchema:
        return DatasetSchema(
            id=self.id,
            fps=self.fps,
            robot_type=self.robot_type,
            level=self.level,
            standard_version=self.standard_version,
            profile=self.profile,
            features=tuple(features),
            layout=self.layout,
            extras=dict(self.extras),
        )
