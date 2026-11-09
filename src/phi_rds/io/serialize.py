"""dataclass ↔ JSON（仅元信息 / 语义 / 统计 / layout）。"""

from __future__ import annotations

import json
import types
import typing
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

T = TypeVar("T")


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj) if not f.name.startswith("_")}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"cannot JSON-encode {type(obj)!r}")


def dump_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cast(value: Any, anno: Any, *, globalns: dict[str, Any] | None = None) -> Any:
    if value is None:
        return None
    if isinstance(anno, str):
        if globalns and anno in globalns:
            anno = globalns[anno]
        else:
            return value

    origin = get_origin(anno)
    args = get_args(anno)

    if origin is tuple or origin is tuple:
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_cast(v, args[0], globalns=globalns) for v in value)
        return tuple(_cast(v, a, globalns=globalns) for v, a in zip(value, args, strict=False))
    if origin is list:
        return [_cast(v, args[0] if args else Any, globalns=globalns) for v in value]
    if origin is dict:
        kt, vt = args if len(args) == 2 else (str, Any)
        return {
            _cast(k, kt, globalns=globalns): _cast(v, vt, globalns=globalns)
            for k, v in value.items()
        }

    # Optional / Union
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _cast(value, non_none[0], globalns=globalns)
        # 多类型：优先 dataclass
        for candidate in non_none:
            if isinstance(candidate, type) and is_dataclass(candidate) and isinstance(value, dict):
                return from_dict(candidate, value)
        return value

    if isinstance(anno, type) and is_dataclass(anno):
        return from_dict(anno, value)
    if isinstance(anno, type) and issubclass(anno, Enum):
        return anno(value)
    return value


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"expected dict for {cls!r}, got {type(data)!r}")
    hints = typing.get_type_hints(cls, include_extras=True)
    globalns = {**getattr(cls, "__dict__", {}), **hints}
    # 把类本身也放进命名空间，便于嵌套
    globalns[cls.__name__] = cls
    data = dict(data)
    # 兼容旧字段
    if cls.__name__ == "DatasetMeta":
        data.pop("layer", None)
        data.pop("lineage", None)
        data.pop("subset", None)
        data.pop("urdf_uri", None)
        data.pop("calib_uri", None)
        if "schema_id" not in data and "layout_id" in data:
            data["schema_id"] = data.pop("layout_id")
        else:
            data.pop("layout_id", None)
    if cls.__name__ == "EpisodeMeta":
        data.pop("source_ref", None)
        data.pop("delete_reason", None)
        data.pop("delete_strategy", None)
        data.pop("semantics_override", None)
        span = data.get("span")
        if isinstance(span, dict) and "parquet" in span:
            extras = dict(span.get("extras") or {})
            extras.update(span.pop("parquet") or {})
            span["extras"] = extras
    if cls.__name__ == "EpisodeStats":
        data.pop("duration_s", None)
        data.pop("num_cameras", None)
        data.pop("has_depth", None)
        data.pop("num_subtasks", None)
    if cls.__name__ == "DatasetStats":
        data.pop("duration_s", None)
    if cls.__name__ == "FeatureStats":
        # 旧字段 p1..p99 → quantiles
        qs = dict(data.get("quantiles") or {})
        for key in ("p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"):
            if key in data:
                val = data.pop(key)
                if val:
                    qs[key] = tuple(val) if isinstance(val, (list, tuple)) else val
        if qs:
            data["quantiles"] = qs
        data.pop("extras", None)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name.startswith("_"):
            continue
        if f.name not in data:
            continue
        anno = hints.get(f.name, f.type)
        kwargs[f.name] = _cast(data[f.name], anno, globalns=globalns)
    return cls(**kwargs)


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return to_jsonable(obj)  # type: ignore[return-value]
