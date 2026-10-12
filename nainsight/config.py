"""YAML configuration with left-to-right merging, dotted command-line overrides and attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class AttrDict(dict):
    """Dictionary whose keys are also attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(o: Any) -> Any:
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars/lists."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] | None = None) -> AttrDict:
    merged: dict = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(path))
    return AttrDict.wrap(apply_overrides(merged, overrides or []))


def resolve(path: str | Path | None, base: Path = REPO_ROOT) -> Path | None:
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()
