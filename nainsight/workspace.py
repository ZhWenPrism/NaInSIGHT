"""Run-time locations.  All data-bearing artefacts live under ``workdir`` (outside the repository);
site adapters are referenced by ``configs/local/paths.yaml`` and are never committed."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import AttrDict, REPO_ROOT, load_config, read_yaml, resolve
from .utils import ensure_dir


class Workspace:
    def __init__(self, paths_file: str | Path):
        self.paths_file = resolve(paths_file)
        raw = read_yaml(self.paths_file)
        self.raw = raw
        self.root = ensure_dir(resolve(raw["workdir"]))

    # ------------------------------------------------------------------ directories
    def dir(self, *parts: str) -> Path:
        return ensure_dir(self.root.joinpath(*parts))

    @property
    def interim(self) -> Path:
        return self.dir("interim")

    @property
    def events(self) -> Path:
        return self.dir("interim", "events")

    @property
    def reference(self) -> Path:
        return self.dir("interim", "reference")

    @property
    def tensors(self) -> Path:
        return self.dir("interim", "tensors")

    @property
    def artifacts(self) -> Path:
        return self.dir("interim", "artifacts")

    @property
    def runs(self) -> Path:
        return self.dir("runs")

    # ------------------------------------------------------------------ site adapters
    def source(self, name: str) -> AttrDict | None:
        entry = (self.raw.get("sources") or {}).get(name)
        if not entry:
            return None
        cfg = AttrDict.wrap(read_yaml(resolve(entry)))
        cfg["_name"] = name
        return cfg

    def site_names(self) -> list[str]:
        return [k for k, v in (self.raw.get("sources") or {}).items() if v and k != "reference"]


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "nainsight.yaml"))
    parser.add_argument("--paths", default=str(REPO_ROOT / "configs" / "local" / "paths.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted configuration overrides, e.g. train.epochs=5")
    return parser


def load_run_context(args) -> tuple[AttrDict, Workspace]:
    cfg = load_config([args.config], args.set)
    return cfg, Workspace(args.paths)
