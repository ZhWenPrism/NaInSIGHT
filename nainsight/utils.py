"""Small shared utilities: seeding, devices, logging, timing and serialisation."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # data-only stages do not need torch
        pass


def resolve_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Stage logger; library modules (``nainsight.*``) log through the same handlers."""
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
    for logger in (logging.getLogger(name), logging.getLogger("nainsight")):
        if logger.handlers:
            continue
        logger.setLevel(logging.INFO)
        for h in handlers:
            logger.addHandler(h)
        logger.propagate = False
    return logging.getLogger(name)


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.perf_counter()
    yield
    logger.info("%s finished in %.1fs", label, time.perf_counter() - t0)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def environment_report() -> dict:
    """Library versions recorded next to every artefact for reproducibility."""
    import platform

    report = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "pyarrow", "yaml"):
        try:
            report[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional dependency
            report[mod] = None
    return report
