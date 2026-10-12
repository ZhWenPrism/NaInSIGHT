"""Patient-cluster bootstrap: resample patients (all their landmarks together) with replacement."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap(df: pd.DataFrame, cluster: str, fn, n_boot: int = 1000, ci: float = 0.95,
                      seed: int = 0) -> dict:
    """fn(frame) -> {key: {metric: value}}; returns {key: {metric: (point, lower, upper)}}."""
    point = fn(df)
    groups = list(df.groupby(cluster).indices.values())
    rng = np.random.default_rng(seed)
    draws: dict = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        res = fn(df.iloc[idx])
        for key, metrics in res.items():
            for name, value in metrics.items():
                draws.setdefault((key, name), []).append(value)
    alpha = (1 - ci) / 2
    out = {}
    for key, metrics in point.items():
        out[key] = {}
        for name, value in metrics.items():
            d = np.asarray(draws.get((key, name), []), dtype=float)
            d = d[np.isfinite(d)]
            lo, hi = (np.quantile(d, alpha), np.quantile(d, 1 - alpha)) if len(d) else (np.nan, np.nan)
            out[key][name] = (value, lo, hi)
    return out


def to_frame(res: dict, setting: str) -> pd.DataFrame:
    rows = []
    for (mode, h, task), metrics in res.items():
        for name, (v, lo, hi) in metrics.items():
            rows.append({"setting": setting, "mode": mode, "horizon_h": h, "task": task, "metric": name,
                         "estimate": v, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)
