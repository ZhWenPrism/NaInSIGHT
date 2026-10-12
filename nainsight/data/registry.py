"""Curated admission registry -> canonical admission frame.

The registry is the per-admission table maintained by the clinical team (one row per surgical admission).
Its adapter section declares the core timestamps and a list of static covariates:

    registry:
      path / format / columns: {stay_key, subject_key, admit_time, surgery_start, surgery_end,
                                discharge_time, procedure_text, age_years, sex_male}
      static: {canonical_name: {source, type: numeric|binary|category, map, lo, hi, fill}}

Only covariates known at the end of surgery may be declared here; the adapter is responsible for
never exposing post-operative fields (they would leak future information into every landmark).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .readers import read_source, to_numeric
from .timeparse import hours_between, to_datetime

CORE = ("stay_key", "subject_key", "admit_time", "surgery_start", "surgery_end", "discharge_time")
TIME_FIELDS = ("admit_time", "surgery_start", "surgery_end", "discharge_time")


def _apply_static(raw: pd.Series, spec: dict) -> pd.Series:
    kind = spec.get("type", "numeric")
    if kind == "numeric":
        out = to_numeric(raw)
        lo, hi = spec.get("lo"), spec.get("hi")
        if lo is not None:
            out = out.where(out >= lo)
        if hi is not None:
            out = out.where(out <= hi)
    elif kind == "binary":
        mapping = {str(k): float(v) for k, v in (spec.get("map") or {}).items()}
        out = raw.astype("string").str.strip().map(mapping).astype(float)
    elif kind == "category":
        out = raw.astype("string").str.strip().replace({"nan": pd.NA, "": pd.NA})
    else:
        raise ValueError(f"unknown static type {kind!r}")
    if "fill" in spec:
        out = out.fillna(spec["fill"])
    return out


def load_registry(site: dict, cache_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (admissions frame with canonical columns, static-column schema {name: type})."""
    spec = site["registry"]
    cache = Path(cache_dir) / f"registry_{site.get('_name', 'site')}.parquet" if cache_dir else None
    static_spec = dict(spec.get("static") or {})
    if cache is not None and cache.exists():
        frame = pd.read_parquet(cache)
    else:
        colmap = dict(spec["columns"])
        for name, s in static_spec.items():
            colmap[f"__static__{name}"] = s["source"]
        frame = read_source({**spec, "columns": colmap}, root=site.get("root"))
        if cache is not None:
            frame.to_parquet(cache)
    out = pd.DataFrame({"stay_key": frame["stay_key"].astype("string"),
                        "subject_key": frame["subject_key"].astype("string")})
    for field in TIME_FIELDS:
        out[field] = to_datetime(frame[field])
    out["procedure_text"] = frame["procedure_text"].astype("string") if "procedure_text" in frame else pd.NA
    out["age_years"] = to_numeric(frame["age_years"])
    sex = frame["sex_male"].astype("string").str.strip()
    sex_map = {str(k): float(v) for k, v in (spec.get("sex_map") or {"1": 1, "1.0": 1, "0": 0, "0.0": 0}).items()}
    out["sex_male"] = sex.map(sex_map).astype(float)
    out["surgery_duration_h"] = hours_between(out["surgery_end"], out["surgery_start"])
    out["preop_stay_days"] = hours_between(out["surgery_start"], out["admit_time"]) / 24.0
    schema = {"age_years": "numeric", "sex_male": "binary", "surgery_duration_h": "numeric",
              "preop_stay_days": "numeric"}
    for name, s in static_spec.items():
        out[name] = _apply_static(frame[f"__static__{name}"], s)
        schema[name] = s.get("type", "numeric")
    out = out.drop_duplicates("stay_key").reset_index(drop=True)
    return out, schema
