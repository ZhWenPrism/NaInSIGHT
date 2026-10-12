"""Robust timestamp parsing for heterogeneous hospital exports (full-width punctuation, split date/time)."""
from __future__ import annotations

import pandas as pd


def _normalise(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.normalize("NFKC").str.strip()
            .str.replace("/", "-", regex=False).replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA, "None": pd.NA}))


def to_datetime(s: pd.Series) -> pd.Series:
    """Parse mixed-format timestamps; unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(_normalise(s), errors="coerce", format="mixed")


def combine_date_time(date: pd.Series, time: pd.Series, default_time: str = "12:00") -> pd.Series:
    """Combine separate date and clock-time columns; a missing clock time falls back to ``default_time``."""
    d = _normalise(date)
    t = time.astype("string").str.normalize("NFKC").str.strip()
    t = t.where(t.str.match(r"^\d{1,2}:\d{1,2}").fillna(False), default_time)
    return pd.to_datetime(d + " " + t, errors="coerce", format="mixed")


def hours_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds() / 3600.0
