"""Generic chunked readers.  A *source spec* declares where a table lives and how its columns map onto
canonical field names, so the same extraction code serves every site:

    path | path_glob : file or glob (relative to the adapter's ``root``)
    exclude_pattern  : regular expression on file names to skip
    format           : csv | csv_headerless | xlsx | parquet
    columns          : {canonical_field: source column name (or 0-based index for headerless csv)}
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd


def source_files(spec: dict, root: str | Path | None = None) -> list[str]:
    base = Path(root) if root else Path(".")
    if spec.get("path"):
        files = [str(base / spec["path"])]
    elif spec.get("path_glob"):
        files = sorted(glob.glob(str(base / spec["path_glob"])))
    else:
        raise ValueError("source spec needs 'path' or 'path_glob'")
    if spec.get("exclude_pattern"):
        pat = re.compile(spec["exclude_pattern"])
        files = [f for f in files if not pat.search(os.path.basename(f))]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing[:3]}")
    return files


def iter_source(spec: dict, fields: Sequence[str] | None = None, root: str | Path | None = None,
                chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """Yield chunks whose columns are the requested canonical ``fields`` (all mapped fields by default).
    Every value is read as string; typed parsing happens downstream so that malformed cells never abort
    a run."""
    colmap: dict = dict(spec["columns"])
    fields = list(fields) if fields is not None else list(colmap)
    absent = [f for f in fields if f not in colmap]
    if absent:
        raise KeyError(f"fields {absent} not mapped in source spec")
    src_cols = [colmap[f] for f in fields]
    unique_src = list(dict.fromkeys(src_cols))
    fmt = spec.get("format", "csv")
    encoding = spec.get("encoding", "utf-8")

    def emit(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: frame[c].values for f, c in zip(fields, src_cols)})

    for path in source_files(spec, root):
        if fmt in ("csv", "csv_headerless"):
            reader = pd.read_csv(path, header=None if fmt == "csv_headerless" else "infer",
                                 usecols=unique_src, dtype=str, encoding=encoding, chunksize=chunksize,
                                 low_memory=False, on_bad_lines="skip")
            for chunk in reader:
                yield emit(chunk)
        elif fmt == "xlsx":
            frame = pd.read_excel(path, usecols=unique_src, dtype=str)
            for start in range(0, len(frame), chunksize):
                yield emit(frame.iloc[start:start + chunksize])
        elif fmt == "parquet":
            frame = pd.read_parquet(path, columns=unique_src)
            for start in range(0, len(frame), chunksize):
                yield emit(frame.iloc[start:start + chunksize].astype(str))
        else:
            raise ValueError(f"unsupported format {fmt!r}")


def read_source(spec: dict, fields: Sequence[str] | None = None, root: str | Path | None = None) -> pd.DataFrame:
    chunks = list(iter_source(spec, fields, root))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=list(fields or spec["columns"]))


def to_numeric(s: pd.Series) -> pd.Series:
    """Numeric coercion tolerant to full-width digits, thousands separators and trailing units."""
    t = s.astype("string").str.normalize("NFKC").str.strip().str.replace(",", "", regex=False)
    t = t.str.extract(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(t, errors="coerce")


def compile_patterns(spec: dict) -> dict:
    """{key: pattern | [patterns]} -> {key: compiled alternation}."""
    out = {}
    for key, pats in (spec or {}).items():
        if pats is None:
            continue
        pats = [pats] if isinstance(pats, str) else list(pats)
        out[key] = re.compile("|".join(f"(?:{p})" for p in pats))
    return out
