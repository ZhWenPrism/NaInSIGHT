"""Event store: long-format clinical events for every stage-1 admission on one clock,

    t = hours since the end of the index operation   (t < 0 before, t > 0 after surgery)

so that landmarks, prediction windows and pre/intra/post-operative phases share a single time axis.
Laboratory results are time-stamped at *report* time (when the value becomes available to clinicians),
which is also the clock used to define the outcomes; using request/specimen time for predictors would
let a specimen drawn before the landmark but reported inside the outcome window leak into the inputs.
"""
from __future__ import annotations

import logging
import re
import warnings

import numpy as np
import pandas as pd

from ..concepts import CONCEPTS, INGREDIENTS, ROUTES
from .dosing import normalise_doses
from .readers import compile_patterns, iter_source, to_numeric
from .timeparse import combine_date_time, to_datetime

LOG = logging.getLogger("nainsight.events")
# adapter patterns may contain capture groups; only matching is needed
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


# ------------------------------------------------------------------------------------- stay windows
def stay_windows(stays: pd.DataFrame, lookback_days: float, max_postop_h: float) -> pd.DataFrame:
    w = stays[["stay_id", "subject_key", "admit_time", "surgery_end", "discharge_time"]].copy()
    w["win_start"] = w["admit_time"] - pd.Timedelta(days=float(lookback_days))
    w["win_end"] = np.minimum(w["discharge_time"], w["surgery_end"] + pd.Timedelta(hours=float(max_postop_h)))
    return w[["stay_id", "subject_key", "win_start", "win_end", "surgery_end"]]


def attach(events: pd.DataFrame, windows: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Link subject-level events to admissions by time window and express time relative to surgery end."""
    if events.empty:
        return events.assign(stay_id=pd.Series(dtype=np.int64), t=pd.Series(dtype=np.float32))
    m = events.merge(windows, on="subject_key", how="inner")
    m = m[(m[time_col] >= m["win_start"]) & (m[time_col] <= m["win_end"])]
    m = m.assign(t=((m[time_col] - m["surgery_end"]).dt.total_seconds() / 3600.0).astype(np.float32))
    return m.drop(columns=["subject_key", "win_start", "win_end", "surgery_end", time_col])


def _plausible(df: pd.DataFrame, ranges: dict | None = None) -> tuple[pd.DataFrame, dict]:
    lo = df["concept"].map(lambda c: (ranges or {}).get(c, {}).get("lo", CONCEPTS[c].lo))
    hi = df["concept"].map(lambda c: (ranges or {}).get(c, {}).get("hi", CONCEPTS[c].hi))
    ok = df["value"].notna() & ((lo.isna()) | (df["value"] >= lo)) & ((hi.isna()) | (df["value"] <= hi))
    dropped = df.loc[~ok, "concept"].value_counts().to_dict()
    return df[ok], dropped


def _match_rules(name: pd.Series, code: pd.Series | None, rules: dict) -> pd.Series:
    """rules: {concept: [{name: regex, code: regex, exclude: regex}, ...]} - first matching concept wins."""
    out = pd.Series(pd.NA, index=name.index, dtype="string")
    for concept, rule_list in rules.items():
        for rule in rule_list:
            m = name.str.contains(rule["name"], regex=True, na=False) if rule.get("name") else pd.Series(True, index=name.index)
            if rule.get("code") and code is not None:
                m &= code.str.contains(rule["code"], regex=True, na=False)
            if rule.get("exclude"):
                m &= ~name.str.contains(rule["exclude"], regex=True, na=False)
            out = out.mask(m & out.isna(), concept)
    return out


def _bin(df: pd.DataFrame, bin_h: float, how: str) -> pd.DataFrame:
    """Aggregate to fixed bins; the token time is the last contributing timestamp (never the bin end),
    so an aggregate is only visible once its last component has been recorded."""
    if df.empty:
        return df
    b = np.floor(df["t"].values / bin_h).astype(np.int64)
    g = df.assign(_b=b).groupby(["stay_id", "concept", "_b"], observed=True)
    out = g.agg(value=("value", how), t=("t", "max")).reset_index().drop(columns="_b")
    return out[["stay_id", "t", "concept", "value"]]


# ------------------------------------------------------------------------------------- laboratory
def extract_labs(site: dict, windows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    lab = site["labs"]
    rules = {k: v for k, v in lab["items"].items()}
    prefilter = re.compile("|".join(r["name"] for rs in rules.values() for r in rs if r.get("name")))
    unit_rules = lab.get("unit_rules") or {}
    subjects = set(windows["subject_key"])
    parts, qc = [], {"rows_matched": 0, "dropped_implausible": {}}
    for src in lab["sources"]:
        layout = src.get("layout", "long")
        if layout == "long":
            for chunk in iter_source(src, root=site.get("root")):
                chunk = chunk[chunk["subject_key"].isin(subjects)]
                name = chunk["item_name"].astype("string").str.normalize("NFKC").str.strip()
                keep = name.str.contains(prefilter, na=False)
                chunk, name = chunk[keep], name[keep]
                if chunk.empty:
                    continue
                code = chunk["item_code"].astype("string").str.strip() if "item_code" in chunk else None
                concept = _match_rules(name, code, rules)
                chunk = chunk.assign(concept=concept).dropna(subset=["concept"])
                if chunk.empty:
                    continue
                value = to_numeric(chunk["value"])
                if unit_rules and "unit" in chunk:
                    unit = chunk["unit"].astype("string").str.normalize("NFKC").str.lower().str.strip()
                    for c, table in unit_rules.items():
                        for u, factor in table.items():
                            sel = chunk["concept"].eq(c) & unit.eq(str(u).lower())
                            value = value.where(~sel, value * float(factor))
                if src.get("time_basis", "report") == "report":
                    time = combine_date_time(chunk["report_date"], chunk["report_time"])
                    if "request_date" in chunk:
                        time = time.fillna(combine_date_time(chunk["request_date"], chunk["request_time"]))
                else:
                    time = combine_date_time(chunk["request_date"], chunk["request_time"])
                parts.append(pd.DataFrame({"subject_key": chunk["subject_key"].values, "time": time.values,
                                           "concept": chunk["concept"].astype(str).values,
                                           "value": value.values, "source": src.get("name", "lab")}))
        elif layout == "wide":
            value_cols = src["value_columns"]                       # {concept: source column}
            spec = {**src, "columns": {**src["columns"], **{f"__v_{c}": col for c, col in value_cols.items()}}}
            for chunk in iter_source(spec, root=site.get("root")):
                chunk = chunk[chunk["subject_key"].isin(subjects)]
                if chunk.empty:
                    continue
                time = combine_date_time(chunk["report_date"], chunk["report_time"])
                for c in value_cols:
                    parts.append(pd.DataFrame({"subject_key": chunk["subject_key"].values, "time": time.values,
                                               "concept": c, "value": to_numeric(chunk[f"__v_{c}"]).values,
                                               "source": src.get("name", "lab_wide")}))
        else:
            raise ValueError(f"unknown lab layout {layout!r}")
    if not parts:
        raise RuntimeError("no laboratory rows matched the adapter rules")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["time", "value"])
    qc["rows_matched"] = int(len(df))
    df, qc["dropped_implausible"] = _plausible(df, lab.get("ranges"))
    df = attach(df, windows).drop_duplicates(["stay_id", "t", "concept", "value"])
    df = df.astype({"concept": "category", "value": np.float32, "source": "category"})
    return df.sort_values(["stay_id", "t"]).reset_index(drop=True), qc


# ------------------------------------------------------------------------------------- vitals / fluid charts
def _extract_charted(site: dict, block: dict, windows: pd.DataFrame, bin_h: float, how: str,
                     ranges: dict | None) -> tuple[pd.DataFrame, dict]:
    subjects = set(windows["subject_key"])
    parts = []
    for src in block["sources"]:
        rules = {k: [{"name": p} for p in ([v] if isinstance(v, str) else v)] for k, v in src["items"].items()}
        prefilter = re.compile("|".join(r["name"] for rs in rules.values() for r in rs))
        for chunk in iter_source(src, root=site.get("root")):
            chunk = chunk[chunk["subject_key"].isin(subjects)]
            item = chunk["item"].astype("string").str.normalize("NFKC").str.strip()
            keep = item.str.contains(prefilter, na=False)
            chunk, item = chunk[keep], item[keep]
            if chunk.empty:
                continue
            concept = _match_rules(item, None, rules)
            parts.append(pd.DataFrame({"subject_key": chunk["subject_key"].values, "time": to_datetime(chunk["time"]).values,
                                       "concept": concept.values, "value": to_numeric(chunk["value"]).values}))
    df = pd.concat(parts, ignore_index=True).dropna(subset=["concept", "time", "value"])
    df["concept"] = df["concept"].astype(str)
    df, dropped = _plausible(df, ranges)
    df = attach(df, windows)
    return df, {"rows": int(len(df)), "dropped_implausible": dropped}


def extract_vitals(site: dict, windows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df, qc = _extract_charted(site, site["vitals"], windows, cfg["events"]["vital_bin_h"], "mean",
                              site["vitals"].get("ranges"))
    # mean arterial pressure from simultaneous systolic/diastolic readings
    bp = df[df["concept"].isin(["VS_SBP", "VS_DBP"])].pivot_table(index=["stay_id", "t"], columns="concept",
                                                                   values="value", aggfunc="mean")
    if {"VS_SBP", "VS_DBP"} <= set(bp.columns):
        bp = bp.dropna()
        bp = bp[bp["VS_SBP"] > bp["VS_DBP"]]
        mp = ((bp["VS_SBP"] + 2.0 * bp["VS_DBP"]) / 3.0).rename("value").reset_index().assign(concept="VS_MAP")
        df = pd.concat([df, mp[["stay_id", "t", "concept", "value"]]], ignore_index=True)
    df = _bin(df, cfg["events"]["vital_bin_h"], "mean")
    return df.astype({"concept": "category", "value": np.float32, "t": np.float32}), qc


def extract_io(site: dict, windows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df, qc = _extract_charted(site, site["fluid_balance"], windows, cfg["events"]["io_bin_h"], "sum",
                              site["fluid_balance"].get("ranges"))
    df = df.drop_duplicates(["stay_id", "t", "concept", "value"])        # the same entry charted in two sources
    df = _bin(df, cfg["events"]["io_bin_h"], "sum")
    return df.astype({"concept": "category", "value": np.float32, "t": np.float32}), qc


# ------------------------------------------------------------------------------------- medications
class IngredientMatcher:
    """Ordered (first match wins) ingredient patterns plus a global exclusion pattern for non-systemic forms."""

    def __init__(self, med: dict):
        self.rules = [(r["ingredient"], re.compile(r["pattern"]), re.compile(r["exclude"]) if r.get("exclude") else None)
                      for r in med["ingredients"]]
        unknown = {i for i, _, _ in self.rules} - set(INGREDIENTS)
        if unknown:
            raise KeyError(f"ingredients not in the canonical registry: {sorted(unknown)}")
        self.any = re.compile("|".join(f"(?:{r['pattern']})" for r in med["ingredients"]))
        self.exclude = re.compile(med["exclude_text"]) if med.get("exclude_text") else None

    def __call__(self, text: pd.Series) -> pd.Series:
        t = text.astype("string").str.normalize("NFKC").fillna("")
        out = pd.Series(pd.NA, index=t.index, dtype="string")
        candidate = t.str.contains(self.any, na=False)
        if self.exclude is not None:
            candidate &= ~t.str.contains(self.exclude, na=False)
        for ing, pat, excl in self.rules:
            m = candidate & out.isna() & t.str.contains(pat, na=False)
            if excl is not None:
                m &= ~t.str.contains(excl, na=False)
            out = out.mask(m, ing)
        return out


def _route(raw: pd.Series, route_map: dict) -> pd.Series:
    r = raw.astype("string").str.normalize("NFKC").str.strip().map({str(k): v for k, v in route_map.items()})
    return r.where(r.isin(ROUTES), "OTHER").fillna("OTHER")


def _administration_log(site: dict, med: dict, windows: pd.DataFrame, matcher: IngredientMatcher) -> pd.DataFrame:
    spec = med["administration_log"]
    subjects = set(windows["subject_key"])
    cls = re.compile(spec["keep_order_class"]) if spec.get("keep_order_class") else None
    flags = {str(f) for f in spec.get("executed_flag_values", [])}
    parts = []
    for chunk in iter_source(spec, root=site.get("root")):
        chunk = chunk[chunk["subject_key"].isin(subjects)]
        if cls is not None:
            chunk = chunk[chunk["order_class"].astype("string").str.contains(cls, na=False)]
        if flags:
            chunk = chunk[chunk["flag"].astype("string").str.strip().isin(flags)]
        if chunk.empty:
            continue
        ing = matcher(chunk["text"])
        chunk = chunk.assign(ingredient=ing).dropna(subset=["ingredient"])
        if chunk.empty:
            continue
        time = to_datetime(chunk["executed_time"]) if "executed_time" in chunk else None
        if "scheduled_time" in chunk:
            sched = to_datetime(chunk["scheduled_time"])
            time = sched if time is None else time.fillna(sched)
        parts.append(pd.DataFrame({"subject_key": chunk["subject_key"].values, "time": time.values,
                                   "ingredient": chunk["ingredient"].astype(str).values,
                                   "value": chunk["value"].values, "unit": chunk["unit"].values,
                                   "text": chunk["text"].values, "route_raw": chunk["route"].values}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _expand_orders(site: dict, med: dict, windows: pd.DataFrame, matcher: IngredientMatcher) -> pd.DataFrame:
    """Order-level source: expand each order into scheduled administrations from its frequency."""
    spec = med["orders"]
    subjects = set(windows["subject_key"])
    freq = {str(k): v for k, v in spec["frequency_hours"].items()}
    skip_types = set(spec.get("exclude_order_types", []))
    horizon = windows.groupby("subject_key")["win_end"].max()
    parts = []
    for chunk in iter_source(spec, root=site.get("root")):
        chunk = chunk[chunk["subject_key"].isin(subjects)]
        if "order_type" in chunk and skip_types:
            chunk = chunk[~chunk["order_type"].astype("string").isin(skip_types)]
        text = chunk["spec"].fillna("").astype(str) + " " + chunk["name"].fillna("").astype(str)
        chunk = chunk.assign(text=text, ingredient=matcher(text)).dropna(subset=["ingredient"])
        if chunk.empty:
            continue
        start = to_datetime(chunk["start_time"])
        end = to_datetime(chunk["end_time"]).fillna(chunk["subject_key"].map(horizon))
        step = chunk["frequency"].astype("string").str.strip().map(freq)
        once = step.isna() | step.astype("string").eq("once")
        step_h = pd.to_numeric(step.where(~once), errors="coerce")
        n = np.where(once, 1, np.clip(np.floor(((end - start).dt.total_seconds() / 3600.0) / step_h) + 1, 1, 400))
        n = np.nan_to_num(n, nan=1).astype(int)
        idx = np.repeat(np.arange(len(chunk)), n)
        k = np.concatenate([np.arange(m) for m in n]) if len(n) else np.array([], dtype=int)
        times = start.values[idx] + (np.nan_to_num(step_h.values[idx], nan=0.0) * k * 3600).astype("timedelta64[s]")
        rep = chunk.iloc[idx]
        parts.append(pd.DataFrame({"subject_key": rep["subject_key"].values, "time": times,
                                   "ingredient": rep["ingredient"].astype(str).values, "value": rep["value"].values,
                                   "unit": rep["unit"].values, "text": rep["text"].values,
                                   "route_raw": rep["route"].values}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def extract_medications(site: dict, windows: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    med = site["medications"]
    matcher = IngredientMatcher(med)
    raw = (_administration_log(site, med, windows, matcher) if med.get("source", "administration_log") == "administration_log"
           else _expand_orders(site, med, windows, matcher))
    raw = raw.dropna(subset=["time"])
    raw = raw.drop_duplicates(["subject_key", "time", "ingredient", "value", "unit"])
    norm = normalise_doses(raw[["ingredient", "value", "unit", "text"]], med.get("unit_aliases"), med.get("defaults"))
    norm.index = raw.index
    df = pd.concat([raw[["subject_key", "time"]], norm.drop(columns=["value", "unit", "text"])], axis=1)
    df["route"] = _route(raw["route_raw"], med.get("routes") or {})
    df = attach(df, windows)
    qc = {"administrations": int(len(df)),
          "dose_convertible_by_concept": df.groupby("concept")["dose_std"].apply(lambda s: round(float(s.notna().mean()), 4)).to_dict(),
          "administrations_by_concept": df["concept"].value_counts().to_dict()}
    df = df.astype({"concept": "category", "ingredient": "category", "dose_unit": "category",
                    "tonicity": "category", "route": "category"})
    for c in ("dose_std", "volume_ml", "na_mmol", "pred_eq_mg"):
        df[c] = df[c].astype(np.float32)
    return df.sort_values(["stay_id", "t"]).reset_index(drop=True), qc
