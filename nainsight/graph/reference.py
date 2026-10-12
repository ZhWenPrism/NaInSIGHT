"""Reference critical-care cohort -> canonical event tables for effect estimation.

The reference adapter declares (all field names canonical, values mapped by the adapter):
  tables.admissions   {subject_key, hadm_key, admit_time, discharge_time}
  tables.patients     {subject_key, age_years | anchor_age + anchor_year}
  tables.icu_stays    {subject_key, hadm_key, icu_key, in_time, out_time}
  tables.diagnoses    {hadm_key, code, version}          tables.procedures {hadm_key, code, version}
  tables.labs         {subject_key, hadm_key, item, time, value}
  tables.med_admin    {subject_key, hadm_key, admin_key, time, medication, event}
  tables.med_detail   {admin_key, dose, unit}             (optional: per-administration dose)
  tables.infusions    {icu_key, start, end, item, amount, unit}
  tables.outputs      {icu_key, time, item, value}
  lab_items           {concept: [item ids]}
  med_ingredients     [{ingredient, pattern, exclude}]   (generic-name patterns, case-insensitive)
  administered_events [event labels counted as administered]
  infusion_items      {item id: {ingredient, kind: ml|g|mg|units|meq|ml_at_pct, pct}}
  urine_items         [item ids]
  subpopulation       {diagnosis: {version: regex}, procedure: {version: regex}}   (stage-2 subgraph)
Times are expressed in hours since ICU admission.  Values stay in reference units until transfer.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..concepts import INGREDIENTS
from ..data.dosing import normalise_doses
from ..data.readers import iter_source, read_source, to_numeric
from ..data.timeparse import to_datetime

LOG = logging.getLogger("nainsight.reference")


def _ids(s: pd.Series) -> pd.Series:
    return to_numeric(s).astype("Int64")


def reference_stays(ref: dict, cfg: dict) -> pd.DataFrame:
    t = ref["tables"]
    root = ref.get("root")
    icu = read_source(t["icu_stays"], root=root)
    icu = pd.DataFrame({"subject_id": _ids(icu["subject_key"]), "hadm_id": _ids(icu["hadm_key"]),
                        "icu_id": _ids(icu["icu_key"]), "in_time": to_datetime(icu["in_time"]),
                        "out_time": to_datetime(icu["out_time"])}).dropna()
    pat = read_source(t["patients"], root=root)
    pat_ids = _ids(pat["subject_key"])
    if "age_years" in pat:
        age = pd.Series(to_numeric(pat["age_years"]).values, index=pat_ids.values)
        icu["age"] = icu["subject_id"].map(age)
    else:
        anchor_age = pd.Series(to_numeric(pat["anchor_age"]).values, index=pat_ids.values)
        anchor_year = pd.Series(to_numeric(pat["anchor_year"]).values, index=pat_ids.values)
        icu["age"] = icu["subject_id"].map(anchor_age) + icu["in_time"].dt.year - icu["subject_id"].map(anchor_year)
    icu["los_h"] = (icu["out_time"] - icu["in_time"]).dt.total_seconds() / 3600.0
    g = cfg["graph"]
    icu = icu[(icu["age"] >= g["min_age_years"]) & (icu["los_h"] >= g["min_icu_stay_h"])]
    # stage-2 sub-population (e.g. intracranial tumour / craniotomy admissions)
    sub = ref.get("subpopulation") or {}
    flagged = set()
    for table, key in (("diagnoses", "diagnosis"), ("procedures", "procedure")):
        if key not in sub or table not in t:
            continue
        d = read_source(t[table], root=root)
        code = d["code"].astype("string").str.replace(".", "", regex=False).str.strip()
        ver = d["version"].astype("string").str.strip()
        hit = pd.Series(False, index=d.index)
        for v, pat_ in sub[key].items():
            hit |= ver.eq(str(v)) & code.str.contains(pat_, regex=True, na=False)
        flagged |= set(_ids(d.loc[hit, "hadm_key"]).dropna().astype(int))
    icu["subpop"] = icu["hadm_id"].astype(int).isin(flagged)
    return icu.reset_index(drop=True)


def _to_stay_time(events: pd.DataFrame, stays: pd.DataFrame, key: str, time_col: str, pad_h: float) -> pd.DataFrame:
    m = events.merge(stays[[key, "icu_id", "in_time", "out_time"]] if key != "icu_id"
                     else stays[["icu_id", "in_time", "out_time"]], on=key, how="inner")
    m = m[(m[time_col] >= m["in_time"] - pd.Timedelta(hours=pad_h)) & (m[time_col] <= m["out_time"] + pd.Timedelta(hours=pad_h))]
    m["t"] = ((m[time_col] - m["in_time"]).dt.total_seconds() / 3600.0).astype(np.float32)
    return m.drop(columns=[c for c in ("in_time", "out_time", time_col) if c in m])


def reference_labs(ref: dict, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    item_map = {int(i): c for c, items in ref["lab_items"].items() for i in items}
    pad = float(cfg["graph"]["baseline_lookback_h"]) + max(cfg["graph"]["lags_h"])
    hadms = set(stays["hadm_id"].astype(int))
    parts = []
    for ch in iter_source(ref["tables"]["labs"], root=ref.get("root"), chunksize=3_000_000):
        item = to_numeric(ch["item"])
        ch = ch[item.isin(item_map)]
        hadm = _ids(ch["hadm_key"])
        ch = ch[hadm.isin(hadms)]
        if ch.empty:
            continue
        parts.append(pd.DataFrame({"hadm_id": _ids(ch["hadm_key"]).astype(int).values,
                                   "time": to_datetime(ch["time"]).values,
                                   "concept": to_numeric(ch["item"]).map(item_map).values,
                                   "value": to_numeric(ch["value"]).values}))
    labs = pd.concat(parts, ignore_index=True).dropna()
    labs = _to_stay_time(labs, stays.assign(hadm_id=stays["hadm_id"].astype(int)), "hadm_id", "time", pad)
    return labs[["icu_id", "t", "concept", "value"]].sort_values(["icu_id", "t"]).reset_index(drop=True)


def reference_drugs(ref: dict, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rules = [(r["ingredient"], re.compile(r["pattern"], re.I), re.compile(r["exclude"], re.I) if r.get("exclude") else None)
             for r in ref["med_ingredients"]]
    unknown = {i for i, _, _ in rules} - set(INGREDIENTS)
    if unknown:
        raise KeyError(f"unknown ingredients {sorted(unknown)}")
    any_pat = re.compile("|".join(f"(?:{r['pattern']})" for r in ref["med_ingredients"]), re.I)
    keep_events = set(ref.get("administered_events", []))
    hadms = set(stays["hadm_id"].astype(int))
    parts = []
    for ch in iter_source(ref["tables"]["med_admin"], root=ref.get("root"), chunksize=2_000_000):
        if keep_events:
            ch = ch[ch["event"].isin(keep_events)]
        ch = ch[_ids(ch["hadm_key"]).isin(hadms)]
        med = ch["medication"].astype("string").fillna("")
        ch, med = ch[med.str.contains(any_pat, na=False)], med[med.str.contains(any_pat, na=False)]
        if ch.empty:
            continue
        ing = pd.Series(pd.NA, index=ch.index, dtype="string")
        for name, pat, excl in rules:
            m = ing.isna() & med.str.contains(pat, na=False)
            if excl is not None:
                m &= ~med.str.contains(excl, na=False)
            ing = ing.mask(m, name)
        ch = ch.assign(ingredient=ing).dropna(subset=["ingredient"])
        parts.append(pd.DataFrame({"hadm_id": _ids(ch["hadm_key"]).astype(int).values,
                                   "admin_key": ch["admin_key"].astype(str).values,
                                   "time": to_datetime(ch["time"]).values,
                                   "ingredient": ch["ingredient"].astype(str).values,
                                   "text": ch["medication"].astype(str).values}))
    drugs = pd.concat(parts, ignore_index=True).dropna(subset=["time"])
    if "med_detail" in ref["tables"]:
        keys = set(drugs["admin_key"])
        det = []
        for ch in iter_source(ref["tables"]["med_detail"], root=ref.get("root"), chunksize=3_000_000):
            ch = ch[ch["admin_key"].astype(str).isin(keys)]
            if len(ch):
                det.append(pd.DataFrame({"admin_key": ch["admin_key"].astype(str).values,
                                         "value": to_numeric(ch["dose"]).values, "unit": ch["unit"].astype(str).values}))
        det = pd.concat(det, ignore_index=True).dropna(subset=["value"]) if det else pd.DataFrame(columns=["admin_key", "value", "unit"])
        # one documented dose per administration (detail tables may repeat it across product rows)
        det = (det[det["value"] > 0].sort_values("value", ascending=False)
               .drop_duplicates("admin_key")[["admin_key", "value", "unit"]])
        drugs = drugs.merge(det, on="admin_key", how="left")
    else:
        drugs["value"], drugs["unit"] = np.nan, ""
    norm = normalise_doses(drugs[["ingredient", "value", "unit", "text"]], ref.get("unit_aliases"), ref.get("dose_defaults"))
    drugs = pd.concat([drugs[["hadm_id", "time"]].reset_index(drop=True), norm[["ingredient", "concept", "dose_std"]]], axis=1)
    drugs = _to_stay_time(drugs, stays.assign(hadm_id=stays["hadm_id"].astype(int)), "hadm_id", "time", 0.0)
    return drugs[["icu_id", "t", "ingredient", "concept", "dose_std"]].sort_values(["icu_id", "t"]).reset_index(drop=True)


def reference_infusions(ref: dict, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    items = {int(k): v for k, v in ref["infusion_items"].items()}
    icu_ids = set(stays["icu_id"].astype(int))
    parts = []
    for ch in iter_source(ref["tables"]["infusions"], root=ref.get("root"), chunksize=2_000_000):
        item = to_numeric(ch["item"])
        ch = ch[item.isin(items) & _ids(ch["icu_key"]).isin(icu_ids)]
        if ch.empty:
            continue
        spec = to_numeric(ch["item"]).astype(int).map(items)
        amount = to_numeric(ch["amount"])
        kind = spec.map(lambda s: s["kind"])
        pct = spec.map(lambda s: float(s.get("pct", np.nan)))
        ingredient = spec.map(lambda s: s["ingredient"])
        dose = np.where(kind.eq("ml_at_pct"), amount * pct / 100.0, amount)       # mL of x % w/v -> g
        volume = np.where(kind.isin(["ml", "ml_at_pct"]), amount, np.nan)
        na_pct = spec.map(lambda s: float(s.get("na_pct", np.nan)))
        na_mmol = np.where(na_pct.notna(), amount * na_pct * 10.0 / 58.44, np.nan)
        parts.append(pd.DataFrame({"icu_id": _ids(ch["icu_key"]).astype(int).values,
                                   "start": to_datetime(ch["start"]).values, "end": to_datetime(ch["end"]).values,
                                   "ingredient": ingredient.values, "dose_std": dose, "volume_ml": volume,
                                   "na_mmol": na_mmol}))
    inf = pd.concat(parts, ignore_index=True).dropna(subset=["start", "end", "dose_std"])
    inf = inf[(inf["dose_std"] > 0) & (inf["end"] >= inf["start"])]
    inf["concept"] = inf["ingredient"].map(lambda i: INGREDIENTS[i][0])
    st = stays[["icu_id", "in_time"]].assign(icu_id=stays["icu_id"].astype(int))
    inf = inf.merge(st, on="icu_id")
    inf["t_start"] = ((inf["start"] - inf["in_time"]).dt.total_seconds() / 3600.0).astype(np.float32)
    inf["t_end"] = ((inf["end"] - inf["in_time"]).dt.total_seconds() / 3600.0).astype(np.float32)
    return inf[["icu_id", "t_start", "t_end", "ingredient", "concept", "dose_std", "volume_ml", "na_mmol"]]


def reference_urine(ref: dict, stays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    items = {int(i) for i in ref["urine_items"]}
    icu_ids = set(stays["icu_id"].astype(int))
    parts = []
    for ch in iter_source(ref["tables"]["outputs"], root=ref.get("root"), chunksize=2_000_000):
        ch = ch[to_numeric(ch["item"]).isin(items) & _ids(ch["icu_key"]).isin(icu_ids)]
        if len(ch):
            parts.append(pd.DataFrame({"icu_id": _ids(ch["icu_key"]).astype(int).values,
                                       "time": to_datetime(ch["time"]).values, "value": to_numeric(ch["value"]).values}))
    u = pd.concat(parts, ignore_index=True).dropna()
    u = u[(u["value"] >= 0) & (u["value"] <= 5000)]
    u = _to_stay_time(u, stays.assign(icu_id=stays["icu_id"].astype(int)), "icu_id", "time", 0.0)
    return u[["icu_id", "t", "value"]].sort_values(["icu_id", "t"]).reset_index(drop=True)
