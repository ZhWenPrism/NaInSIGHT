"""Static covariates and fitted preprocessors.

All statistics (medians, scales, quantile cut-points, category levels) are estimated on the *training*
split only and then frozen; validation, temporal-test and external cohorts are transformed with them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..concepts import INGREDIENTS

PREOP_LABS = ("LAB_NA", "LAB_K", "LAB_CL", "LAB_GLU", "LAB_CREA", "LAB_UREA", "LAB_ALB", "LAB_HGB",
              "LAB_WBC", "LAB_PLT")
PREOP_DRUGS = {"preop_mannitol": ["MANNITOL"], "preop_glucocorticoid": ["DEXAMETHASONE", "GLUCOCORTICOID_OTHER"],
               "preop_any_aed": ["CARBAMAZEPINE", "OXCARBAZEPINE", "VALPROATE", "LEVETIRACETAM", "OTHER_AED"],
               "preop_oxcarbazepine": ["OXCARBAZEPINE"], "preop_valproate": ["VALPROATE"],
               "preop_hypertonic_saline": ["HYPERTONIC_SALINE"], "preop_desmopressin": ["DESMOPRESSIN"],
               "preop_diuretic": ["LOOP_DIURETIC", "THIAZIDE", "K_SPARING_DIURETIC"]}


def preop_summaries(stays: pd.DataFrame, labs: pd.DataFrame, meds: pd.DataFrame, io: pd.DataFrame) -> pd.DataFrame:
    """Covariates observable before skin incision: last pre-operative laboratory values, medication
    exposure flags and charted urine output per day."""
    start = (-stays.set_index("stay_id")["surgery_duration_h"]).rename("start_rel")
    out = pd.DataFrame(index=stays["stay_id"].values)
    pre_l = labs.join(start, on="stay_id")
    pre_l = pre_l[pre_l["t"] < pre_l["start_rel"]].sort_values(["stay_id", "t"])
    last = pre_l.groupby(["stay_id", "concept"], observed=True)["value"].last().unstack()
    for c in PREOP_LABS:
        out[f"preop_last_{c.lower()}"] = last[c] if c in last else np.nan
    if "LAB_NA" in set(pre_l["concept"].astype(str)):
        out["preop_min_lab_na"] = pre_l[pre_l["concept"] == "LAB_NA"].groupby("stay_id")["value"].min()
    pre_m = meds.join(start, on="stay_id")
    pre_m = pre_m[pre_m["t"] < pre_m["start_rel"]]
    for name, concepts in PREOP_DRUGS.items():
        ids = set(pre_m.loc[pre_m["concept"].isin(concepts), "stay_id"])
        out[name] = out.index.isin(ids).astype(float)
    pre_io = io.join(start, on="stay_id")
    pre_io = pre_io[(pre_io["t"] < pre_io["start_rel"]) & (pre_io["concept"] == "IO_URINE")]
    if len(pre_io):
        span = (pre_io.groupby("stay_id")["t"].agg(lambda s: max(s.max() - s.min(), 24.0)) / 24.0)
        out["preop_urine_ml_per_day"] = pre_io.groupby("stay_id")["value"].sum() / span
    out.index.name = "stay_id"
    return out.reset_index()


@dataclass
class StaticPreprocessor:
    """numeric: median imputation + missing indicator + robust scaling;  binary: missing indicator + 0-fill;
    category: one-hot over levels seen >= ``min_count`` times in training (+ 'other')."""
    schema: dict
    min_count: int = 20
    stats: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "StaticPreprocessor":
        for col, kind in self.schema.items():
            s = df[col]
            if kind == "numeric":
                v = pd.to_numeric(s, errors="coerce")
                med, scale = robust_center_scale(v)
                self.stats[col] = {"kind": kind, "median": med, "scale": scale, "has_missing": bool(v.isna().any())}
            elif kind == "binary":
                self.stats[col] = {"kind": kind, "has_missing": bool(pd.to_numeric(s, errors="coerce").isna().any())}
            else:
                vc = s.astype("string").value_counts()
                self.stats[col] = {"kind": "category", "levels": [str(x) for x in vc[vc >= self.min_count].index]}
        self.columns = list(self.transform(df.head(1)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, st in self.stats.items():
            s = df[col] if col in df else pd.Series(np.nan, index=df.index)
            if st["kind"] == "numeric":
                v = pd.to_numeric(s, errors="coerce")
                out[col] = ((v.fillna(st["median"]) - st["median"]) / st["scale"]).clip(-8, 8)
                if st["has_missing"]:
                    out[f"{col}__missing"] = v.isna().astype(float)
            elif st["kind"] == "binary":
                v = pd.to_numeric(s, errors="coerce")
                out[col] = v.fillna(0.0).clip(0, 1)
                if st["has_missing"]:
                    out[f"{col}__missing"] = v.isna().astype(float)
            else:
                v = s.astype("string")
                for lvl in st["levels"]:
                    out[f"{col}={lvl}"] = (v == lvl).fillna(False).astype(float)      # missing -> all-zero one-hot
                out[f"{col}=other"] = (~v.isin(st["levels"]) & v.notna()).fillna(False).astype(float)
        frame = pd.DataFrame(out, index=df.index).astype(np.float32)
        if self.columns:
            frame = frame.reindex(columns=self.columns, fill_value=0.0)
        if not np.isfinite(frame.values).all():
            raise ValueError("static preprocessing produced non-finite values")
        return frame

    def state(self) -> dict:
        return {"schema": self.schema, "min_count": self.min_count, "stats": self.stats, "columns": self.columns}

    @classmethod
    def from_state(cls, st: dict) -> "StaticPreprocessor":
        return cls(schema=st["schema"], min_count=st["min_count"], stats=st["stats"], columns=st["columns"])


def robust_center_scale(values: pd.Series) -> tuple[float, float]:
    """(median, IQR / 1.349); falls back to the SD, then to 1, when the IQR is degenerate."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return 0.0, 1.0
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
    scale = float(q3 - q1) / 1.349
    if not (np.isfinite(scale) and scale > 0):
        sd = float(v.std()) if len(v) > 1 else float("nan")
        scale = sd if np.isfinite(sd) and sd > 0 else 1.0
    return float(med), float(scale)


def fit_value_scaler(labs: pd.DataFrame, vitals: pd.DataFrame, io: pd.DataFrame, meds: pd.DataFrame) -> dict:
    """Per-concept robust centre/scale for measured values; per-ingredient centre/scale of log-dose."""
    scaler = {}
    for frame in (labs, vitals, io):
        for c, g in frame.groupby("concept", observed=True):
            scaler[str(c)] = robust_center_scale(g["value"].astype(float))
    for ing, g in meds.groupby("ingredient", observed=True):
        scaler[f"dose::{ing}"] = robust_center_scale(np.log1p(g["dose_std"].astype(float)))
    return scaler


def fit_dose_cutpoints(meds: pd.DataFrame, levels: int) -> dict:
    """Within-ingredient quantile cut-points -> dose levels 1..K (0 = dose unknown)."""
    qs = np.linspace(0, 1, levels + 1)[1:-1]
    cuts = {}
    for ing, g in meds.groupby("ingredient", observed=True):
        d = g["dose_std"].dropna()
        d = d[d > 0]
        cuts[str(ing)] = [float(x) for x in np.unique(np.quantile(d, qs))] if len(d) >= levels * 5 else []
    return cuts


def dose_levels(meds: pd.DataFrame, cuts: dict) -> np.ndarray:
    lvl = np.zeros(len(meds), dtype=np.int8)
    ing = meds["ingredient"].astype(str).values
    dose = meds["dose_std"].values.astype(float)
    for name in np.unique(ing):
        sel = (ing == name) & np.isfinite(dose)
        c = cuts.get(name, [])
        lvl[sel] = (np.searchsorted(np.asarray(c), dose[sel], side="right") + 1) if c else 1
    return lvl


def ingredient_concepts() -> dict:
    return {ing: v[0] for ing, v in INGREDIENTS.items()}
