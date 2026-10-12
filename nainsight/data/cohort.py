"""Cohort construction.

Stage 1 (structural, registry only)              Stage 2 (requires the event store)
  source admissions in the study period             no documented pre-operative hyponatraemia
  adults                                            >= 1 evaluable 24-h or 48-h landmark
  tumour resection                                  >= 1 landmark evaluable at both horizons
  plausible surgical record                         one index admission per patient
  consistent admission / surgery / discharge times  temporal split (+ grouped validation subset)
  post-operative stay long enough for the first rolling landmark

Every step is logged into a flow table (remaining / excluded admissions and patients).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .timeparse import hours_between


class FlowLog:
    def __init__(self, site: str):
        self.site, self.rows = site, []

    def add(self, step: str, frame: pd.DataFrame, before: int | None = None) -> None:
        n = len(frame)
        prev = self.rows[-1]["remaining"] if self.rows and before is None else (before if before is not None else n)
        key = "subject_id" if "subject_id" in frame else "subject_key"
        self.rows.append({"site": self.site, "step": step, "remaining": n, "excluded": prev - n,
                          "patients": int(frame[key].nunique()) if key in frame else np.nan})

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if len(df):
            df["retained_pct"] = (100.0 * df["remaining"] / df["remaining"].iloc[0]).round(1)
        return df


def structural_eligibility(reg: pd.DataFrame, cfg: dict, site: dict) -> tuple[pd.DataFrame, FlowLog]:
    c = cfg["cohort"]
    rules = site.get("cohort_rules", {}) or {}
    log = FlowLog(site.get("_name", "site"))
    y0, y1 = c["study_period"]
    df = reg[reg["admit_time"].dt.year.between(y0, y1)].copy()
    log.add("source admissions in study period", df, before=len(df))

    df = df[df["age_years"] >= c["min_age_years"]]
    log.add("adult (age >= minimum)", df)

    if rules.get("resection_include"):
        text = df["procedure_text"].fillna("").astype(str)
        keep = text.str.contains(rules["resection_include"], regex=True)
        if rules.get("resection_exclude"):
            keep &= ~text.str.contains(rules["resection_exclude"], regex=True)
        df = df[keep]
    log.add("tumour resection", df)

    lo, hi = c["surgery_duration_h"]
    ok = df["surgery_start"].notna() & df["surgery_end"].notna() & df["surgery_duration_h"].between(lo, hi)
    df = df[ok]
    log.add("plausible surgical record", df)

    tol = pd.Timedelta(hours=float(c.get("admit_tolerance_h", 24)))
    ok = (df["admit_time"].notna() & df["discharge_time"].notna()
          & (df["surgery_start"] >= df["admit_time"] - tol) & (df["surgery_end"] <= df["discharge_time"]))
    df = df[ok]
    log.add("consistent admission, surgery and discharge timestamps", df)

    df = df.assign(postop_stay_h=hours_between(df["discharge_time"], df["surgery_end"]))
    df = df[df["postop_stay_h"] >= c["min_postop_stay_h"]]
    log.add("post-operative stay >= first rolling landmark + gap + 48 h", df)
    return df.reset_index(drop=True), log


def assign_internal_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integer stay/subject identifiers; the crosswalk to source keys stays in the local work directory."""
    df = df.sort_values(["admit_time", "stay_key"]).reset_index(drop=True)
    df["stay_id"] = np.arange(len(df), dtype=np.int64)
    df["subject_id"] = pd.factorize(df["subject_key"])[0].astype(np.int64)
    crosswalk = df[["stay_id", "subject_id", "stay_key", "subject_key"]].copy()
    return df, crosswalk


def finalize_cohort(stage1: pd.DataFrame, preop_hypo: pd.Series, lmk: pd.DataFrame, cfg: dict,
                    log: FlowLog, seed: int, site_role: str = "development") -> pd.DataFrame:
    """Apply the outcome-dependent criteria, keep one index admission per patient and split."""
    c = cfg["cohort"]
    df = stage1.copy()
    df = df[~df["stay_id"].map(preop_hypo).fillna(False).astype(bool)]
    log.add("no documented pre-operative hyponatraemia", df)

    risk_cols = [col for col in lmk.columns if col.startswith("risk_")]
    any_eval = lmk[risk_cols].notna().any(axis=1)
    both_eval = lmk[risk_cols].notna().all(axis=1)
    df = df[df["stay_id"].isin(lmk.loc[any_eval, "stay_id"])]
    log.add("at least one evaluable 24-h or 48-h landmark", df)
    df = df[df["stay_id"].isin(lmk.loc[both_eval, "stay_id"])]
    log.add("final common 24-/48-h landmark cohort", df)

    if c.get("index_admission", "first_eligible") == "first_eligible":
        df = df.sort_values(["subject_id", "admit_time"]).drop_duplicates("subject_id", keep="first")
        log.add("index admission per patient", df)

    df = df.sort_values("stay_id").reset_index(drop=True)
    if site_role != "development":
        df["split"] = site_role
        return df
    sp = c["split"]
    year = df["admit_time"].dt.year
    df["split"] = np.where(year.between(*sp["train_years"]), "train",
                           np.where(year.isin(list(range(sp["test_years"][0], sp["test_years"][-1] + 1))), "test", "unused"))
    # a patient never contributes to both development and temporal test data
    train_subjects = set(df.loc[df["split"] == "train", "subject_id"])
    df.loc[(df["split"] == "test") & df["subject_id"].isin(train_subjects), "split"] = "unused"

    # grouped, outcome-stratified validation subset inside the development years
    stay_event = lmk.groupby("stay_id")[risk_cols].max().max(axis=1).reindex(df["stay_id"]).fillna(0).values
    rng = np.random.default_rng(seed)
    train_idx = np.where(df["split"].values == "train")[0]
    subj = df["subject_id"].values[train_idx]
    ev = stay_event[train_idx]
    val_subjects = set()
    for stratum in (0, 1):
        s = np.unique(subj[ev == stratum])
        rng.shuffle(s)
        val_subjects |= set(s[: int(round(sp["val_fraction"] * len(s)))])
    is_val = (df["split"].values == "train") & df["subject_id"].isin(val_subjects).values
    df.loc[is_val, "split"] = "val"
    return df
