"""Landmarks and outcomes.

Clock: t = hours since the end of surgery.  Two prediction modes share one definition of the outcome.

  surgery-end baseline   landmark t = 0, windows (0, 24] and (0, 48]
  rolling landmarks      t = 24, 48, 72, ... (<= last_h), a 6-h actionable gap, windows (t+6, t+30] and (t+6, t+54]

For every landmark and horizon h (window W = (t + gap, t + gap + h]):
  risk_h   1 if any serum sodium reported in W is < 135 mmol/L, 0 if >= 1 result in W and none < 135,
           missing if no sodium result is reported in W (not ascertainable);
  nadir_h  minimum sodium in W, only when every 24-h segment of W contains >= 1 result
           (a 0-24-h event keeps the 48-h *risk* positive but does not establish the 48-h nadir);
  sev_h    ordinal class of nadir_h: 0 normal (>= 135), 1 mild [130, 135), 2 moderate-severe (< 130).
No rolling landmark is generated at or after the first post-operative hyponatraemia result, and a landmark
requires the whole 48-h window to lie within the hospital stay (discharge is never treated as negative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sodium_events(labs: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    na = labs[labs["concept"].isin(concepts)][["stay_id", "t", "value"]]
    return na.sort_values(["stay_id", "t"]).reset_index(drop=True)


def preop_hyponatraemia(stays: pd.DataFrame, na: pd.DataFrame, cfg: dict) -> pd.Series:
    rule = cfg["cohort"]["preop_sodium"]
    thr = float(rule["threshold"])
    s = stays.set_index("stay_id")
    start_rel = -s["surgery_duration_h"]                               # surgery start on the clock
    earliest = start_rel - s["preop_stay_days"].clip(lower=0) * 24.0 - float(rule["lookback_days"]) * 24.0
    m = na.join(start_rel.rename("start_rel"), on="stay_id").join(earliest.rename("earliest"), on="stay_id")
    m = m[(m["t"] < m["start_rel"]) & (m["t"] >= m["earliest"])]
    if rule.get("rule", "any") == "last":
        flag = m.groupby("stay_id")["value"].last() < thr
    else:
        flag = m.groupby("stay_id")["value"].min() < thr
    return flag.reindex(s.index, fill_value=False).astype(bool)


def _severity(nadir: float, cut: tuple[float, float]) -> float:
    if np.isnan(nadir):
        return np.nan
    return 0.0 if nadir >= cut[0] else (1.0 if nadir >= cut[1] else 2.0)


def build_landmarks(stays: pd.DataFrame, na: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    oc = cfg["outcome"]
    thr = float(oc["threshold"])
    cut = tuple(float(x) for x in oc["severity_cutpoints"])
    horizons = [int(h) for h in oc["horizons_h"]]
    seg = float(oc.get("segment_h", 24))
    hmax = max(horizons)
    plans = []
    if oc["baseline"]["enabled"]:
        plans.append(("baseline", [float(oc["baseline"]["time_h"])], float(oc["baseline"]["gap_h"])))
    if oc["rolling"]["enabled"]:
        r = oc["rolling"]
        plans.append(("rolling", list(np.arange(r["first_h"], r["last_h"] + 1e-9, r["step_h"], dtype=float)),
                      float(r["gap_h"])))

    post = na[na["t"] > 0]
    by_stay = {sid: (g["t"].values.astype(float), g["value"].values.astype(float))
               for sid, g in post.groupby("stay_id", sort=False)}
    empty = (np.empty(0), np.empty(0))
    rows = []
    for st in stays[["stay_id", "postop_stay_h"]].itertuples(index=False):
        times, vals = by_stay.get(st.stay_id, empty)
        below = np.where(vals < thr)[0]
        first_event = times[below[0]] if len(below) else np.inf
        lmk_id = 0
        for mode, lmk_times, gap in plans:
            for t in lmk_times:
                if oc.get("require_full_window_in_hospital", True) and t + gap + hmax > st.postop_stay_h:
                    continue
                if mode == "rolling" and t >= first_event:
                    continue
                row = {"stay_id": st.stay_id, "lmk_id": lmk_id, "mode": mode, "t_lmk": t, "gap_h": gap}
                for h in horizons:
                    lo_t, hi_t = t + gap, t + gap + h
                    a, b = np.searchsorted(times, lo_t, "right"), np.searchsorted(times, hi_t, "right")
                    w = vals[a:b]
                    row[f"n_na_{h}"] = len(w)
                    row[f"risk_{h}"] = np.nan if len(w) == 0 else float((w < thr).any())
                    covered = len(w) > 0
                    if oc.get("nadir_requires_segment_coverage", True):
                        edges = np.arange(lo_t, hi_t + 1e-9, seg)
                        for e0, e1 in zip(edges[:-1], edges[1:]):
                            if not ((times > e0) & (times <= e1)).any():
                                covered = False
                                break
                    nadir = float(w.min()) if covered else np.nan
                    row[f"nadir_{h}"] = nadir
                    row[f"sev_{h}"] = _severity(nadir, cut)
                rows.append(row)
                lmk_id += 1
    lmk = pd.DataFrame(rows)
    return lmk


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def outcome_composition(cohort: pd.DataFrame, lmk: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Landmark-level outcome composition per split, mode and horizon (descriptive, Table-2 style)."""
    m = lmk.merge(cohort[["stay_id", "subject_id", "split"]], on="stay_id")
    rows = []
    for (split, mode), g in m.groupby(["split", "mode"]):
        for h in horizons:
            e = g[g[f"risk_{h}"].notna()]
            k, n = int(e[f"risk_{h}"].sum()), len(e)
            pat = e.groupby("subject_id")[f"risk_{h}"].max()
            lo, hi = wilson(k, n)
            plo, phi = wilson(int(pat.sum()), len(pat))
            sev = e[f"sev_{h}"].value_counts().reindex([0.0, 1.0, 2.0]).fillna(0).astype(int).tolist()
            rows.append({"split": split, "mode": mode, "horizon_h": h, "patients": int(g["subject_id"].nunique()),
                         "landmarks": n, "positive_landmarks": k,
                         "positive_pct": round(100 * k / max(n, 1), 1), "positive_ci": f"{100*lo:.1f}-{100*hi:.1f}",
                         "patients_with_event": int(pat.sum()),
                         "patients_with_event_pct": round(100 * pat.mean(), 1) if len(pat) else np.nan,
                         "patients_with_event_ci": f"{100*plo:.1f}-{100*phi:.1f}",
                         "severity_normal_mild_modsev": "/".join(map(str, sev))})
    return pd.DataFrame(rows)
