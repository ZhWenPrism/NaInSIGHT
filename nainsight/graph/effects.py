"""Dose- and lag-resolved medication effect estimation in the reference cohort.

Anchor design (self-controlled, within ICU stay):
  anchors a = k * step (h since ICU admission); exposure window [a, a + step)
  exposure x[d, k]   1 if intervention d was given in the window at within-ingredient dose level k
                     (administration records, and infusion amounts pro-rated to the window)
  baseline Y_l(a-)   last value of state l in [a - lookback, a]   (urine: output rate in [a - step, a))
  outcome Y_l(a+tau) value nearest to a + tau within max(min_h, frac * tau), observed after a
                     (urine: output rate in [a + tau - step, a + tau))
  Delta Y_l^(tau) = Y_l(a+tau) - Y_l(a-)

For every state l and lag tau a within-stay (fixed-effects) linear model is fitted

  Delta Y = sum_{d,k} beta^{(k,tau)}_{d,l} x[d,k] + sum_m alpha^{(tau)}_{l,m} z_m(a-) + sum_m gamma_m miss_m + u_stay + e

z_m are standardised baselines of all states (self term = regression to the mean, cross terms = lab-to-lab
propagation).  Standard errors are cluster-robust by patient.  beta forms the effect spectrum
e_{d->l} = [beta^{(k,tau)}]_{k,tau}; alpha forms the temporal propagation matrices.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..concepts import DRUG_NODES, STATE_NODES

LOG = logging.getLogger("nainsight.effects")
STATE_OF_CONCEPT = {"LAB_NA": "NA", "LAB_K": "K", "LAB_CL": "CL", "LAB_GLU": "GLU", "LAB_UREA": "UREA",
                    "LAB_CREA": "CREA", "LAB_ALB": "ALB"}


def quantile_levels(values: pd.Series, groups: pd.Series, levels: int) -> tuple[np.ndarray, dict]:
    """Within-group quantile levels 1..K; a missing dose is assigned the median level."""
    qs = np.linspace(0, 1, levels + 1)[1:-1]
    out = np.zeros(len(values), dtype=np.int8)
    cuts = {}
    v = values.values.astype(float)
    g = groups.astype(str).values
    for name in np.unique(g):
        sel = g == name
        pos = v[sel]
        ok = np.isfinite(pos) & (pos > 0)
        c = np.unique(np.quantile(pos[ok], qs)) if ok.sum() >= levels * 5 else np.array([])
        cuts[name] = c.tolist()
        lvl = np.full(sel.sum(), int(np.ceil(levels / 2)), dtype=np.int8)
        if len(c):
            lvl[ok] = np.searchsorted(c, pos[ok], side="right") + 1
        out[sel] = lvl
    return out, cuts


def _infusion_windows(inf: pd.DataFrame, step: float) -> pd.DataFrame:
    """Split each infusion over the exposure windows it overlaps, pro-rating the amount by overlap."""
    if inf.empty:
        return pd.DataFrame(columns=["icu_id", "k", "concept", "amount"])
    k0 = np.floor(inf["t_start"].clip(lower=0).values / step).astype(int)
    k1 = np.floor(inf["t_end"].clip(lower=0).values / step).astype(int)
    span = np.clip(k1 - k0 + 1, 1, 200)
    ridx = np.repeat(np.arange(len(inf)), span)
    kk = k0[ridx] + np.concatenate([np.arange(s) for s in span])
    ts, te = inf["t_start"].values[ridx], inf["t_end"].values[ridx]
    dur = np.maximum(te - ts, 1e-6)
    overlap = np.clip(np.minimum(te, (kk + 1) * step) - np.maximum(ts, kk * step), 0, None)
    frac = np.where(te - ts > 1e-6, overlap / dur, 1.0)
    win = pd.DataFrame({"icu_id": inf["icu_id"].values[ridx].astype(int), "k": kk,
                        "concept": inf["concept"].values[ridx], "amount": inf["dose_std"].values[ridx] * frac})
    return win[win["amount"] > 0].groupby(["icu_id", "k", "concept"], as_index=False)["amount"].sum()


def anchor_panel(stays: pd.DataFrame, labs: pd.DataFrame, drugs: pd.DataFrame, infusions: pd.DataFrame,
                 urine: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    g = cfg["graph"]
    step, lb, K = float(g["anchor_step_h"]), float(g["baseline_lookback_h"]), int(g["dose_levels"])
    lags = [float(x) for x in g["lags_h"]]
    st = stays.assign(icu_id=stays["icu_id"].astype(int), subject_id=stays["subject_id"].astype(int))
    n_k = np.maximum(np.floor((st["los_h"].values - step) / step).astype(int), 0)
    rep = np.repeat(np.arange(len(st)), n_k)
    k = np.concatenate([np.arange(1, n + 1) for n in n_k]) if len(st) else np.array([], int)
    panel = pd.DataFrame({"icu_id": st["icu_id"].values[rep], "subject_id": st["subject_id"].values[rep],
                          "subpop": st["subpop"].values[rep], "k": k, "a": (k * step).astype(np.float32)})

    # ---------------------------------------------------------------- exposures (max level per window)
    d = drugs[drugs["t"] >= 0].copy()
    d["lvl"], cuts_d = quantile_levels(d["dose_std"], d["ingredient"], K)
    d["k"] = np.floor(d["t"].values / step).astype(int)
    d["icu_id"] = d["icu_id"].astype(int)
    win = _infusion_windows(infusions, step)
    win["lvl"], cuts_i = quantile_levels(win["amount"], win["concept"], K)
    expo = pd.concat([d[["icu_id", "k", "concept", "lvl"]], win[["icu_id", "k", "concept", "lvl"]]], ignore_index=True)
    expo["concept"] = expo["concept"].astype(str)
    expo = expo.pivot_table(index=["icu_id", "k"], columns="concept", values="lvl", aggfunc="max").reset_index()
    expo.columns.name = None
    level_cols = [c for c in expo.columns if c not in ("icu_id", "k")]
    panel = panel.merge(expo, on=["icu_id", "k"], how="left")
    new = {}
    for dn in DRUG_NODES:
        col = panel[dn].fillna(0).values if dn in panel else np.zeros(len(panel))
        for lv in range(1, K + 1):
            new[f"x__{dn}__{lv}"] = (col == lv).astype(np.int8)
    panel = pd.concat([panel.drop(columns=level_cols), pd.DataFrame(new, index=panel.index)], axis=1)
    panel["_row"] = np.arange(len(panel))
    new = {}

    # ---------------------------------------------------------------- baselines and outcomes (laboratory)
    labs = labs.assign(state=labs["concept"].map(STATE_OF_CONCEPT)).dropna(subset=["state"])
    tol_min, tol_frac = float(g["outcome_tolerance"]["min_h"]), float(g["outcome_tolerance"]["frac"])
    for s, ls in labs.groupby("state"):
        ls = (ls[["icu_id", "t", "value"]].assign(icu_id=lambda x: x["icu_id"].astype(int))
              .rename(columns={"t": "_t"}).astype({"_t": np.float32}).sort_values("_t"))
        ls["_tobs"] = ls["_t"].values
        q = panel[["_row", "icu_id", "a"]].assign(_t=panel["a"].astype(np.float32)).sort_values("_t")
        base = pd.merge_asof(q, ls, on="_t", by="icu_id", direction="backward", tolerance=np.float32(lb))
        new[f"b__{s}"] = base.set_index("_row")["value"].reindex(panel["_row"]).values
        for tau in lags:
            q = panel[["_row", "icu_id", "a"]].assign(_t=(panel["a"] + tau).astype(np.float32)).sort_values("_t")
            hit = pd.merge_asof(q, ls, on="_t", by="icu_id", direction="nearest",
                                tolerance=np.float32(max(tol_min, tol_frac * tau)))
            hit = hit.set_index("_row").reindex(panel["_row"])
            ok = hit["_tobs"].values > hit["a"].values
            new[f"y__{s}__{int(tau)}"] = np.where(ok, hit["value"].values, np.nan)

    # ---------------------------------------------------------------- urine output rate (mL/h) per window
    u = urine[urine["t"] >= 0].assign(k=lambda x: np.floor(x["t"].values / step).astype(int),
                                       icu_id=lambda x: x["icu_id"].astype(int))
    u = (u.groupby(["icu_id", "k"])["value"].sum() / step).rename("rate").reset_index()
    for name, shift in [("b__URINE", -1)] + [(f"y__URINE__{int(t)}", int(round(t / step)) - 1) for t in lags]:
        look = panel[["_row", "icu_id", "k"]].assign(k=panel["k"] + shift).merge(u, on=["icu_id", "k"], how="left")
        new[name] = look.set_index("_row")["rate"].reindex(panel["_row"]).values
    panel = pd.concat([panel.drop(columns="_row"), pd.DataFrame(new, index=panel.index)], axis=1)
    meta = {"dose_cutpoints_administrations": cuts_d, "dose_cutpoints_infusion_windows": cuts_i,
            "n_anchors": int(len(panel)), "n_stays": int(panel["icu_id"].nunique()),
            "baseline_mean": {s: float(panel[f"b__{s}"].mean()) for s in STATE_NODES if f"b__{s}" in panel},
            "baseline_sd": {s: float(panel[f"b__{s}"].std()) for s in STATE_NODES if f"b__{s}" in panel},
            "exposure_prevalence": {dn: float(panel[[f"x__{dn}__{lv}" for lv in range(1, K + 1)]].max(axis=1).mean())
                                    for dn in DRUG_NODES}}
    return panel, meta


def _demean(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-group demeaning (absorbs the stay fixed effect); singleton groups are dropped."""
    order = np.argsort(groups, kind="stable")
    X, y, g = X[order], y[order], groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    counts = np.diff(np.r_[starts, len(g)])
    Xm = np.add.reduceat(X, starts, axis=0) / counts[:, None]
    ym = np.add.reduceat(y, starts) / counts
    rep = np.repeat(np.arange(len(starts)), counts)
    keep = np.repeat(counts >= 2, counts)
    return (X - Xm[rep])[keep], (y - ym[rep])[keep], order[keep]


def fit_effects(panel: pd.DataFrame, cfg: dict, stage: str, seed: int) -> pd.DataFrame:
    g = cfg["graph"]
    K = int(g["dose_levels"])
    lags = [int(x) for x in g["lags_h"]]
    xcols = [f"x__{d}__{lv}" for d in DRUG_NODES for lv in range(1, K + 1)]
    bcols = [f"b__{s}" for s in STATE_NODES]
    base = panel[bcols].astype(float)
    mu, sd = base.mean(), base.std().replace(0, 1.0).fillna(1.0)
    Z = ((base - mu) / sd).fillna(0.0).values
    M = base.isna().values.astype(float)
    Xall = np.hstack([panel[xcols].values.astype(float), Z, M])
    names = xcols + [f"z__{s}" for s in STATE_NODES] + [f"m__{s}" for s in STATE_NODES]
    rng = np.random.default_rng(seed)
    rows = []
    for s in STATE_NODES:
        for tau in lags:
            ycol = f"y__{s}__{tau}"
            if ycol not in panel:
                continue
            y_all = (panel[ycol] - panel[f"b__{s}"]).values.astype(float)
            idx = np.flatnonzero(np.isfinite(y_all))
            if len(idx) > g["max_anchors_per_fit"]:                    # subsample whole stays
                stays = panel["icu_id"].values[idx]
                uniq = np.unique(stays)
                take = rng.choice(uniq, max(1, int(len(uniq) * g["max_anchors_per_fit"] / len(idx))), replace=False)
                idx = idx[np.isin(stays, take)]
            if len(idx) < 200:
                LOG.warning("[%s] %s tau=%dh: only %d usable anchors, skipped", stage, s, tau, len(idx))
                continue
            X, y, order = _demean(Xall[idx], y_all[idx], panel["icu_id"].values[idx])
            live = X.std(axis=0) > 0
            Xl = X[:, live]
            xtx = Xl.T @ Xl + float(g["ridge"]) * len(y) * np.eye(Xl.shape[1])
            inv = np.linalg.pinv(xtx)
            beta = inv @ (Xl.T @ y)
            resid = y - Xl @ beta
            clusters = panel["subject_id"].values[idx][order]
            scores = pd.DataFrame(Xl * resid[:, None]).groupby(clusters).sum().values
            G = scores.shape[0]
            V = inv @ (scores.T @ scores) @ inv * (G / max(G - 1, 1))
            se = np.sqrt(np.clip(np.diag(V), 0, None))
            full_b, full_se = np.full(len(names), np.nan), np.full(len(names), np.nan)
            full_b[live], full_se[live] = beta, se
            n_exp = (Xall[idx][:, : len(xcols)] > 0).sum(axis=0)
            for j, name in enumerate(names):
                kind, *rest = name.split("__")
                if kind == "m":
                    continue
                if kind == "x":
                    ne = int(n_exp[j])
                    ok = ne >= g["min_exposed"]
                    rows.append({"stage": stage, "src_type": "drug", "src": rest[0], "level": int(rest[1]), "dst": s,
                                 "lag_h": tau, "beta": full_b[j] if ok else np.nan, "se": full_se[j] if ok else np.nan,
                                 "n_exposed": ne, "n_rows": len(y), "n_clusters": G})
                else:
                    rows.append({"stage": stage, "src_type": "state", "src": rest[0], "level": 0, "dst": s,
                                 "lag_h": tau, "beta": full_b[j], "se": full_se[j], "n_exposed": len(y),
                                 "n_rows": len(y), "n_clusters": G})
            LOG.info("[%s] %s tau=%dh: rows=%d clusters=%d", stage, s, tau, len(y), G)
    return pd.DataFrame(rows)
