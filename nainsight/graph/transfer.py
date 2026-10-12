"""Transfer of reference-cohort effects to the development cohort.

1. stage-2 sub-population: estimates from the neurosurgical sub-population are combined with the
   full-cohort estimates by empirical Bayes (random-effects precision weighting; the between-population
   variance tau^2 is a method-of-moments estimate per (state, lag) block);
2. unit conversion: reference units -> development units per state node (Delta Y scales linearly);
3. time-step conversion: spectra over the reference lag grid are interpolated (piecewise-linear in lag,
   held constant beyond the last estimated lag) to the lags used by the development landmarks.
Output: prior matrices B[drug, level, lag, state] and A[lag, state_dst, state_src] with standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..concepts import DRUG_NODES, STATE_NODES

KEY = ["src_type", "src", "level", "dst", "lag_h"]


def shrink_subpopulation(eff_all: pd.DataFrame, eff_sub: pd.DataFrame) -> pd.DataFrame:
    m = eff_all[KEY + ["beta", "se", "n_exposed"]].merge(
        eff_sub[KEY + ["beta", "se", "n_exposed"]], on=KEY, how="left", suffixes=("_all", "_sub"))
    m["tau2"] = 0.0
    both = np.isfinite(m["beta_all"]) & np.isfinite(m["beta_sub"]) & np.isfinite(m["se_all"]) & np.isfinite(m["se_sub"])
    for (stype, dst, lag), idx in m.groupby(["src_type", "dst", "lag_h"]).groups.items():
        sel = m.index.isin(idx) & both
        if sel.sum() >= 3:
            dev = (m.loc[sel, "beta_sub"] - m.loc[sel, "beta_all"]) ** 2 - m.loc[sel, "se_sub"] ** 2 - m.loc[sel, "se_all"] ** 2
            m.loc[m.index.isin(idx), "tau2"] = max(0.0, float(dev.mean()))
    w_sub = np.where(both, 1.0 / np.maximum(m["se_sub"] ** 2, 1e-12), 0.0)
    v_all = m["se_all"] ** 2 + m["tau2"]
    w_all = np.where(np.isfinite(m["beta_all"]), 1.0 / np.maximum(v_all, 1e-12), 0.0)
    num = w_sub * np.nan_to_num(m["beta_sub"]) + w_all * np.nan_to_num(m["beta_all"])
    den = w_sub + w_all
    m["beta"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    m["se"] = np.where(den > 0, np.sqrt(1.0 / np.where(den > 0, den, 1.0)), np.nan)
    m["weight_sub"] = np.where(den > 0, w_sub / np.where(den > 0, den, 1.0), 0.0)
    return m


def convert_units(eff: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """Delta Y of state ``dst`` is expressed in development units (multiplicative unit factors)."""
    out = eff.copy()
    f = out["dst"].map(lambda s: float(factors.get(s, 1.0)))
    for c in [c for c in out.columns if c.startswith("beta") or c.startswith("se")]:
        out[c] = out[c] * f
    return out


def interpolate_lags(values: np.ndarray, ref_lags: list, dev_lags: list) -> np.ndarray:
    """values[..., T_ref] -> [..., T_dev]; NaN cells are interpolated from finite neighbours."""
    ref = np.asarray(ref_lags, dtype=float)
    out = np.full(values.shape[:-1] + (len(dev_lags),), np.nan)
    flat_in = values.reshape(-1, values.shape[-1])
    flat_out = out.reshape(-1, len(dev_lags))
    for i, row in enumerate(flat_in):
        ok = np.isfinite(row)
        if ok.sum() == 0:
            continue
        flat_out[i] = np.interp(dev_lags, ref[ok], row[ok]) if ok.sum() > 1 else row[ok][0]
    return out


def prior_matrices(eff: pd.DataFrame, cfg: dict) -> dict:
    g = cfg["graph"]
    K = int(g["dose_levels"])
    ref_lags = [int(x) for x in g["lags_h"]]
    dev_lags = [int(x) for x in g["dev_lags_h"]]
    D, S, T = len(DRUG_NODES), len(STATE_NODES), len(ref_lags)
    di, si, ti = ({n: i for i, n in enumerate(DRUG_NODES)}, {n: i for i, n in enumerate(STATE_NODES)},
                  {n: i for i, n in enumerate(ref_lags)})
    B, Bse = np.full((D, K, T, S), np.nan), np.full((D, K, T, S), np.nan)
    A, Ase = np.full((T, S, S), np.nan), np.full((T, S, S), np.nan)
    for r in eff.itertuples(index=False):
        if r.lag_h not in ti or r.dst not in si:
            continue
        if r.src_type == "drug" and r.src in di and 1 <= r.level <= K:
            B[di[r.src], r.level - 1, ti[r.lag_h], si[r.dst]] = r.beta
            Bse[di[r.src], r.level - 1, ti[r.lag_h], si[r.dst]] = r.se
        elif r.src_type == "state" and r.src in si:
            A[ti[r.lag_h], si[r.dst], si[r.src]] = r.beta
            Ase[ti[r.lag_h], si[r.dst], si[r.src]] = r.se
    B_dev = np.moveaxis(interpolate_lags(np.moveaxis(B, 2, -1), ref_lags, dev_lags), -1, 2)
    Bse_dev = np.moveaxis(interpolate_lags(np.moveaxis(Bse, 2, -1), ref_lags, dev_lags), -1, 2)
    A_dev = np.moveaxis(interpolate_lags(np.moveaxis(A, 0, -1), ref_lags, dev_lags), -1, 0)
    Ase_dev = np.moveaxis(interpolate_lags(np.moveaxis(Ase, 0, -1), ref_lags, dev_lags), -1, 0)
    return {"B": B, "B_se": Bse, "A": A, "A_se": Ase, "B_dev": B_dev, "B_se_dev": Bse_dev, "A_dev": A_dev,
            "A_se_dev": Ase_dev, "ref_lags": np.asarray(ref_lags), "dev_lags": np.asarray(dev_lags),
            "drug_nodes": np.asarray(DRUG_NODES), "state_nodes": np.asarray(STATE_NODES)}
