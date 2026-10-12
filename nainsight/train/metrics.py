"""Discrimination, calibration, regression and ordinal metrics on landmark-level predictions."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

SEVERITY = ("normal", "mild", "moderate_severe")


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(int), p[m]
    out = {"n": int(len(y)), "n_pos": int(y.sum())}
    if len(np.unique(y)) < 2:
        return {**out, "AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan}
    out.update(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p), Brier=brier_score_loss(y, p))
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.c_[np.ones_like(lp), lp]
    beta = np.zeros(2)
    for _ in range(50):                                            # logistic recalibration (Newton)
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = mu * (1 - mu) + 1e-9
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-8 * np.eye(2), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    out.update(calibration_intercept=beta[0], calibration_slope=beta[1])
    return out


def regression_metrics(y: np.ndarray, mu: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[m], mu[m]
    if len(y) < 2:
        return {"n": int(len(y)), "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "bias": np.nan}
    err = mu - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "bias": float(err.mean())}


def ordinal_metrics(y: np.ndarray, P: np.ndarray, per_class: bool = False) -> dict:
    m = np.isfinite(y) & np.isfinite(P).all(axis=1)
    y, P = y[m].astype(int), P[m]
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out
    pred = P.argmax(axis=1)
    aucs = [roc_auc_score((y == c).astype(int), P[:, c]) for c in range(3) if 0 < (y == c).sum() < len(y)]
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (cohen_kappa_score(y, pred, labels=[0, 1, 2], weights="quadratic")
                 if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else np.nan)
    out.update(macro_AUROC=float(np.mean(aucs)) if aucs else np.nan,
               macro_F1=f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
               balanced_accuracy=balanced_accuracy_score(y, pred), weighted_kappa=kappa)
    if per_class:
        pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        for c, name in enumerate(SEVERITY):
            out.update({f"{name}_precision": pr[c], f"{name}_recall": rc[c], f"{name}_F1": f1[c], f"{name}_support": int(sup[c])})
    return out


def landmark_metrics(df: pd.DataFrame, horizons: list[int], per_class: bool = False) -> dict:
    """{(mode, horizon, task): {metric: value}} for one prediction frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)   # degenerate bootstrap resamples
        return _landmark_metrics(df, horizons, per_class)


def _landmark_metrics(df: pd.DataFrame, horizons: list[int], per_class: bool) -> dict:
    res = {}
    for mode, g in df.groupby("mode"):
        for h in horizons:
            res[(mode, h, "risk")] = binary_metrics(g[f"y_risk_{h}"].values, g[f"p_risk_{h}"].values)
            res[(mode, h, "nadir")] = regression_metrics(g[f"y_nadir_{h}"].values, g[f"mu_nadir_{h}"].values)
            P = g[[f"p_sev{c}_{h}" for c in range(3)]].values
            res[(mode, h, "severity")] = ordinal_metrics(g[f"y_sev_{h}"].values, P, per_class)
    return res


def validation_score(res: dict) -> float:
    aucs = [v["AUROC"] for (mode, h, task), v in res.items() if task == "risk" and np.isfinite(v.get("AUROC", np.nan))]
    return float(np.mean(aucs)) if aucs else float("nan")
