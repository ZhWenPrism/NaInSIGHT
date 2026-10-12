"""Knowledge-graph assembly for GNN pre-training.

Nodes (concepts.GRAPH_NODES): intervention nodes followed by physiological state nodes.
Edges
  type 0  drug -> state    attribute = effect spectrum (z-statistics over dose level x lag) + summaries
  type 1  state -> drug    reverse message direction of type-0 edges (same attribute, separate type)
  type 2  state -> state   temporal propagation over lags (alpha z-statistics) + summaries
Self-supervision targets: three-class direction of each drug -> state edge (decrease / no clear
effect / increase) from the precision-weighted mean effect.
"""
from __future__ import annotations

import numpy as np

from ..concepts import DRUG_NODES, GRAPH_NODES, STATE_NODES

N_SUMMARY = 4


def _z(beta: np.ndarray, se: np.ndarray, clip: float = 10.0) -> np.ndarray:
    z = np.where(np.isfinite(beta) & np.isfinite(se) & (se > 0), beta / np.where(se > 0, se, 1.0), np.nan)
    return np.clip(z, -clip, clip)


def build_graph(pri: dict, meta: dict) -> dict:
    """``pri``: prior matrices on the reference lag grid (B, B_se, A, A_se); ``meta``: anchor-panel summaries."""
    B, Bse, A, Ase = pri["B"], pri["B_se"], pri["A"], pri["A_se"]
    D, K, T, S = B.shape
    edge_dim = K * T + N_SUMMARY
    sd = np.array([meta.get("baseline_sd", {}).get(s, 1.0) or 1.0 for s in STATE_NODES])
    mean = np.array([meta.get("baseline_mean", {}).get(s, 0.0) or 0.0 for s in STATE_NODES])
    prev = np.array([meta.get("exposure_prevalence", {}).get(d, 0.0) or 0.0 for d in DRUG_NODES])

    # ---------------------------------------------------------------- node features
    x = np.zeros((len(GRAPH_NODES), 6), dtype=np.float32)
    x[:D, 0] = 1.0
    x[D:, 1] = 1.0
    x[:D, 2] = np.sqrt(prev)
    zB = _z(B, Bse)
    x[:D, 3] = np.nanmean(np.abs(np.nan_to_num(zB)), axis=(1, 2, 3)) / 10.0
    x[D:, 4] = (mean - mean.mean()) / (mean.std() + 1e-6)
    x[D:, 5] = np.log(np.maximum(sd, 1e-6)) / 5.0

    src, dst, attr, etype, label = [], [], [], [], []
    for d in range(D):
        for s in range(S):
            spec = zB[d, :, :, s]                                  # (K, T)
            if not np.isfinite(spec).any():
                continue
            b, e = B[d, :, :, s], Bse[d, :, :, s]
            ok = np.isfinite(b) & np.isfinite(e) & (e > 0)
            w = np.where(ok, 1.0 / np.where(ok, e, 1.0) ** 2, 0.0)
            pooled = (w * np.nan_to_num(b)).sum() / w.sum()
            pooled_z = pooled * np.sqrt(w.sum())
            a = np.zeros(edge_dim, dtype=np.float32)
            a[: K * T] = np.nan_to_num(spec).reshape(-1) / 10.0
            a[K * T:] = [ok.mean(), np.sign(pooled), np.nanmax(np.abs(spec)) / 10.0, np.tanh(pooled / sd[s])]
            cls = 2 if pooled_z > 1.96 else (0 if pooled_z < -1.96 else 1)
            for (u, v, t, lab) in ((d, D + s, 0, cls), (D + s, d, 1, -100)):
                src.append(u); dst.append(v); attr.append(a); etype.append(t); label.append(lab)
    zA = _z(A, Ase)                                                # (T, S_dst, S_src)
    for m in range(S):
        for s in range(S):
            spec = zA[:, s, m]
            if not np.isfinite(spec).any():
                continue
            a = np.zeros(edge_dim, dtype=np.float32)
            a[:T] = np.nan_to_num(spec) / 10.0
            fin = A[:, s, m][np.isfinite(A[:, s, m])]
            a[K * T:] = [np.isfinite(spec).mean(), np.sign(fin.mean()) if len(fin) else 0.0,
                         np.nanmax(np.abs(spec)) / 10.0, np.tanh((fin.mean() if len(fin) else 0.0) / sd[s])]
            src.append(D + m); dst.append(D + s); attr.append(a); etype.append(2); label.append(-100)
    return {"x": x, "edge_index": np.asarray([src, dst], dtype=np.int64),
            "edge_attr": np.asarray(attr, dtype=np.float32), "edge_type": np.asarray(etype, dtype=np.int64),
            "sign_label": np.asarray(label, dtype=np.int64), "node_names": np.asarray(GRAPH_NODES),
            "n_drug": np.asarray(D), "reliability": node_reliability(zB, D, S)}


def node_reliability(zB: np.ndarray, D: int, S: int) -> np.ndarray:
    """Share of a node's effect cells that are statistically clear (|z| > 1.96) - used by the adapter gate."""
    clear = np.abs(np.nan_to_num(zB)) > 1.96
    finite = np.isfinite(zB)
    rel = np.zeros(D + S, dtype=np.float32)
    rel[:D] = clear.sum(axis=(1, 2, 3)) / np.maximum(finite.sum(axis=(1, 2, 3)), 1)
    rel[D:] = clear.sum(axis=(0, 1, 2)) / np.maximum(finite.sum(axis=(0, 1, 2)), 1)
    return rel
