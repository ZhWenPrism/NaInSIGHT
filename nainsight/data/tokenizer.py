"""Clinical event tokenisation.

Each admission becomes one causal sequence of event tokens ordered on the surgery-end clock.  A *landmark
query token* is inserted at every landmark time, after all events with t <= landmark; the network reads its
predictions from the hidden state of these query tokens, so one forward pass serves every landmark of a stay
and no token later than a landmark can influence that landmark's prediction.

Token fields
  concept     canonical concept id (specials: PAD, LMK_BASELINE, LMK_ROLLING, UNK)
  ingredient  medication ingredient id (0 = not a medication)
  modality    special | lab | vital | io | drug
  route       medication route id (0 = none)
  phase       1 pre-operative | 2 intra-operative | 3 post-operative (0 = special token)
  dose_level  within-ingredient quantile level 1..K (0 = unknown dose / not a medication)
  value       robust z-score of the measured value or of the log-dose (0 when absent), clipped
  value_mask  1 when ``value`` carries information
  time        hours since surgery end
  na_load     asinh(sodium load / 50 mmol) of the administration (0 for sodium-free products)
  node        knowledge-graph node id (n_nodes = none)

Per landmark the tokenizer also builds the exposure profile that conditions the graph-prior adapter
(``node_feat``: n_nodes x F) and the last observed serum sodium (reference level for the nadir head).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..concepts import CONCEPTS, GRAPH_NODES, INGREDIENTS, MODALITIES, NODE_INDEX, ROUTES, STATE_NODES, STATE_SOURCE
from .features import dose_levels

SPECIALS = ("[PAD]", "[LMK_BASELINE]", "[LMK_ROLLING]", "[UNK]")
N_NODES = len(GRAPH_NODES)
N_DRUG = N_NODES - len(STATE_NODES)
LMK_FIELDS = ("risk", "nadir", "sev")
_MODALITY_OF = {"labs": "lab", "vitals": "vital", "io": "io"}


def node_feature_layout(levels: int) -> dict:
    """Column layout of the per-node exposure profile (drug block followed by state block)."""
    d0 = 3 + levels + 1                                        # exposure 6/24/72 h + dose one-hot (0..K)
    lay = {"exp6": 0, "exp24": 1, "exp72": 2, "dose0": 3, "recency_drug": d0, "count_drug": d0 + 1,
           "na_load": d0 + 2}
    s0 = d0 + 3
    lay.update({"z_last": s0, "measured": s0 + 1, "z_delta": s0 + 2, "recency_state": s0 + 3,
                "count_state": s0 + 4, "dim": s0 + 5})
    return lay


@dataclass
class ClinicalEventTokenizer:
    max_len: int = 1024
    preop_window_h: float = 168.0
    value_clip: float = 6.0
    dose_levels: int = 4
    min_concept_count: int = 20
    concept_vocab: dict = field(default_factory=dict)
    ingredient_vocab: dict = field(default_factory=dict)
    scaler: dict = field(default_factory=dict)
    dose_cuts: dict = field(default_factory=dict)
    urine_ref: tuple = (1500.0, 750.0)

    # ------------------------------------------------------------------ fitting (training split only)
    def fit(self, events: dict, scaler: dict, dose_cuts: dict) -> "ClinicalEventTokenizer":
        counts = pd.concat([events[k]["concept"].astype(str) for k in ("labs", "vitals", "io", "meds")]).value_counts()
        kept = sorted(c for c, n in counts.items() if n >= self.min_concept_count and c in CONCEPTS)
        self.concept_vocab = {s: i for i, s in enumerate(SPECIALS)}
        self.concept_vocab.update({c: i + len(SPECIALS) for i, c in enumerate(kept)})
        ing = sorted(set(events["meds"]["ingredient"].astype(str)) & set(INGREDIENTS))
        self.ingredient_vocab = {"[NONE]": 0, "[UNK]": 1, **{c: i + 2 for i, c in enumerate(ing)}}
        self.scaler, self.dose_cuts = scaler, dose_cuts
        urine = events["io"][events["io"]["concept"].astype(str) == "IO_URINE"]
        if len(urine):
            per_day = urine.groupby([urine["stay_id"], np.floor(urine["t"] / 24.0)])["value"].sum()
            q1, med, q3 = per_day.quantile([0.25, 0.5, 0.75])
            self.urine_ref = (float(med), float(max((q3 - q1) / 1.349, 100.0)))
        return self

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("max_len", "preop_window_h", "value_clip", "dose_levels",
                                               "min_concept_count", "concept_vocab", "ingredient_vocab",
                                               "scaler", "dose_cuts", "urine_ref")}

    @classmethod
    def from_state(cls, st: dict) -> "ClinicalEventTokenizer":
        st = dict(st)
        st["urine_ref"] = tuple(st["urine_ref"])
        st["scaler"] = {k: tuple(v) for k, v in st["scaler"].items()}
        return cls(**st)

    @property
    def node_feat_dim(self) -> int:
        return node_feature_layout(self.dose_levels)["dim"]

    # ------------------------------------------------------------------ helpers
    def _z(self, key: str, values: np.ndarray) -> np.ndarray:
        c, s = self.scaler.get(key, (0.0, 1.0))
        return np.clip((values - c) / (s if s > 0 else 1.0), -self.value_clip, self.value_clip)

    def _event_table(self, events: dict, stay_ids: np.ndarray) -> pd.DataFrame:
        """All modalities in one token table (one row per token, before windowing)."""
        frames = []
        for key in ("labs", "vitals", "io"):
            e = events[key]
            e = e[e["stay_id"].isin(stay_ids)]
            if e.empty:
                continue
            conc = e["concept"].astype(str).values
            val = e["value"].values.astype(np.float64)
            z = np.empty(len(e), dtype=np.float32)
            for c in np.unique(conc):
                sel = conc == c
                z[sel] = self._z(c, val[sel])
            frames.append(pd.DataFrame({
                "stay_id": e["stay_id"].values, "t": e["t"].values.astype(np.float32), "concept": conc,
                "ingredient": "[NONE]", "modality": _MODALITY_OF[key], "route": "", "dose_level": 0,
                "value": z, "value_mask": 1, "na_load": 0.0}))
        m = events["meds"]
        m = m[m["stay_id"].isin(stay_ids)]
        if not m.empty:
            ing = m["ingredient"].astype(str).values
            logd = np.log1p(m["dose_std"].values.astype(np.float64))
            z = np.zeros(len(m), dtype=np.float32)
            for name in np.unique(ing):
                sel = ing == name
                z[sel] = np.nan_to_num(self._z(f"dose::{name}", logd[sel]), nan=0.0)
            frames.append(pd.DataFrame({
                "stay_id": m["stay_id"].values, "t": m["t"].values.astype(np.float32),
                "concept": m["concept"].astype(str).values, "ingredient": ing, "modality": "drug",
                "route": m["route"].astype(str).values, "dose_level": dose_levels(m, self.dose_cuts),
                "value": z, "value_mask": np.isfinite(logd).astype(np.int8),
                "na_load": np.arcsinh(np.nan_to_num(m["na_mmol"].values.astype(np.float64)) / 50.0).astype(np.float32)}))
        return pd.concat(frames, ignore_index=True)

    def _encode_columns(self, tok: pd.DataFrame, start_rel: pd.Series) -> dict:
        unk_c, unk_i = self.concept_vocab["[UNK]"], self.ingredient_vocab["[UNK]"]
        mod = {m: i for i, m in enumerate(MODALITIES)}
        route = {r: i + 1 for i, r in enumerate(ROUTES)}
        node_of = {c: NODE_INDEX[CONCEPTS[c].node] for c in CONCEPTS if CONCEPTS[c].node in NODE_INDEX}
        sr = tok["stay_id"].map(start_rel).values
        t = tok["t"].values
        return {
            "concept": tok["concept"].map(self.concept_vocab).fillna(unk_c).astype(np.int32).values,
            "ingredient": tok["ingredient"].map(self.ingredient_vocab).fillna(unk_i).astype(np.int32).values,
            "modality": tok["modality"].map(mod).fillna(1).astype(np.int8).values,
            "route": tok["route"].map(route).fillna(0).astype(np.int8).values,
            "phase": np.where(t < sr, 1, np.where(t <= 0, 2, 3)).astype(np.int8),
            "dose_level": tok["dose_level"].values.astype(np.int8),
            "value": tok["value"].values.astype(np.float32),
            "value_mask": tok["value_mask"].values.astype(np.int8),
            "time": t.astype(np.float32),
            "na_load": tok["na_load"].values.astype(np.float32),
            "node": tok["concept"].map(node_of).fillna(N_NODES).astype(np.int16).values,
        }

    # ------------------------------------------------------------------ landmark exposure profile
    def _node_features(self, t: float, drug: dict, state: dict) -> np.ndarray:
        """Drug nodes: exposure in the last 6/24/72 h, one-hot of the highest dose level in 24 h, recency,
        administration count and sodium load (24 h).  State nodes: last z-value, measured flag, change from
        the previous value, recency and measurement count (24 h).  Urine output uses 24-h totals."""
        lay = node_feature_layout(self.dose_levels)
        f = np.zeros((N_NODES, lay["dim"]), dtype=np.float32)
        f[:N_DRUG, lay["recency_drug"]] = 1.0                 # 'never exposed' unless observed
        f[N_DRUG:, lay["recency_state"]] = 1.0
        for n, (tt, lv, na) in drug.items():
            past = tt <= t
            if not past.any():
                continue
            tp, lp, nap = tt[past], lv[past], na[past]
            w24 = tp > t - 24
            f[n, lay["exp6"]] = float((tp > t - 6).any())
            f[n, lay["exp24"]] = float(w24.any())
            f[n, lay["exp72"]] = float((tp > t - 72).any())
            if w24.any():
                f[n, lay["dose0"] + int(lp[w24].max())] = 1.0
                f[n, lay["na_load"]] = np.arcsinh(nap[w24].sum() / 100.0)
            f[n, lay["recency_drug"]] = np.log1p(t - tp.max()) / np.log1p(168.0)
            f[n, lay["count_drug"]] = np.log1p(w24.sum()) / np.log(10.0)
        for j, sname in enumerate(STATE_NODES):
            if sname not in state:
                continue
            n = N_DRUG + j
            tt, zz, raw = state[sname]
            past = tt <= t
            if not past.any():
                continue
            tp, zp, rp = tt[past], zz[past], raw[past]
            if sname == "URINE":
                c, s = self.urine_ref
                last24, prev24 = rp[tp > t - 24].sum(), rp[(tp > t - 48) & (tp <= t - 24)].sum()
                f[n, lay["z_last"]] = np.clip((last24 - c) / s, -self.value_clip, self.value_clip)
                f[n, lay["measured"]] = float((tp > t - 24).any())
                f[n, lay["z_delta"]] = np.clip((last24 - prev24) / s, -self.value_clip, self.value_clip)
            else:
                f[n, lay["z_last"]], f[n, lay["measured"]] = zp[-1], 1.0
                f[n, lay["z_delta"]] = zp[-1] - zp[-2] if len(zp) > 1 else 0.0
            f[n, lay["recency_state"]] = np.log1p(t - tp.max()) / np.log1p(168.0)
            f[n, lay["count_state"]] = np.log1p((tp > t - 24).sum()) / np.log(10.0)
        return f

    # ------------------------------------------------------------------ encode one split
    def encode(self, stays: pd.DataFrame, events: dict, lmk: pd.DataFrame, static: np.ndarray,
               horizons: list[int]) -> dict:
        stays = stays.sort_values("stay_id").reset_index(drop=True)
        ids = stays["stay_id"].values
        start_rel = (-stays.set_index("stay_id")["surgery_duration_h"]).astype(np.float32)
        lmk = lmk[lmk["stay_id"].isin(ids)].sort_values(["stay_id", "t_lmk"]).reset_index(drop=True)
        tok = self._event_table(events, ids)
        tok = tok[tok["t"] >= tok["stay_id"].map(start_rel) - self.preop_window_h]
        tok = tok[tok["t"] <= tok["stay_id"].map(lmk.groupby("stay_id")["t_lmk"].max()).fillna(-np.inf)]
        tok = tok.sort_values(["stay_id", "t"], kind="mergesort").reset_index(drop=True)
        cols = self._encode_columns(tok, start_rel)

        meds = events["meds"][events["meds"]["stay_id"].isin(ids)]
        drug_tab = pd.DataFrame({"stay_id": meds["stay_id"].values, "t": meds["t"].values,
                                 "node": meds["concept"].astype(str).map(NODE_INDEX).values,
                                 "lvl": dose_levels(meds, self.dose_cuts),
                                 "na": np.nan_to_num(meds["na_mmol"].values.astype(float))}).dropna(subset=["node"])
        drug_groups = dict(tuple(drug_tab.groupby("stay_id")))
        src_of = {s: STATE_SOURCE[s] for s in STATE_NODES}
        state_tab = pd.concat([events["labs"], events["io"]], ignore_index=True)
        state_tab = state_tab[state_tab["stay_id"].isin(ids) & state_tab["concept"].astype(str).isin(list(src_of.values()))]
        state_groups = dict(tuple(state_tab.groupby("stay_id")))
        sodium = events["labs"][events["labs"]["stay_id"].isin(ids)
                                & events["labs"]["concept"].astype(str).isin(["LAB_NA", "LAB_NA_BG"])]
        sodium_groups = {sid: g.sort_values("t") for sid, g in sodium.groupby("stay_id")}

        tok_groups = np.split(np.arange(len(tok)), np.searchsorted(tok["stay_id"].values, ids[1:]))
        lmk_groups = dict(tuple(lmk.groupby("stay_id")))
        fields = list(cols)
        out_tok = {k: [] for k in fields}
        tok_off, lmk_off, lmk_pos, node_feat, na_last, na_mask = [0], [0], [], [], [], []
        targets = {f"{f}_{h}": [] for f in LMK_FIELDS for h in horizons}
        meta = {"lmk_time": [], "lmk_mode": [], "lmk_id": [], "lmk_stay": []}
        n_truncated = 0
        for i, sid in enumerate(ids):
            idx = tok_groups[i]
            lg = lmk_groups[sid]
            times = cols["time"][idx]
            ins = np.searchsorted(times, lg["t_lmk"].values.astype(np.float32), side="right")
            n_ev, n_l = len(idx), len(lg)
            drop = max(0, n_ev + n_l - self.max_len)
            n_truncated += drop > 0
            keep_ev = idx[drop:]                                   # oldest events are dropped first
            ins = np.maximum(ins - drop, 0)
            seq_len = len(keep_ev) + n_l
            lmk_slots = ins + np.arange(n_l)
            is_lmk = np.zeros(seq_len, dtype=bool)
            is_lmk[lmk_slots] = True
            ev_slots = np.flatnonzero(~is_lmk)
            for k in fields:
                arr = np.zeros(seq_len, dtype=cols[k].dtype)
                arr[ev_slots] = cols[k][keep_ev]
                out_tok[k].append(arr)
            rolling = lg["mode"].values == "rolling"
            out_tok["concept"][-1][lmk_slots] = np.where(rolling, self.concept_vocab["[LMK_ROLLING]"],
                                                         self.concept_vocab["[LMK_BASELINE]"])
            out_tok["time"][-1][lmk_slots] = lg["t_lmk"].values.astype(np.float32)
            out_tok["node"][-1][lmk_slots] = N_NODES
            tok_off.append(tok_off[-1] + seq_len)
            lmk_off.append(lmk_off[-1] + n_l)
            lmk_pos.append(lmk_slots.astype(np.int32))

            dg = drug_groups.get(sid)
            drug = {} if dg is None else {int(n): (g["t"].values, g["lvl"].values, g["na"].values)
                                          for n, g in dg.groupby("node")}
            sg = state_groups.get(sid)
            state = {}
            if sg is not None:
                for sname, src in src_of.items():
                    g = sg[sg["concept"].astype(str) == src].sort_values("t")
                    if len(g):
                        v = g["value"].values.astype(float)
                        state[sname] = (g["t"].values, self._z(src, v).astype(np.float32), v)
            na_g = sodium_groups.get(sid)
            for r in lg.itertuples(index=False):
                node_feat.append(self._node_features(float(r.t_lmk), drug, state))
                past = na_g["t"].values <= r.t_lmk if na_g is not None else np.zeros(0, dtype=bool)
                na_last.append(float(na_g["value"].values[past][-1]) if past.any() else 140.0)
                na_mask.append(int(past.any()))
                meta["lmk_time"].append(r.t_lmk)
                meta["lmk_mode"].append(int(r.mode == "rolling"))
                meta["lmk_id"].append(r.lmk_id)
                meta["lmk_stay"].append(i)
                for f in LMK_FIELDS:
                    for h in horizons:
                        targets[f"{f}_{h}"].append(getattr(r, f"{f}_{h}"))

        pack = {f"tok_{k}": np.concatenate(v) for k, v in out_tok.items()}
        pack.update({
            "stay_id": ids.astype(np.int64),
            "subject_id": stays["subject_id"].values.astype(np.int64),
            "tok_offsets": np.asarray(tok_off, dtype=np.int64),
            "lmk_offsets": np.asarray(lmk_off, dtype=np.int64),
            "lmk_pos": np.concatenate(lmk_pos).astype(np.int32),
            "lmk_time": np.asarray(meta["lmk_time"], dtype=np.float32),
            "lmk_mode": np.asarray(meta["lmk_mode"], dtype=np.int8),
            "lmk_id": np.asarray(meta["lmk_id"], dtype=np.int32),
            "lmk_stay": np.asarray(meta["lmk_stay"], dtype=np.int64),
            "lmk_na_last": np.asarray(na_last, dtype=np.float32),
            "lmk_na_mask": np.asarray(na_mask, dtype=np.int8),
            "node_feat": np.stack(node_feat).astype(np.float32),
            "static": static.astype(np.float32),
            "n_truncated": np.asarray(n_truncated),
        })
        for k, v in targets.items():
            pack[f"y_{k}"] = np.asarray(v, dtype=np.float32)
        return pack
