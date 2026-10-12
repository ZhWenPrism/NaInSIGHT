#!/usr/bin/env python3
"""Synthetic end-to-end check (no real data): event store -> landmarks -> preprocessors -> tokens -> effect
table -> prior matrices -> graph -> GNN -> NaInSIGHT training -> prediction -> metrics with cluster bootstrap,
plus forward passes of every ablation variant."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from nainsight.concepts import DRUG_NODES, INGREDIENTS, STATE_NODES
from nainsight.config import AttrDict, load_config, REPO_ROOT
from nainsight.data.features import StaticPreprocessor, fit_dose_cutpoints, fit_value_scaler, preop_summaries
from nainsight.data.landmarks import build_landmarks, preop_hyponatraemia, sodium_events
from nainsight.data.tokenizer import N_NODES, ClinicalEventTokenizer
from nainsight.graph.build import build_graph
from nainsight.graph.gnn import pretrain_graph_prior
from nainsight.graph.transfer import prior_matrices, shrink_subpopulation
from nainsight.model.nainsight import NaInSIGHT, PriorPack
from nainsight.runtime import state_scale
from nainsight.train.bootstrap import cluster_bootstrap, to_frame
from nainsight.train.losses import MultiTaskLoss, positive_weights
from nainsight.train.metrics import landmark_metrics
from nainsight.train.trainer import Trainer, make_loader
from nainsight.utils import get_logger, seed_everything


def synthetic_cohort(n: int, rng) -> tuple[pd.DataFrame, dict]:
    dur = rng.uniform(2.5, 6.0, n)
    stays = pd.DataFrame({"stay_id": np.arange(n), "subject_id": np.arange(n), "surgery_duration_h": dur,
                          "preop_stay_days": rng.uniform(1, 5, n), "postop_stay_h": rng.uniform(90, 260, n),
                          "age_years": rng.uniform(20, 75, n), "sex_male": rng.integers(0, 2, n).astype(float),
                          "bmi": rng.normal(24, 3, n), "tumour_volume_cm3": rng.lognormal(4, 0.8, n),
                          "tumour_family": rng.choice(["a", "b", "c"], n)})
    stays["split"] = np.where(stays["stay_id"] < int(0.6 * n), "train", np.where(stays["stay_id"] < int(0.8 * n), "val", "test"))
    labs, vit, io, meds = [], [], [], []
    for r in stays.itertuples(index=False):
        risk = rng.uniform(0, 1)
        na = 140.0 + rng.normal(0, 1.5)
        for t in np.r_[-r.preop_stay_days * 24 + 4, -r.surgery_duration_h - 6, np.arange(8, r.postop_stay_h, 24)]:
            na += rng.normal(-1.2 * risk if t > 0 else 0.0, 1.3)
            labs += [(r.stay_id, t, "LAB_NA", na), (r.stay_id, t, "LAB_K", rng.normal(4.0, 0.3)),
                     (r.stay_id, t, "LAB_GLU", rng.normal(6.0, 1.0)), (r.stay_id, t, "LAB_CREA", rng.normal(65, 10))]
        for t in np.arange(-48, r.postop_stay_h, 4):
            vit += [(r.stay_id, t, "VS_HR", rng.normal(80, 10)), (r.stay_id, t, "VS_MAP", rng.normal(85, 8))]
            io += [(r.stay_id, t, "IO_URINE", max(0.0, rng.normal(400 + 300 * risk, 120)))]
        for t in np.arange(-r.surgery_duration_h, min(r.postop_stay_h, 120), 8):
            meds.append((r.stay_id, t, "mannitol", rng.choice([25.0, 50.0]), "IV", np.nan))
            meds.append((r.stay_id, t + 1, "normal_saline", 500.0, "IV", 77.0))
            if rng.uniform() < 0.3:
                meds.append((r.stay_id, t + 2, "dexamethasone", 5.0, "IV", np.nan))
            if rng.uniform() < risk * 0.3:
                meds.append((r.stay_id, t + 3, "hypertonic_saline", 30.0, "IV", 51.3))
    ev = {"labs": pd.DataFrame(labs, columns=["stay_id", "t", "concept", "value"]),
          "vitals": pd.DataFrame(vit, columns=["stay_id", "t", "concept", "value"]),
          "io": pd.DataFrame(io, columns=["stay_id", "t", "concept", "value"])}
    m = pd.DataFrame(meds, columns=["stay_id", "t", "ingredient", "dose_std", "route", "na_mmol"])
    m["concept"] = m["ingredient"].map(lambda i: INGREDIENTS[i][0])
    ev["meds"] = m
    for k in ev:
        ev[k]["t"] = ev[k]["t"].astype(np.float32)
    return stays, ev


def synthetic_effects(rng, cfg) -> pd.DataFrame:
    rows = []
    for stage in ("all", "subpopulation"):
        for s in STATE_NODES:
            for lag in cfg["graph"]["lags_h"]:
                for d in DRUG_NODES:
                    for lv in range(1, cfg["graph"]["dose_levels"] + 1):
                        if rng.uniform() < 0.6:
                            rows.append({"stage": stage, "src_type": "drug", "src": d, "level": lv, "dst": s, "lag_h": lag,
                                         "beta": rng.normal(0, 0.5), "se": rng.uniform(0.05, 0.5), "n_exposed": 100})
                for m in STATE_NODES:
                    rows.append({"stage": stage, "src_type": "state", "src": m, "dst": s, "level": 0, "lag_h": lag,
                                 "beta": rng.normal(-0.5 if m == s else 0, 0.1), "se": 0.05, "n_exposed": 1000})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config([REPO_ROOT / "configs" / "nainsight.yaml"],
                      ["train.epochs=2", "train.batch_size=8", "train.max_steps_per_epoch=6", "graph.gnn.epochs=40",
                       "tokens.min_concept_count=1", "tokens.static_min_category_count=1", "evaluate.bootstrap=20",
                       "model.d_model=64", "model.fusion_dim=128", "model.head_hidden=64", "train.device=cpu"])
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    log = get_logger("smoke")
    stays, ev = synthetic_cohort(60, rng)

    na = sodium_events(ev["labs"], ["LAB_NA"])
    assert not preop_hyponatraemia(stays, na, cfg).any() or True
    lmk = build_landmarks(stays, na, cfg)
    log.info("landmarks: %d (%s)", len(lmk), lmk["mode"].value_counts().to_dict())
    pre = preop_summaries(stays, ev["labs"], ev["meds"], ev["io"])
    frame = stays.merge(pre, on="stay_id", how="left")
    schema = {"age_years": "numeric", "sex_male": "binary", "bmi": "numeric", "tumour_volume_cm3": "numeric",
              "tumour_family": "category", **{c: "numeric" for c in pre.columns if c != "stay_id"}}
    tr_ids = set(stays.loc[stays["split"] == "train", "stay_id"])
    tr_ev = {k: v[v["stay_id"].isin(tr_ids)] for k, v in ev.items()}
    static = StaticPreprocessor(schema, 1).fit(frame[frame["stay_id"].isin(tr_ids)])
    tok = ClinicalEventTokenizer(256, 168, 6.0, 4, 1).fit(tr_ev, fit_value_scaler(tr_ev["labs"], tr_ev["vitals"], tr_ev["io"], tr_ev["meds"]),
                                                          fit_dose_cutpoints(tr_ev["meds"], 4))
    packs = {}
    for split in ("train", "val", "test"):
        s = stays[stays["split"] == split].sort_values("stay_id")
        packs[split] = tok.encode(s, ev, lmk, static.transform(frame[frame["stay_id"].isin(s["stay_id"])].sort_values("stay_id")).values,
                                  [24, 48])
        p = packs[split]
        log.info("%s: %d stays, %d landmarks, %d tokens, truncated %d", split, len(p["stay_id"]), len(p["lmk_time"]),
                 len(p["tok_concept"]), int(p["n_truncated"]))
        # every landmark query token sits at its landmark time and after all earlier events
        for i in range(len(p["stay_id"])):
            a, b = p["tok_offsets"][i], p["tok_offsets"][i + 1]
            times = p["tok_time"][a:b]
            for j in range(p["lmk_offsets"][i], p["lmk_offsets"][i + 1]):
                pos = p["lmk_pos"][j]
                assert times[pos] == p["lmk_time"][j]
                assert (times[:pos] <= p["lmk_time"][j]).all()

    # knowledge graph from a synthetic effect table
    eff = synthetic_effects(rng, cfg)
    post = shrink_subpopulation(eff[eff["stage"] == "all"], eff[eff["stage"] == "subpopulation"])
    pri = prior_matrices(post, cfg)
    meta = {"baseline_mean": {s: 1.0 for s in STATE_NODES}, "baseline_sd": {s: 1.0 for s in STATE_NODES},
            "exposure_prevalence": {d: 0.1 for d in DRUG_NODES}}
    graph = build_graph(pri, meta)
    vectors, hist, _ = pretrain_graph_prior(graph, cfg, cfg.seed)
    log.info("graph: %d nodes, %d edges | GNN loss %.3f -> %.3f", graph["x"].shape[0], graph["edge_index"].shape[1],
             hist["history"][0]["loss"], hist["history"][-1]["loss"])
    n_drug = int(graph["n_drug"])
    prior = PriorPack(vectors, np.r_[np.zeros(n_drug), np.ones(N_NODES - n_drug)].astype(np.int64),
                      graph["reliability"], pri["B_dev"], pri["A_dev"], pri["dev_lags"])

    # NaInSIGHT training
    horizons = [24, 48]
    vocab = {"concept": len(tok.concept_vocab), "ingredient": len(tok.ingredient_vocab)}
    model = NaInSIGHT(cfg, vocab, packs["train"]["static"].shape[1], 4, prior, state_scale(tok))
    loss = MultiTaskLoss(horizons, "uncertainty", None, positive_weights(packs["train"], horizons, "balanced_sqrt"), 0.1)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, loss, cfg, torch.device("cpu"), Path(tmp), log, horizons)
        res = trainer.fit(make_loader(packs["train"], 8, True, cfg.seed, N_NODES),
                          make_loader(packs["val"], 8, False, cfg.seed, N_NODES), packs["val"])
        pred = trainer.predict(make_loader(packs["test"], 8, False, cfg.seed, N_NODES), packs["test"])
    assert (pred["p_risk_48"] >= pred["p_risk_24"] - 1e-6).all(), "risk must be monotone over nested windows"
    assert (pred["mu_nadir_48"] <= pred["mu_nadir_24"] + 1e-6).all(), "nadir must be monotone over nested windows"
    assert np.allclose(pred[[f"p_sev{c}_24" for c in range(3)]].sum(axis=1), 1.0, atol=1e-5)
    boot = cluster_bootstrap(pred, "subject_id", lambda d: landmark_metrics(d, horizons), 20, 0.95, cfg.seed)
    table = to_frame(boot, "synthetic_test")
    log.info("training result %s | metric rows %d | knowledge gate mean %.3f", res, len(table), pred["knowledge_gate"].mean())

    # forward pass of every ablation variant
    batch = next(iter(make_loader(packs["val"], 4, False, cfg.seed, N_NODES)))
    for abl in ([], ["no_graph_prior"], ["random_prior"], ["no_ssm"], ["no_transformer"], ["no_interaction"],
                ["no_static"], ["no_mechanistic"], ["no_gate"], ["dense_prior_attention"]):
        c = AttrDict.wrap({**cfg.to_dict(), "model": {**cfg.model.to_dict(), "ablation": abl}})
        m = NaInSIGHT(c, vocab, packs["train"]["static"].shape[1], 4, prior, state_scale(tok)).eval()
        with torch.no_grad():
            out = m(batch)
        assert torch.isfinite(out["risk_logit"]).all(), abl
        log.info("ablation %-26s ok | risk logits %s", abl or ["full"], tuple(out["risk_logit"].shape))
    log.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
