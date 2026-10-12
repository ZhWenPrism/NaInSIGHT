"""Helpers shared by the training and evaluation entry points."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .concepts import STATE_NODES, STATE_SOURCE
from .data.tokenizer import ClinicalEventTokenizer
from .model.nainsight import NaInSIGHT, PriorPack


def state_scale(tok: ClinicalEventTokenizer) -> np.ndarray:
    """Development-cohort spread of each state node (raw units) used to express mechanistic projections in z."""
    out = []
    for s in STATE_NODES:
        out.append(tok.urine_ref[1] if s == "URINE" else tok.scaler.get(STATE_SOURCE[s], (0.0, 1.0))[1])
    return np.asarray(out, dtype=np.float32)


def load_prior(reference_dir: Path, cfg: dict) -> PriorPack | None:
    if "no_graph_prior" in set(cfg["model"].get("ablation") or []):
        return None
    if not (Path(reference_dir) / "prior_vectors.npz").exists():
        raise FileNotFoundError("knowledge prior not found: run build_reference_graph.py and pretrain_graph_prior.py, "
                                "or train with model.ablation=[no_graph_prior]")
    return PriorPack.load(reference_dir)


def build_model(cfg: dict, preproc: dict, n_static: int, prior: PriorPack | None) -> NaInSIGHT:
    tok = ClinicalEventTokenizer.from_state(preproc["tokenizer"])
    return NaInSIGHT(cfg, {"concept": len(tok.concept_vocab), "ingredient": len(tok.ingredient_vocab)}, n_static,
                    tok.dose_levels, prior, state_scale(tok), n_horizons=len(cfg["outcome"]["horizons_h"]))
