#!/usr/bin/env python3
"""Self-supervised pre-training of the edge-conditioned GNN -> knowledge prior vectors (+ random control)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from nainsight.graph.gnn import pretrain_graph_prior, random_prior_like
from nainsight.utils import get_logger, save_json, seed_everything
from nainsight.workspace import common_parser, load_run_context


def main() -> None:
    ap = common_parser(__doc__)
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    seed_everything(cfg.seed)
    out = ws.reference
    log = get_logger("gnn", out / "pretrain_graph_prior.log")
    graph = dict(np.load(out / "graph.npz", allow_pickle=False))
    vectors, hist, state = pretrain_graph_prior(graph, cfg, cfg.seed, device="cpu")
    np.savez(out / "prior_vectors.npz", vectors=vectors, random=random_prior_like(vectors, cfg.seed),
             node_names=graph["node_names"])
    torch.save(state, out / "gnn_encoder.pt")
    save_json(hist, out / "gnn_history.json")
    log.info("knowledge prior vectors: %s", vectors.shape)


if __name__ == "__main__":
    main()
