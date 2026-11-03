import numpy as np
import pytest

from nainsight.concepts import DRUG_NODES, STATE_NODES
from nainsight.config import load_config
from nainsight.graph.build import build_graph, N_SUMMARY
from nainsight.graph.gnn import pretrain_graph_prior


def empty_graph(with_signal=False):
    d, s = len(DRUG_NODES), len(STATE_NODES)
    prior = {"B": np.full((d, 2, 3, s), np.nan), "B_se": np.full((d, 2, 3, s), np.nan),
             "A": np.full((3, s, s), np.nan), "A_se": np.full((3, s, s), np.nan)}
    if with_signal:
        prior["B"][0, 0, 0, 0] = .5
        prior["B_se"][0, 0, 0, 0] = .1
    return build_graph(prior, {})


def test_empty_graph_preserves_edge_feature_width():
    graph = empty_graph()
    assert graph["edge_attr"].shape == (0, 2 * 3 + N_SUMMARY)
    assert graph["edge_index"].shape == (2, 0)
    assert np.isfinite(graph["x"]).all()
    assert np.count_nonzero(graph["reliability"]) == 0


def test_pretraining_without_drug_edges_reports_missing_signal():
    graph = empty_graph()
    cfg = load_config(["configs/nainsight.yaml"]).to_dict()
    with pytest.raises(ValueError, match="drug-to-state"):
        pretrain_graph_prior(graph, cfg, seed=0)


def test_pretraining_with_observed_edges_still_produces_finite_vectors():
    graph = empty_graph(with_signal=True)
    cfg = load_config(["configs/nainsight.yaml"]).to_dict()
    cfg["graph"]["gnn"].update(epochs=2, hidden=16, out_dim=4, layers=1, dropout=0.)
    vectors, log, state = pretrain_graph_prior(graph, cfg, seed=7)
    assert graph["edge_attr"].shape == (2, 10)
    assert vectors.shape == (len(graph["node_names"]), 4)
    assert np.isfinite(vectors).all() and log["history"] and state
