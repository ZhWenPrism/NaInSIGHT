import numpy as np
import pytest

from nainsight.model.nainsight import PriorPack


@pytest.mark.parametrize(("field", "bad", "match"), [
    ("n_drug", 1.5, "n_drug"),
    ("vectors", np.ones((3, 4)), "align with graph nodes"),
    ("reliability", [0.3], "align with graph nodes"),
    ("B_dev", np.zeros((1, 2, 2, 1)), "align with graph nodes and development lags"),
    ("A_dev", np.zeros((1, 2, 1)), "align with graph nodes and development lags"),
])
def test_incompatible_prior_archives_fail_at_loading(tmp_path, field, bad, match):
    graph = {"n_drug": 1, "node_names": ["drug", "state"], "reliability": [0.2, 0.3]}
    effects = {"B_dev": np.zeros((1, 2, 1, 1)), "A_dev": np.zeros((1, 1, 1)), "dev_lags": [6]}
    vectors = {"vectors": np.ones((2, 4))}
    for name, data in (("graph", graph), ("prior_matrices", effects), ("prior_vectors", vectors)):
        if field in data:
            data[field] = bad
        np.savez(tmp_path / f"{name}.npz", **data)
    with pytest.raises(ValueError, match=match):
        PriorPack.load(tmp_path)
