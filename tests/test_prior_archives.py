import numpy as np
import pytest

from nainsight.model.nainsight import PriorPack


@pytest.mark.parametrize("key", ["vectors", "missing"])
def test_prior_loader_closes_all_archives(tmp_path, monkeypatch, key):
    np.savez(tmp_path / "graph.npz", n_drug=1, node_names=["drug", "state"], reliability=[0.1, 0.8])
    np.savez(tmp_path / "prior_matrices.npz", B_dev=np.zeros((1, 1, 1, 1)),
             A_dev=np.zeros((1, 1, 1)), dev_lags=[24])
    np.savez(tmp_path / "prior_vectors.npz", vectors=np.ones((2, 3)))
    opened = []
    load = np.load

    def record(*args, **kwargs):
        archive = load(*args, **kwargs)
        opened.append(archive)
        return archive

    monkeypatch.setattr(np, "load", record)
    if key == "missing":
        with pytest.raises(KeyError, match="missing"):
            PriorPack.load(tmp_path, key=key)
    else:
        prior = PriorPack.load(tmp_path, key=key)
        np.testing.assert_array_equal(prior.vectors, np.ones((2, 3)))
        np.testing.assert_array_equal(prior.node_type, [0, 1])

    assert len(opened) == 3
    assert all(archive.zip is None for archive in opened)
