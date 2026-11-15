import pandas as pd
import pytest

from nainsight.concepts import DRUG_NODES, STATE_NODES
from nainsight.graph.transfer import prior_matrices, shrink_subpopulation


def effects():
    return pd.DataFrame([dict(src_type="drug", src=DRUG_NODES[0], level=1,
                              dst=STATE_NODES[0], lag_h=6, beta=1.0, se=0.5, n_exposed=20)])


def test_effect_merge_requires_a_unique_key_in_each_population():
    one = effects()
    duplicate = pd.concat([one, one.assign(beta=2.0)], ignore_index=True)
    for left, right in ((one, duplicate), (duplicate, one)):
        with pytest.raises(pd.errors.MergeError, match="not unique"):
            shrink_subpopulation(left, right)
    assert len(shrink_subpopulation(one, one)) == 1


def test_duplicate_effect_rows_cannot_overwrite_a_tensor_cell():
    one = effects()
    cfg = {"graph": {"dose_levels": 1, "lags_h": [6], "dev_lags_h": [6]}}
    assert prior_matrices(one, cfg)["B"][0, 0, 0, 0] == 1.0
    with pytest.raises(ValueError, match="effect rows must be unique"):
        prior_matrices(pd.concat([one, one.assign(beta=2.0)], ignore_index=True), cfg)
