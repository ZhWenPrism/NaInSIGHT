import numpy as np
import pandas as pd
import pytest

from nainsight.train.bootstrap import cluster_bootstrap


def _stat(frame):
    return {"all": {"mean": frame["value"].mean()}}


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"patient": [], "value": []}),
    pd.DataFrame({"value": [1]}),
    pd.DataFrame({"patient": ["p", None], "value": [1, 9]}),
])
def test_invalid_cluster_inputs_fail_before_evaluating_metrics(frame):
    calls = []
    with pytest.raises(ValueError, match="cluster ID"):
        cluster_bootstrap(frame, "patient", lambda x: calls.append(x), n_boot=2)
    assert not calls


@pytest.mark.parametrize("options", [{"n_boot": -1}, {"n_boot": 1.5}, {"n_boot": True},
                                     {"ci": 0}, {"ci": 1}, {"ci": float("nan")}])
def test_invalid_bootstrap_settings_are_rejected(options):
    with pytest.raises(ValueError):
        cluster_bootstrap(pd.DataFrame({"patient": ["p"], "value": [1]}), "patient", _stat, **options)


def test_unused_categories_do_not_create_empty_patient_draws():
    frame = pd.DataFrame({"patient": pd.Categorical(["a", "a"], categories=["a", "unused"]),
                          "value": [1., 3.]})
    seen = []

    def statistic(sample):
        seen.append(len(sample))
        return _stat(sample)

    result = cluster_bootstrap(frame, "patient", statistic, n_boot=20, seed=3)
    assert seen == [2] * 21
    assert result["all"]["mean"] == (2., 2., 2.)


def test_zero_draws_keeps_point_estimates_without_an_interval():
    result = cluster_bootstrap(pd.DataFrame({"patient": ["p"], "value": [3]}),
                               "patient", _stat, n_boot=0)["all"]["mean"]
    assert result[0] == 3 and np.isnan(result[1:]).all()
