import numpy as np
import pytest

from nainsight.train.metrics import ordinal_metrics


@pytest.mark.parametrize("label", [1.5, 3])
def test_ordinal_metrics_reject_invalid_observed_labels(label):
    with pytest.raises(ValueError, match="severity labels must be 0, 1 or 2"):
        ordinal_metrics(np.array([0, label]), np.array([[0.9, 0.1, 0], [0.1, 0.2, 0.7]]))


def test_ordinal_metrics_reject_extra_class_column():
    with pytest.raises(ValueError, match="one row per label and three columns"):
        ordinal_metrics(np.array([0, 1]), np.array([[0.9, 0.1, 0, 0], [0, 0.1, 0, 0.9]]))


def test_ordinal_metrics_keep_pairwise_missing_value_filtering():
    y = np.array([np.nan, 0, 1, 2])
    scores = np.array([[0.1, 0.2, 0.7], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, np.nan, 1.0]])
    result = ordinal_metrics(y, scores)
    assert result["n"] == 2
    assert result["weighted_kappa"] == 1.0
