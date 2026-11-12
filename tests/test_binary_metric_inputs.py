import numpy as np
import pytest

from nainsight.train.metrics import binary_metrics


def test_binary_metrics_reject_fractional_labels():
    with pytest.raises(ValueError, match="binary labels must be 0 or 1"):
        binary_metrics(np.array([0.9, 1.0]), np.array([0.2, 0.8]))


def test_binary_metrics_reject_broadcasting_one_label_over_multiple_predictions():
    with pytest.raises(ValueError, match="aligned 1D arrays"):
        binary_metrics(np.array([0]), np.array([0.2, 0.8]))


def test_binary_metrics_keep_pairwise_missing_value_filtering():
    result = binary_metrics(np.array([0.0, np.nan, 1.0]), np.array([0.2, 0.8, np.nan]))
    assert result["n"] == 1
    assert result["n_pos"] == 0
    assert result["Brier"] == pytest.approx(0.04)
