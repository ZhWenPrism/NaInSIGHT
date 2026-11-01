import numpy as np
import pytest

from nainsight.train.metrics import binary_metrics, ordinal_metrics, regression_metrics


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_group_still_has_a_brier_score(label):
    y = np.full(3, label)
    p = np.array([0.1, 0.3, 0.8])
    result = binary_metrics(y, p)
    assert result["Brier"] == pytest.approx(np.mean((y - p) ** 2))
    assert np.isnan(result["AUROC"]) and np.isnan(result["AUPRC"])


def test_one_observation_retains_regression_errors():
    result = regression_metrics(np.array([5.0]), np.array([7.0]))
    assert result["MAE"] == 2.0 and result["RMSE"] == 2.0 and result["bias"] == 2.0
    assert np.isnan(result["R2"])


def test_constant_prediction_has_defined_quadratic_kappa():
    result = ordinal_metrics(np.array([0, 1, 2]), np.tile([1.0, 0.0, 0.0], (3, 1)))
    assert result["weighted_kappa"] == pytest.approx(0.0)


def test_empty_regression_group_still_returns_missing_metrics():
    result = regression_metrics(np.array([]), np.array([]))
    assert result["n"] == 0 and np.isnan(result["MAE"])

