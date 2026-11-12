import numpy as np
import pytest

from nainsight.train.metrics import regression_metrics


def test_regression_errors_do_not_wrap_on_unsigned_inputs():
    result = regression_metrics(np.array([140, 120], dtype=np.uint8),
                                np.array([120, 120], dtype=np.uint8))
    assert result == pytest.approx({"n": 2, "MAE": 10, "RMSE": np.sqrt(200), "R2": -1, "bias": -10})


def test_regression_metrics_reject_unaligned_inputs():
    with pytest.raises(ValueError, match="aligned 1D arrays"):
        regression_metrics(np.array([140]), np.array([120, 130]))
