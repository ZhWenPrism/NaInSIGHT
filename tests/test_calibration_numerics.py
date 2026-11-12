import numpy as np

from nainsight.train.metrics import binary_metrics


def test_binary_recalibration_avoids_overflow_on_nearly_tied_scores():
    with np.errstate(over="raise"):
        result = binary_metrics(np.array([0, 1, 0]), np.array([0.477, 0.47699, 0.48034]))
    assert np.isfinite(result["calibration_intercept"])
    assert np.isfinite(result["calibration_slope"])
