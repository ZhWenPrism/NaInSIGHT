import numpy as np
import pytest

from nainsight.graph.transfer import interpolate_lags


@pytest.mark.parametrize("ref", [[12., 6.], [6., 6.], [6., np.nan], [6., np.inf], [[6., 12.]], []])
def test_invalid_reference_lag_grid_is_rejected(ref):
    with pytest.raises(ValueError, match="ref_lags"):
        interpolate_lags(np.array([[6., 12.]]), ref, [9.])


def test_empty_destination_grid_preserves_leading_dimensions():
    assert interpolate_lags(np.ones((2, 3)), [0., 6., 12.], []).shape == (2, 0)


def test_interpolation_keeps_query_order_and_finite_neighbours():
    values = np.array([[0., np.nan, 24.], [np.nan, np.nan, np.nan]])
    out = interpolate_lags(values, [0., 12., 24.], [18., 6., 30., -1.])
    np.testing.assert_allclose(out[0], [18., 6., 24., 0.])
    assert np.isnan(out[1]).all()
