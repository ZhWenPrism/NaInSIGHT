import numpy as np
import pytest

from nainsight.train.losses import positive_weights


def test_misspelled_weight_mode_cannot_fall_back_to_sqrt():
    pack = {"y_risk_24": np.array([0., 0., 0., 0., 1., np.nan])}
    with pytest.raises(ValueError, match="positive class weighting"):
        positive_weights(pack, [24], "balancd")
    for mode, expected in [("none", 1.), ("balanced", 4.), ("balanced_sqrt", 2.)]:
        assert positive_weights(pack, [24], mode) == [expected]
