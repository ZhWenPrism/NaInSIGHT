import numpy as np
import pytest

from nainsight.data.dataset import validate_pack
from test_pack_offsets import pack_data


@pytest.mark.parametrize("field", [
    "static", "tok_time", "tok_value", "tok_value_mask", "tok_na_load",
    "lmk_time", "lmk_mode", "lmk_na_last", "lmk_na_mask", "node_feat",
])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_inputs_are_rejected(field, value):
    pack = pack_data()
    pack[field] = pack[field].astype(float)
    pack[field].flat[0] = value
    with pytest.raises(ValueError, match=field):
        validate_pack(pack)


def test_missing_outcome_labels_remain_supported():
    pack = pack_data()
    pack["y_risk_24"][:] = np.nan
    validate_pack(pack)
