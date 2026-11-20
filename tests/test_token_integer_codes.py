import numpy as np
import pytest
from nainsight.data.dataset import LONG_FIELDS, validate_pack
from test_pack_offsets import pack_data


@pytest.mark.parametrize("field", sorted(LONG_FIELDS))
@pytest.mark.parametrize("value", [.5, -1., np.nan, float(2**63)])
def test_invalid_codes_are_rejected_before_collator_conversion(field, value):
    pack = pack_data()
    pack["tok_" + field][0] = value
    with pytest.raises(ValueError, match="integer codes"):
        validate_pack(pack)


def test_integral_float_codes_remain_supported():
    validate_pack(pack_data())
