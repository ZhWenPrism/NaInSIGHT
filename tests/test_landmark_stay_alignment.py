import numpy as np
import pytest

from nainsight.data.dataset import validate_pack
from test_pack_offsets import pack_data


@pytest.mark.parametrize("indices", [[1, 0], [-1, 1], [0, 2], [0], [0., 1.], [[0], [1]]])
def test_landmarks_cannot_be_assigned_to_a_different_stay(indices):
    pack = pack_data()
    pack["lmk_stay"] = np.asarray(indices)
    with pytest.raises(ValueError, match="lmk_stay"):
        validate_pack(pack)


def test_landmark_stay_indices_are_required():
    pack = pack_data()
    pack.pop("lmk_stay", None)
    with pytest.raises(ValueError, match="lmk_stay"):
        validate_pack(pack)


def test_integer_landmark_indices_preserve_valid_stay_mapping():
    pack = pack_data()
    pack["lmk_stay"] = np.array([0, 1], dtype=np.uint64)
    validate_pack(pack)
