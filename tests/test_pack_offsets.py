import numpy as np
import pytest

from nainsight.data.dataset import TOKEN_FIELDS, load_pack


def pack_data():
    return {
        "stay_id": np.array([11, 21]),
        "subject_id": np.array([1, 2]),
        "static": np.zeros((2, 1)),
        "tok_offsets": np.array([0, 2, 5]),
        "lmk_offsets": np.array([0, 1, 2]),
        **{f"tok_{field}": np.zeros(5) for field in TOKEN_FIELDS},
        "lmk_pos": np.array([1, 2]),
        "lmk_time": np.array([0., 1.]),
        "lmk_mode": np.zeros(2, dtype=np.int8),
        "lmk_na_last": np.zeros(2),
        "lmk_na_mask": np.zeros(2),
        "node_feat": np.zeros((2, 3, 4)),
        "y_risk_24": np.zeros(2),
    }


@pytest.mark.parametrize(("field", "values"), [
    ("tok_offsets", [0, 4, 3]),
    ("tok_offsets", [1, 2, 5]),
    ("tok_offsets", [0, 2, 4]),
    ("tok_offsets", [0., 2., 5.]),
    ("lmk_offsets", [0, 2]),
    ("lmk_offsets", [0, 1, 3]),
    ("lmk_pos", [2, 2]),
    ("lmk_pos", [-1, 2]),
    ("tok_time", [0., 1., 2., 3.]),
    ("subject_id", [1]),
    ("subject_id", [[1], [2]]),
    ("static", [[1.0]]),
    ("static", [1.0, 2.0]),
])
def test_load_pack_rejects_misaligned_stay_slices(tmp_path, field, values):
    pack = pack_data()
    pack[field] = np.array(values)
    path = tmp_path / "bad.npz"
    np.savez(path, **pack)

    with pytest.raises(ValueError, match="tok_offsets" if field == "tok_time" else field):
        load_pack(path)


def test_load_pack_accepts_valid_stay_slices(tmp_path):
    path = tmp_path / "valid.npz"
    np.savez(path, **pack_data())

    pack = load_pack(path)

    assert pack["tok_offsets"].tolist() == [0, 2, 5]
    assert pack["lmk_pos"].tolist() == [1, 2]
