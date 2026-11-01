import numpy as np

from nainsight.utils import load_json, save_json


def test_numpy_scalars_round_trip_in_json_artifacts(tmp_path):
    path = tmp_path / "nested" / "result.json"
    save_json({"passed": np.bool_(True), "failed": np.bool_(False),
               "count": np.int64(3), "score": np.float32(0.5)}, path)
    assert load_json(path) == {"passed": True, "failed": False, "count": 3, "score": 0.5}

