import pytest

from nainsight.utils import load_json, save_json


def test_failed_serialization_preserves_previous_artifact(tmp_path):
    path = tmp_path / "result.json"
    save_json({"score": 0.75}, path)
    previous = path.read_bytes()
    with pytest.raises(TypeError):
        save_json({"score": 0.5, "unsupported": object()}, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    save_json({"score": 0.8}, path)
    assert load_json(path) == {"score": 0.8}


def test_failed_first_write_leaves_no_partial_artifact(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(TypeError):
        save_json({"unsupported": object()}, path)
    assert list(tmp_path.iterdir()) == []
