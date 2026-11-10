import pandas as pd
import pytest

from nainsight.data.readers import iter_source


@pytest.mark.parametrize("chunksize", [0, -1, 1.5, True])
def test_invalid_chunk_sizes_fail_before_reading_a_source(tmp_path, monkeypatch, chunksize):
    calls = []
    def read_excel(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame({"original": ["a", "b"]})
    monkeypatch.setattr(pd, "read_excel", read_excel)
    path = tmp_path / "events.xlsx"
    path.touch()
    spec = {"path": path.name, "format": "xlsx", "columns": {"subject_key": "original"}}
    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        list(iter_source(spec, root=tmp_path, chunksize=chunksize))
    assert not calls
