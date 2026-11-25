import pandas as pd
import pytest
from nainsight.data import readers


def test_closing_a_partial_source_iterator_releases_its_csv_handle(tmp_path, monkeypatch):
    path = tmp_path / "source.csv"
    path.write_text("value\n1\n2\n", encoding="utf-8")
    reader = pd.read_csv(path, dtype=str, chunksize=1)
    handle = reader.handles.handle
    monkeypatch.setattr(readers.pd, "read_csv", lambda *args, **kwargs: reader)
    stream = readers.iter_source({"path": str(path), "columns": {"canonical": "value"}}, chunksize=1)
    try:
        assert next(stream)["canonical"].tolist() == ["1"]
        assert not handle.closed
        stream.close()
        assert handle.closed
    finally:
        stream.close()
        reader.close()


def test_mapping_failure_also_closes_the_reader(tmp_path, monkeypatch):
    path = tmp_path / "source.csv"
    path.write_text("other\n1\n2\n", encoding="utf-8")
    reader = pd.read_csv(path, dtype=str, chunksize=1)
    handle = reader.handles.handle
    monkeypatch.setattr(readers.pd, "read_csv", lambda *args, **kwargs: reader)
    stream = readers.iter_source({"path": str(path), "columns": {"canonical": "missing"}}, chunksize=1)
    try:
        with pytest.raises(KeyError):
            next(stream)
        assert handle.closed
    finally:
        stream.close()
        reader.close()
