import pandas as pd

from nainsight.data.readers import read_source


def test_parquet_missing_values_do_not_become_subject_identifiers(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({"subject": ["001", None, "nan"], "value": [1.0, None, 3.0]}).to_parquet(path, index=False)
    spec = {"path": path.name, "format": "parquet", "columns": {"subject_key": "subject", "reading": "value"}}
    frame = read_source(spec, root=tmp_path)
    assert frame["subject_key"].iloc[0] == "001"
    assert pd.isna(frame["subject_key"].iloc[1])
    assert frame["subject_key"].iloc[2] == "nan"
    assert pd.isna(frame["reading"].iloc[1])
    assert frame["reading"].iloc[0] == "1.0"
