import pandas as pd
import pytest
from nainsight.data.readers import read_source


@pytest.mark.parametrize("fmt", ["csv", "csv_headerless", "xlsx"])
def test_literal_identifiers_survive_table_readers(tmp_path, fmt):
    path = tmp_path / ("source.xlsx" if fmt == "xlsx" else "source.csv")
    frame = pd.DataFrame({"identifier": ["001", "NA", "nan", None, "NULL"]})
    if fmt == "xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False, header=fmt != "csv_headerless")
    spec = {"path": path.name, "format": fmt,
            "columns": {"subject_key": 0 if fmt == "csv_headerless" else "identifier"}}
    result = read_source(spec, root=tmp_path)["subject_key"]
    assert result.iloc[:3].tolist() == ["001", "NA", "nan"]
    assert pd.isna(result.iloc[3])
    assert result.iloc[4] == "NULL"
