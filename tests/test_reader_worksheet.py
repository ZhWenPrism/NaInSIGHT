import pandas as pd
import pytest

from nainsight.data.readers import iter_source


def test_source_selects_configured_excel_worksheet(tmp_path):
    path = tmp_path / "source.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as book:
        pd.DataFrame({"identifier": ["001"]}).to_excel(book, sheet_name="First", index=False)
        pd.DataFrame({"identifier": ["002", "003"]}).to_excel(book, sheet_name="Events", index=False)
    spec = {"path": path.name, "format": "xlsx", "columns": {"key": "identifier"}}
    assert [chunk["key"].tolist() for chunk in iter_source(spec, root=tmp_path)] == [["001"]]
    for sheet in ("Events", 1):
        chunks = iter_source({**spec, "sheet_name": sheet}, root=tmp_path, chunksize=1)
        assert [chunk["key"].tolist() for chunk in chunks] == [["002"], ["003"]]
    with pytest.raises(ValueError, match="sheet_name"):
        list(iter_source({**spec, "sheet_name": None}, root=tmp_path))
