import pytest

from nainsight.data.readers import source_files


def test_unmatched_source_glob_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no source files"):
        source_files({"path_glob": "*.csv"}, tmp_path)


def test_excluding_every_source_file_is_an_error(tmp_path):
    (tmp_path / "excluded.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no source files"):
        source_files({"path_glob": "*.csv", "exclude_pattern": "excluded"}, tmp_path)


def test_source_glob_retains_sorted_matching_files(tmp_path):
    for name in ("b.csv", "a.csv"):
        (tmp_path / name).write_text("value\n1\n", encoding="utf-8")
    assert source_files({"path_glob": "*.csv"}, tmp_path) == [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]

