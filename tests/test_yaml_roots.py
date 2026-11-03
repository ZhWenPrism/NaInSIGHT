import pytest

from nainsight.config import load_config, read_yaml


@pytest.mark.parametrize("content", ["false", "true", "0", "42", "[]", "[1, 2]", "text"])
def test_nonmapping_yaml_root_reports_file(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping") as error:
        read_yaml(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("content", ["", "# empty configuration", "{}", "null"])
def test_empty_yaml_configuration_remains_supported(tmp_path, content):
    path = tmp_path / "empty.yaml"
    path.write_text(content, encoding="utf-8")
    assert read_yaml(path) == {}


def test_valid_nested_values_are_preserved(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  enabled: false\n  seeds: [1, 2]\n  epochs: 0\n", encoding="utf-8")
    assert read_yaml(path) == {"train": {"enabled": False, "seeds": [1, 2], "epochs": 0}}


def test_invalid_override_file_cannot_be_silently_ignored(tmp_path):
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("train: {epochs: 5}", encoding="utf-8")
    override.write_text("false", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config([base, override])
