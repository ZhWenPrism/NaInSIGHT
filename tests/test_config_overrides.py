import pytest

from nainsight.config import apply_overrides


@pytest.mark.parametrize("item", ["=2", ".a=2", "a.=2", "a..b=2", "   =2", "train.epochs.next=2"])
def test_invalid_override_path_is_rejected(item):
    cfg = {"train": {"epochs": 3}}
    with pytest.raises(ValueError, match="override"):
        apply_overrides(cfg, [item])
    assert cfg == {"train": {"epochs": 3}}


def test_valid_overrides_preserve_original_and_parse_values():
    cfg = {"train": {"epochs": 3}}
    result = apply_overrides(cfg, ["train.epochs=5", "new.values=[1, 2]", "new.active=true"])
    assert result == {"train": {"epochs": 5}, "new": {"values": [1, 2], "active": True}}
    assert cfg == {"train": {"epochs": 3}}
