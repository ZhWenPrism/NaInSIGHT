from unittest.mock import patch

import pandas as pd
import pytest

from nainsight.data.events import IngredientMatcher, _expand_orders


@pytest.mark.parametrize("interval", [0, -2, float("inf"), float("nan"), "invalid", None, True])
def test_invalid_frequency_intervals_fail_before_reading_sources(interval):
    windows = pd.DataFrame({"subject_key": ["synthetic"], "win_end": pd.to_datetime(["2020-01-03"])})
    med = {"ingredients": [{"ingredient": "mannitol", "pattern": "mannitol"}],
           "orders": {"frequency_hours": {"bad": interval}}}
    with patch("nainsight.data.events.iter_source", return_value=iter([])) as reader:
        with pytest.raises(ValueError, match="frequency"):
            _expand_orders({}, med, windows, IngredientMatcher(med))
    reader.assert_not_called()


def test_valid_repeating_and_once_orders_preserve_administration_times():
    windows = pd.DataFrame({"subject_key": ["synthetic"], "win_end": pd.to_datetime(["2020-01-03"])})
    med = {"ingredients": [{"ingredient": "mannitol", "pattern": "mannitol"}],
           "orders": {"frequency_hours": {"q6": "6", "single": "once"}}}
    chunk = pd.DataFrame({"subject_key": ["synthetic"]*2, "spec": [""]*2, "name": ["mannitol"]*2,
                          "start_time": ["2020-01-02"]*2, "end_time": ["2020-01-02 12:00"]*2,
                          "frequency": ["q6", "single"], "value": ["1"]*2, "unit": ["g"]*2,
                          "route": ["IV"]*2})
    with patch("nainsight.data.events.iter_source", return_value=iter([chunk])):
        result = _expand_orders({}, med, windows, IngredientMatcher(med))
    assert result["time"].tolist() == list(pd.to_datetime([
        "2020-01-02", "2020-01-02 06:00", "2020-01-02 12:00", "2020-01-02"], format="mixed"))
