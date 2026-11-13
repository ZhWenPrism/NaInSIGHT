import pandas as pd

from nainsight.data.timeparse import combine_date_time


def test_only_missing_clock_times_receive_the_default():
    dates = pd.Series(["2026-01-02"] * 6)
    times = pd.Series(["０８：３０", None, "  ", "NaN", "bad time", "26:10"])
    actual = combine_date_time(dates, times)
    expected = pd.Series(pd.to_datetime([
        "2026-01-02 08:30", "2026-01-02 12:00", "2026-01-02 12:00",
        "2026-01-02 12:00", None, None,
    ], format="mixed"))
    pd.testing.assert_series_equal(actual, expected)
