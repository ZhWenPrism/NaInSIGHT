import numpy as np
import pandas as pd

from nainsight.data.features import StaticPreprocessor


def test_refit_rebuilds_categories_and_missing_indicators():
    schema = {"value": "numeric", "kind": "category", "flag": "binary"}
    first = pd.DataFrame({"value": [1, np.nan], "kind": ["old", "old"], "flag": [1, np.nan]})
    second = pd.DataFrame({"value": [3, 5], "kind": ["new", "new"], "flag": [0, 1]})
    reused = StaticPreprocessor(schema, min_count=1).fit(first).fit(second)
    fresh = StaticPreprocessor(schema, min_count=1).fit(second)
    assert reused.columns == fresh.columns
    pd.testing.assert_frame_equal(reused.transform(second), fresh.transform(second))
    assert "kind=new" in reused.columns and "kind=old" not in reused.columns
    assert not any(c.endswith("__missing") for c in reused.columns)
