import pandas as pd

from nainsight.data.features import StaticPreprocessor


def test_exported_state_cannot_change_the_fitted_preprocessor():
    frame = pd.DataFrame({"x": [0., 2., 4.], "site": ["a", "a", "b"]})
    preprocessor = StaticPreprocessor({"x": "numeric", "site": "category"}, min_count=1).fit(frame)
    expected = preprocessor.transform(frame)
    state = preprocessor.state()
    state["stats"]["x"]["median"] = 999.
    state["stats"]["site"]["levels"].clear()
    state["columns"].clear()
    state["schema"]["x"] = "category"
    pd.testing.assert_frame_equal(preprocessor.transform(frame), expected)


def test_restored_state_does_not_alias_the_checkpoint_dictionary():
    frame = pd.DataFrame({"x": [0., 2., 4.]})
    original = StaticPreprocessor({"x": "numeric"}).fit(frame)
    state = original.state()
    restored = StaticPreprocessor.from_state(state)
    state["stats"]["x"]["median"] = 999.
    state["columns"].clear()
    pd.testing.assert_frame_equal(restored.transform(frame), original.transform(frame))
