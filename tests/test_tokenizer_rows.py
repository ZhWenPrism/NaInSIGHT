import numpy as np
import pandas as pd
import pytest

from nainsight.data.tokenizer import ClinicalEventTokenizer


def inputs(with_events=True):
    stays = pd.DataFrame({"stay_id": [2, 1], "subject_id": [102, 101], "surgery_duration_h": [2., 2.]})
    obs = pd.DataFrame({"stay_id": [1, 2], "concept": ["LAB_NA", "LAB_NA"], "t": [-1., -1.], "value": [140., 141.]})
    if not with_events:
        obs = obs.iloc[:0]
    events = {"labs": obs, "vitals": obs.iloc[:0].copy(), "io": obs.iloc[:0].copy(),
              "meds": pd.DataFrame(columns=["stay_id", "concept", "ingredient", "t", "dose_std", "route", "na_mmol"])}
    landmarks = pd.DataFrame({"stay_id": [2, 1], "t_lmk": [0., 0.], "mode": ["baseline", "baseline"],
                             "lmk_id": [0, 1], "risk_24": [0., 1.], "nadir_24": [140., 134.], "sev_24": [0., 1.]})
    tok = ClinicalEventTokenizer(min_concept_count=1).fit(events, {}, {})
    return tok, stays, events, landmarks


def test_static_rows_follow_stays_when_tokenizer_sorts_them():
    tok, stays, events, landmarks = inputs()
    pack = tok.encode(stays, events, landmarks, np.array([[20.], [10.]]), [24])
    np.testing.assert_array_equal(pack["stay_id"], [1, 2])
    np.testing.assert_array_equal(pack["static"].ravel(), [10., 20.])
    np.testing.assert_array_equal(pack["subject_id"], [101, 102])
    np.testing.assert_array_equal(pack["y_risk_24"], [1., 0.])


def test_static_row_count_must_match_stays():
    tok, stays, events, landmarks = inputs()
    with pytest.raises(ValueError, match="static"):
        tok.encode(stays, events, landmarks, np.zeros((1, 1)), [24])
