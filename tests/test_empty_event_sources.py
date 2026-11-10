from unittest.mock import patch

import pandas as pd
import pytest

from nainsight.data.events import attach, extract_io, extract_medications, extract_vitals


def _windows():
    return pd.DataFrame({"stay_id": [0], "subject_key": ["synthetic"],
                         "win_start": pd.to_datetime(["2020-01-01"]),
                         "win_end": pd.to_datetime(["2020-01-03"]),
                         "surgery_end": pd.to_datetime(["2020-01-02"])})


@pytest.mark.parametrize("extractor, block, concept", [
    (extract_vitals, "vitals", "VS_HR"), (extract_io, "fluid_balance", "IO_URINE")])
def test_no_charted_rows_preserves_the_event_schema(extractor, block, concept):
    config = {"events": {"vital_bin_h": 1., "io_bin_h": 1.}}
    site = {block: {"sources": [{"items": {concept: "matched"}}]}}
    chunk = pd.DataFrame({"subject_key": ["outside"], "item": ["unmatched"],
                          "time": ["2020-01-02"], "value": ["1"]})
    with patch("nainsight.data.events.iter_source", return_value=iter([chunk])):
        events, qc = extractor(site, _windows(), config)
    assert events.empty
    assert set(events) == {"stay_id", "t", "concept", "value"}
    assert qc == {"rows": 0, "dropped_implausible": {}}


@pytest.mark.parametrize("source", ["administration_log", "orders"])
def test_no_medication_rows_returns_a_normalised_empty_table(source):
    site = {"medications": {"source": source, "ingredients": [{"ingredient": "mannitol", "pattern": "mannitol"}],
                             "administration_log": {}, "orders": {"frequency_hours": {"daily": 24}}}}
    chunk = pd.DataFrame(columns=["subject_key", "text", "name", "spec"])
    with patch("nainsight.data.events.iter_source", return_value=iter([chunk])):
        events, qc = extract_medications(site, _windows(), {})
    assert events.empty
    assert {"stay_id", "t", "concept", "ingredient", "dose_std", "route"} <= set(events)
    assert not {"subject_key", "time"} & set(events)
    assert qc["administrations"] == 0


def test_empty_attach_uses_the_same_identity_columns_as_nonempty_attach():
    frame = pd.DataFrame({"subject_key": ["synthetic"], "time": pd.to_datetime(["2020-01-02"]), "value": [1.]})
    assert set(attach(frame.iloc[:0], _windows())) == set(attach(frame, _windows()))
