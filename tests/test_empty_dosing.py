import pandas as pd

from nainsight.data.dosing import normalise_doses


def test_empty_medication_table_retains_the_normalised_schema():
    columns = ["ingredient", "value", "unit", "text", "stay_id"]
    frame = pd.DataFrame(columns=columns)
    result = normalise_doses(frame)
    assert result.empty
    assert set(result) == set(columns) | {
        "concept", "dose_std", "dose_unit", "volume_ml", "na_mmol", "tonicity", "pred_eq_mg"}
    pd.testing.assert_frame_equal(frame, pd.DataFrame(columns=columns))
