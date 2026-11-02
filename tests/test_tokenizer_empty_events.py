import numpy as np

from test_tokenizer_rows import inputs


def test_landmark_only_sequences_are_encoded_without_observations():
    tok, stays, events, landmarks = inputs(with_events=False)
    pack = tok.encode(stays, events, landmarks, np.array([[20.], [10.]]), [24])
    np.testing.assert_array_equal(pack["tok_offsets"], [0, 1, 2])
    np.testing.assert_array_equal(pack["tok_concept"], [tok.concept_vocab["[LMK_BASELINE]"]] * 2)
    np.testing.assert_array_equal(pack["lmk_pos"], [0, 0])
    np.testing.assert_array_equal(pack["lmk_na_mask"], [0, 0])
    assert np.isfinite(pack["node_feat"]).all()
