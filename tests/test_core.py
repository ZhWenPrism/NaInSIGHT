"""Unit tests for dose normalisation, outcome construction and causality of the temporal backbone."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nainsight.data.dosing import normalise_doses
from nainsight.data.landmarks import build_landmarks
from nainsight.model.backbone import NaFormerBackbone
from nainsight.model.ssm import sequential_scan, ssd_scan

UNIT_ALIASES = {"tablet": "unit_dose"}


@pytest.mark.parametrize("ingredient,value,unit,text,dose,na", [
    ("mannitol", 250, "ml", "*20%250ml", 50.0, None),
    ("mannitol", 100, "ml", "100ml:20g(20%)", 20.0, None),
    ("hypertonic_saline", 30, "ml", "*10% 10ml:1g", 30.0, 51.335),
    ("sodium_chloride", 500, "ml", "*0.9%500ml", 500.0, 77.002),
    ("dextrose_water", 500, "ml", "*5% 500ml:25g", 500.0, 0.0),
    ("dexamethasone", 1, "ml", "1ml:5mg", 5.0, None),
    ("desmopressin", 0.1, "mg", "0.1mg*30", 100.0, None),
    ("potassium_chloride", 10, "ml", "*1.5g:10ml", 1.5, None),
    ("levetiracetam", 1, "tablet", "0.5g*30", 500.0, None),
    ("pituitrin", 1, "ml", "1ml 6iu", 6.0, None),
])
def test_dose_conversion(ingredient, value, unit, text, dose, na):
    r = normalise_doses(pd.DataFrame([(ingredient, value, unit, text)], columns=["ingredient", "value", "unit", "text"]),
                        UNIT_ALIASES).iloc[0]
    assert r["dose_std"] == pytest.approx(dose, rel=1e-3)
    if na is not None:
        assert r["na_mmol"] == pytest.approx(na, rel=1e-3)


def test_unconvertible_dose_is_missing_not_zero():
    r = normalise_doses(pd.DataFrame([("oxcarbazepine", 5, "ml", "@100ml")], columns=["ingredient", "value", "unit", "text"]))
    assert np.isnan(r["dose_std"].iloc[0])


def test_glucocorticoid_prednisone_equivalent():
    r = normalise_doses(pd.DataFrame([("dexamethasone", 5, "mg", ""), ("methylprednisolone", 40, "mg", "")],
                                     columns=["ingredient", "value", "unit", "text"]))
    assert r["pred_eq_mg"].tolist() == pytest.approx([33.333, 50.0], rel=1e-3)


OUTCOME = {"outcome": {"threshold": 135, "severity_cutpoints": [135, 130], "horizons_h": [24, 48], "segment_h": 24,
                       "baseline": {"enabled": True, "time_h": 0, "gap_h": 0},
                       "rolling": {"enabled": True, "first_h": 24, "last_h": 168, "step_h": 24, "gap_h": 6},
                       "require_full_window_in_hospital": True, "nadir_requires_segment_coverage": True}}


def test_landmarks_and_outcomes():
    stays = pd.DataFrame({"stay_id": [0], "postop_stay_h": [200.0]})
    na = pd.DataFrame({"stay_id": [0] * 5, "t": [10, 40, 60, 90, 130], "value": [140, 138, 137, 133, 129]})
    lmk = build_landmarks(stays, na, OUTCOME).set_index("t_lmk")
    assert list(lmk.index) == [0.0, 24.0, 48.0, 72.0]             # nothing at/after the first event (t = 90)
    assert lmk.loc[48.0, "risk_48"] == 1 and lmk.loc[48.0, "nadir_48"] == 133 and lmk.loc[48.0, "sev_48"] == 1
    assert lmk.loc[72.0, "risk_48"] == 1 and np.isnan(lmk.loc[72.0, "nadir_48"])   # second segment unobserved


def test_no_landmark_without_full_window():
    stays = pd.DataFrame({"stay_id": [0], "postop_stay_h": [100.0]})
    na = pd.DataFrame({"stay_id": [0, 0], "t": [20, 70], "value": [139, 141]})
    lmk = build_landmarks(stays, na, OUTCOME)
    assert lmk["t_lmk"].max() == 24.0                              # 48 + 6 + 48 > 100 h stay


def test_chunked_scan_matches_recurrence():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2, 130, 3, 4, generator=g, dtype=torch.float64)
    A = -torch.rand(2, 130, 3, generator=g, dtype=torch.float64)
    B = torch.randn(2, 130, 3, 5, generator=g, dtype=torch.float64)
    C = torch.randn(2, 130, 3, 5, generator=g, dtype=torch.float64)
    assert torch.allclose(ssd_scan(X, A, B, C, 32), sequential_scan(X, A, B, C), atol=1e-10)


def test_backbone_is_causal():
    torch.manual_seed(0)
    bb = NaFormerBackbone(32, 2, {"d_state": 8, "expand": 2, "headdim": 16, "d_conv": 4, "chunk": 16},
                          {"n_heads": 4, "ff_mult": 2, "time_buckets_h": [1, 4, 24, 72]},
                          {"fine_h": 6, "coarse_h": 24, "n_heads": 4}, 0.0, set()).eval()
    t = torch.sort(torch.rand(2, 80) * 150 - 50, dim=1).values
    gap = torch.cat([torch.zeros(2, 1), t[:, 1:] - t[:, :-1]], dim=1)
    mask = torch.ones(2, 80, dtype=torch.bool)
    x = torch.randn(2, 80, 32)
    x2 = x.clone()
    x2[:, 50:] += 5.0
    with torch.no_grad():
        assert torch.equal(bb(x, gap, t, mask)[:, :50], bb(x2, gap, t, mask)[:, :50])
