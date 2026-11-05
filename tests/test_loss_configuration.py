import numpy as np
import pytest
import torch

from nainsight.train.losses import MultiTaskLoss


@pytest.mark.parametrize("options", [
    {"weighting": "uncertain"},
    {"pos_weight": [1.]},
    {"pos_weight": [1., float("nan")]},
    {"weighting": "fixed", "fixed_weights": {"risk": 1.}},
    {"weighting": "fixed", "fixed_weights": {"risk": -1., "severity": 1., "nadir": 1.}},
    {"consistency_weight": -1.},
    {"threshold": float("inf")},
])
def test_invalid_loss_options_fail_before_training(options):
    with pytest.raises(ValueError):
        MultiTaskLoss([24, 48], **options)


def test_positive_weights_accept_arrays_and_do_not_alias_the_caller():
    weights = np.array([1., 2.], dtype=np.float32)
    loss = MultiTaskLoss([24, 48], pos_weight=weights)
    weights[:] = -1
    torch.testing.assert_close(loss.pos_weight, torch.tensor([1., 2.]))


def test_default_and_explicit_weights_compute_the_same_loss():
    targets = {f"{task}_{h}": torch.tensor([0., 1.]) for task in ("risk", "sev", "nadir") for h in (24, 48)}
    out = {"risk_logit": torch.zeros(2, 2), "sev_logits": torch.zeros(2, 2, 2),
           "nadir_mu": torch.zeros(2, 2), "nadir_sigma": torch.ones(2, 2)}
    default, _ = MultiTaskLoss([24, 48])(out, targets)
    explicit, _ = MultiTaskLoss([24, 48], pos_weight=np.ones(2))(out, targets)
    torch.testing.assert_close(default, explicit)
