import logging
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nainsight.train.trainer import Trainer


@pytest.mark.parametrize("name,value", [
    ("lr", 0), ("lr", np.nan), ("lr", np.inf), ("weight_decay", -1), ("weight_decay", np.nan),
    ("grad_clip", -1), ("grad_clip", 0), ("grad_clip", np.nan), ("grad_clip", True),
    ("warmup_frac", -.1), ("warmup_frac", 1.1), ("warmup_frac", np.nan),
])
def test_invalid_settings_fail_before_optimizer_creation(tmp_path, name, value):
    config = {"train": {"epochs": 1, "patience": 1, "lr": .001, "weight_decay": .01,
                        "grad_clip": 1., "warmup_frac": .05}}
    config["train"][name] = value
    with patch("nainsight.train.trainer.torch.optim.AdamW") as optimizer:
        with pytest.raises(ValueError, match=name):
            Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), config, torch.device("cpu"),
                    tmp_path, logging.getLogger(__name__), [24])
        optimizer.assert_not_called()


@pytest.mark.parametrize("fraction", [0., 1.])
def test_warmup_fraction_endpoints_are_valid(tmp_path, fraction):
    config = {"train": {"epochs": 1, "patience": 1, "lr": .001, "weight_decay": 0.,
                        "grad_clip": 1., "warmup_frac": fraction}}
    Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), config, torch.device("cpu"),
            tmp_path, logging.getLogger(__name__), [24])
