import logging

import pytest
import torch

from nainsight.train.trainer import Trainer


@pytest.mark.parametrize("name", ["epochs", "patience", "max_steps_per_epoch"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_budgets_fail_at_trainer_initialization(tmp_path, name, value):
    config = {"train": {"epochs": 1, "patience": 1, "weight_decay": .01, "lr": .001, "grad_clip": 1.}}
    config["train"][name] = value
    with pytest.raises(ValueError, match=name):
        Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), config, torch.device("cpu"),
                tmp_path, logging.getLogger(__name__), [24])
