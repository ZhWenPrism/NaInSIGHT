import logging
from unittest.mock import Mock

import pytest
import torch

from nainsight.train.trainer import Trainer


@pytest.mark.parametrize("train,val", [([], [1]), ([1], []), ([], [])])
def test_empty_loaders_do_not_train_or_write_a_checkpoint(tmp_path, train, val):
    config = {"train": {"epochs": 1, "patience": 1, "weight_decay": .01,
                        "lr": .001, "grad_clip": 1., "warmup_frac": .05}}
    trainer = Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), config,
                      torch.device("cpu"), tmp_path, logging.getLogger(__name__), [24])
    trainer.train_epoch = Mock(side_effect=AssertionError("training started"))
    with pytest.raises(ValueError, match="at least one batch"):
        trainer.fit(train, val, {})
    trainer.train_epoch.assert_not_called()
    assert not (tmp_path / "best.pt").exists()
