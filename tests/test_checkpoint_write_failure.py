import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from nainsight.train.trainer import Trainer


@pytest.mark.parametrize("score", [.8, np.nan])
def test_best_and_fallback_checkpoint_failures_preserve_existing_file(tmp_path, monkeypatch, score):
    config = {"train": {"epochs": 1, "patience": 1, "lr": .001, "weight_decay": 0.,
                        "grad_clip": 1., "warmup_frac": .05}}
    trainer = Trainer(torch.nn.Linear(1, 1), torch.nn.MSELoss(), config, torch.device("cpu"),
                      tmp_path, logging.getLogger(__name__), [24])
    trainer.train_epoch = Mock(return_value={"total": 0.})
    trainer.predict = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr("nainsight.train.trainer.landmark_metrics", lambda *a: {})
    monkeypatch.setattr("nainsight.train.trainer.validation_score", lambda *a: score)
    target = tmp_path / "best.pt"
    target.write_bytes(b"previous checkpoint")

    def fail(state, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise OSError("serialization failed")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="serialization failed"):
        trainer.fit([1], [1], {})
    assert target.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.glob(".best.pt.*")) == []
