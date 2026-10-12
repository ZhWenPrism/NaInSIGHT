#!/usr/bin/env python3
"""Landmark-level predictions and metrics with patient-cluster bootstrap confidence intervals."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from nainsight.config import AttrDict, read_yaml
from nainsight.data.dataset import load_pack
from nainsight.data.tokenizer import N_NODES
from nainsight.runtime import build_model, load_prior
from nainsight.train.bootstrap import cluster_bootstrap, to_frame
from nainsight.train.losses import MultiTaskLoss
from nainsight.train.metrics import landmark_metrics
from nainsight.train.trainer import Trainer, make_loader
from nainsight.utils import get_logger, load_json, resolve_device, seed_everything
from nainsight.workspace import Workspace, common_parser


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--run-name", default="nainsight")
    ap.add_argument("--splits", nargs="*", default=["development_test"],
                    help="tensor files to evaluate, e.g. development_test external_external")
    args = ap.parse_args()
    ws = Workspace(args.paths)
    run_dir = ws.dir("runs", args.run_name)
    cfg = AttrDict.wrap(read_yaml(run_dir / "config.yaml"))
    seed_everything(cfg.seed)
    log = get_logger("evaluate", run_dir / "evaluate.log")
    pre = load_json(run_dir / "preprocessors.json")
    prior_dir = run_dir / "prior" if (run_dir / "prior").exists() else ws.reference
    horizons = [int(h) for h in cfg.outcome.horizons_h]
    device = resolve_device(cfg.train.device)
    tables = []
    for name in args.splits:
        pack = load_pack(ws.tensors / f"{name}.npz")
        model = build_model(cfg, pre, pack["static"].shape[1], load_prior(prior_dir, cfg))
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        trainer = Trainer(model, MultiTaskLoss(horizons), cfg, device, run_dir, log, horizons)
        pred = trainer.predict(make_loader(pack, cfg.train.batch_size, False, cfg.seed, N_NODES), pack)
        pred.to_parquet(run_dir / f"predictions_{name}.parquet")
        res = cluster_bootstrap(pred, "subject_id", lambda d: landmark_metrics(d, horizons, per_class=True),
                                int(cfg.evaluate.bootstrap), float(cfg.evaluate.ci), int(cfg.seed))
        frame = to_frame(res, name)
        frame.to_csv(run_dir / f"metrics_{name}.csv", index=False)
        tables.append(frame)
        main = frame[frame["metric"].isin(["AUROC", "AUPRC", "Brier", "MAE", "RMSE", "R2", "bias", "macro_AUROC",
                                           "macro_F1", "balanced_accuracy", "weighted_kappa"])]
        log.info("%s\n%s", name, main.round(3).to_string(index=False))
    pd.concat(tables, ignore_index=True).to_csv(run_dir / "metrics_all.csv", index=False)


if __name__ == "__main__":
    main()
