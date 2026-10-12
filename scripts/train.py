#!/usr/bin/env python3
"""Train NaInSIGHT (or an ablation variant: --set model.ablation=[...])."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from nainsight.data.dataset import load_pack
from nainsight.data.tokenizer import N_NODES
from nainsight.runtime import build_model, load_prior
from nainsight.train.losses import MultiTaskLoss, positive_weights
from nainsight.train.trainer import Trainer, make_loader
from nainsight.utils import environment_report, get_logger, load_json, resolve_device, save_json, seed_everything
from nainsight.workspace import common_parser, load_run_context


def main() -> None:
    ap = common_parser(__doc__)
    ap.add_argument("--run-name", default="nainsight")
    ap.add_argument("--site", default="development")
    args = ap.parse_args()
    cfg, ws = load_run_context(args)
    seed_everything(cfg.seed)
    run_dir = ws.dir("runs", args.run_name)
    log = get_logger("train", run_dir / "train.log")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)
    save_json(environment_report(), run_dir / "environment.json")

    pre = load_json(ws.artifacts / "preprocessors.json")
    train = load_pack(ws.tensors / f"{args.site}_train.npz")
    val = load_pack(ws.tensors / f"{args.site}_val.npz")
    horizons = [int(h) for h in cfg.outcome.horizons_h]
    prior = load_prior(ws.reference, cfg)
    model = build_model(cfg, pre, train["static"].shape[1], prior)
    # freeze the exact preprocessors and knowledge prior used by this run next to its checkpoint
    shutil.copy2(ws.artifacts / "preprocessors.json", run_dir / "preprocessors.json")
    if prior is not None:
        (run_dir / "prior").mkdir(exist_ok=True)
        for name in ("graph.npz", "prior_matrices.npz", "prior_vectors.npz"):
            shutil.copy2(ws.reference / name, run_dir / "prior" / name)
    log.info("parameters: %.2fM trainable | ablation: %s", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
             list(cfg.model.ablation or []))
    lc = cfg.train.loss
    loss = MultiTaskLoss(horizons, lc.weighting, lc.fixed_weights, positive_weights(train, horizons, lc.pos_weight),
                         lc.consistency_weight, cfg.outcome.threshold)
    device = resolve_device(cfg.train.device)
    trainer = Trainer(model, loss, cfg, device, run_dir, log, horizons)
    result = trainer.fit(make_loader(train, cfg.train.batch_size, True, cfg.seed, N_NODES),
                         make_loader(val, cfg.train.batch_size, False, cfg.seed, N_NODES), val)
    save_json(result, run_dir / "train_result.json")
    log.info("best validation score %.4f at epoch %d", result["best_val_score"], result["best_epoch"])


if __name__ == "__main__":
    main()
