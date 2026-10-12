"""Training loop: AdamW with decoupled weight decay (no decay on norms, biases, embeddings and gates),
linear warm-up + cosine schedule, gradient clipping, early stopping on the validation score (mean risk
AUROC over prediction modes and horizons), best-checkpoint saving and JSON-lines logging."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import Collator, LengthBucketSampler, StayDataset
from ..model.heads import CoherentMultiTaskHead
from .metrics import landmark_metrics, validation_score


def make_loader(pack: dict, batch_size: int, shuffle: bool, seed: int, n_nodes: int) -> DataLoader:
    ds = StayDataset(pack)
    sampler = LengthBucketSampler(ds.lengths, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=Collator(pack, n_nodes), num_workers=0)


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out


def _param_groups(modules: list[torch.nn.Module], weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for mod in modules:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(k in name for k in ("norm", "emb", "bias", "log_vars", "g_local", "g_global",
                                                     "A_log", "dt_bias", "gap_gain", "calib", ".D", "na_default")):
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    def __init__(self, model, loss_fn, cfg: dict, device: torch.device, run_dir: Path, logger, horizons: list[int]):
        self.model, self.loss_fn, self.cfg = model.to(device), loss_fn.to(device), cfg
        self.device, self.run_dir, self.log, self.horizons = device, Path(run_dir), logger, horizons
        tc = cfg["train"]
        self.opt = torch.optim.AdamW(_param_groups([self.model, self.loss_fn], tc["weight_decay"]), lr=tc["lr"])
        self.grad_clip = tc["grad_clip"]
        self.max_steps = tc.get("max_steps_per_epoch")

    def _scheduler(self, steps_per_epoch: int):
        tc = self.cfg["train"]
        total = max(1, steps_per_epoch * tc["epochs"])
        warm = max(1, int(tc["warmup_frac"] * total))

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            progress = min(1.0, (step - warm) / max(1, total - warm))
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.opt, factor)

    def train_epoch(self, loader, sched) -> dict:
        self.model.train(); self.loss_fn.train()
        agg, n = {}, 0
        for step, batch in enumerate(loader):
            if self.max_steps and step >= self.max_steps:
                break
            b = to_device(batch, self.device)
            out = self.model(b)
            loss, parts = self.loss_fn(out, b["targets"])
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.loss_fn.parameters()), self.grad_clip)
            self.opt.step()
            sched.step()
            parts["total"] = float(loss.detach())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    @torch.no_grad()
    def predict(self, loader, pack: dict) -> pd.DataFrame:
        self.model.eval()
        frames = []
        for batch in loader:
            b = to_device(batch, self.device)
            out = self.model(b)
            rows = batch["lmk_index"].numpy()
            stay_row = pack["lmk_stay"][rows]
            f = {"lmk_index": rows, "stay_id": pack["stay_id"][stay_row], "subject_id": pack["subject_id"][stay_row],
                 "mode": np.where(pack["lmk_mode"][rows] == 1, "rolling", "baseline"), "t_lmk": pack["lmk_time"][rows]}
            p = torch.sigmoid(out["risk_logit"]).cpu().numpy()
            ps = CoherentMultiTaskHead.severity_probs(out["sev_logits"]).cpu().numpy()
            mu, sig = out["nadir_mu"].cpu().numpy(), out["nadir_sigma"].cpu().numpy()
            for i, h in enumerate(self.horizons):
                f[f"p_risk_{h}"] = p[:, i]
                for c in range(3):
                    f[f"p_sev{c}_{h}"] = ps[:, i, c]
                f[f"mu_nadir_{h}"], f[f"sigma_nadir_{h}"] = mu[:, i], sig[:, i]
                for task in ("risk", "sev", "nadir"):
                    f[f"y_{task}_{h}"] = pack[f"y_{task}_{h}"][rows]
            if "knowledge_gate" in out:
                f["knowledge_gate"] = out["knowledge_gate"].cpu().numpy()
            frames.append(pd.DataFrame(f))
        return pd.concat(frames, ignore_index=True).sort_values("lmk_index").reset_index(drop=True)

    def fit(self, train_loader, val_loader, val_pack: dict) -> dict:
        tc = self.cfg["train"]
        steps = len(train_loader) if not self.max_steps else min(len(train_loader), self.max_steps)
        sched = self._scheduler(steps)
        best, best_epoch, wait = -np.inf, -1, 0
        log_path = self.run_dir / "train_log.jsonl"
        for epoch in range(tc["epochs"]):
            t0 = time.time()
            tr = self.train_epoch(train_loader, sched)
            pred = self.predict(val_loader, val_pack)
            res = landmark_metrics(pred, self.horizons)
            score = validation_score(res)
            rec = {"epoch": epoch, "train": tr, "val_score": score, "seconds": round(time.time() - t0, 1),
                   "val": {f"{m}|{h}|{t}": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                            for k, v in d.items()} for (m, h, t), d in res.items()}}
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.log.info("epoch %d | loss %.4f | val score %.4f | %.0fs", epoch, tr.get("total", np.nan), score, rec["seconds"])
            if np.isfinite(score) and score > best:
                best, best_epoch, wait = score, epoch, 0
                torch.save({"model": self.model.state_dict(), "loss": self.loss_fn.state_dict(), "epoch": epoch,
                            "val_score": score}, self.run_dir / "best.pt")
            else:
                wait += 1
                if wait >= tc["patience"]:
                    self.log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                    break
        if best_epoch < 0:                                           # no finite validation score: keep last
            torch.save({"model": self.model.state_dict(), "loss": self.loss_fn.state_dict(), "epoch": epoch,
                        "val_score": float("nan")}, self.run_dir / "best.pt")
        return {"best_val_score": best, "best_epoch": best_epoch}
