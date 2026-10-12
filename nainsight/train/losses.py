"""Masked multi-task objective.

  risk        BCE with logits per horizon on ascertainable landmarks (optional positive-class weight)
  severity    cumulative-logit (ordinal) BCE on [y >= mild, y >= moderate-severe]
  nadir       heteroscedastic Gaussian negative log-likelihood
  consistency (risk probability - P(nadir < threshold | mu, sigma))^2 on every landmark (label-free)
Task weighting: homoscedastic uncertainty (learned log-variances) or fixed weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = ("risk", "severity", "nadir")


class MultiTaskLoss(nn.Module):
    def __init__(self, horizons: list[int], weighting: str = "uncertainty", fixed_weights: dict | None = None,
                 pos_weight: list[float] | None = None, consistency_weight: float = 0.1, threshold: float = 135.0):
        super().__init__()
        self.horizons, self.weighting, self.threshold = list(horizons), weighting, threshold
        self.fixed = fixed_weights or {"risk": 1.0, "severity": 1.0, "nadir": 0.5}
        self.consistency_weight = consistency_weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight or [1.0] * len(horizons), dtype=torch.float32))
        self.log_vars = nn.Parameter(torch.zeros(len(TASKS))) if weighting == "uncertainty" else None

    def forward(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        dev = out["risk_logit"].device
        parts = {t: [] for t in TASKS}
        for i, h in enumerate(self.horizons):
            y = targets[f"risk_{h}"].to(dev)
            m = torch.isfinite(y)
            if m.any():
                parts["risk"].append(F.binary_cross_entropy_with_logits(
                    out["risk_logit"][m, i], y[m], pos_weight=self.pos_weight[i]))
            y = targets[f"sev_{h}"].to(dev)
            m = torch.isfinite(y)
            if m.any():
                cum = torch.stack([(y[m] >= 1).float(), (y[m] >= 2).float()], dim=-1)
                parts["severity"].append(F.binary_cross_entropy_with_logits(out["sev_logits"][m, i], cum))
            y = targets[f"nadir_{h}"].to(dev)
            m = torch.isfinite(y)
            if m.any():
                mu, sig = out["nadir_mu"][m, i], out["nadir_sigma"][m, i]
                parts["nadir"].append((torch.log(sig) + 0.5 * ((y[m] - mu) / sig) ** 2).mean())
        losses = {t: torch.stack(v).mean() if v else torch.zeros((), device=dev) for t, v in parts.items()}
        p_risk = torch.sigmoid(out["risk_logit"])
        p_nadir = 0.5 * (1 + torch.erf((self.threshold - out["nadir_mu"]) / (out["nadir_sigma"] * math.sqrt(2.0))))
        losses["consistency"] = ((p_risk - p_nadir) ** 2).mean()
        if self.weighting == "uncertainty":
            total = sum(torch.exp(-self.log_vars[j]) * losses[t] + self.log_vars[j]
                        for j, t in enumerate(TASKS) if parts[t])
        else:
            total = sum(self.fixed[t] * losses[t] for t in TASKS if parts[t])
        total = total + self.consistency_weight * losses["consistency"]
        return total, {k: float(v.detach()) for k, v in losses.items()}


def positive_weights(pack: dict, horizons: list[int], mode: str) -> list[float]:
    out = []
    for h in horizons:
        y = pack[f"y_risk_{h}"]
        y = y[(y == 0) | (y == 1)]
        pos, neg = float((y == 1).sum()), float((y == 0).sum())
        ratio = neg / max(pos, 1.0)
        out.append(1.0 if mode == "none" else (ratio if mode == "balanced" else math.sqrt(ratio)))
    return out
