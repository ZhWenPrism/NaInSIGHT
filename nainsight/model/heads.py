"""Coherent multi-task output layer.

Structural constraints built into the parameterisation (H = horizons, nested windows):
  risk        logit_48 = logit_24 + softplus(.)                -> P(event in 48 h) >= P(event in 24 h)
  severity    cumulative logits [logit_h, logit_h - softplus(.) - 0.5]
              -> P(severity >= mild) is the risk probability itself; P(moderate-severe) <= P(any event)
  nadir       mu_24 = reference + delta, mu_48 = mu_24 - softplus(.)  -> predicted 48-h nadir <= 24-h nadir;
              reference = last observed sodium before the landmark (learned constant if none);
              heteroscedastic: sigma_h = softplus(.) + floor
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoherentMultiTaskHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_horizons: int, dropout: float, sigma_floor: float = 0.5):
        super().__init__()
        self.H, self.sigma_floor = n_horizons, sigma_floor
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.risk_base = nn.Linear(hidden, 1)
        self.risk_inc = nn.Linear(hidden, max(n_horizons - 1, 1))
        self.sev_gap = nn.Linear(hidden, n_horizons)
        self.nadir_delta = nn.Linear(hidden, 1)
        self.nadir_inc = nn.Linear(hidden, max(n_horizons - 1, 1))
        self.nadir_sigma = nn.Linear(hidden, n_horizons)
        self.na_default = nn.Parameter(torch.tensor(140.0))
        nn.init.constant_(self.sev_gap.bias, 1.0)
        nn.init.constant_(self.nadir_sigma.bias, 1.5)                 # initial sigma ~ 2.2 mmol/L

    def forward(self, h: torch.Tensor, na_last: torch.Tensor, na_mask: torch.Tensor) -> dict:
        u = self.trunk(h)
        risk = [self.risk_base(u)]
        inc = F.softplus(self.risk_inc(u))
        for i in range(self.H - 1):
            risk.append(risk[-1] + inc[:, i: i + 1])
        risk = torch.cat(risk, dim=-1)                                            # n H
        sev2 = risk - F.softplus(self.sev_gap(u)) - 0.5
        ref = torch.where(na_mask > 0, na_last, self.na_default.expand_as(na_last)).unsqueeze(-1)
        mu = [ref + self.nadir_delta(u)]
        dec = F.softplus(self.nadir_inc(u))
        for i in range(self.H - 1):
            mu.append(mu[-1] - dec[:, i: i + 1])
        mu = torch.cat(mu, dim=-1)
        sigma = F.softplus(self.nadir_sigma(u)) + self.sigma_floor
        return {"risk_logit": risk, "sev_logits": torch.stack([risk, sev2], dim=-1), "nadir_mu": mu, "nadir_sigma": sigma}

    @staticmethod
    def severity_probs(sev_logits: torch.Tensor) -> torch.Tensor:
        p1, p2 = torch.sigmoid(sev_logits[..., 0]), torch.sigmoid(sev_logits[..., 1])
        return torch.stack([1 - p1, (p1 - p2).clamp(min=0), p2], dim=-1)
