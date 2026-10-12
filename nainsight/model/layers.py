"""Shared building blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Time2Vec(nn.Module):
    """Continuous-time encoding: one linear (trend) and d-1 periodic components with learned frequencies."""

    def __init__(self, d: int, scale_h: float = 24.0):
        super().__init__()
        self.scale = scale_h
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, d - 1)
        with torch.no_grad():                                   # spread periods from hours to weeks
            self.per.weight.copy_(torch.exp(torch.linspace(math.log(0.15), math.log(6.0), d - 1)).unsqueeze(-1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = (t / self.scale).unsqueeze(-1)
        return torch.cat([self.lin(x), torch.sin(self.per(x))], dim=-1)


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))
