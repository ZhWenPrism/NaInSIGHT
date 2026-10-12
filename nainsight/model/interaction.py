"""Causal multi-resolution interaction between the local (SSM) and global (Transformer) branches.

Each branch summarises its states in fixed time bins (bins aligned on the surgery-end clock).  A token may
only read bins that are *complete* before its own bin starts, so the exchange never looks ahead:
  local  <- global summaries at the coarse resolution (e.g. 24 h)  : slow context for fine dynamics
  global <- local  summaries at the fine resolution   (e.g. 6 h)   : recent detail for long-range attention
Updates enter through learned sigmoid gates initialised close to zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalBinCrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, resolution_h: float, dropout: float, max_bins: int = 512):
        super().__init__()
        self.r, self.h, self.dh, self.max_bins = float(resolution_h), n_heads, d // n_heads, max_bins
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.out = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, xq: torch.Tensor, xs: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, l, d = xq.shape
        valid = pad_mask.bool()
        origin = torch.floor(t.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True) / self.r) * self.r
        origin = torch.nan_to_num(origin, posinf=0.0)
        bins = torch.floor((t - origin) / self.r).long().clamp(0, self.max_bins - 1).masked_fill(~valid, 0)
        nb = int(bins.max().item()) + 1
        w = valid.to(xs.dtype)
        summ = torch.zeros(b, nb, d, device=xs.device, dtype=xs.dtype).scatter_add_(
            1, bins.unsqueeze(-1).expand(-1, -1, d), xs * w.unsqueeze(-1))
        cnt = torch.zeros(b, nb, device=xs.device, dtype=xs.dtype).scatter_add_(1, bins, w)
        summary = summ / cnt.clamp(min=1.0).unsqueeze(-1)
        allowed = (torch.arange(nb, device=xs.device).view(1, 1, nb) < bins.unsqueeze(-1)) & (cnt > 0).unsqueeze(1)
        has = allowed.any(dim=-1)                                                        # b l
        q = self.q(self.nq(xq)).view(b, l, self.h, self.dh).transpose(1, 2)              # b h l dh
        k, v = self.kv(self.nk(summary)).view(b, nb, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)                           # b h l nb
        logits = logits.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        logits = torch.where(has.unsqueeze(1).unsqueeze(-1), logits, torch.zeros_like(logits))
        attn = self.drop(torch.softmax(logits, dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(b, l, d)
        return self.out(y) * has.unsqueeze(-1).to(y.dtype)


class MultiResolutionInteraction(nn.Module):
    def __init__(self, d: int, n_heads: int, fine_h: float, coarse_h: float, dropout: float):
        super().__init__()
        self.local_from_global = CausalBinCrossAttention(d, n_heads, coarse_h, dropout)
        self.global_from_local = CausalBinCrossAttention(d, n_heads, fine_h, dropout)
        self.g_local = nn.Parameter(torch.full((d,), -2.0))
        self.g_global = nn.Parameter(torch.full((d,), -2.0))

    def forward(self, h_loc, h_glob, t, pad_mask):
        d_loc = self.local_from_global(h_loc, h_glob, t, pad_mask)
        d_glob = self.global_from_local(h_glob, h_loc, t, pad_mask)
        return h_loc + torch.sigmoid(self.g_local) * d_loc, h_glob + torch.sigmoid(self.g_global) * d_glob
