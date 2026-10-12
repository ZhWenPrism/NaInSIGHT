"""NaFormer dual-scale temporal backbone: stages of [selective SSM | causal Transformer | interaction]."""
from __future__ import annotations

import torch
import torch.nn as nn

from .interaction import MultiResolutionInteraction
from .ssm import SelectiveSSM
from .transformer import CausalTransformerBlock, TimeBucketBias, causal_attention_mask


class SSMBlock(nn.Module):
    def __init__(self, d: int, cfg: dict, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ssm = SelectiveSSM(d, cfg["d_state"], cfg["expand"], cfg["headdim"], cfg["d_conv"], cfg["chunk"],
                                cfg.get("time_aware", True))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gap, pad_mask):
        return x + self.drop(self.ssm(self.norm(x), gap, pad_mask)) * pad_mask.unsqueeze(-1)


class NaFormerBackbone(nn.Module):
    def __init__(self, d: int, n_stages: int, ssm_cfg: dict, tf_cfg: dict, inter_cfg: dict, dropout: float,
                 ablation: set):
        super().__init__()
        self.use_local = "no_ssm" not in ablation
        self.use_global = "no_transformer" not in ablation
        self.use_inter = "no_interaction" not in ablation and self.use_local and self.use_global
        self.local = nn.ModuleList([SSMBlock(d, ssm_cfg, dropout) for _ in range(n_stages)]) if self.use_local else None
        self.glob = nn.ModuleList([CausalTransformerBlock(d, tf_cfg["n_heads"], tf_cfg["ff_mult"], dropout)
                                   for _ in range(n_stages)]) if self.use_global else None
        self.time_bias = TimeBucketBias(tf_cfg["n_heads"], tf_cfg["time_buckets_h"]) if self.use_global else None
        self.inter = nn.ModuleList([MultiResolutionInteraction(d, inter_cfg["n_heads"], inter_cfg["fine_h"],
                                                               inter_cfg["coarse_h"], dropout)
                                    for _ in range(n_stages)]) if self.use_inter else None
        n_branches = int(self.use_local) + int(self.use_global)
        if n_branches == 0:
            raise ValueError("at least one temporal branch must be enabled")
        self.merge = nn.Sequential(nn.Linear(n_branches * d, d), nn.LayerNorm(d))
        self.n_stages = n_stages

    def forward(self, x: torch.Tensor, gap: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor):
        pm = pad_mask.to(x.dtype)
        h_loc, h_glob = x, x
        bias = (self.time_bias(t) + causal_attention_mask(pad_mask)) if self.use_global else None
        for s in range(self.n_stages):
            if self.use_local:
                h_loc = self.local[s](h_loc, gap, pm)
            if self.use_global:
                h_glob = self.glob[s](h_glob, bias)
            if self.use_inter:
                h_loc, h_glob = self.inter[s](h_loc, h_glob, t, pad_mask)
        parts = ([h_loc] if self.use_local else []) + ([h_glob] if self.use_global else [])
        return self.merge(torch.cat(parts, dim=-1))
