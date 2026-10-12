"""Global branch: causal Transformer with a learned, per-head bias over log-spaced elapsed-time buckets."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBucketBias(nn.Module):
    def __init__(self, n_heads: int, boundaries_h: list[float]):
        super().__init__()
        self.register_buffer("bounds", torch.tensor(boundaries_h, dtype=torch.float32), persistent=False)
        self.bias = nn.Embedding(len(boundaries_h) + 1, n_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dt = (t.unsqueeze(-1) - t.unsqueeze(-2)).abs()                 # b l l
        return self.bias(torch.bucketize(dt, self.bounds)).permute(0, 3, 1, 2)   # b h l l


def causal_attention_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """Additive mask (b, 1, l, l): key j visible to query i iff j <= i and j is a real token."""
    b, l = pad_mask.shape
    causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=pad_mask.device))
    allowed = causal.unsqueeze(0) & pad_mask.bool().unsqueeze(1)
    allowed = allowed | torch.eye(l, dtype=torch.bool, device=pad_mask.device).unsqueeze(0)  # never empty rows
    mask = torch.zeros(b, 1, l, l, device=pad_mask.device)
    return mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh, self.p = n_heads, d // n_heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.p if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, l, d))


class CausalTransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), attn_bias))
        return x + self.drop(self.ff(self.n2(x)))
