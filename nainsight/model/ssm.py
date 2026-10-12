"""Selective state-space local branch (input-dependent, time-aware discretisation).

Per head h (scalar decay, Mamba-2 style structured state-space duality):
    s_t = exp(dt_t * A_h) s_{t-1} + dt_t * B_t x_t^T          (state  N x P)
    y_t = C_t^T s_t + D_h x_t
dt_t = softplus(W x_t + b) * (1 + softplus(g_h) * log(1 + gap_t)) grows with the real time elapsed since the
previous event, so irregular sampling is reflected in how much state is retained; padded steps get dt = 0
(no decay, no write).  The scan is computed exactly with the chunked SSD algorithm (quadratic inside chunks,
linear recurrence across chunk boundaries).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


def segsum(x: torch.Tensor) -> torch.Tensor:
    """x[..., T] -> S[..., T, T] with S[i, j] = sum_{k=j+1..i} x_k (i >= j), -inf above the diagonal."""
    T = x.size(-1)
    xx = x.unsqueeze(-1).expand(*x.shape, T)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    xx = xx.masked_fill(~strict, 0.0)
    out = torch.cumsum(xx, dim=-2)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return out.masked_fill(~lower, float("-inf"))


def ssd_scan(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, chunk: int) -> torch.Tensor:
    """X (b,l,h,p) discretised input, A (b,l,h) log-decay (<= 0), B/C (b,l,h,n) -> Y (b,l,h,p)."""
    b, l, h, p = X.shape
    n = B.shape[-1]
    pad = (-l) % chunk
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    c = X.shape[1] // chunk
    X = X.reshape(b, c, chunk, h, p)
    B = B.reshape(b, c, chunk, h, n)
    C = C.reshape(b, c, chunk, h, n)
    A = A.reshape(b, c, chunk, h).permute(0, 3, 1, 2)                       # b h c l
    A_cum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))                                                 # b h c l l
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp(A_cum[..., -1:] - A_cum)                        # b h c l
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cum[..., -1], (1, 0))))           # b h (c+1) (c+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(A_cum))
    return (y_diag + y_off).reshape(b, c * chunk, h, p)[:, :l]


def sequential_scan(X, A, B, C):
    """Reference recurrence used by the tests."""
    b, l, h, p = X.shape
    s = torch.zeros(b, h, p, B.shape[-1], dtype=X.dtype, device=X.device)
    ys = []
    for t in range(l):
        s = torch.exp(A[:, t]).unsqueeze(-1).unsqueeze(-1) * s + torch.einsum("bhp,bhn->bhpn", X[:, t], B[:, t])
        ys.append(torch.einsum("bhpn,bhn->bhp", s, C[:, t]))
    return torch.stack(ys, dim=1)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, headdim: int = 32, d_conv: int = 4,
                 chunk: int = 64, time_aware: bool = True, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "expand * d_model must be divisible by headdim"
        self.nheads, self.headdim, self.d_state, self.chunk = self.d_inner // headdim, headdim, d_state, chunk
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))           # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.time_aware = time_aware
        self.gap_gain = nn.Parameter(torch.zeros(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, gap_h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        z, xbc, dt = torch.split(self.in_proj(x), [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :l].transpose(1, 2))
        xs, Bm, Cm = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        if self.time_aware:
            dt = dt * (1.0 + F.softplus(self.gap_gain) * torch.log1p(gap_h).unsqueeze(-1))
        dt = dt * pad_mask.unsqueeze(-1).to(dt.dtype)
        A = -torch.exp(self.A_log.float())
        xs = xs.reshape(b, l, self.nheads, self.headdim)
        Bh = Bm.unsqueeze(2).expand(b, l, self.nheads, self.d_state)
        Ch = Cm.unsqueeze(2).expand(b, l, self.nheads, self.d_state)
        y = ssd_scan(xs * dt.unsqueeze(-1), A * dt, Bh, Ch, self.chunk)
        y = y + xs * self.D.view(1, 1, -1, 1)
        y = self.norm(y.reshape(b, l, self.d_inner) * F.silu(z))
        return self.out_proj(y)
