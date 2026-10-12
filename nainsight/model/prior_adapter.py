"""Patient-conditioned graph-prior adapter.

  prior embedding      u_n = LN(MLP(p_n) + E_type[n] + E_node[n])          p_n: GNN knowledge prior vector
  patient conditioning u_n' = u_n * (1 + gamma(f_n)) + beta(f_n)            f_n: node exposure profile at the
                                                                            landmark (+ mechanistic projection)
  sparse cross-attention  query = backbone state of the landmark token, keys/values = {u_n'}; only the top-k
                          logits per head are kept (the rest are masked) -> context c
  adaptive knowledge gate g = sigmoid(W [z, c, r]),  r = attention-weighted reliability of the attended nodes
  graph-prior state       s = g * c

Mechanistic projection: the prior effect matrices give the change of every state node expected from the
current exposures and current state deviations,
  dY_hat[tau, s] = sum_{d,k} B[d,k,tau,s] e[d,k] + sum_m A[tau,s,m] z_m      (development units)
expressed in development z-units and appended to the state-node profiles; a per-state calibration
factor (initialised at 1) is learned.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MechanisticProjector(nn.Module):
    def __init__(self, B: torch.Tensor, A: torch.Tensor, state_scale: torch.Tensor, lay: dict, n_drug: int):
        super().__init__()
        self.register_buffer("B", torch.nan_to_num(B.float()))            # D K T S
        self.register_buffer("A", torch.nan_to_num(A.float()))            # T S S
        self.register_buffer("scale", state_scale.float().clamp(min=1e-6))
        self.calib = nn.Parameter(torch.ones(state_scale.numel()))
        self.lay, self.n_drug, self.K = lay, n_drug, B.shape[1]

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        lay, D, K = self.lay, self.n_drug, self.K
        d0 = lay["dose0"]
        e = node_feat[:, :D, d0 + 1: d0 + 1 + K].clone()                      # 24-h exposure at level k
        unknown = node_feat[:, :D, d0]                                         # exposure with unknown dose
        e[..., (K + 1) // 2 - 1] = e[..., (K + 1) // 2 - 1] + unknown
        z = node_feat[:, D:, lay["z_last"]] * node_feat[:, D:, lay["measured"]]
        delta = torch.einsum("ndk,dkts->nts", e, self.B) + torch.einsum("nm,tsm->nts", z, self.A)
        return delta * self.calib / self.scale                               # n T S (z-units)


class PatientConditionedGraphPriorAdapter(nn.Module):
    def __init__(self, d: int, prior_vectors: torch.Tensor, node_type: torch.Tensor, reliability: torch.Tensor,
                 node_feat_dim: int, n_heads: int, top_k: int | None, gate_bias: float, trainable: bool,
                 dropout: float, mechanistic: MechanisticProjector | None, use_gate: bool = True):
        super().__init__()
        n_nodes, d_prior = prior_vectors.shape
        self.prior = nn.Parameter(prior_vectors.float().clone(), requires_grad=trainable)
        self.register_buffer("node_type", node_type.long())
        self.register_buffer("reliability", reliability.float())
        self.prior_proj = nn.Sequential(nn.Linear(d_prior, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(2, d)
        self.node_emb = nn.Embedding(n_nodes, d)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        self.mech = mechanistic
        n_mech = 0 if mechanistic is None else mechanistic.B.shape[2]
        self.n_drug = int((node_type == 0).sum())
        self.film = nn.Sequential(nn.Linear(node_feat_dim + n_mech, d), nn.GELU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.nu, self.nq = nn.LayerNorm(d), nn.LayerNorm(d)
        self.h, self.dh, self.top_k = n_heads, d // n_heads, top_k
        self.q, self.k, self.v, self.out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.use_gate = use_gate
        self.gate = nn.Sequential(nn.Linear(2 * d + 1, d), nn.GELU(), nn.Linear(d, d))
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, z: torch.Tensor, node_feat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        n, N, _ = node_feat.shape
        u0 = self.nu(self.prior_proj(self.prior) + self.type_emb(self.node_type) + self.node_emb.weight)   # N d
        cond = node_feat
        mech = None
        if self.mech is not None:
            mech = self.mech(node_feat)                                        # n T S
            pad = torch.zeros(n, self.n_drug, mech.shape[1], device=z.device, dtype=z.dtype)
            cond = torch.cat([node_feat, torch.cat([pad, mech.transpose(1, 2)], dim=1)], dim=-1)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        u = u0.unsqueeze(0) * (1.0 + gamma) + beta                             # n N d
        q = self.q(self.nq(z)).view(n, self.h, self.dh)
        k = self.k(u).view(n, N, self.h, self.dh).transpose(1, 2)              # n h N dh
        v = self.v(u).view(n, N, self.h, self.dh).transpose(1, 2)
        logits = torch.einsum("nhd,nhjd->nhj", q, k) / math.sqrt(self.dh)
        if self.top_k is not None and self.top_k < N:
            kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        ctx = self.out(torch.einsum("nhj,nhjd->nhd", self.drop(attn), v).reshape(n, -1))
        rel = (attn.mean(1) * self.reliability.unsqueeze(0)).sum(-1, keepdim=True)
        g = torch.sigmoid(self.gate(torch.cat([z, ctx, rel], dim=-1))) if self.use_gate else torch.ones_like(ctx)
        return g * ctx, {"prior_attention": attn.mean(1), "knowledge_gate": g.mean(-1), "mechanistic": mech}
