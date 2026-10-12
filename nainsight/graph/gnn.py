"""Edge-conditioned graph encoder and self-supervised pre-training of the knowledge prior vectors.

Message passing  m_{u->v} = MLP([h_v, h_u, e_uv]) * sigmoid(W [h_v, h_u, e_uv]),  e_uv = W_e a_uv + E_type
                 h_v <- LayerNorm(h_v + MLP([h_v, mean_u m_{u->v}]))
Pre-training objectives on drug -> state edges (full-batch, the graph is small):
  * masked spectrum reconstruction  - attributes of a random subset of edges (and of their reverse
    edges) are hidden and reconstructed from the endpoint embeddings (MSE);
  * effect-direction classification - decrease / no clear effect / increase (class-balanced CE);
  * edge-consistency contrast        - true (u, v, a_uv) triples versus triples whose spectrum is taken
    from another edge (BCE).
The encoder output for every node (unmasked graph, eval mode) is the knowledge prior vector.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG = logging.getLogger("nainsight.gnn")


class EdgeConditionedLayer(nn.Module):
    def __init__(self, d: int, d_edge: int, n_types: int, dropout: float):
        super().__init__()
        self.edge_in = nn.Linear(d_edge, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.msg = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Linear(3 * d, d)
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, edge_index, edge_attr, edge_type):
        src, dst = edge_index
        e = self.edge_in(edge_attr) + self.type_emb(edge_type)
        z = torch.cat([h[dst], h[src], e], dim=-1)
        m = self.msg(z) * torch.sigmoid(self.gate(z))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones(dst.numel(), device=h.device, dtype=h.dtype)).clamp(min=1.0)
        return self.norm(h + self.upd(torch.cat([h, agg / deg.unsqueeze(-1)], dim=-1)))


class GraphPriorEncoder(nn.Module):
    def __init__(self, n_nodes: int, d_node: int, d_edge: int, hidden: int, out_dim: int, layers: int,
                 n_types: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_node, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.node_id = nn.Embedding(n_nodes, hidden)
        nn.init.normal_(self.node_id.weight, std=0.02)
        self.layers = nn.ModuleList([EdgeConditionedLayer(hidden, d_edge, n_types, dropout) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x, edge_index, edge_attr, edge_type):
        h = self.inp(x) + self.node_id.weight
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_type)
        return self.out(h)


class PretrainHeads(nn.Module):
    def __init__(self, d: int, d_edge: int):
        super().__init__()
        self.recon = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d_edge))
        self.sign = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3))
        self.link = nn.Sequential(nn.Linear(2 * d + d_edge, d), nn.GELU(), nn.Linear(d, 1))

    @staticmethod
    def pair(hu, hv):
        return torch.cat([hu, hv, hu * hv], dim=-1)


def pretrain_graph_prior(graph: dict, cfg: dict, seed: int, device: str = "cpu") -> tuple[np.ndarray, dict, dict]:
    gc = cfg["graph"]["gnn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    x = torch.tensor(graph["x"], dtype=torch.float32, device=dev)
    ei = torch.tensor(graph["edge_index"], dtype=torch.long, device=dev)
    ea = torch.tensor(graph["edge_attr"], dtype=torch.float32, device=dev)
    et = torch.tensor(graph["edge_type"], dtype=torch.long, device=dev)
    lab = torch.tensor(graph["sign_label"], dtype=torch.long, device=dev)
    d_edge = ea.shape[1]
    fwd = torch.nonzero(et == 0).squeeze(-1)                       # drug -> state edges
    # index of the reverse (state -> drug) edge of every forward edge
    pairs = {(int(u), int(v)): i for i, (u, v) in enumerate(graph["edge_index"].T)}
    rev = torch.tensor([pairs.get((int(graph["edge_index"][1, i]), int(graph["edge_index"][0, i])), -1)
                        for i in fwd.tolist()], dtype=torch.long, device=dev)

    enc = GraphPriorEncoder(x.shape[0], x.shape[1], d_edge + 1, gc["hidden"], gc["out_dim"], gc["layers"],
                            dropout=gc["dropout"]).to(dev)
    heads = PretrainHeads(gc["out_dim"], d_edge).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=gc["lr"], weight_decay=gc["weight_decay"])
    counts = torch.bincount(lab[fwd], minlength=3).float().clamp(min=1.0)
    cls_w = (counts.sum() / (3.0 * counts)).to(dev)
    w = gc["loss_weights"]
    history = []
    for epoch in range(int(gc["epochs"])):
        enc.train(); heads.train()
        n_mask = max(1, int(round(gc["mask_rate"] * len(fwd))))
        mi = torch.tensor(rng.choice(len(fwd), n_mask, replace=False), device=dev)
        masked = fwd[mi]
        attr = torch.cat([ea, torch.zeros(len(ea), 1, device=dev)], dim=-1)
        hide = torch.cat([masked, rev[mi][rev[mi] >= 0]])
        attr[hide] = 0.0
        attr[hide, -1] = 1.0                                       # mask flag channel
        h = enc(x, ei, attr, et)
        hu, hv = h[ei[0, masked]], h[ei[1, masked]]
        loss_recon = F.mse_loss(heads.recon(heads.pair(hu, hv)), ea[masked])
        hu_all, hv_all = h[ei[0, fwd]], h[ei[1, fwd]]
        loss_sign = F.cross_entropy(heads.sign(heads.pair(hu_all, hv_all)), lab[fwd], weight=cls_w)
        perm = torch.tensor(rng.permutation(len(fwd)), device=dev)
        pos = heads.link(torch.cat([hu_all, hv_all, ea[fwd]], dim=-1)).squeeze(-1)
        neg = heads.link(torch.cat([hu_all, hv_all, ea[fwd][perm]], dim=-1)).squeeze(-1)
        loss_link = F.binary_cross_entropy_with_logits(torch.cat([pos, neg]),
                                                       torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        loss = w["recon"] * loss_recon + w["sign"] * loss_sign + w["link"] * loss_link
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(heads.parameters()), 1.0)
        opt.step()
        if epoch % 50 == 0 or epoch == int(gc["epochs"]) - 1:
            rec = {"epoch": epoch, "loss": loss.item(), "recon": loss_recon.item(), "sign": loss_sign.item(),
                   "link": loss_link.item()}
            history.append(rec)
            LOG.info("gnn epoch %d | %s", epoch, {k: round(v, 4) for k, v in rec.items() if k != "epoch"})
    enc.eval()
    with torch.no_grad():
        attr = torch.cat([ea, torch.zeros(len(ea), 1, device=dev)], dim=-1)
        vectors = enc(x, ei, attr, et).cpu().numpy()
    return vectors, {"history": history}, {k: v.cpu() for k, v in enc.state_dict().items()}


def random_prior_like(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Ablation control: Gaussian vectors with the per-dimension mean/sd of the learned prior."""
    rng = np.random.default_rng(seed + 7919)
    return (rng.standard_normal(vectors.shape) * vectors.std(axis=0, keepdims=True)
            + vectors.mean(axis=0, keepdims=True)).astype(np.float32)
