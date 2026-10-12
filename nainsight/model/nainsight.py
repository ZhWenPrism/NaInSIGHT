"""NaInSIGHT: clinical event embedding -> NaFormer backbone -> landmark readout
-> [graph-prior adapter | static encoder | mode and landmark-time embeddings] -> fusion -> coherent heads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..concepts import MODALITIES, PHASES, ROUTES
from ..data.tokenizer import node_feature_layout
from .backbone import NaFormerBackbone
from .embedding import ClinicalEventEmbedding
from .heads import CoherentMultiTaskHead
from .layers import Time2Vec
from .prior_adapter import MechanisticProjector, PatientConditionedGraphPriorAdapter


@dataclass
class PriorPack:
    vectors: np.ndarray          # N x d_prior
    node_type: np.ndarray        # N   (0 drug, 1 state)
    reliability: np.ndarray      # N
    B: np.ndarray                # D x K x T x S   (development units and lags)
    A: np.ndarray                # T x S x S
    dev_lags: np.ndarray         # T

    @classmethod
    def load(cls, graph_dir: str | Path, vectors_file: str = "prior_vectors.npz", key: str = "vectors") -> "PriorPack":
        graph_dir = Path(graph_dir)
        g = np.load(graph_dir / "graph.npz", allow_pickle=False)
        pm = np.load(graph_dir / "prior_matrices.npz", allow_pickle=False)
        vec = np.load(graph_dir / vectors_file, allow_pickle=False)[key]
        n_drug = int(g["n_drug"])
        node_type = np.r_[np.zeros(n_drug), np.ones(len(g["node_names"]) - n_drug)].astype(np.int64)
        return cls(vec, node_type, g["reliability"], pm["B_dev"], pm["A_dev"], pm["dev_lags"])


class NaInSIGHT(nn.Module):
    def __init__(self, cfg: dict, vocab_sizes: dict, n_static: int, dose_levels: int, prior: PriorPack | None,
                 state_scale: np.ndarray | None = None, n_horizons: int = 2):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        abl = set(m.get("ablation") or [])
        self.ablation = abl
        self.embed = ClinicalEventEmbedding(vocab_sizes["concept"], vocab_sizes["ingredient"], len(MODALITIES),
                                            len(ROUTES), len(PHASES), dose_levels, d, m["dropout"])
        self.backbone = NaFormerBackbone(d, m["n_stages"], m["ssm"], m["transformer"], m["interaction"], m["dropout"], abl)
        lay = node_feature_layout(dose_levels)
        self.use_prior = prior is not None and "no_graph_prior" not in abl
        if self.use_prior:
            vectors = prior.vectors
            if "random_prior" in abl:
                rng = np.random.default_rng(cfg.get("seed", 0) + 7919)
                vectors = (rng.standard_normal(vectors.shape) * vectors.std(0, keepdims=True)
                           + vectors.mean(0, keepdims=True)).astype(np.float32)
            mech = None
            if m["adapter"].get("mechanistic", True) and "no_mechanistic" not in abl:
                scale = torch.tensor(state_scale if state_scale is not None else np.ones(prior.B.shape[-1]))
                mech = MechanisticProjector(torch.tensor(prior.B), torch.tensor(prior.A), scale, lay,
                                            int((prior.node_type == 0).sum()))
            top_k = None if "dense_prior_attention" in abl else m["adapter"]["top_k"]
            self.adapter = PatientConditionedGraphPriorAdapter(
                d, torch.tensor(vectors), torch.tensor(prior.node_type), torch.tensor(prior.reliability),
                lay["dim"], m["adapter"]["n_heads"], top_k, m["adapter"]["gate_bias_init"],
                m["adapter"].get("prior_trainable", False), m["dropout"], mech, use_gate="no_gate" not in abl)
        self.use_static = "no_static" not in abl and n_static > 0
        if self.use_static:
            self.static_enc = nn.Sequential(nn.Linear(n_static, d), nn.GELU(), nn.Dropout(m["dropout"]), nn.Linear(d, d))
        self.mode_emb = nn.Embedding(2, d)
        self.lmk_time = Time2Vec(d, scale_h=24.0)
        fuse_in = d * (3 + int(self.use_prior) + int(self.use_static))
        self.fusion = nn.Sequential(nn.LayerNorm(fuse_in), nn.Linear(fuse_in, m["fusion_dim"]), nn.GELU(),
                                    nn.Dropout(m["dropout"]), nn.Linear(m["fusion_dim"], m["fusion_dim"]), nn.GELU())
        self.heads = CoherentMultiTaskHead(m["fusion_dim"], m["head_hidden"], n_horizons, m["dropout"],
                                           cfg["train"]["loss"].get("nadir_sigma_floor", 0.5))

    def forward(self, b: dict) -> dict:
        x, gap = self.embed(b)
        z = self.backbone(x, gap, b["time"], b["pad_mask"])
        zr = z[b["lmk_batch"], b["lmk_pos"]]                                   # landmark query states
        parts, aux = [zr], {}
        if self.use_prior:
            s, aux = self.adapter(zr, b["node_feat"])
            parts.append(s)
        if self.use_static:
            parts.append(self.static_enc(b["static"])[b["lmk_batch"]])
        parts += [self.mode_emb(b["lmk_mode"]), self.lmk_time(b["lmk_time"])]
        out = self.heads(self.fusion(torch.cat(parts, dim=-1)), b["lmk_na_last"], b["lmk_na_mask"])
        out.update({k: v for k, v in aux.items() if v is not None})
        return out
