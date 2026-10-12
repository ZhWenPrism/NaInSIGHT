"""Clinical event embedding: one vector per token from concept, ingredient, modality, route, phase, dose level,
value (with concept-conditioned FiLM), sodium load and continuous time (absolute and inter-event gap)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import Time2Vec


class ClinicalEventEmbedding(nn.Module):
    def __init__(self, n_concepts: int, n_ingredients: int, n_modalities: int, n_routes: int, n_phases: int,
                 n_dose_levels: int, d: int, dropout: float):
        super().__init__()
        self.concept = nn.Embedding(n_concepts, d, padding_idx=0)
        self.ingredient = nn.Embedding(n_ingredients, d, padding_idx=0)
        self.modality = nn.Embedding(n_modalities, d)
        self.route = nn.Embedding(n_routes + 1, d, padding_idx=0)
        self.phase = nn.Embedding(n_phases + 1, d)
        self.dose = nn.Embedding(n_dose_levels + 1, d, padding_idx=0)
        self.value = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.t_abs = Time2Vec(d // 2, scale_h=24.0)
        self.t_gap = Time2Vec(d - d // 2, scale_h=6.0)
        self.time_proj = nn.Linear(d, d)
        self.na_load = nn.Linear(1, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def gaps(time: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Hours elapsed since the previous token of the same sequence (0 for the first token and padding)."""
        prev = torch.cat([time[:, :1], time[:, :-1]], dim=1)
        return ((time - prev).clamp(min=0.0) * pad_mask).float()

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.concept(b["concept"])
        gamma, beta = self.film(c).chunk(2, dim=-1)
        v = self.value(torch.stack([b["value"], b["value_mask"]], dim=-1)) * (1.0 + gamma) + beta
        gap = self.gaps(b["time"], b["pad_mask"].float())
        te = self.time_proj(torch.cat([self.t_abs(b["time"]), self.t_gap(torch.log1p(gap))], dim=-1))
        e = (c + self.ingredient(b["ingredient"]) + self.modality(b["modality"]) + self.route(b["route"])
             + self.phase(b["phase"]) + self.dose(b["dose_level"]) + v + te + self.na_load(b["na_load"].unsqueeze(-1)))
        e = self.drop(self.norm(e)) * b["pad_mask"].unsqueeze(-1)
        return e, gap
