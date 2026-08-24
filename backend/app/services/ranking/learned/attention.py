from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DistanceAwareAttention(nn.Module):
    """Attend to recent history with distance as an additive logit penalty.

    Distances are expected in the same unit as ``distance_scale``. Negative and
    negative-infinite values are treated as zero; NaN, positive infinity, and
    values above the scale are treated as ``distance_scale``. Thus malformed or
    extreme distances deterministically receive a penalty without producing
    NaN/Inf outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int = 10,
        rho: float = 0.5,
        distance_scale: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if rho < 0:
            raise ValueError("rho must be non-negative")
        if distance_scale <= 0:
            raise ValueError("distance_scale must be positive")

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.rho = float(rho)
        self.distance_scale = float(distance_scale)
        self.relevance_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        history_hidden: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != (self.embed_dim,):
            raise ValueError(f"query must have shape ({self.embed_dim},)")
        if history_hidden.ndim != 2 or history_hidden.shape[1] != self.embed_dim:
            raise ValueError(
                f"history_hidden must have shape (sequence, {self.embed_dim})"
            )
        if distances.ndim != 1 or distances.shape[0] != history_hidden.shape[0]:
            raise ValueError("distances and history_hidden must have the same length")
        if history_hidden.shape[0] == 0:
            return torch.zeros_like(query), query.new_empty((0,))

        window_hidden = history_hidden[-self.window_size :]
        window_distances = distances[-self.window_size :].to(
            device=query.device, dtype=query.dtype
        )
        clean_distances = torch.nan_to_num(
            window_distances,
            nan=self.distance_scale,
            posinf=self.distance_scale,
            neginf=0.0,
        ).clamp(min=0.0, max=self.distance_scale)
        normalized_distance = clean_distances / self.distance_scale

        relevance_logits = self.relevance_projection(window_hidden) @ query
        attention_logits = relevance_logits - self.rho * normalized_distance
        weights = F.softmax(attention_logits, dim=0)
        context = (weights.unsqueeze(1) * window_hidden).sum(dim=0)
        return context, weights
