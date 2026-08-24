from __future__ import annotations

import torch
from torch import nn


class BilinearFusion(nn.Module):
    """Combine an objective score with learned user-candidate interactions."""

    def __init__(self, embed_dim: int, init_identity: bool = True) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = embed_dim
        self.interaction_matrix = nn.Parameter(torch.empty(embed_dim, embed_dim))
        if init_identity:
            nn.init.eye_(self.interaction_matrix)
        else:
            nn.init.normal_(self.interaction_matrix, std=0.01)
        self.projection = nn.Linear(2, 1)

    def forward(
        self,
        user_vec: torch.Tensor,
        candidate_vec: torch.Tensor,
        objective_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_vectors = (user_vec.ndim == 2 and user_vec.shape[1] == self.embed_dim)
        if not expected_vectors or candidate_vec.shape != user_vec.shape:
            raise ValueError(
                f"user_vec and candidate_vec must have shape (batch, {self.embed_dim})"
            )
        if objective_score.shape != (user_vec.shape[0], 1):
            raise ValueError("objective_score must have shape (batch, 1)")

        interaction = (
            (user_vec @ self.interaction_matrix) * candidate_vec
        ).sum(dim=1, keepdim=True)
        probability = torch.sigmoid(
            self.projection(torch.cat((objective_score, interaction), dim=1))
        )
        return probability, interaction
