from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from backend.app.services.ranking.learned.attention import DistanceAwareAttention
from backend.app.services.ranking.learned.fusion import BilinearFusion


class LearnedRankingModel(nn.Module):
    """Rank candidate vectors, optionally conditioned on real user histories.

    ``history_seqs`` and ``history_distances`` contain one variable-length tensor
    per candidate. Non-empty histories require aligned distances. Only the most
    recent ``max_history`` items are encoded, so overlong inputs have stable and
    documented behavior.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        max_history: int = 10,
        rho: float = 0.5,
        distance_scale: float = 10_000.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2:
            raise ValueError("embed_dim must be a positive even integer")
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.embed_dim = embed_dim
        self.max_history = max_history
        self.bilinear_fusion = BilinearFusion(embed_dim)
        self.history_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.distance_attention = DistanceAwareAttention(
            embed_dim=embed_dim,
            window_size=max_history,
            rho=rho,
            distance_scale=distance_scale,
        )
        self.sequence_projection = nn.Linear(embed_dim, 1)
        self.score_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        user_vec: torch.Tensor,
        candidate_vec: torch.Tensor,
        objective_score: torch.Tensor,
        history_seqs: Sequence[torch.Tensor | None] | None = None,
        history_distances: Sequence[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = user_vec.shape[0] if user_vec.ndim > 0 else 0
        fusion_score, interaction = self.bilinear_fusion(
            user_vec, candidate_vec, objective_score
        )

        if history_seqs is None:
            if history_distances is not None:
                raise ValueError("history_distances requires history_seqs")
            sequence_score = objective_score.new_zeros((batch_size, 1))
        else:
            if len(history_seqs) != batch_size:
                raise ValueError("history_seqs must contain one item per candidate")
            if history_distances is not None and len(history_distances) != batch_size:
                raise ValueError(
                    "history_distances must contain one item per candidate"
                )
            sequence_score = self._encode_histories(
                history_seqs,
                history_distances,
                device=user_vec.device,
                dtype=user_vec.dtype,
            )

        features = torch.cat((objective_score, fusion_score, sequence_score), dim=1)
        final_score = self.score_head(features)
        return final_score, interaction, sequence_score

    def _encode_histories(
        self,
        history_seqs: Sequence[torch.Tensor | None],
        history_distances: Sequence[torch.Tensor | None] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scores: list[torch.Tensor] = []
        for index, sequence in enumerate(history_seqs):
            distances = (
                history_distances[index] if history_distances is not None else None
            )
            if sequence is None:
                if distances is not None:
                    raise ValueError("history_distances requires a history sequence")
                scores.append(torch.zeros((), device=device, dtype=dtype))
                continue
            if not isinstance(sequence, torch.Tensor) or (
                sequence.ndim != 2 or sequence.shape[1] != self.embed_dim
            ):
                raise ValueError(
                    f"history sequence must have shape (sequence, {self.embed_dim})"
                )
            if distances is not None and (
                not isinstance(distances, torch.Tensor)
                or distances.ndim != 1
                or distances.shape[0] != sequence.shape[0]
            ):
                raise ValueError(
                    "history sequence and distances must have the same length "
                    "and distances must be one-dimensional"
                )
            if sequence.shape[0] == 0:
                scores.append(torch.zeros((), device=device, dtype=dtype))
                continue
            if distances is None:
                raise ValueError("non-empty history requires history_distances")

            recent_sequence = sequence[-self.max_history :].to(
                device=device, dtype=dtype
            )
            recent_distances = distances[-self.max_history :]
            encoded, _ = self.history_encoder(recent_sequence.unsqueeze(0))
            encoded = encoded.squeeze(0)
            context, _ = self.distance_attention(
                encoded[-1], encoded, recent_distances
            )
            scores.append(torch.sigmoid(self.sequence_projection(context)).squeeze(0))

        if not scores:
            return torch.empty((0, 1), device=device, dtype=dtype)
        return torch.stack(scores).unsqueeze(1)
