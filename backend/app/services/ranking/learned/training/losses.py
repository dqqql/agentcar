"""Differentiable ranking objectives without privacy-related claims."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_logistic_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rank every strictly more-relevant item above every less-relevant item."""

    scores = scores.reshape(-1)
    labels = labels.reshape(-1).to(device=scores.device, dtype=scores.dtype)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    differences = labels[:, None] - labels[None, :]
    positive, negative = torch.where(differences > 0)
    if positive.numel() == 0:
        return scores.sum() * 0.0
    return F.softplus(-(scores[positive] - scores[negative])).mean()


def listwise_softmax_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy between predicted and relevance-derived list distributions."""

    scores = scores.reshape(-1)
    labels = labels.reshape(-1).to(device=scores.device, dtype=scores.dtype)
    if scores.shape != labels.shape or scores.numel() == 0:
        raise ValueError("scores and labels must be aligned and non-empty")
    target = torch.softmax(labels, dim=0)
    return -(target * torch.log_softmax(scores, dim=0)).sum()
