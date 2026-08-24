from typing import TYPE_CHECKING, Any

from backend.app.services.ranking.service import RankingService, build_ranking_service

if TYPE_CHECKING:
    from backend.app.services.ranking.learned.reranker import (
        LearnedReranker,
        build_model_inputs,
    )


def __getattr__(name: str) -> Any:
    """Expose learned helpers lazily so off mode does not import the model stack."""

    if name in {"LearnedReranker", "build_model_inputs"}:
        from backend.app.services.ranking.learned import reranker

        return getattr(reranker, name)
    raise AttributeError(name)


__all__ = [
    "RankingService",
    "build_ranking_service",
    "LearnedReranker",
    "build_model_inputs",
]
