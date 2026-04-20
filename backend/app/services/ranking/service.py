from __future__ import annotations

from backend.app.models.ranking import RankingRequest, RankingResult


class RankingService:
    """Scaffold for the core ranking layer."""

    def rank_candidates(self, request: RankingRequest) -> RankingResult:
        candidate_pool = request.candidate_pool
        return RankingResult(
            source_candidate_pool_path=candidate_pool.result_file_path,
            debug_meta={
                "status": "scaffold_only",
                "message": "Ranking algorithm has not been implemented yet.",
                "spot_candidate_count": len(candidate_pool.spot_candidates),
                "food_candidate_count": len(candidate_pool.food_candidates),
                "hotel_candidate_count": len(candidate_pool.hotel_candidates),
            },
        )


def build_ranking_service() -> RankingService:
    return RankingService()
