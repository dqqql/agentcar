from __future__ import annotations

from backend.app.models.adapter import CandidatePoi, CandidatePoolMeta, CandidatePoolResult
from backend.app.models.extract import (
    AlgorithmInput,
    SearchContext,
    SequenceModelInput,
    SubjectivePreferenceInput,
)
from backend.app.models.ranking import RankingRequest, RankingResult
from backend.app.services.ranking.service import RankingService, build_ranking_service


def candidate(poi_id: str, *, tags: list[str], distance: int = 1000) -> CandidatePoi:
    return CandidatePoi(
        poi_id=poi_id,
        poi_type="spot",
        source_dataset="place",
        name=poi_id,
        center_distance_m=distance,
        rating=4.5,
        popularity=100,
        tags=tags,
    )


def request_for(
    candidates: list[CandidatePoi],
    *,
    preference_terms: list[str] | None = None,
    history: list[str] | None = None,
) -> RankingRequest:
    history = history or []
    algorithm_input = AlgorithmInput(
        search_context=SearchContext(destination_text="北京"),
        subjective_preference=SubjectivePreferenceInput(
            preference_terms=preference_terms or []
        ),
        sequence_model_input=SequenceModelInput(
            has_history=bool(history),
            historical_poi_ids=history,
        ),
    )
    return RankingRequest(
        candidate_pool=CandidatePoolResult(
            result_file_path="mock.json",
            algorithm_input=algorithm_input,
            meta=CandidatePoolMeta(search_radius_m=5000),
            spot_candidates=candidates,
        )
    )


def test_ranking_preserves_public_contract() -> None:
    result = build_ranking_service().rank_candidates(
        request_for([candidate("a", tags=["历史"])])
    )
    assert isinstance(result, RankingResult)
    assert result.source_candidate_pool_path == "mock.json"
    assert result.ranked_spot_candidates[0].candidate.poi_id == "a"
    assert result.ranked_food_candidates == []
    assert result.ranked_hotel_candidates == []


def test_ranking_top_k_and_ties_are_deterministic() -> None:
    service = RankingService(top_k=2)
    candidates = [
        candidate("c", tags=[], distance=1000),
        candidate("a", tags=[], distance=1000),
        candidate("b", tags=[], distance=1000),
    ]
    first = service.rank_candidates(request_for(candidates))
    second = service.rank_candidates(request_for(list(reversed(candidates))))
    assert [item.poi_id for item in first.ranked_spot_candidates] == ["a", "b"]
    assert [item.poi_id for item in second.ranked_spot_candidates] == ["a", "b"]
    assert [item.rank for item in first.ranked_spot_candidates] == [1, 2]


def test_preference_match_changes_ranking_and_breakdown_is_truthful() -> None:
    result = RankingService().rank_candidates(
        request_for(
            [candidate("matched", tags=["博物馆"]), candidate("other", tags=["公园"])],
            preference_terms=["博物馆"],
        )
    )
    assert result.ranked_spot_candidates[0].poi_id == "matched"
    breakdown = result.ranked_spot_candidates[0].score_breakdown
    expected = breakdown["alpha"] * breakdown["objective"] + (
        1 - breakdown["alpha"]
    ) * breakdown["subjective"]
    assert abs(result.ranked_spot_candidates[0].score - expected) < 0.001
    assert breakdown["sequence_weight"] == 0.0


def test_recent_history_penalizes_repeated_poi() -> None:
    result = RankingService(sequence_weight=0.2).rank_candidates(
        request_for(
            [candidate("recent", tags=[]), candidate("fresh", tags=[])],
            history=["old", "older", "recent"],
        )
    )
    assert [item.poi_id for item in result.ranked_spot_candidates] == ["fresh", "recent"]
    fresh, recent = result.ranked_spot_candidates
    assert fresh.score_breakdown["sequence"] == 1.0
    assert recent.score_breakdown["sequence"] == 0.0
    assert result.debug_meta["sequence_active"] is True
