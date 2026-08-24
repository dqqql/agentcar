from __future__ import annotations

import math
import threading
import time

import torch

from backend.app.models.adapter import CandidatePoi, CandidatePoolMeta, CandidatePoolResult
from backend.app.models.extract import (
    AlgorithmInput,
    SearchContext,
    SequenceModelInput,
    SubjectivePreferenceInput,
)
from backend.app.models.ranking import RankingRequest, RankingResult
from backend.app.services.ranking.learned.checkpoint import (
    CheckpointLoadResult,
    CheckpointMetadata,
)
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


class StubModel(torch.nn.Module):
    def __init__(self, outputs: list[list[float]] | Exception):
        super().__init__()
        self.outputs = outputs
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, user_vec, *args, **kwargs):
        if isinstance(self.outputs, Exception):
            raise self.outputs
        self.batch_sizes.append(user_vec.shape[0])
        values = self.outputs[self.calls]
        self.calls += 1
        return (
            torch.tensor(values[: user_vec.shape[0]], dtype=torch.float32).reshape(-1, 1),
            None,
            None,
        )


def loaded(model: torch.nn.Module) -> CheckpointLoadResult:
    metadata = CheckpointMetadata(
        model_version="v1",
        embed_dim=4,
        max_history=3,
        vocabulary={},
        vocabulary_size=0,
        epoch=1,
    )
    return CheckpointLoadResult(
        loaded=True, model=model, metadata=metadata, version="v1", sha256="abc"
    )


def test_off_is_exact_rule_behavior_and_never_calls_loader() -> None:
    baseline_request = request_for(
        [candidate("a", tags=[]), candidate("b", tags=[])]
    )
    off_request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    baseline = RankingService().rank_candidates(baseline_request)
    called = False

    def loader(*args):
        nonlocal called
        called = True
        raise AssertionError

    result = RankingService(model_mode="off", checkpoint_loader=loader).rank_candidates(
        off_request
    )
    assert result.model_dump() == baseline.model_dump()
    assert off_request.model_dump() == baseline_request.model_dump()
    assert called is False


def test_shadow_runs_model_but_only_adds_diagnostics() -> None:
    baseline_request = request_for(
        [candidate("a", tags=[]), candidate("b", tags=[])]
    )
    shadow_request = request_for(
        [candidate("a", tags=[]), candidate("b", tags=[])]
    )
    baseline = RankingService().rank_candidates(baseline_request)
    result = RankingService(
        model_mode="shadow", checkpoint_loader=lambda *_: loaded(StubModel([[0, 1]]))
    ).rank_candidates(shadow_request)
    baseline_payload = baseline.model_dump()
    shadow_payload = result.model_dump()
    baseline_debug = baseline_payload.pop("debug_meta")
    shadow_debug = shadow_payload.pop("debug_meta")
    assert shadow_payload == baseline_payload
    assert shadow_request.model_dump() == baseline_request.model_dump()
    assert {key: shadow_debug[key] for key in baseline_debug} == baseline_debug
    assert set(shadow_debug) - set(baseline_debug) == {
        "model_mode",
        "model_version",
        "checkpoint_hash",
        "rerank_candidate_count",
        "fallback_reason",
        "inference_ms",
        "ranking_changed",
    }
    assert result.debug_meta["model_mode"] == "shadow"
    assert result.debug_meta["ranking_changed"] is True
    assert result.debug_meta["rerank_candidate_count"] == 2


def test_shadow_reports_false_when_counterfactual_order_is_unchanged() -> None:
    request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    baseline = RankingService().rank_candidates(request.model_copy(deep=True))
    result = RankingService(
        model_mode="shadow", checkpoint_loader=lambda *_: loaded(StubModel([[1, 0]]))
    ).rank_candidates(request.model_copy(deep=True))
    assert result.ranked_spot_candidates == baseline.ranked_spot_candidates
    assert result.debug_meta["ranking_changed"] is False


def test_rerank_blends_top_n_without_changing_candidate_set_or_tail() -> None:
    request = request_for([candidate(x, tags=[]) for x in ["a", "b", "c"]])
    baseline = RankingService().rank_candidates(request.model_copy(deep=True))
    result = RankingService(
        model_mode="rerank",
        model_top_n=2,
        model_blend_weight=1,
        checkpoint_loader=lambda *_: loaded(StubModel([[0, 1]])),
    ).rank_candidates(request.model_copy(deep=True))
    before = [x.poi_id for x in baseline.ranked_spot_candidates]
    after = [x.poi_id for x in result.ranked_spot_candidates]
    assert after == [before[1], before[0], before[2]]
    assert sorted(after) == sorted(before)
    assert result.ranked_spot_candidates[2] == baseline.ranked_spot_candidates[2]
    assert (
        result.ranked_spot_candidates[0].score_breakdown["rule"]
        == baseline.ranked_spot_candidates[1].score
    )
    assert result.debug_meta["ranking_changed"] is True


def test_loader_failure_falls_back_with_diagnostics() -> None:
    failure = CheckpointLoadResult(loaded=False, fallback_reason="missing_file")
    request = request_for([candidate("a", tags=[])])
    baseline = RankingService().rank_candidates(request.model_copy(deep=True))
    result = RankingService(
        model_mode="rerank", checkpoint_loader=lambda *_: failure
    ).rank_candidates(request.model_copy(deep=True))
    assert result.ranked_spot_candidates == baseline.ranked_spot_candidates
    assert result.debug_meta["fallback_reason"] == "missing_file"


def test_group_two_failure_is_transactional() -> None:
    request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    request.candidate_pool.food_candidates = [
        candidate("food-a", tags=[]).model_copy(update={"poi_type": "food"}),
        candidate("food-b", tags=[]).model_copy(update={"poi_type": "food"}),
    ]
    baseline = RankingService().rank_candidates(request.model_copy(deep=True))
    model = StubModel([[0, 1], [math.nan, 0]])
    result = RankingService(
        model_mode="rerank", checkpoint_loader=lambda *_: loaded(model)
    ).rank_candidates(request.model_copy(deep=True))
    assert result.ranked_spot_candidates == baseline.ranked_spot_candidates
    assert result.ranked_food_candidates == baseline.ranked_food_candidates
    assert result.debug_meta["fallback_reason"] == "nonfinite_score"


def test_model_never_sees_or_restores_candidates_excluded_by_rule_top_k() -> None:
    request = request_for([candidate(x, tags=[]) for x in ["a", "b", "c"]])
    model = StubModel([[0, 1]])
    result = RankingService(
        top_k=2,
        model_mode="rerank",
        model_top_n=10,
        checkpoint_loader=lambda *_: loaded(model),
    ).rank_candidates(request)
    assert model.batch_sizes == [2]
    assert {item.poi_id for item in result.ranked_spot_candidates} == {"a", "b"}


def test_blocked_inference_times_out_to_exact_rule_result() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, user_vec, *args, **kwargs):
            self.calls += 1
            entered.set()
            release.wait(1.0)
            return torch.zeros((user_vec.shape[0], 1)), None, None

    request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    baseline = RankingService().rank_candidates(request.model_copy(deep=True))
    model = BlockingModel()
    service = RankingService(
        model_mode="rerank",
        model_timeout_ms=20,
        checkpoint_loader=lambda *_: loaded(model),
    )
    started = time.perf_counter()
    try:
        result = service.rank_candidates(request.model_copy(deep=True))
        elapsed = time.perf_counter() - started
        assert entered.wait(0.2)
        assert elapsed < 0.5
        assert result.ranked_spot_candidates == baseline.ranked_spot_candidates
        assert result.ranked_food_candidates == baseline.ranked_food_candidates
        assert result.ranked_hotel_candidates == baseline.ranked_hotel_candidates
        assert result.debug_meta["fallback_reason"] == "timeout"
        second = service.rank_candidates(request.model_copy(deep=True))
        assert second.ranked_spot_candidates == baseline.ranked_spot_candidates
        assert second.debug_meta["fallback_reason"] == "timeout"
        assert model.calls == 1
    finally:
        release.set()
