from __future__ import annotations

import math
import threading
import time

import pytest
import torch

from backend.app.services.ranking.learned.checkpoint import CheckpointMetadata
from backend.app.services.ranking.learned.reranker import (
    LearnedReranker,
    build_model_inputs,
)
from tests.test_ranking_service import candidate, request_for


class FixedModel(torch.nn.Module):
    def __init__(self, scores: list[float] | Exception):
        super().__init__()
        self.scores = scores

    def forward(self, *args, **kwargs):
        if isinstance(self.scores, Exception):
            raise self.scores
        count = args[0].shape[0]
        return torch.tensor(self.scores[:count]).reshape(-1, 1), None, None


@pytest.fixture
def metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        model_version="v1",
        embed_dim=4,
        max_history=3,
        vocabulary={"museum": 0, "park": 1, "old": 2},
        vocabulary_size=3,
        epoch=1,
    )


def test_feature_builder_is_deterministic_and_handles_id_only_history(metadata) -> None:
    request = request_for(
        [candidate("museum", tags=["museum"]), candidate("park", tags=["park"])],
        preference_terms=["museum"],
        history=["old"],
    )
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    first = build_model_inputs(ranked, request.candidate_pool.algorithm_input, metadata)
    second = build_model_inputs(ranked, request.candidate_pool.algorithm_input, metadata)
    assert torch.equal(first.user_vec, second.user_vec)
    assert torch.equal(first.candidate_vec, second.candidate_vec)
    assert first.history_seqs[0] is not None
    assert first.history_distances[0].tolist() == [0.0]


def test_reranker_validates_shape_nonfinite_exception_and_elapsed_timeout(metadata) -> None:
    request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    algorithm_input = request.candidate_pool.algorithm_input

    assert LearnedReranker(FixedModel([math.nan, 0.2]), metadata).score(
        ranked, algorithm_input
    ).fallback_reason == "nonfinite_score"
    assert LearnedReranker(FixedModel(RuntimeError("boom")), metadata).score(
        ranked, algorithm_input
    ).fallback_reason == "inference_error"
    assert LearnedReranker(FixedModel([0.1]), metadata).score(
        ranked, algorithm_input
    ).fallback_reason == "invalid_output_shape"

    ticks = iter([1.0, 1.2])
    timed = LearnedReranker(
        FixedModel([0.1, 0.2]), metadata, timeout_ms=10, clock=lambda: next(ticks)
    ).score(ranked, algorithm_input)
    assert timed.fallback_reason == "timeout"


def test_timeout_returns_before_blocked_daemon_worker_and_trips_circuit(metadata) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.worker_daemon: bool | None = None

        def forward(self, user_vec, *args, **kwargs):
            self.calls += 1
            self.worker_daemon = threading.current_thread().daemon
            entered.set()
            try:
                release.wait(1.0)
                return torch.zeros((user_vec.shape[0], 1)), None, None
            finally:
                finished.set()

    request = request_for([candidate("a", tags=[])])
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    model = BlockingModel()
    reranker = LearnedReranker(model, metadata, timeout_ms=20)
    started = time.perf_counter()
    try:
        first = reranker.score(ranked, request.candidate_pool.algorithm_input)
        elapsed = time.perf_counter() - started
        assert entered.wait(0.2)
        assert first.fallback_reason == "timeout"
        assert elapsed < 0.5
        assert model.worker_daemon is True

        second = reranker.score(ranked, request.candidate_pool.algorithm_input)
        assert second.fallback_reason == "timeout"
        assert model.calls == 1
    finally:
        release.set()
        assert finished.wait(0.5)


def test_simultaneous_scores_reserve_worker_before_start(metadata, monkeypatch) -> None:
    start_reserved = threading.Event()
    allow_start = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    original_start = threading.Thread.start
    intercept_lock = threading.Lock()
    intercepted = False
    model_calls = 0

    def controlled_start(thread: threading.Thread) -> None:
        nonlocal intercepted
        should_pause = False
        if thread.name == "learned-ranking-inference":
            with intercept_lock:
                if not intercepted:
                    intercepted = True
                    should_pause = True
        if should_pause:
            start_reserved.set()
            assert allow_start.wait(0.5)
        original_start(thread)

    class BlockingModel(torch.nn.Module):
        def forward(self, user_vec, *args, **kwargs):
            nonlocal model_calls
            with intercept_lock:
                model_calls += 1
            entered.set()
            try:
                release.wait(1.0)
                return torch.zeros((user_vec.shape[0], 1)), None, None
            finally:
                worker_finished.set()

    monkeypatch.setattr(threading.Thread, "start", controlled_start)
    request = request_for([candidate("a", tags=[])])
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    reranker = LearnedReranker(BlockingModel(), metadata, timeout_ms=100)
    results = []
    first = threading.Thread(
        target=lambda: results.append(
            reranker.score(ranked, request.candidate_pool.algorithm_input)
        )
    )
    second = threading.Thread(
        target=lambda: results.append(
            reranker.score(ranked, request.candidate_pool.algorithm_input)
        )
    )
    try:
        first.start()
        assert start_reserved.wait(0.2)
        second.start()
        time.sleep(0.05)
        allow_start.set()
        assert entered.wait(0.2)
        first.join(0.5)
        second.join(0.5)
        assert len(results) == 2
        assert all(result.fallback_reason == "timeout" for result in results)
        assert model_calls == 1
    finally:
        allow_start.set()
        release.set()
        first.join(0.5)
        second.join(0.5)
        assert worker_finished.wait(0.5)


@pytest.mark.parametrize(
    "malformed",
    [
        torch.tensor([[1 + 2j], [3 + 4j]]),
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 0]]),
            torch.tensor([0.1, 0.2]),
            (2, 1),
            check_invariants=True,
        ),
    ],
)
def test_malformed_tensor_output_fails_without_apparent_timeout(
    metadata, malformed
) -> None:
    class MalformedModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return malformed, None, None

    request = request_for([candidate("a", tags=[]), candidate("b", tags=[])])
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    result = LearnedReranker(MalformedModel(), metadata, timeout_ms=500).score(
        ranked, request.candidate_pool.algorithm_input
    )
    assert result.fallback_reason in {"inference_error", "invalid_output"}


def test_model_is_moved_to_cpu_and_normalized_once_at_construction(metadata) -> None:
    class NormalizationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0
            self.eval_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls += 1
            return self

        def eval(self):
            self.eval_calls += 1
            return self

        def forward(self, user_vec, *args, **kwargs):
            return torch.zeros((user_vec.shape[0], 1)), None, None

    request = request_for([candidate("a", tags=[])])
    ranked = __import__(
        "backend.app.services.ranking.service", fromlist=["RankingService"]
    ).RankingService().rank_candidates(request).ranked_spot_candidates
    model = NormalizationModel()
    reranker = LearnedReranker(model, metadata)
    assert reranker.score(ranked, request.candidate_pool.algorithm_input).scores == [0.0]
    assert reranker.score(ranked, request.candidate_pool.algorithm_input).scores == [0.0]
    assert model.to_calls == 1
    assert model.eval_calls == 1
