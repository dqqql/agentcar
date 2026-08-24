from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from backend.app.services.ranking.learned.checkpoint import (
    MAX_CHECKPOINT_BYTES,
    CheckpointMetadata,
    load_learned_checkpoint,
)
from backend.app.services.ranking.learned.model import LearnedRankingModel
from backend.app.services.ranking.learned.reranker import build_model_inputs
from backend.app.services.ranking.learned.training.data import (
    TrainingDataset,
    TrainingQuery,
    build_vocabulary,
    chronological_split,
)
from backend.app.services.ranking.learned.training.losses import (
    listwise_softmax_loss,
    pairwise_logistic_loss,
)
from backend.app.services.ranking.learned.training.trainer import (
    RankingMetrics,
    TrainerConfig,
    evaluate_rankings,
    train_reranker,
)
from tests.test_ranking_service import candidate, request_for


def _metadata_for(train: TrainingDataset) -> CheckpointMetadata:
    vocabulary = build_vocabulary(train)
    return CheckpointMetadata(
        model_version="v1",
        embed_dim=8,
        max_history=3,
        vocabulary=vocabulary,
        vocabulary_size=len(vocabulary),
        epoch=0,
        dataset=train.name,
    )


def _query(day: int, *, labels: list[float] | None = None) -> TrainingQuery:
    labels = labels or [1.0, 0.0]
    request = request_for(
        [
            candidate(f"museum-{day}", tags=["museum"]),
            candidate(f"park-{day}", tags=["park"]),
        ],
        preference_terms=["museum"],
        history=["old"],
    )
    from backend.app.services.ranking.service import RankingService

    ranked = RankingService().rank_candidates(request).ranked_spot_candidates
    return TrainingQuery(
        query_id=f"q-{day}",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
        algorithm_input=request.candidate_pool.algorithm_input,
        candidates=ranked,
        labels=labels,
        categories=["culture", "nature"],
    )


def _zero_signal_query(day: int) -> TrainingQuery:
    raw = _query(day, labels=[0.0, 0.0]).model_dump()
    first = raw["candidates"][0]
    for ranked in raw["candidates"]:
        ranked["score"] = 0.5
        ranked["score_breakdown"]["objective"] = 0.5
        ranked["candidate"]["name"] = "identical"
        ranked["candidate"]["tags"] = ["identical"]
        ranked["candidate"]["address"] = first["candidate"]["address"]
        ranked["candidate"]["source_dataset"] = first["candidate"]["source_dataset"]
    return TrainingQuery.model_validate(raw)


def test_training_record_is_bounded_and_requires_aligned_candidates() -> None:
    with pytest.raises(ValidationError):
        TrainingQuery.model_validate(
            {**_query(0).model_dump(), "labels": [1.0]}
        )
    with pytest.raises(ValidationError):
        TrainingQuery.model_validate(
            {**_query(0).model_dump(), "labels": [2.0, 0.0]}
        )
    with pytest.raises(ValidationError):
        TrainingQuery.model_validate(
            {**_query(0).model_dump(), "query_id": "x" * 257}
        )

    oversized = _query(0).model_dump()
    oversized["candidates"][0]["candidate"]["tags"] = ["tag"] * 65
    with pytest.raises(ValidationError, match="tags"):
        TrainingQuery.model_validate(oversized)

    hostile = _query(0).model_dump()
    hostile["algorithm_input"]["sequence_model_input"]["historical_poi_ids"] = [
        "old"
    ] * 101
    with pytest.raises(ValidationError, match="history"):
        TrainingQuery.model_validate(hostile)

    nonfinite = _query(0).model_dump()
    nonfinite["candidates"][0]["score"] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        TrainingQuery.model_validate(nonfinite)

    future = _query(0).model_dump()
    future["timestamp"] = datetime(2200, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValidationError, match="timestamp"):
        TrainingQuery.model_validate(future)

    coercive = _query(0).model_dump()
    coercive["labels"] = ["1", 0.0]
    with pytest.raises(ValidationError):
        TrainingQuery.model_validate(coercive)

    coercive = _query(0).model_dump()
    coercive["algorithm_input"]["search_context"]["search_radius_m"] = "5000"
    with pytest.raises(ValidationError):
        TrainingQuery.model_validate(coercive)

    nested_extra = _query(0).model_dump()
    nested_extra["algorithm_input"]["search_context"]["search_radus_m"] = 5000
    with pytest.raises(ValidationError, match="extra"):
        TrainingQuery.model_validate(nested_extra)

    candidate_extra = _query(0).model_dump()
    candidate_extra["candidates"][0]["candidate"]["misspelled_tag"] = "x"
    with pytest.raises(ValidationError, match="extra"):
        TrainingQuery.model_validate(candidate_extra)


def test_dataset_reader_rejects_oversized_file_before_read(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.train_learned_reranker as script_module

    path = tmp_path / "oversized.json"
    path.write_bytes(b"x" * 33)
    monkeypatch.setattr(script_module, "MAX_DATASET_BYTES", 32)
    with pytest.raises(ValueError, match="too large"):
        script_module._read_dataset(path)


def test_vocabulary_is_derived_from_training_split_only(tmp_path: Path) -> None:
    train = TrainingDataset(name="train", queries=[_query(0)])
    later_data = _query(1).model_dump()
    later_data["candidates"][0]["candidate"]["tags"] = ["future-only"]
    later = TrainingQuery.model_validate(later_data)
    vocabulary = build_vocabulary(train)
    assert "old" in vocabulary
    assert "future-only" not in vocabulary

    metadata = CheckpointMetadata(
        model_version="v1",
        embed_dim=8,
        max_history=3,
        vocabulary=vocabulary,
        vocabulary_size=len(vocabulary),
        epoch=0,
        dataset="training-only-vocabulary",
    )
    result = train_reranker(
        train,
        TrainingDataset(name="validation", queries=[later]),
        TrainingDataset(name="test", queries=[_query(2)]),
        metadata,
        tmp_path / "best.pt",
        TrainerConfig(epochs=1),
    )
    payload = torch.load(result.checkpoint_path, weights_only=True)
    assert "future-only" not in payload["metadata"]["vocabulary"]


def test_direct_trainer_rejects_validation_or_test_enriched_vocabulary(
    tmp_path: Path,
) -> None:
    train = TrainingDataset(name="train", queries=[_query(0)])
    validation = TrainingDataset(name="validation", queries=[_query(1)])
    test = TrainingDataset(name="test", queries=[_query(2)])
    enriched = build_vocabulary(train)
    enriched["future-only"] = len(enriched)
    metadata = CheckpointMetadata(
        model_version="v1",
        embed_dim=4,
        max_history=3,
        vocabulary=enriched,
        vocabulary_size=len(enriched),
        epoch=0,
        dataset="leaky",
    )
    with pytest.raises(ValueError, match="training vocabulary"):
        train_reranker(
            train,
            validation,
            test,
            metadata,
            tmp_path / "leaky.pt",
            TrainerConfig(epochs=1),
        )


def test_chronological_split_has_strict_nonoverlapping_time_boundaries() -> None:
    dataset = TrainingDataset(name="chronological", queries=[_query(i) for i in range(10)])
    split = chronological_split(dataset, validation_fraction=0.2, test_fraction=0.2)
    assert len(split.train.queries) == 6
    assert len(split.validation.queries) == 2
    assert len(split.test.queries) == 2
    assert max(q.timestamp for q in split.train.queries) < min(
        q.timestamp for q in split.validation.queries
    )
    assert max(q.timestamp for q in split.validation.queries) < min(
        q.timestamp for q in split.test.queries
    )
    with pytest.raises(ValueError, match="distinct timestamps"):
        repeated = [_query(0), _query(1), _query(1)]
        repeated[1] = repeated[1].model_copy(update={"query_id": "q-1-a", "split_group_id": "g-1-a"})
        repeated[2] = repeated[2].model_copy(update={"query_id": "q-1-b", "split_group_id": "g-1-b"})
        chronological_split(
            TrainingDataset(name="small", queries=repeated)
        )


def test_chronological_split_keeps_coherent_multi_query_groups_whole() -> None:
    queries = [_query(i) for i in range(6)]
    same_group_second = _query(1).model_copy(
        update={
            "query_id": "q-0-second",
            "split_group_id": "session-0",
            "timestamp": queries[0].timestamp,
        }
    )
    queries[0] = queries[0].model_copy(update={"split_group_id": "session-0"})
    dataset = TrainingDataset(
        name="grouped", queries=[queries[0], same_group_second, *queries[2:]]
    )
    split = chronological_split(dataset)
    placements = [
        part
        for part in (split.train, split.validation, split.test)
        if any(query.split_group_id == "session-0" for query in part.queries)
    ]
    assert len(placements) == 1
    assert sum(
        query.split_group_id == "session-0" for query in placements[0].queries
    ) == 2

    incoherent = same_group_second.model_copy(
        update={"timestamp": queries[0].timestamp + timedelta(days=1)}
    )
    with pytest.raises(ValidationError, match="same timestamp"):
        TrainingDataset(name="bad-group", queries=[queries[0], incoherent])


def test_training_uses_exact_shared_builder_and_nonempty_history(monkeypatch, tmp_path: Path) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    calls = []

    def recording_builder(candidates, algorithm_input, metadata):
        result = build_model_inputs(candidates, algorithm_input, metadata)
        calls.append(result)
        return result

    monkeypatch.setattr(trainer_module, "build_model_inputs", recording_builder)
    queries = [_query(i) for i in range(6)]
    train = TrainingDataset(name="train", queries=queries[:2])
    result = train_reranker(
        train,
        TrainingDataset(name="validation", queries=queries[2:4]),
        TrainingDataset(name="test", queries=queries[4:]),
        _metadata_for(train),
        tmp_path / "best.pt",
        TrainerConfig(epochs=1, learning_rate=0.01, seed=7, patience=1),
    )
    assert calls
    assert all(item.history_seqs and item.history_seqs[0] is not None for item in calls)
    assert all(item.history_seqs[0].numel() > 0 for item in calls)
    assert result.history_gradient_norm > 0
    assert math.isfinite(result.history_gradient_norm)
    assert result.history_parameter_update_norm > 0
    assert result.checkpoint_path == tmp_path / "best.pt"


def test_training_refuses_missing_or_unknown_history(tmp_path: Path) -> None:
    for history in ([], ["unknown-history"]):
        raw = _query(0).model_dump()
        raw["algorithm_input"]["sequence_model_input"]["historical_poi_ids"] = history
        bad = TrainingQuery.model_validate(raw)
        train = TrainingDataset(name="train", queries=[bad])
        metadata = _metadata_for(train)
        if history:
            vocabulary = {
                token: index
                for index, token in enumerate(
                    token
                    for token in metadata.vocabulary
                    if token != "unknown-history"
                )
            }
            metadata = CheckpointMetadata(
                **{
                    **metadata.model_dump(),
                    "vocabulary": vocabulary,
                    "vocabulary_size": len(vocabulary),
                }
            )
        with pytest.raises(ValueError, match="history"):
            train_reranker(
                train,
                TrainingDataset(name="validation", queries=[_query(1)]),
                TrainingDataset(name="test", queries=[_query(2)]),
                metadata,
                tmp_path / "bad.pt",
                TrainerConfig(epochs=1),
            )
        assert not (tmp_path / "bad.pt").exists()


def test_training_rejects_cross_split_identity_or_timestamp_leakage(tmp_path: Path) -> None:
    train = TrainingDataset(name="train", queries=[_query(2)])
    validation = TrainingDataset(name="validation", queries=[_query(1)])
    test = TrainingDataset(name="test", queries=[_query(3)])
    with pytest.raises(ValueError, match="strictly later"):
        train_reranker(train, validation, test, _metadata_for(train), tmp_path / "bad.pt")

    validation = TrainingDataset(
        name="validation",
        queries=[_query(3).model_copy(update={"query_id": train.queries[0].query_id})],
    )
    with pytest.raises(ValueError, match="query IDs"):
        train_reranker(train, validation, test, _metadata_for(train), tmp_path / "bad.pt")

    validation = TrainingDataset(
        name="validation",
        queries=[_query(1).model_copy(update={"split_group_id": train.queries[0].split_group_id})],
    )
    with pytest.raises(ValueError, match="split group"):
        train_reranker(train, validation, test, _metadata_for(train), tmp_path / "bad.pt")


def test_dataset_allows_duplicate_split_group_at_same_timestamp() -> None:
    first = _query(0)
    second = _query(1).model_copy(
        update={
            "query_id": "another-query",
            "split_group_id": first.split_group_id,
            "timestamp": first.timestamp,
        }
    )
    dataset = TrainingDataset(name="same-session", queries=[first, second])
    assert len(dataset.queries) == 2


def test_ranking_losses_are_finite_differentiable_and_handle_no_pairs() -> None:
    scores = torch.tensor([0.8, 0.1, 0.4], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.5])
    pairwise = pairwise_logistic_loss(scores, labels)
    listwise = listwise_softmax_loss(scores, labels)
    loss = pairwise + listwise
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    tied = torch.tensor([0.1, 0.2], requires_grad=True)
    zero = pairwise_logistic_loss(tied, torch.ones(2))
    zero.backward()
    assert float(zero.detach()) == 0.0
    assert tied.grad is not None


def test_mixed_zero_signal_and_informative_queries_train_but_all_zero_fails(
    tmp_path: Path,
) -> None:
    mixed = TrainingDataset(
        name="train", queries=[_zero_signal_query(0), _query(1)]
    )
    result = train_reranker(
        mixed,
        TrainingDataset(name="validation", queries=[_query(2)]),
        TrainingDataset(name="test", queries=[_query(3)]),
        _metadata_for(mixed),
        tmp_path / "mixed.pt",
        TrainerConfig(epochs=1),
    )
    assert result.history_gradient_norm > 0

    all_zero = TrainingDataset(name="train-zero", queries=[_zero_signal_query(0)])
    with pytest.raises(FloatingPointError, match="non-zero"):
        train_reranker(
            all_zero,
            TrainingDataset(name="validation", queries=[_query(1)]),
            TrainingDataset(name="test", queries=[_query(2)]),
            _metadata_for(all_zero),
            tmp_path / "zero.pt",
            TrainerConfig(epochs=1),
        )
    assert not (tmp_path / "zero.pt").exists()


def test_metrics_are_correct_bounded_and_deterministic_for_ties() -> None:
    metrics = evaluate_rankings(
        scores_by_query=[[0.5, 0.5, 0.1], []],
        labels_by_query=[[1.0, 0.0, 1.0], []],
        candidate_ids_by_query=[["a", "b", "c"], []],
        categories_by_query=[["x", "x", "y"], []],
        k=2,
        catalog_size=4,
        inference_latency_ms=[2.0, float("nan")],
    )
    assert metrics == RankingMetrics(
        recall_at_k=0.25,
        ndcg_at_k=0.3065735963827292,
        mrr=0.5,
        coverage=0.5,
        diversity=0.25,
        inference_latency_ms=1.0,
    )
    assert all(math.isfinite(value) for value in metrics.model_dump().values())
    with pytest.raises(ValidationError):
        RankingMetrics.model_validate(
            {**metrics.model_dump(), "coverage": float("nan")}
        )

    negative = evaluate_rankings(
        scores_by_query=[[-2.0, -1.0]],
        labels_by_query=[[0.0, 1.0]],
        candidate_ids_by_query=[["a", "b"]],
        categories_by_query=[["x", "y"]],
        k=1,
        catalog_size=2,
        inference_latency_ms=[0.0],
    )
    assert negative.recall_at_k == 1.0
    assert negative.ndcg_at_k == 1.0

    common = dict(
        scores_by_query=[[0.1]],
        labels_by_query=[[1.0]],
        candidate_ids_by_query=[["a"]],
        categories_by_query=[["x"]],
        inference_latency_ms=[0.0],
    )
    with pytest.raises(ValueError, match="k"):
        evaluate_rankings(**common, k=0, catalog_size=1)
    with pytest.raises(ValueError, match="catalog_size"):
        evaluate_rankings(**common, k=1, catalog_size=-1)


@pytest.mark.parametrize(
    "bad_label", [True, float("nan"), float("inf"), -0.1, "1"]
)
def test_metrics_reject_invalid_relevance_labels(bad_label) -> None:
    with pytest.raises(ValueError, match="labels"):
        evaluate_rankings(
            scores_by_query=[[0.1]],
            labels_by_query=[[bad_label]],
            candidate_ids_by_query=[["a"]],
            categories_by_query=[["x"]],
            k=1,
            catalog_size=1,
            inference_latency_ms=[0.0],
        )


def test_metrics_reject_catalog_smaller_than_observed_candidates() -> None:
    with pytest.raises(ValueError, match="catalog_size"):
        evaluate_rankings(
            scores_by_query=[[0.2, 0.1]],
            labels_by_query=[[1.0, 0.0]],
            candidate_ids_by_query=[["a", "b"]],
            categories_by_query=[["x", "y"]],
            k=2,
            catalog_size=1,
            inference_latency_ms=[0.0],
        )


def test_training_is_reproducible_and_checkpoint_matches_safe_schema(tmp_path: Path) -> None:
    queries = [_query(i) for i in range(6)]
    train = TrainingDataset(name="train", queries=queries[:2])
    args = (
        train,
        TrainingDataset(name="validation", queries=queries[2:4]),
        TrainingDataset(name="test", queries=queries[4:]),
        _metadata_for(train),
    )
    config = TrainerConfig(epochs=2, learning_rate=0.01, seed=19, patience=2)
    first = train_reranker(*args, tmp_path / "one.pt", config)
    second = train_reranker(*args, tmp_path / "two.pt", config)
    assert first.validation_metrics.model_dump(exclude={"inference_latency_ms"}) == second.validation_metrics.model_dump(exclude={"inference_latency_ms"})
    assert first.test_metrics.model_dump(exclude={"inference_latency_ms"}) == second.test_metrics.model_dump(exclude={"inference_latency_ms"})
    assert first.validation_metrics.inference_latency_ms >= 0
    assert second.test_metrics.inference_latency_ms >= 0

    assert first.sha256 == second.sha256
    assert first.checkpoint_path.read_bytes() == second.checkpoint_path.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()

    payload = torch.load(first.checkpoint_path, map_location="cpu", weights_only=True)
    assert set(payload) == {"model_state_dict", "metadata"}
    assert "optimizer_state_dict" not in repr(payload.keys())
    assert all("latency" not in key for key in payload["metadata"]["metrics"])
    assert all(type(value) is torch.Tensor for value in payload["model_state_dict"].values())
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in payload["model_state_dict"].values())
    assert first.metadata_path.read_text(encoding="utf-8")
    loaded = load_learned_checkpoint(first.checkpoint_path, first.sha256)
    assert loaded.loaded and loaded.sha256 == first.sha256
    assert list(tmp_path.glob("*.pt")) == [tmp_path / "one.pt", tmp_path / "two.pt"]


def test_future_test_catalog_cannot_change_validation_or_checkpoint(tmp_path: Path) -> None:
    queries = [_query(i) for i in range(5)]
    train = TrainingDataset(name="train", queries=queries[:2])
    validation = TrainingDataset(name="validation", queries=queries[2:4])
    future_unseen = TrainingDataset(name="test", queries=[queries[4]])

    reused_raw = queries[4].model_dump()
    reused_raw["candidates"][0]["poi_id"] = queries[0].candidates[0].poi_id
    reused_raw["candidates"][0]["candidate"]["poi_id"] = queries[0].candidates[0].poi_id
    reused_raw["candidates"][1]["poi_id"] = queries[0].candidates[1].poi_id
    reused_raw["candidates"][1]["candidate"]["poi_id"] = queries[0].candidates[1].poi_id
    future_reused = TrainingDataset(
        name="test", queries=[TrainingQuery.model_validate(reused_raw)]
    )

    config = TrainerConfig(epochs=1, seed=31)
    unseen = train_reranker(
        train, validation, future_unseen, _metadata_for(train), tmp_path / "unseen.pt", config
    )
    reused = train_reranker(
        train, validation, future_reused, _metadata_for(train), tmp_path / "reused.pt", config
    )

    assert unseen.validation_metrics.model_dump(
        exclude={"inference_latency_ms"}
    ) == reused.validation_metrics.model_dump(exclude={"inference_latency_ms"})
    assert unseen.test_metrics.coverage != reused.test_metrics.coverage
    assert unseen.metadata_path.read_bytes() == reused.metadata_path.read_bytes()
    assert unseen.sha256 == reused.sha256
    assert unseen.checkpoint_path.read_bytes() == reused.checkpoint_path.read_bytes()


def test_no_history_parameter_update_preserves_existing_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    output.write_bytes(b"prior-valid-checkpoint")
    sidecar.write_bytes(b'{"prior": true}')
    prior_checkpoint = output.read_bytes()
    prior_sidecar = sidecar.read_bytes()

    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)
    queries = [_query(i) for i in range(3)]
    train = TrainingDataset(name="train", queries=[queries[0]])
    with pytest.raises(FloatingPointError, match="did not update"):
        train_reranker(
            train,
            TrainingDataset(name="validation", queries=[queries[1]]),
            TrainingDataset(name="test", queries=[queries[2]]),
            _metadata_for(train),
            output,
            TrainerConfig(epochs=1),
        )
    assert output.read_bytes() == prior_checkpoint
    assert sidecar.read_bytes() == prior_sidecar


@pytest.mark.parametrize("failure_point", ["stage_second", "replace_first", "replace_second"])
@pytest.mark.parametrize("with_prior", [False, True])
def test_pair_publication_rolls_back_every_failure_boundary(
    tmp_path: Path, monkeypatch, failure_point: str, with_prior: bool
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    checkpoint = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    if with_prior:
        checkpoint.write_bytes(b"old-checkpoint")
        sidecar.write_bytes(b"old-sidecar")

    real_stage = trainer_module._stage_bytes
    real_replace = trainer_module._replace_path
    stage_calls = 0
    replace_calls = 0

    def failing_stage(path, content, role):
        nonlocal stage_calls
        stage_calls += 1
        if failure_point == "stage_second" and stage_calls == 2:
            raise OSError("injected staging failure")
        return real_stage(path, content, role)

    def failing_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if failure_point == "replace_first" and replace_calls == 1:
            raise OSError("injected first replace failure")
        if failure_point == "replace_second" and replace_calls == 2:
            raise OSError("injected second replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(trainer_module, "_stage_bytes", failing_stage)
    monkeypatch.setattr(trainer_module, "_replace_path", failing_replace)
    with pytest.raises(OSError, match="injected"):
        trainer_module._publish_artifact_pair(
            checkpoint, b"new-checkpoint", sidecar, b"new-sidecar"
        )

    if with_prior:
        assert checkpoint.read_bytes() == b"old-checkpoint"
        assert sidecar.read_bytes() == b"old-sidecar"
    else:
        assert not checkpoint.exists()
        assert not sidecar.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_persistent_rollback_fault_retains_recovery_backup(
    tmp_path: Path, monkeypatch
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    checkpoint = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    checkpoint.write_bytes(b"old-checkpoint")
    sidecar.write_bytes(b"old-sidecar")
    real_replace = trainer_module._replace_path
    calls = 0

    def persistent_failure(source, destination):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("persistent replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(trainer_module, "_replace_path", persistent_failure)
    with pytest.raises(OSError, match=r"recovery.*\.bak"):
        trainer_module._publish_artifact_pair(
            checkpoint, b"new-checkpoint", sidecar, b"new-sidecar"
        )

    backups = list(tmp_path.glob(".*.bak"))
    assert any(path.read_bytes() == b"old-checkpoint" for path in backups)
    assert sidecar.read_bytes() == b"old-sidecar"
    assert not list(tmp_path.glob(".*.tmp"))


def test_oversized_prior_artifact_rejects_before_live_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    checkpoint = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    checkpoint.write_bytes(b"12345")
    sidecar.write_bytes(b"old-sidecar")
    monkeypatch.setattr(trainer_module, "MAX_ARTIFACT_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds"):
        trainer_module._publish_artifact_pair(
            checkpoint, b"new-checkpoint", sidecar, b"new-sidecar"
        )
    assert checkpoint.read_bytes() == b"12345"
    assert sidecar.read_bytes() == b"old-sidecar"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_schema_valid_large_sidecar_can_publish_and_republish(tmp_path: Path) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    vocabulary = {
        f"{index:05d}-{'x' * 100}": index for index in range(50_000)
    }
    metadata = CheckpointMetadata(
        model_version="v1",
        embed_dim=4,
        max_history=3,
        vocabulary=vocabulary,
        vocabulary_size=len(vocabulary),
        epoch=1,
        dataset="maximum-valid-vocabulary",
    )
    sidecar_bytes = json.dumps(
        metadata.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    assert 4 * 1024 * 1024 < len(sidecar_bytes) <= MAX_CHECKPOINT_BYTES

    checkpoint = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    trainer_module._publish_artifact_pair(
        checkpoint, b"checkpoint-one", sidecar, sidecar_bytes
    )
    trainer_module._publish_artifact_pair(
        checkpoint, b"checkpoint-two", sidecar, sidecar_bytes
    )
    assert checkpoint.read_bytes() == b"checkpoint-two"
    assert sidecar.read_bytes() == sidecar_bytes
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_oversized_new_artifact_rejects_before_live_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    checkpoint = tmp_path / "best.pt"
    sidecar = tmp_path / "best.pt.metadata.json"
    checkpoint.write_bytes(b"old")
    sidecar.write_bytes(b"old")
    monkeypatch.setattr(trainer_module, "MAX_ARTIFACT_BYTES", 4)
    with pytest.raises(ValueError, match="new checkpoint exceeds"):
        trainer_module._publish_artifact_pair(
            checkpoint, b"12345", sidecar, b"new"
        )
    assert checkpoint.read_bytes() == b"old"
    assert sidecar.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.bak"))


def test_stage_bytes_cleans_partial_file_on_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    def partial_then_fail(stream, content):
        stream.write(content[:3])
        raise OSError("partial write failure")

    monkeypatch.setattr(
        trainer_module, "_write_staged_content", partial_then_fail
    )
    with pytest.raises(OSError, match="partial write"):
        trainer_module._stage_bytes(tmp_path / "best.pt", b"abcdef", "checkpoint")
    assert not list(tmp_path.iterdir())


def test_training_restores_python_and_torch_rng_after_success_and_exception(
    tmp_path: Path, monkeypatch
) -> None:
    train = TrainingDataset(name="train", queries=[_query(0)])
    validation = TrainingDataset(name="validation", queries=[_query(1)])
    test = TrainingDataset(name="test", queries=[_query(2)])
    metadata = _metadata_for(train)

    random.seed(12345)
    torch.manual_seed(54321)
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    train_reranker(
        train,
        validation,
        test,
        metadata,
        tmp_path / "success.pt",
        TrainerConfig(epochs=1),
    )
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    def fail_step(self, closure=None):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(torch.optim.Adam, "step", fail_step)
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected optimizer"):
        train_reranker(
            train,
            validation,
            test,
            metadata,
            tmp_path / "failure.pt",
            TrainerConfig(epochs=1),
        )
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_state_capture_failure_releases_training_lock(
    tmp_path: Path, monkeypatch
) -> None:
    import backend.app.services.ranking.learned.training.trainer as trainer_module

    train = TrainingDataset(name="train", queries=[_query(0)])
    validation = TrainingDataset(name="validation", queries=[_query(1)])
    test = TrainingDataset(name="test", queries=[_query(2)])
    metadata = _metadata_for(train)
    real_get_rng_state = torch.random.get_rng_state
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("capture failed")
        return real_get_rng_state()

    monkeypatch.setattr(torch.random, "get_rng_state", fail_once)
    with pytest.raises(RuntimeError, match="capture failed"):
        train_reranker(
            train,
            validation,
            test,
            metadata,
            tmp_path / "first.pt",
            TrainerConfig(epochs=1),
        )

    outcome: list[BaseException | None] = []

    def second_call() -> None:
        try:
            train_reranker(
                train,
                validation,
                test,
                metadata,
                tmp_path / "second.pt",
                TrainerConfig(epochs=1),
            )
            outcome.append(None)
        except BaseException as error:
            outcome.append(error)

    worker = threading.Thread(target=second_call, daemon=True)
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), "training lock leaked after capture failure"
    assert outcome == [None]
    assert (tmp_path / "second.pt").exists()


def test_smoke_cli_is_explicitly_synthetic_and_normal_mode_requires_paths(tmp_path: Path) -> None:
    script = Path("scripts/train_learned_reranker.py")
    smoke = subprocess.run(
        [sys.executable, str(script), "--smoke-test", "--output", str(tmp_path / "smoke.pt")],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert "SYNTHETIC SMOKE ONLY" in smoke.stdout
    assert "not recommendation evidence" in smoke.stdout.lower()
    assert (tmp_path / "smoke.pt").is_file()
    assert json.loads((tmp_path / "smoke.pt.metadata.json").read_text(encoding="utf-8"))["model_version"] == "v1"

    normal = subprocess.run(
        [sys.executable, str(script)], text=True, capture_output=True, timeout=30
    )
    assert normal.returncode != 0
    assert "--dataset" in normal.stderr and "--output" in normal.stderr
