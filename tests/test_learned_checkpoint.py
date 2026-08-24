from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from backend.app.core.config import Settings
from backend.app.services.ranking.learned import checkpoint as checkpoint_module
from backend.app.services.ranking.learned.checkpoint import (
    LEGACY_SYNTHETIC_SHA256,
    inspect_legacy_checkpoint,
    load_learned_checkpoint,
)
from backend.app.services.ranking.learned.model import LearnedRankingModel


def _metadata(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model_version": "v1",
        "embed_dim": 8,
        "max_history": 10,
        "vocabulary": {"museum": 0, "park": 1},
        "vocabulary_size": 2,
        "epoch": 3,
        "metrics": {"validation_loss": 0.25},
        "dataset": "unit-test",
    }
    value.update(updates)
    return value


def _write_checkpoint(
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
    state_dict: dict[str, object] | None = None,
) -> str:
    model = LearnedRankingModel(embed_dim=8, max_history=10)
    torch.save(
        {
            "metadata": metadata if metadata is not None else _metadata(),
            "model_state_dict": (
                state_dict if state_dict is not None else model.state_dict()
            ),
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_valid_corrected_checkpoint_on_cpu(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path)

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is True
    assert isinstance(result.model, LearnedRankingModel)
    assert result.model.training is False
    assert result.version == "v1"
    assert result.sha256 == digest
    assert result.metadata is not None
    assert result.metadata.vocabulary_size == 2


def test_deserialization_is_weights_only_and_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path)
    real_load = torch.load
    calls: list[dict[str, object]] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    assert load_learned_checkpoint(path, digest).loaded
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("missing", "missing_file"),
        ("hash", "hash_mismatch"),
        ("corrupt", "corrupt_checkpoint"),
    ],
)
def test_file_failures_return_safe_fallback(
    tmp_path: Path, case: str, expected_reason: str
) -> None:
    path = tmp_path / "reranker.pt"
    digest = "0" * 64
    if case == "hash":
        digest = _write_checkpoint(path)
        digest = "f" * 64
    elif case == "corrupt":
        path.write_bytes(b"not a torch checkpoint")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.model is None
    assert result.fallback_reason == expected_reason


def test_hash_is_checked_before_torch_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    _write_checkpoint(path)

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("torch.load must not run before hash validation")

    monkeypatch.setattr(torch, "load", forbidden_load)
    result = load_learned_checkpoint(path, "0" * 64)
    assert result.fallback_reason == "hash_mismatch"


def test_path_replacement_cannot_change_verified_deserialization_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    replacement = tmp_path / "replacement.pt"
    digest = _write_checkpoint(path, metadata=_metadata(epoch=3))
    _write_checkpoint(replacement, metadata=_metadata(epoch=99))
    replacement_bytes = replacement.read_bytes()
    real_safe_load = checkpoint_module._safe_torch_load

    def replace_path_then_load(source: object) -> object:
        path.write_bytes(replacement_bytes)
        return real_safe_load(source)

    monkeypatch.setattr(checkpoint_module, "_safe_torch_load", replace_path_then_load)

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is True
    assert result.metadata is not None
    assert result.metadata.epoch == 3


def test_oversized_checkpoint_bytes_fall_back_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    path.write_bytes(b"x" * 33)
    monkeypatch.setattr(checkpoint_module, "MAX_CHECKPOINT_BYTES", 32, raising=False)

    result = load_learned_checkpoint(
        path, hashlib.sha256(path.read_bytes()).hexdigest()
    )

    assert result.loaded is False
    assert result.fallback_reason == "checkpoint_too_large"


def test_excessive_aggregate_tensor_payload_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path)
    monkeypatch.setattr(
        checkpoint_module, "MAX_TOTAL_TENSOR_ELEMENTS", 1, raising=False
    )

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.fallback_reason == "resource_limit"


def test_filesystem_oserror_returns_safe_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reranker.pt"
    path.write_bytes(b"unreadable")

    def denied_open(*args: object, **kwargs: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied_open)

    result = load_learned_checkpoint(path, "0" * 64)

    assert result.loaded is False
    assert result.fallback_reason == "unreadable_file"


def test_incompatible_model_version_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path, metadata=_metadata(model_version="v999"))
    result = load_learned_checkpoint(path, digest)
    assert result.loaded is False
    assert result.fallback_reason == "incompatible_version"


def test_wrong_parameter_names_fall_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    state = dict(LearnedRankingModel(embed_dim=8).state_dict())
    state["unexpected.weight"] = state.pop(next(iter(state)))
    digest = _write_checkpoint(path, state_dict=state)
    assert load_learned_checkpoint(path, digest).fallback_reason == "parameter_names"


def test_empty_checkpoint_root_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    torch.save({}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert load_learned_checkpoint(path, digest).fallback_reason == "schema_error"


def test_empty_metadata_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path, metadata={})

    assert load_learned_checkpoint(path, digest).fallback_reason == "schema_error"


def test_empty_state_dict_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path, state_dict={})

    assert load_learned_checkpoint(path, digest).fallback_reason == "parameter_names"


def test_wrong_parameter_shape_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    state = dict(LearnedRankingModel(embed_dim=8).state_dict())
    name = next(iter(state))
    state[name] = torch.zeros(1)
    digest = _write_checkpoint(path, state_dict=state)
    assert load_learned_checkpoint(path, digest).fallback_reason == "parameter_shape"


def test_non_tensor_parameter_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    state: dict[str, object] = dict(LearnedRankingModel(embed_dim=8).state_dict())
    state[next(iter(state))] = "hostile"
    digest = _write_checkpoint(path, state_dict=state)
    assert load_learned_checkpoint(path, digest).fallback_reason == "parameter_type"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_state_tensor_falls_back(
    tmp_path: Path, bad_value: float
) -> None:
    path = tmp_path / "reranker.pt"
    state = dict(LearnedRankingModel(embed_dim=8).state_dict())
    name = next(iter(state))
    state[name] = state[name].clone()
    state[name].view(-1)[0] = bad_value
    digest = _write_checkpoint(path, state_dict=state)

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.fallback_reason == "parameter_type"


def test_wrong_state_tensor_dtype_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "reranker.pt"
    state = dict(LearnedRankingModel(embed_dim=8).state_dict())
    name = next(iter(state))
    state[name] = state[name].to(torch.float64)
    digest = _write_checkpoint(path, state_dict=state)

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.fallback_reason == "parameter_type"


def test_parameter_subclass_is_not_accepted_as_a_plain_state_tensor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reranker.pt"
    state = dict(LearnedRankingModel(embed_dim=8).state_dict())
    name = next(iter(state))
    state[name] = torch.nn.Parameter(state[name].clone())
    digest = _write_checkpoint(path, state_dict=state)

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.fallback_reason == "parameter_type"


@pytest.mark.parametrize(
    "vocabulary",
    [
        {1: 0},
        {"museum": "0"},
        {"museum": 0, "park": 2},
        {"museum": 0, "park": 0},
    ],
    ids=["non-string-key", "non-integer-value", "noncontiguous", "duplicate"],
)
def test_hostile_vocabulary_entries_fall_back(
    tmp_path: Path, vocabulary: dict[object, object]
) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(
        path,
        metadata=_metadata(vocabulary=vocabulary, vocabulary_size=len(vocabulary)),
    )

    result = load_learned_checkpoint(path, digest)

    assert result.loaded is False
    assert result.fallback_reason == "schema_error"


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(vocabulary_size=3),
        _metadata(vocabulary={f"tag-{index}": index for index in range(50_001)}, vocabulary_size=50_001),
        _metadata(metrics={"loss": float("nan")}),
        _metadata(dataset="x" * 257),
        _metadata(extra_hostile_field={"nested": "hostile"}),
    ],
)
def test_bad_or_oversized_metadata_falls_back(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    path = tmp_path / "reranker.pt"
    digest = _write_checkpoint(path, metadata=metadata)
    result = load_learned_checkpoint(path, digest)
    assert result.loaded is False
    assert result.fallback_reason == "schema_error"


def test_canonical_legacy_inspection_never_deserializes_optimizer_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("backend/app/services/ranking/checkpoints/best.pt")

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("canonical legacy inspection must be hash-only")

    with monkeypatch.context() as context:
        context.setattr(torch, "load", forbidden_load)
        inspection = inspect_legacy_checkpoint(path)
        load_result = load_learned_checkpoint(path, LEGACY_SYNTHETIC_SHA256)

    assert inspection.valid is True
    assert inspection.classification == "legacy/synthetic/smoke-only"
    assert inspection.model_parameter_count == 17
    assert inspection.epoch == 2
    assert load_result.loaded is False
    assert load_result.model is None
    assert load_result.fallback_reason == "legacy_not_production"
    assert load_result.version == "v0.1.0-synthetic"


def test_noncanonical_optimizer_heavy_legacy_file_is_rejected_without_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "config": {},
            "embed_dim": 8,
            "epoch": 2,
            "metrics": {},
            "model_state_dict": {
                f"parameter-{index}": torch.zeros(1) for index in range(17)
            },
            "num_tags": 0,
            "optimizer_state_dict": {
                "state": {0: {"momentum": torch.zeros(1024, 1024)}}
            },
            "poi_list": [],
            "saved_at": "2026-01-01T00:00:00+00:00",
            "tag_vocab": {},
        },
        path,
    )

    def forbidden_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("noncanonical legacy files must not be deserialized")

    monkeypatch.setattr(torch, "load", forbidden_load)

    inspection = inspect_legacy_checkpoint(path)

    assert inspection.valid is False
    assert inspection.fallback_reason == "hash_mismatch"


def test_arbitrary_legacy_digest_does_not_claim_known_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "config": {},
            "embed_dim": 8,
            "epoch": 2,
            "metrics": {},
            "model_state_dict": {
                f"parameter-{index}": torch.zeros(1) for index in range(17)
            },
            "num_tags": 0,
            "optimizer_state_dict": "ignored",
            "poi_list": [],
            "saved_at": "2026-01-01T00:00:00+00:00",
            "tag_vocab": {},
        },
        path,
    )
    inspection = inspect_legacy_checkpoint(path)

    assert inspection.valid is False
    assert inspection.classification == "legacy/unknown/smoke-only"
    assert inspection.fallback_reason == "hash_mismatch"


def test_pydantic_v2_dependency_is_explicit() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "pydantic>=2,<3" in requirements


def test_invalid_ranking_environment_is_safely_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANKING_MODEL_MODE", "unsafe")
    monkeypatch.setenv("RANKING_MODEL_TOP_N", "999999")
    monkeypatch.setenv("RANKING_MODEL_BLEND_WEIGHT", "nan")

    settings = Settings()

    assert settings.ranking_model_mode == "off"
    assert settings.ranking_model_top_n == 100
    assert settings.ranking_model_blend_weight == 0.5


def test_settings_read_ranking_environment_at_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANKING_MODEL_MODE", "shadow")
    monkeypatch.setenv("RANKING_MODEL_PATH", "models/custom.pt")
    monkeypatch.setenv("RANKING_MODEL_SHA256", "a" * 64)
    monkeypatch.setenv("RANKING_MODEL_TOP_N", "12")
    monkeypatch.setenv("RANKING_MODEL_BLEND_WEIGHT", "0.35")

    settings = Settings()

    assert settings.ranking_model_mode == "shadow"
    assert settings.ranking_model_path == Path("models/custom.pt")
    assert settings.ranking_model_sha256 == "a" * 64
    assert settings.ranking_model_top_n == 12
    assert settings.ranking_model_blend_weight == 0.35
