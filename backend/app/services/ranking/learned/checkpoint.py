"""Safe, fail-closed loading for learned-ranking checkpoints.

Training code should save exactly two root keys: ``model_state_dict`` and
``metadata``.  Metadata is JSON-safe and versioned; optimizer state is never
part of the production format.  Every failure is represented by a result so a
bad optional model cannot prevent application startup.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from backend.app.services.ranking.learned.model import LearnedRankingModel

SUPPORTED_MODEL_VERSION = "v1"
LEGACY_SYNTHETIC_VERSION = "v0.1.0-synthetic"
LEGACY_SYNTHETIC_SHA256 = (
    "93850b0e5569b09230f8b9fd6d70efd4db7b1d99bf1551b08dd6979ed43a84fb"
)
MAX_VOCABULARY_SIZE = 50_000
MAX_MODEL_PARAMETERS = 128
MAX_EMBED_DIM = 1_024
MAX_HISTORY = 100
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_TENSOR_COUNT = 128
MAX_TOTAL_TENSOR_ELEMENTS = 25_000_000
MAX_TOTAL_TENSOR_BYTES = 128 * 1024 * 1024


class CheckpointMetadata(BaseModel):
    """Bounded JSON-safe metadata shared by loading and future training code."""

    model_version: StrictStr
    embed_dim: StrictInt
    max_history: StrictInt
    vocabulary: dict[StrictStr, StrictInt]
    vocabulary_size: StrictInt
    epoch: StrictInt
    metrics: dict[StrictStr, float] = Field(default_factory=dict)
    dataset: StrictStr = "unknown"

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("model_version")
    @classmethod
    def _bounded_version(cls, value: str) -> str:
        if not 1 <= len(value) <= 64:
            raise ValueError("model_version length is invalid")
        return value

    @field_validator("embed_dim")
    @classmethod
    def _bounded_embed_dim(cls, value: int) -> int:
        if value <= 0 or value > MAX_EMBED_DIM or value % 2:
            raise ValueError("embed_dim must be a bounded positive even integer")
        return value

    @field_validator("max_history")
    @classmethod
    def _bounded_history(cls, value: int) -> int:
        if not 1 <= value <= MAX_HISTORY:
            raise ValueError("max_history is out of bounds")
        return value

    @field_validator("vocabulary", mode="before")
    @classmethod
    def _bounded_vocabulary_container(cls, value: object) -> object:
        if not isinstance(value, dict) or len(value) > MAX_VOCABULARY_SIZE:
            raise ValueError("vocabulary is not a bounded mapping")
        return value

    @field_validator("vocabulary")
    @classmethod
    def _valid_vocabulary(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not 1 <= len(key) <= 128 for key in value):
            raise ValueError("vocabulary keys must be bounded non-empty strings")
        if sorted(value.values()) != list(range(len(value))):
            raise ValueError("vocabulary indices must be unique and contiguous")
        return value

    @field_validator("vocabulary_size")
    @classmethod
    def _bounded_vocabulary_size(cls, value: int) -> int:
        if not 0 <= value <= MAX_VOCABULARY_SIZE:
            raise ValueError("vocabulary_size is out of bounds")
        return value

    @field_validator("epoch")
    @classmethod
    def _bounded_epoch(cls, value: int) -> int:
        if not 0 <= value <= 10_000_000:
            raise ValueError("epoch is out of bounds")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def _bounded_metrics_container(cls, value: object) -> object:
        if not isinstance(value, dict) or len(value) > 64:
            raise ValueError("metrics is not a bounded mapping")
        return value

    @field_validator("metrics")
    @classmethod
    def _valid_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not 1 <= len(key) <= 128 for key in value):
            raise ValueError("metric keys must be bounded non-empty strings")
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("metrics must be finite")
        return value

    @field_validator("dataset")
    @classmethod
    def _bounded_dataset(cls, value: str) -> str:
        if not 1 <= len(value) <= 256:
            raise ValueError("dataset must be a bounded non-empty string")
        return value

    @model_validator(mode="after")
    def _vocabulary_size_matches(self) -> "CheckpointMetadata":
        if self.vocabulary_size != len(self.vocabulary):
            raise ValueError("vocabulary_size does not match vocabulary")
        return self


@dataclass(frozen=True)
class CheckpointLoadResult:
    loaded: bool
    model: LearnedRankingModel | None = None
    fallback_reason: str | None = None
    version: str | None = None
    sha256: str | None = None
    metadata: CheckpointMetadata | None = None


@dataclass(frozen=True)
class LegacyCheckpointInspection:
    valid: bool
    classification: str = "legacy/unknown/smoke-only"
    version: str = "unknown-legacy"
    model_parameter_count: int | None = None
    epoch: int | None = None
    sha256: str | None = None
    fallback_reason: str | None = None


def _safe_torch_load(source: io.BytesIO) -> object:
    return torch.load(source, map_location="cpu", weights_only=True)


def _read_verified_bytes(
    path: Path, expected_sha256: str
) -> tuple[bytes | None, str | None, str | None]:
    """Read once and verify the exact immutable bytes under a strict size cap.

    The caller-provided SHA-256 allowlist is the trust gate. Deserialization, if
    appropriate for the artifact type, must use only the returned bytes.
    """

    try:
        with path.open("rb") as stream:
            checkpoint_bytes = stream.read(MAX_CHECKPOINT_BYTES + 1)
    except FileNotFoundError:
        return None, None, "missing_file"
    except OSError:
        return None, None, "unreadable_file"
    if len(checkpoint_bytes) > MAX_CHECKPOINT_BYTES:
        return None, None, "checkpoint_too_large"

    actual = hashlib.sha256(checkpoint_bytes).hexdigest()
    expected = (
        expected_sha256.strip().lower()
        if isinstance(expected_sha256, str)
        else ""
    )
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        return None, actual, "invalid_expected_hash"
    if not hmac.compare_digest(actual, expected):
        return None, actual, "hash_mismatch"
    return checkpoint_bytes, actual, None


def _fallback(
    reason: str, *, digest: str | None = None, version: str | None = None
) -> CheckpointLoadResult:
    return CheckpointLoadResult(
        loaded=False, fallback_reason=reason, sha256=digest, version=version
    )


def _tensor_resources_within_bounds(state: dict[object, object]) -> bool:
    tensor_count = 0
    total_elements = 0
    total_bytes = 0
    for value in state.values():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        total_elements += value.numel()
        total_bytes += value.numel() * value.element_size()
        if (
            tensor_count > MAX_TOTAL_TENSOR_COUNT
            or total_elements > MAX_TOTAL_TENSOR_ELEMENTS
            or total_bytes > MAX_TOTAL_TENSOR_BYTES
        ):
            return False
    return True


def load_learned_checkpoint(
    path: str | Path, expected_sha256: str
) -> CheckpointLoadResult:
    """Load a corrected reranker, returning a safe fallback for every failure.

    ``weights_only=True`` is not a complete hostile-resource sandbox. Tensor
    budgets are post-deserialization acceptance limits; SHA allowlisting remains
    the trust boundary. The known legacy digest is rejected before deserialization.
    """

    checkpoint_path = Path(path)
    checkpoint_bytes, digest, read_error = _read_verified_bytes(
        checkpoint_path, expected_sha256
    )
    if read_error:
        return _fallback(read_error, digest=digest)
    if digest == LEGACY_SYNTHETIC_SHA256:
        return _fallback(
            "legacy_not_production",
            digest=digest,
            version=LEGACY_SYNTHETIC_VERSION,
        )
    if checkpoint_bytes is None:
        return _fallback("unreadable_file", digest=digest)
    try:
        payload = _safe_torch_load(io.BytesIO(checkpoint_bytes))
    except Exception:
        return _fallback("corrupt_checkpoint", digest=digest)

    if (
        not isinstance(payload, dict)
        or len(payload) != 2
        or set(payload) != {"metadata", "model_state_dict"}
    ):
        return _fallback("schema_error", digest=digest)
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_metadata, dict) or len(raw_metadata) > 16:
        return _fallback("schema_error", digest=digest)
    try:
        metadata = CheckpointMetadata.model_validate(raw_metadata)
    except (ValidationError, TypeError, ValueError):
        return _fallback("schema_error", digest=digest)
    if metadata.model_version != SUPPORTED_MODEL_VERSION:
        return _fallback(
            "incompatible_version", digest=digest, version=metadata.model_version
        )

    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or len(state) > MAX_MODEL_PARAMETERS:
        return _fallback("schema_error", digest=digest, version=metadata.model_version)
    if not _tensor_resources_within_bounds(state):
        return _fallback(
            "resource_limit", digest=digest, version=metadata.model_version
        )
    try:
        model = LearnedRankingModel(
            embed_dim=metadata.embed_dim, max_history=metadata.max_history
        )
    except (TypeError, ValueError):
        return _fallback("schema_error", digest=digest, version=metadata.model_version)
    expected_state = model.state_dict()
    if not all(isinstance(name, str) and 1 <= len(name) <= 256 for name in state):
        return _fallback("parameter_names", digest=digest, version=metadata.model_version)
    if set(state) != set(expected_state):
        return _fallback("parameter_names", digest=digest, version=metadata.model_version)
    for name, expected_tensor in expected_state.items():
        tensor = state[name]
        if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
            return _fallback(
                "parameter_type", digest=digest, version=metadata.model_version
            )
        if tensor.dtype != expected_tensor.dtype or tensor.device.type != "cpu":
            return _fallback(
                "parameter_type", digest=digest, version=metadata.model_version
            )
        if tensor.shape != expected_tensor.shape:
            return _fallback(
                "parameter_shape", digest=digest, version=metadata.model_version
            )
        if (
            tensor.is_floating_point() or tensor.is_complex()
        ) and not torch.isfinite(tensor).all().item():
            return _fallback("parameter_type", digest=digest, version=metadata.model_version)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError):
        return _fallback(
            "incompatible_state", digest=digest, version=metadata.model_version
        )
    model.eval()
    return CheckpointLoadResult(
        loaded=True,
        model=model,
        version=metadata.model_version,
        sha256=digest,
        metadata=metadata,
    )


def inspect_legacy_checkpoint(path: str | Path) -> LegacyCheckpointInspection:
    """Identify the one canonical legacy artifact without deserializing it.

    The hard-coded digest uniquely identifies the inspected file, so the returned
    facts are immutable known facts about that artifact. Unknown legacy files are
    rejected before ``torch.load``; their optimizer payloads are never materialized.
    This inspection function never returns or constructs a model.
    """

    checkpoint_path = Path(path)
    checkpoint_bytes, digest, read_error = _read_verified_bytes(
        checkpoint_path, LEGACY_SYNTHETIC_SHA256
    )
    if read_error:
        return LegacyCheckpointInspection(
            False, sha256=digest, fallback_reason=read_error
        )
    if checkpoint_bytes is None:
        return LegacyCheckpointInspection(
            False, sha256=digest, fallback_reason="unreadable_file"
        )
    return LegacyCheckpointInspection(
        True,
        classification="legacy/synthetic/smoke-only",
        version=LEGACY_SYNTHETIC_VERSION,
        model_parameter_count=17,
        epoch=2,
        sha256=digest,
    )
