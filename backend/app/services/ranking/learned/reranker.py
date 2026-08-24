"""Deterministic feature construction and safe CPU inference for reranking."""

from __future__ import annotations

import hashlib
import math
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from backend.app.models.extract import AlgorithmInput
from backend.app.models.ranking import RankedCandidate
from backend.app.services.ranking.learned.checkpoint import CheckpointMetadata


@dataclass(frozen=True)
class ModelInputs:
    user_vec: torch.Tensor
    candidate_vec: torch.Tensor
    objective_score: torch.Tensor
    history_seqs: list[torch.Tensor | None]
    history_distances: list[torch.Tensor | None]


@dataclass(frozen=True)
class ScoreResult:
    scores: list[float] | None
    inference_ms: float
    fallback_reason: str | None = None


def _tokens(values: Sequence[str | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not value:
            continue
        result.extend(
            token
            for token in re.split(r"[\s,\uFF0C\u3001/|;\uFF1B]+", value.strip().lower())
            if token
        )
    return result


def _vector(tokens: Sequence[str], metadata: CheckpointMetadata) -> torch.Tensor:
    """Map checkpoint-vocabulary terms into a fixed-width vector deterministically.

    Vocabulary indices select dimensions and SHA-256 supplies a stable sign when
    the vocabulary is wider than the embedding. Python's randomized ``hash`` is
    intentionally never used.
    """

    vector = torch.zeros(metadata.embed_dim, dtype=torch.float32)
    for token in tokens:
        vocabulary_index = metadata.vocabulary.get(token)
        if vocabulary_index is None:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        sign = 1.0 if digest[0] & 1 else -1.0
        vector[vocabulary_index % metadata.embed_dim] += sign
    norm = torch.linalg.vector_norm(vector)
    if norm.item() > 0:
        vector = vector / norm
    return vector


def build_model_inputs(
    candidates: Sequence[RankedCandidate],
    algorithm_input: AlgorithmInput,
    metadata: CheckpointMetadata,
) -> ModelInputs:
    """Build the shared production/training representation.

    Production history currently contains POI IDs but no historical tags or
    distances. Known IDs are therefore embedded through the checkpoint vocabulary
    and their unavailable distances are represented as zero.
    """

    preference = algorithm_input.subjective_preference
    user_tokens = _tokens(
        [
            *preference.preference_terms,
            *preference.travel_styles,
            *preference.spot_keywords,
            *preference.food_keywords,
            *preference.hotel_keywords,
        ]
    )
    user = _vector(user_tokens, metadata)
    user_vec = user.repeat(len(candidates), 1)

    candidate_vectors: list[torch.Tensor] = []
    for ranked in candidates:
        poi = ranked.candidate
        candidate_vectors.append(
            _vector(
                _tokens([*poi.tags, poi.name, poi.address, poi.source_dataset]),
                metadata,
            )
        )
    candidate_vec = (
        torch.stack(candidate_vectors)
        if candidate_vectors
        else torch.empty((0, metadata.embed_dim), dtype=torch.float32)
    )
    objective_score = torch.tensor(
        [[float(item.score_breakdown.get("objective", item.score))] for item in candidates],
        dtype=torch.float32,
    )

    history_ids = algorithm_input.sequence_model_input.historical_poi_ids[
        -metadata.max_history :
    ]
    if history_ids:
        history = torch.stack(
            [_vector([poi_id.lower()], metadata) for poi_id in history_ids]
        )
        distances = torch.zeros(len(history_ids), dtype=torch.float32)
        history_seqs = [history.clone() for _ in candidates]
        history_distances = [distances.clone() for _ in candidates]
    else:
        history_seqs = [None for _ in candidates]
        history_distances = [None for _ in candidates]
    return ModelInputs(
        user_vec=user_vec,
        candidate_vec=candidate_vec,
        objective_score=objective_score,
        history_seqs=history_seqs,
        history_distances=history_distances,
    )


class LearnedReranker:
    """Run inference behind a bounded wait with a daemon-worker circuit breaker."""

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: CheckpointMetadata,
        *,
        timeout_ms: float = 100.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model.to("cpu")
        self.model.eval()
        self.metadata = metadata
        self.timeout_ms = max(0.0, float(timeout_ms))
        self.clock = clock
        self._state_lock = threading.Lock()
        self._active_worker: threading.Thread | None = None

    def score(
        self,
        candidates: Sequence[RankedCandidate],
        algorithm_input: AlgorithmInput,
    ) -> ScoreResult:
        if not candidates:
            return ScoreResult([], 0.0)
        try:
            inputs = build_model_inputs(candidates, algorithm_input, self.metadata)
        except Exception:
            return ScoreResult(None, 0.0, "inference_error")

        result_queue: queue.Queue[ScoreResult] = queue.Queue(maxsize=1)
        with self._state_lock:
            if self._active_worker is not None:
                if self._active_worker.is_alive():
                    return ScoreResult(None, 0.0, "timeout")
                self._active_worker = None
            worker = threading.Thread(
                target=self._infer,
                args=(inputs, len(candidates), result_queue),
                name="learned-ranking-inference",
                daemon=True,
            )
            self._active_worker = worker
            # Starting while holding the lock makes the reservation observable as
            # alive before another caller can inspect or replace it.
            try:
                worker.start()
            except Exception:
                self._active_worker = None
                return ScoreResult(None, 0.0, "inference_error")

        started = self.clock()
        try:
            result = result_queue.get(timeout=self.timeout_ms / 1000.0)
        except queue.Empty:
            elapsed = max(0.0, (self.clock() - started) * 1000.0)
            if not math.isfinite(elapsed):
                elapsed = self.timeout_ms
            return ScoreResult(None, elapsed, "timeout")

        elapsed = max(0.0, (self.clock() - started) * 1000.0)
        with self._state_lock:
            if self._active_worker is worker:
                self._active_worker = None
        if not math.isfinite(elapsed):
            return ScoreResult(None, 0.0, "timeout")
        if elapsed > self.timeout_ms:
            return ScoreResult(None, elapsed, "timeout")
        return ScoreResult(result.scores, elapsed, result.fallback_reason)

    def _infer(
        self,
        inputs: ModelInputs,
        candidate_count: int,
        result_queue: queue.Queue[ScoreResult],
    ) -> None:
        """Run forward and all output validation in the daemon worker."""

        try:
            with torch.inference_mode():
                output = self.model(
                    inputs.user_vec,
                    inputs.candidate_vec,
                    inputs.objective_score,
                    inputs.history_seqs,
                    inputs.history_distances,
                )
            score_tensor = output[0] if isinstance(output, tuple) else output
        except Exception:
            self._publish(result_queue, ScoreResult(None, 0.0, "inference_error"))
            return
        try:
            if not isinstance(score_tensor, torch.Tensor) or score_tensor.shape != (
                candidate_count,
                1,
            ):
                self._publish(
                    result_queue,
                    ScoreResult(None, 0.0, "invalid_output_shape"),
                )
                return
            if (
                score_tensor.layout != torch.strided
                or not score_tensor.is_floating_point()
            ):
                self._publish(
                    result_queue, ScoreResult(None, 0.0, "invalid_output")
                )
                return
            flattened = (
                score_tensor.detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(-1)
            )
            scores = [float(value) for value in flattened]
            if any(not math.isfinite(value) for value in scores):
                self._publish(
                    result_queue, ScoreResult(None, 0.0, "nonfinite_score")
                )
                return
            self._publish(result_queue, ScoreResult(scores, 0.0))
        except Exception:
            self._publish(result_queue, ScoreResult(None, 0.0, "invalid_output"))

    @staticmethod
    def _publish(
        result_queue: queue.Queue[ScoreResult], result: ScoreResult
    ) -> None:
        """Publish at most once; a late result after timeout is safe to ignore."""

        try:
            result_queue.put_nowait(result)
        except queue.Full:
            pass
