"""Small, reproducible CPU training and evaluation pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.ranking.learned.checkpoint import (
    MAX_CHECKPOINT_BYTES,
    CheckpointMetadata,
)
from backend.app.services.ranking.learned.model import LearnedRankingModel
from backend.app.services.ranking.learned.reranker import build_model_inputs
from backend.app.services.ranking.learned.training.data import (
    TrainingDataset,
    build_vocabulary,
)
from backend.app.services.ranking.learned.training.losses import (
    listwise_softmax_loss,
    pairwise_logistic_loss,
)

_TRAINING_LOCK = threading.Lock()
MAX_ARTIFACT_BYTES = MAX_CHECKPOINT_BYTES
COPY_CHUNK_BYTES = 64 * 1024


class TrainerConfig(BaseModel):
    epochs: int = Field(default=20, ge=1, le=10_000)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    patience: int = Field(default=3, ge=1, le=10_000)
    k: int = Field(default=10, ge=1, le=1_000)

    model_config = ConfigDict(extra="forbid")


class RankingMetrics(BaseModel):
    """Macro metrics across queries.

    Coverage is unique recommended candidate IDs divided by the supplied
    time-visible ``catalog_size``. The trainer uses train+validation candidates
    for validation and train+validation+test candidates for the later test set,
    so future test IDs cannot influence validation selection or its artifact.
    Diversity is the macro average of distinct categories / recommendations in
    each top-K list. Empty queries contribute zero. Latency is the macro mean in
    milliseconds, with invalid samples bounded to zero.
    """

    recall_at_k: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ndcg_at_k: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mrr: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    diversity: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    inference_latency_ms: float = Field(
        ge=0.0, le=3_600_000.0, allow_inf_nan=False
    )

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    metadata_path: Path
    sha256: str
    best_epoch: int
    validation_metrics: RankingMetrics
    test_metrics: RankingMetrics
    history_gradient_norm: float
    history_parameter_update_norm: float


def _safe_number(value: float, *, upper: float | None = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    value = max(0.0, value)
    return min(upper, value) if upper is not None else value


def evaluate_rankings(
    *,
    scores_by_query: Sequence[Sequence[float]],
    labels_by_query: Sequence[Sequence[float]],
    candidate_ids_by_query: Sequence[Sequence[str]],
    categories_by_query: Sequence[Sequence[str]],
    k: int,
    catalog_size: int,
    inference_latency_ms: Sequence[float],
) -> RankingMetrics:
    """Compute deterministic macro metrics; original index breaks score ties."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if catalog_size < 0:
        raise ValueError("catalog_size must be non-negative")
    count = len(scores_by_query)
    if not (
        len(labels_by_query)
        == len(candidate_ids_by_query)
        == len(categories_by_query)
        == len(inference_latency_ms)
        == count
    ):
        raise ValueError("all evaluation inputs must contain the same query count")
    recalls: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    diversities: list[float] = []
    recommended: set[str] = set()
    observed_candidates: set[str] = set()
    latencies: list[float] = []
    for scores, labels, ids, categories, latency in zip(
        scores_by_query,
        labels_by_query,
        candidate_ids_by_query,
        categories_by_query,
        inference_latency_ms,
    ):
        if not (len(scores) == len(labels) == len(ids) == len(categories)):
            raise ValueError("query score, label, ID, and category lists must align")
        numeric_scores = [float(score) for score in scores]
        if any(not math.isfinite(score) for score in numeric_scores):
            raise ValueError("prediction scores must be finite")
        numeric_labels: list[float] = []
        for label in labels:
            if isinstance(label, bool) or not isinstance(label, (int, float)):
                raise ValueError("labels must be numeric, not boolean or string")
            numeric_label = float(label)
            if not math.isfinite(numeric_label) or not 0.0 <= numeric_label <= 1.0:
                raise ValueError("labels must be finite relevance values in [0, 1]")
            numeric_labels.append(numeric_label)
        observed_candidates.update(ids)
        order = sorted(
            range(len(numeric_scores)),
            key=lambda index: (-numeric_scores[index], index),
        )
        top = order[:k]
        relevant_total = sum(1 for label in numeric_labels if label > 0)
        recalls.append(sum(1 for index in top if numeric_labels[index] > 0) / relevant_total if relevant_total else 0.0)
        gains = [numeric_labels[index] for index in top]
        dcg = sum((2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sorted(numeric_labels, reverse=True)[:k]
        idcg = sum((2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        first = next((rank + 1 for rank, index in enumerate(order) if numeric_labels[index] > 0), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        recommended.update(ids[index] for index in top)
        diversities.append(len({categories[index] for index in top}) / len(top) if top else 0.0)
        latencies.append(_safe_number(float(latency), upper=3_600_000.0))
    if len(observed_candidates) > catalog_size:
        raise ValueError(
            "catalog_size cannot be smaller than observed unique candidate IDs"
        )
    divisor = max(1, count)
    return RankingMetrics(
        recall_at_k=_safe_number(sum(recalls) / divisor),
        ndcg_at_k=_safe_number(sum(ndcgs) / divisor),
        mrr=_safe_number(sum(reciprocal_ranks) / divisor),
        coverage=_safe_number(len(recommended) / catalog_size if catalog_size > 0 else 0.0),
        diversity=_safe_number(sum(diversities) / divisor),
        inference_latency_ms=_safe_number(
            sum(latencies) / divisor, upper=3_600_000.0
        ),
    )


def _evaluate_model(
    model: LearnedRankingModel,
    dataset: TrainingDataset,
    metadata: CheckpointMetadata,
    k: int,
    catalog_size: int,
) -> RankingMetrics:
    scores: list[list[float]] = []
    labels: list[list[float]] = []
    ids: list[list[str]] = []
    categories: list[list[str]] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for query in dataset.queries:
            inputs = build_model_inputs(query.candidates, query.algorithm_input, metadata)
            started = time.perf_counter()
            output, _, _ = model(
                inputs.user_vec,
                inputs.candidate_vec,
                inputs.objective_score,
                inputs.history_seqs,
                inputs.history_distances,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            scores.append([float(value) for value in output.reshape(-1)])
            labels.append(query.labels)
            ids.append([candidate.poi_id for candidate in query.candidates])
            categories.append(query.categories)
            latencies.append(elapsed)
    return evaluate_rankings(
        scores_by_query=scores,
        labels_by_query=labels,
        candidate_ids_by_query=ids,
        categories_by_query=categories,
        k=k,
        catalog_size=catalog_size,
        inference_latency_ms=latencies,
    )


def _temporary_file(path: Path, role: str, suffix: str) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.{role}.", suffix=f".{suffix}", dir=path.parent
    )
    return descriptor, Path(name)


def _write_staged_content(stream: object, content: bytes) -> None:
    stream.write(content)  # type: ignore[attr-defined]


def _stage_bytes(path: Path, content: bytes, role: str) -> Path:
    descriptor, staged = _temporary_file(path, role, "tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            _write_staged_content(stream, content)
            stream.flush()
            os.fsync(stream.fileno())
        return staged
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        staged.unlink(missing_ok=True)
        raise


def _stage_existing_backup(path: Path, role: str, maximum_bytes: int) -> Path:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise OSError(f"cannot inspect existing artifact {path}: {error}") from error
    if size > maximum_bytes:
        raise ValueError(
            f"existing artifact {path} exceeds backup limit {maximum_bytes} bytes"
        )
    descriptor, backup = _temporary_file(path, role, "bak")
    copied = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            with path.open("rb") as source:
                while True:
                    chunk = source.read(COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > maximum_bytes:
                        raise ValueError(
                            f"existing artifact {path} exceeds backup limit "
                            f"{maximum_bytes} bytes"
                        )
                    destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        return backup
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        backup.unlink(missing_ok=True)
        raise


def _replace_path(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _publish_artifact_pair(
    checkpoint_path: Path,
    checkpoint_bytes: bytes,
    metadata_path: Path,
    metadata_bytes: bytes,
) -> None:
    """Recoverably publish a checkpoint and its audit-mirror JSON sidecar.

    Both files are fully serialized and staged before either live path changes.
    Backups allow rollback across caught replacement failures. Two fixed Windows
    filenames cannot provide crash-level pair atomicity; the checkpoint's
    internal metadata is authoritative and the JSON sidecar is an audit mirror.
    """

    for role, content in (
        ("checkpoint", checkpoint_bytes),
        ("metadata sidecar", metadata_bytes),
    ):
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"new {role} exceeds artifact limit {MAX_ARTIFACT_BYTES} bytes"
            )
    staged: list[Path] = []
    backups: dict[Path, Path] = {}
    retained_recovery_backups: set[Path] = set()
    existed = {
        checkpoint_path: checkpoint_path.exists(),
        metadata_path: metadata_path.exists(),
    }
    replaced: list[Path] = []
    try:
        staged_checkpoint = _stage_bytes(
            checkpoint_path, checkpoint_bytes, "checkpoint"
        )
        staged.append(staged_checkpoint)
        staged_metadata = _stage_bytes(metadata_path, metadata_bytes, "metadata")
        staged.append(staged_metadata)
        for live_path, role in (
            (checkpoint_path, "backup-checkpoint"),
            (metadata_path, "backup-metadata"),
        ):
            if existed[live_path]:
                backup = _stage_existing_backup(
                    live_path, role, MAX_ARTIFACT_BYTES
                )
                backups[live_path] = backup
                staged.append(backup)

        _replace_path(staged_checkpoint, checkpoint_path)
        replaced.append(checkpoint_path)
        _replace_path(staged_metadata, metadata_path)
        replaced.append(metadata_path)
    except Exception as publish_error:
        recovery_errors: list[tuple[Path, Path | None, BaseException]] = []
        for live_path in reversed(replaced):
            backup = backups.get(live_path)
            try:
                if backup is not None and backup.exists():
                    _replace_path(backup, live_path)
                elif not existed[live_path]:
                    live_path.unlink(missing_ok=True)
            except BaseException as recovery_error:
                recovery_errors.append((live_path, backup, recovery_error))
                if backup is not None and backup.exists():
                    retained_recovery_backups.add(backup)
        if recovery_errors:
            details = "; ".join(
                f"{live}: recovery backup {backup}: {error}"
                for live, backup, error in recovery_errors
            )
            raise OSError(
                f"artifact publication failed ({publish_error}); recovery failed; "
                f"{details}"
            ) from publish_error
        raise
    finally:
        for temporary in staged:
            if temporary not in retained_recovery_backups:
                temporary.unlink(missing_ok=True)


@contextmanager
def _isolated_training_state(seed: int):
    """Serialize and restore process-global RNG/determinism settings."""

    with _TRAINING_LOCK:
        python_random_state = random.getstate()
        torch_random_state = torch.random.get_rng_state().clone()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        previous_threads = torch.get_num_threads()
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
            yield
        finally:
            restoration_errors: list[BaseException] = []
            for restore in (
                lambda: random.setstate(python_random_state),
                lambda: torch.random.set_rng_state(torch_random_state),
                lambda: torch.set_num_threads(previous_threads),
                lambda: torch.use_deterministic_algorithms(previous_determinism),
            ):
                try:
                    restore()
                except BaseException as error:
                    restoration_errors.append(error)
            if restoration_errors:
                raise RuntimeError(
                    "failed to restore one or more process-global training states: "
                    + "; ".join(str(error) for error in restoration_errors)
                ) from restoration_errors[0]


def train_reranker(
    train: TrainingDataset,
    validation: TrainingDataset,
    test: TrainingDataset,
    metadata: CheckpointMetadata,
    output_path: str | Path,
    config: TrainerConfig | None = None,
) -> TrainingResult:
    """Train on train only, select on validation, and evaluate test once."""

    config = config or TrainerConfig()
    split_ids = [
        {query.query_id for query in dataset.queries}
        for dataset in (train, validation, test)
    ]
    if (
        split_ids[0] & split_ids[1]
        or split_ids[0] & split_ids[2]
        or split_ids[1] & split_ids[2]
    ):
        raise ValueError("train, validation, and test query IDs must be disjoint")
    split_groups = [
        {query.split_group_id for query in dataset.queries}
        for dataset in (train, validation, test)
    ]
    if (
        split_groups[0] & split_groups[1]
        or split_groups[0] & split_groups[2]
        or split_groups[1] & split_groups[2]
    ):
        raise ValueError(
            "train, validation, and test split group identities must be disjoint"
        )
    if not (
        max(query.timestamp for query in train.queries)
        < min(query.timestamp for query in validation.queries)
        and max(query.timestamp for query in validation.queries)
        < min(query.timestamp for query in test.queries)
    ):
        raise ValueError(
            "validation must be strictly later than train and test strictly later than validation"
        )
    for query in train.queries:
        history_ids = query.algorithm_input.sequence_model_input.historical_poi_ids
        if not history_ids:
            raise ValueError(
                f"training query {query.query_id!r} requires non-empty history"
            )
        unknown = [
            poi_id for poi_id in history_ids if poi_id.lower() not in metadata.vocabulary
        ]
        if unknown:
            raise ValueError(
                f"training query {query.query_id!r} has unmatched history IDs"
            )
        history_inputs = build_model_inputs(
            query.candidates, query.algorithm_input, metadata
        )
        if any(
            sequence is None
            or distances is None
            or sequence.shape[0] == 0
            or distances.shape[0] != sequence.shape[0]
            or not torch.isfinite(sequence).all().item()
            or not torch.isfinite(distances).all().item()
            for sequence, distances in zip(
                history_inputs.history_seqs, history_inputs.history_distances
            )
        ):
            raise ValueError(
                f"training query {query.query_id!r} has unusable or unaligned history"
            )
    training_vocabulary = build_vocabulary(train)
    if (
        metadata.vocabulary != training_vocabulary
        or metadata.vocabulary_size != len(training_vocabulary)
    ):
        raise ValueError(
            "metadata vocabulary must exactly match the training vocabulary"
        )
    with _isolated_training_state(config.seed):
        model = LearnedRankingModel(embed_dim=metadata.embed_dim, max_history=metadata.max_history)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        initial_history_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith(
                ("history_encoder.", "distance_attention.", "sequence_projection.")
            )
        }
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: RankingMetrics | None = None
        best_epoch = 0
        best_ndcg = -1.0
        stale = 0
        history_gradient_norm = 0.0
        validation_catalog_size = len(
            {
                candidate.poi_id
                for dataset in (train, validation)
                for query in dataset.queries
                for candidate in query.candidates
            }
        )
        test_catalog_size = len(
            {
                candidate.poi_id
                for dataset in (train, validation, test)
                for query in dataset.queries
                for candidate in query.candidates
            }
        )

        for epoch in range(1, config.epochs + 1):
            model.train()
            for query in train.queries:
                inputs = build_model_inputs(query.candidates, query.algorithm_input, metadata)
                predictions, _, _ = model(
                    inputs.user_vec,
                    inputs.candidate_vec,
                    inputs.objective_score,
                    inputs.history_seqs,
                    inputs.history_distances,
                )
                labels = torch.tensor(query.labels, dtype=torch.float32)
                loss = pairwise_logistic_loss(predictions, labels) + listwise_softmax_loss(predictions, labels)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                history_parameters = list(model.history_encoder.parameters()) + list(model.distance_attention.parameters()) + list(model.sequence_projection.parameters())
                history_gradients = [
                    parameter.grad
                    for parameter in history_parameters
                    if parameter.grad is not None
                ]
                if not history_gradients or any(
                    not torch.isfinite(gradient).all().item()
                    for gradient in history_gradients
                ):
                    raise FloatingPointError(
                        "history path did not receive finite gradients"
                    )
                gradient_norm = sum(
                    float(gradient.norm()) for gradient in history_gradients
                )
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("history gradient norm is non-finite")
                history_gradient_norm = max(history_gradient_norm, gradient_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            validation_metrics = _evaluate_model(
                model,
                validation,
                metadata,
                config.k,
                validation_catalog_size,
            )
            if validation_metrics.ndcg_at_k > best_ndcg:
                best_ndcg = validation_metrics.ndcg_at_k
                best_metrics = validation_metrics
                best_epoch = epoch
                best_state = {name: tensor.detach().to(device="cpu", dtype=torch.float32).clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

        if best_state is None or best_metrics is None:
            raise RuntimeError("training produced no finite validation model")
        if history_gradient_norm <= 0:
            raise FloatingPointError(
                "training did not produce any non-zero history gradient"
            )
        history_parameter_update_norm = sum(
            float((best_state[name] - initial).norm())
            for name, initial in initial_history_state.items()
        )
        if (
            not math.isfinite(history_parameter_update_norm)
            or history_parameter_update_norm <= 0
        ):
            raise FloatingPointError("history parameters did not update")
        model.load_state_dict(best_state, strict=True)
        test_metrics = _evaluate_model(
            model, test, metadata, config.k, test_catalog_size
        )
        saved_metadata = CheckpointMetadata(
            model_version="v1",
            embed_dim=metadata.embed_dim,
            max_history=metadata.max_history,
            vocabulary=metadata.vocabulary,
            vocabulary_size=metadata.vocabulary_size,
            epoch=best_epoch,
            metrics={
                "validation_recall_at_k": best_metrics.recall_at_k,
                "validation_ndcg_at_k": best_metrics.ndcg_at_k,
                "validation_mrr": best_metrics.mrr,
                "validation_coverage": best_metrics.coverage,
                "validation_diversity": best_metrics.diversity,
            },
            dataset=f"{train.name}|{validation.name}|{test.name}"[:256],
        )
        output = Path(output_path)
        payload = {"model_state_dict": best_state, "metadata": saved_metadata.model_dump(mode="json")}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        checkpoint_bytes = buffer.getvalue()
        digest = hashlib.sha256(checkpoint_bytes).hexdigest()
        metadata_path = output.with_name(f"{output.name}.metadata.json")
        metadata_bytes = json.dumps(
            saved_metadata.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _publish_artifact_pair(
            output, checkpoint_bytes, metadata_path, metadata_bytes
        )
        return TrainingResult(
            output,
            metadata_path,
            digest,
            best_epoch,
            best_metrics,
            test_metrics,
            history_gradient_norm,
            history_parameter_update_norm,
        )
