#!/usr/bin/env python
"""Train the corrected reranker from JSON, or run an explicit synthetic smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.adapter import CandidatePoi
from backend.app.models.extract import AlgorithmInput, SearchContext, SequenceModelInput, SubjectivePreferenceInput
from backend.app.models.ranking import RankedCandidate
from backend.app.services.ranking.learned.checkpoint import CheckpointMetadata, load_learned_checkpoint
from backend.app.services.ranking.learned.training.data import (
    MAX_QUERIES,
    TrainingDataset,
    build_vocabulary,
    chronological_split,
)
from backend.app.services.ranking.learned.training.trainer import TrainerConfig, train_reranker

MAX_DATASET_BYTES = 16 * 1024 * 1024


def _synthetic_dataset() -> TrainingDataset:
    queries = []
    for day in range(6):
        algorithm_input = AlgorithmInput(
            search_context=SearchContext(),
            subjective_preference=SubjectivePreferenceInput(preference_terms=["museum"]),
            sequence_model_input=SequenceModelInput(has_history=True, historical_poi_ids=["old"]),
        )
        ranked = [
            RankedCandidate(
                poi_id=f"museum-{day}", poi_type="spot", score=0.8, rank=1,
                score_breakdown={"objective": 0.8},
                candidate=CandidatePoi(poi_id=f"museum-{day}", poi_type="spot", source_dataset="synthetic-smoke", name="museum", tags=["museum"]),
            ),
            RankedCandidate(
                poi_id=f"park-{day}", poi_type="spot", score=0.2, rank=2,
                score_breakdown={"objective": 0.2},
                candidate=CandidatePoi(poi_id=f"park-{day}", poi_type="spot", source_dataset="synthetic-smoke", name="park", tags=["park"]),
            ),
        ]
        queries.append({
            "query_id": f"synthetic-{day}",
            "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day),
            "algorithm_input": algorithm_input,
            "candidates": ranked,
            "labels": [1.0, 0.0],
            "categories": ["culture", "nature"],
        })
    return TrainingDataset(name="synthetic-smoke-only", queries=queries)


def _read_dataset(path: Path) -> TrainingDataset:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_DATASET_BYTES + 1)
    except OSError as error:
        raise ValueError(f"cannot read dataset file: {error}") from error
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError(
            f"dataset file is too large (maximum {MAX_DATASET_BYTES} bytes)"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dataset file must be UTF-8") from error
    if path.suffix.lower() == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > MAX_QUERIES:
            raise ValueError("dataset contains too many records")
        queries = [json.loads(line) for line in lines]
        return TrainingDataset(name=path.stem, queries=queries)
    payload = json.loads(text)
    if isinstance(payload, list):
        payload = {"name": path.stem, "queries": payload}
    return TrainingDataset.model_validate(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the corrected learned reranker")
    parser.add_argument("--smoke-test", action="store_true", help="run fixed tiny synthetic shape/integration smoke only")
    parser.add_argument("--dataset", type=Path, help="real JSON or JSONL chronological dataset")
    parser.add_argument("--output", type=Path, help="explicit best checkpoint output path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.smoke_test and (args.dataset is None or args.output is None):
        parser.error("normal mode requires both --dataset and --output; synthetic data is never used implicitly")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.smoke_test:
        print("SYNTHETIC SMOKE ONLY - tensor/integration check, not recommendation evidence")
        dataset = _synthetic_dataset()
        if args.output is None:
            temporary = tempfile.TemporaryDirectory(prefix="learned-reranker-smoke-")
            output = Path(temporary.name) / "smoke.pt"
        else:
            output = args.output
        epochs = min(args.epochs, 2)
        embed_dim = 4
        max_history = 3
    else:
        dataset = _read_dataset(args.dataset)
        output = args.output
        epochs = args.epochs
        embed_dim = 64
        max_history = 10
    split = chronological_split(dataset)
    vocabulary = build_vocabulary(split.train)
    metadata = CheckpointMetadata(
        model_version="v1",
        embed_dim=embed_dim,
        max_history=max_history,
        vocabulary=vocabulary,
        vocabulary_size=len(vocabulary),
        epoch=0,
        dataset=dataset.name,
    )
    result = train_reranker(split.train, split.validation, split.test, metadata, output, TrainerConfig(epochs=epochs, seed=args.seed, patience=2))
    loaded = load_learned_checkpoint(result.checkpoint_path, result.sha256)
    if not loaded.loaded:
        raise RuntimeError(f"saved checkpoint verification failed: {loaded.fallback_reason}")
    print(json.dumps({"classification": "synthetic-smoke-only" if args.smoke_test else "dataset-evaluation", "validation": result.validation_metrics.model_dump(), "test": result.test_metrics.model_dump(), "checkpoint": str(result.checkpoint_path), "sha256": result.sha256}, ensure_ascii=False, indent=2))
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
