"""Bounded training records and leakage-resistant chronological splitting."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from backend.app.models.extract import AlgorithmInput
from backend.app.models.ranking import RankedCandidate
from backend.app.services.ranking.learned.checkpoint import MAX_VOCABULARY_SIZE
from backend.app.services.ranking.learned.reranker import tokenize_feature_values

MAX_QUERIES = 100_000
MAX_CANDIDATES_PER_QUERY = 1_000
MAX_FEATURE_STRINGS = 128
MAX_TAGS = 64
MAX_MAPPING_ITEMS = 64
MAX_HISTORY_ITEMS = 100
MAX_STRING_LENGTH = 512


def _forbid_extra(value: Any, allowed: set[str], path: str) -> None:
    if not isinstance(value, dict):
        return
    extras = set(value) - allowed
    if extras:
        raise ValueError(f"extra fields are forbidden at {path}: {sorted(extras)!r}")


def _strict_number(value: Any, field: str, *, integer: bool = False) -> None:
    expected = int if integer else (int, float)
    if value is not None and (isinstance(value, bool) or not isinstance(value, expected)):
        raise ValueError(f"{field} must use a strict numeric JSON value")


def _bounded_string(value: str | None, field: str, maximum: int = MAX_STRING_LENGTH) -> None:
    if value is not None and not 1 <= len(value) <= maximum:
        raise ValueError(f"{field} strings must contain 1..{maximum} characters")


def _bounded_string_list(
    values: list[str], field: str, maximum_items: int = MAX_FEATURE_STRINGS
) -> None:
    if len(values) > maximum_items:
        raise ValueError(f"{field} contains too many items")
    for value in values:
        _bounded_string(value, field, 128)


def _finite_number(value: Any, field: str, *, lower: float, upper: float) -> None:
    if value is None:
        return
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} must be finite and in [{lower}, {upper}]")


class TrainingQuery(BaseModel):
    """One timestamped ranking session.

    ``split_group_id`` is the indivisible session identity used for leakage
    checks. It defaults to ``query_id`` for backward-compatible one-query
    sessions; callers with a separate session identity should provide it.
    """

    query_id: StrictStr
    split_group_id: StrictStr | None = None
    timestamp: datetime
    algorithm_input: AlgorithmInput
    candidates: list[RankedCandidate] = Field(min_length=2, max_length=MAX_CANDIDATES_PER_QUERY)
    labels: list[float] = Field(min_length=2, max_length=MAX_CANDIDATES_PER_QUERY)
    categories: list[StrictStr] = Field(min_length=2, max_length=MAX_CANDIDATES_PER_QUERY)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _strict_nested_schema(cls, value: Any) -> Any:
        """Reject nested extras and coercive scalar types before base DTO parsing."""

        if not isinstance(value, dict):
            return value
        _forbid_extra(
            value,
            {
                "query_id",
                "split_group_id",
                "timestamp",
                "algorithm_input",
                "candidates",
                "labels",
                "categories",
            },
            "query",
        )
        labels = value.get("labels")
        if isinstance(labels, list):
            for label in labels:
                _strict_number(label, "labels")

        algorithm = value.get("algorithm_input")
        _forbid_extra(
            algorithm,
            {
                "search_context",
                "objective_weights",
                "subjective_preference",
                "fusion_config",
                "sequence_model_input",
            },
            "algorithm_input",
        )
        if isinstance(algorithm, dict):
            search = algorithm.get("search_context")
            _forbid_extra(
                search,
                {
                    "destination_text",
                    "center_location",
                    "search_radius_m",
                    "candidate_types",
                    "date_texts",
                    "people_count",
                },
                "algorithm_input.search_context",
            )
            if isinstance(search, dict):
                _strict_number(search.get("search_radius_m"), "search_radius_m", integer=True)
                _strict_number(search.get("people_count"), "people_count", integer=True)
                center = search.get("center_location")
                _forbid_extra(center, {"lng", "lat"}, "center_location")
                if isinstance(center, dict):
                    _strict_number(center.get("lng"), "longitude")
                    _strict_number(center.get("lat"), "latitude")
            weights = algorithm.get("objective_weights")
            _forbid_extra(
                weights,
                {"rating_weight", "distance_weight", "popularity_weight"},
                "objective_weights",
            )
            if isinstance(weights, dict):
                for key, item in weights.items():
                    _strict_number(item, key)
            preference = algorithm.get("subjective_preference")
            _forbid_extra(
                preference,
                {
                    "destination",
                    "budget_text",
                    "budget_min_cny",
                    "budget_max_cny",
                    "people_count",
                    "spot_keywords",
                    "food_keywords",
                    "hotel_keywords",
                    "travel_styles",
                    "preference_terms",
                },
                "subjective_preference",
            )
            if isinstance(preference, dict):
                for key in ("budget_min_cny", "budget_max_cny", "people_count"):
                    _strict_number(preference.get(key), key, integer=True)
            fusion = algorithm.get("fusion_config")
            _forbid_extra(fusion, {"alpha"}, "fusion_config")
            if isinstance(fusion, dict):
                _strict_number(fusion.get("alpha"), "alpha")
            sequence = algorithm.get("sequence_model_input")
            _forbid_extra(
                sequence,
                {"has_history", "historical_poi_ids", "time_context"},
                "sequence_model_input",
            )
            if isinstance(sequence, dict) and "has_history" in sequence:
                if type(sequence["has_history"]) is not bool:
                    raise ValueError("has_history must be a strict boolean")

        candidates = value.get("candidates")
        if isinstance(candidates, list):
            for ranked in candidates:
                _forbid_extra(
                    ranked,
                    {
                        "poi_id",
                        "poi_type",
                        "score",
                        "rank",
                        "score_breakdown",
                        "candidate",
                    },
                    "ranked_candidate",
                )
                if not isinstance(ranked, dict):
                    continue
                _strict_number(ranked.get("score"), "score")
                _strict_number(ranked.get("rank"), "rank", integer=True)
                breakdown = ranked.get("score_breakdown")
                if isinstance(breakdown, dict):
                    for key, item in breakdown.items():
                        _strict_number(item, f"score_breakdown.{key}")
                poi = ranked.get("candidate")
                _forbid_extra(
                    poi,
                    {
                        "poi_id",
                        "poi_type",
                        "source_dataset",
                        "name",
                        "address",
                        "longitude",
                        "latitude",
                        "center_distance_m",
                        "rating",
                        "popularity",
                        "price_value_cny",
                        "review_count",
                        "tags",
                        "objective_features",
                        "source_provider",
                    },
                    "candidate",
                )
                if isinstance(poi, dict):
                    for key in (
                        "longitude",
                        "latitude",
                        "rating",
                        "popularity",
                        "price_value_cny",
                    ):
                        _strict_number(poi.get(key), key)
                    for key in ("center_distance_m", "review_count"):
                        _strict_number(poi.get(key), key, integer=True)
                    features = poi.get("objective_features")
                    if isinstance(features, dict):
                        for key, item in features.items():
                            if item is not None and not isinstance(item, str):
                                _strict_number(item, f"objective_features.{key}")
        return value

    @field_validator("query_id")
    @classmethod
    def _bounded_query_id(cls, value: str) -> str:
        if not 1 <= len(value) <= 256:
            raise ValueError("query_id must contain 1..256 characters")
        return value

    @field_validator("split_group_id")
    @classmethod
    def _bounded_split_group_id(cls, value: str | None) -> str | None:
        if value is not None and not 1 <= len(value) <= 256:
            raise ValueError("split_group_id must contain 1..256 characters")
        return value

    @field_validator("timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        utc_year = value.astimezone(timezone.utc).year
        if not 1970 <= utc_year <= 2100:
            raise ValueError("timestamp must be between 1970 and 2100 UTC")
        return value

    @field_validator("labels")
    @classmethod
    def _bounded_labels(cls, values: list[float]) -> list[float]:
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("labels must be finite relevance values in [0, 1]")
        return values

    @field_validator("categories")
    @classmethod
    def _bounded_categories(cls, values: list[str]) -> list[str]:
        if any(not 1 <= len(value) <= 128 for value in values):
            raise ValueError("categories must contain bounded non-empty strings")
        return values

    @model_validator(mode="after")
    def _aligned_fields(self) -> "TrainingQuery":
        if self.split_group_id is None:
            self.split_group_id = self.query_id
        size = len(self.candidates)
        if len(self.labels) != size or len(self.categories) != size:
            raise ValueError("candidates, labels, and categories must be aligned")
        self._validate_nested_input()
        return self

    def _validate_nested_input(self) -> None:
        algorithm = self.algorithm_input
        search = algorithm.search_context
        for field in ("destination_text",):
            _bounded_string(getattr(search, field), f"search_context.{field}")
        _bounded_string_list(search.candidate_types, "candidate_types", 32)
        _bounded_string_list(search.date_texts, "date_texts", 128)
        _finite_number(search.search_radius_m, "search_radius_m", lower=0, upper=10_000_000)
        _finite_number(search.people_count, "people_count", lower=1, upper=1_000_000)
        _finite_number(search.center_location.lng, "longitude", lower=-180, upper=180)
        _finite_number(search.center_location.lat, "latitude", lower=-90, upper=90)

        preference = algorithm.subjective_preference
        for field in ("destination", "budget_text"):
            _bounded_string(getattr(preference, field), f"preference.{field}")
        for field in (
            "spot_keywords",
            "food_keywords",
            "hotel_keywords",
            "travel_styles",
            "preference_terms",
        ):
            _bounded_string_list(getattr(preference, field), field)
        for field in ("budget_min_cny", "budget_max_cny"):
            _finite_number(getattr(preference, field), field, lower=0, upper=1_000_000_000)
        _finite_number(preference.people_count, "people_count", lower=1, upper=1_000_000)

        history = algorithm.sequence_model_input
        _bounded_string_list(
            history.historical_poi_ids, "history", MAX_HISTORY_ITEMS
        )
        _bounded_string_list(history.time_context, "time_context", MAX_HISTORY_ITEMS)
        for field, value in algorithm.objective_weights.model_dump().items():
            _finite_number(value, field, lower=0, upper=1)
        _finite_number(algorithm.fusion_config.alpha, "alpha", lower=0, upper=1)

        for ranked in self.candidates:
            _bounded_string(ranked.poi_id, "poi_id", 256)
            _bounded_string(ranked.poi_type, "poi_type", 64)
            _finite_number(ranked.score, "score", lower=-1_000_000, upper=1_000_000)
            _finite_number(ranked.rank, "rank", lower=0, upper=1_000_000)
            if len(ranked.score_breakdown) > MAX_MAPPING_ITEMS:
                raise ValueError("score_breakdown contains too many items")
            for key, value in ranked.score_breakdown.items():
                _bounded_string(key, "score_breakdown key", 128)
                _finite_number(value, "score_breakdown value", lower=-1e12, upper=1e12)
            poi = ranked.candidate
            for field in (
                "poi_id",
                "poi_type",
                "source_dataset",
                "name",
                "address",
                "source_provider",
            ):
                _bounded_string(getattr(poi, field), f"candidate.{field}")
            _bounded_string_list(poi.tags, "candidate tags", MAX_TAGS)
            for field, lower, upper in (
                ("longitude", -180, 180),
                ("latitude", -90, 90),
                ("center_distance_m", 0, 1e9),
                ("rating", 0, 100),
                ("popularity", 0, 1e12),
                ("price_value_cny", 0, 1e12),
                ("review_count", 0, 1e12),
            ):
                _finite_number(getattr(poi, field), field, lower=lower, upper=upper)
            if len(poi.objective_features) > MAX_MAPPING_ITEMS:
                raise ValueError("objective_features contains too many items")
            for key, value in poi.objective_features.items():
                _bounded_string(key, "objective_features key", 128)
                if isinstance(value, str):
                    _bounded_string(value, "objective_features string")
                elif value is not None:
                    _finite_number(value, "objective_features value", lower=-1e12, upper=1e12)


class TrainingDataset(BaseModel):
    name: StrictStr
    queries: list[TrainingQuery] = Field(min_length=1, max_length=MAX_QUERIES)

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def _bounded_name(cls, value: str) -> str:
        if not 1 <= len(value) <= 256:
            raise ValueError("dataset name must contain 1..256 characters")
        return value

    @model_validator(mode="after")
    def _unique_query_ids(self) -> "TrainingDataset":
        ids = [query.query_id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("query_id values must be unique within a dataset")
        group_timestamps: dict[str | None, datetime] = {}
        for query in self.queries:
            previous = group_timestamps.setdefault(
                query.split_group_id, query.timestamp
            )
            if previous != query.timestamp:
                raise ValueError(
                    "all records in a split_group_id must have the same timestamp"
                )
        return self


class DatasetSplit(BaseModel):
    train: TrainingDataset
    validation: TrainingDataset
    test: TrainingDataset

    model_config = ConfigDict(extra="forbid")


def chronological_split(
    dataset: TrainingDataset,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> DatasetSplit:
    """Split whole timestamp buckets into train, validation, and later test sets.

    Equal timestamps are never divided across sets. At least three distinct
    timestamps are required, and boundaries are strictly increasing, preventing
    timestamp overlap/leakage. Repeated records in a coherent session group
    share one timestamp and therefore cannot cross a boundary. The test
    partition is always the latest data; user identity may recur because the
    leakage boundary is session/group identity, not user ID.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation and test fractions must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")

    buckets: dict[datetime, list[TrainingQuery]] = {}
    for query in dataset.queries:
        buckets.setdefault(query.timestamp, []).append(query)
    timestamps = sorted(buckets)
    if len(timestamps) < 3:
        raise ValueError("at least three distinct timestamps are required")

    total = len(timestamps)
    test_count = max(1, round(total * test_fraction))
    validation_count = max(1, round(total * validation_fraction))
    if test_count + validation_count >= total:
        raise ValueError("dataset is too small for non-empty chronological splits")
    train_end = total - validation_count - test_count
    validation_end = total - test_count

    def collect(selected: list[datetime]) -> list[TrainingQuery]:
        return [query for timestamp in selected for query in buckets[timestamp]]

    return DatasetSplit(
        train=TrainingDataset(name=f"{dataset.name}:train", queries=collect(timestamps[:train_end])),
        validation=TrainingDataset(name=f"{dataset.name}:validation", queries=collect(timestamps[train_end:validation_end])),
        test=TrainingDataset(name=f"{dataset.name}:test", queries=collect(timestamps[validation_end:])),
    )


def build_vocabulary(dataset: TrainingDataset) -> dict[str, int]:
    """Build deterministic vocabulary from the training partition only.

    Validation/test terms are intentionally absent and follow the shared
    inference builder's unknown-token behavior.
    """

    tokens: set[str] = set()
    for query in dataset.queries:
        preference = query.algorithm_input.subjective_preference
        tokens.update(
            tokenize_feature_values(
                [
                    *preference.preference_terms,
                    *preference.travel_styles,
                    *preference.spot_keywords,
                    *preference.food_keywords,
                    *preference.hotel_keywords,
                ]
            )
        )
        tokens.update(
            poi_id.lower()
            for poi_id in query.algorithm_input.sequence_model_input.historical_poi_ids
        )
        for ranked in query.candidates:
            poi = ranked.candidate
            tokens.update(
                tokenize_feature_values(
                    [*poi.tags, poi.name, poi.address, poi.source_dataset]
                )
            )
    if len(tokens) > MAX_VOCABULARY_SIZE:
        raise ValueError("training vocabulary exceeds checkpoint bounds")
    return {token: index for index, token in enumerate(sorted(tokens))}
