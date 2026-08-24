from __future__ import annotations

import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from backend.app.models.adapter import CandidatePoi
from backend.app.models.extract import AlgorithmInput
from backend.app.models.ranking import RankedCandidate, RankingRequest, RankingResult


class RankingService:
    """Deterministic multi-stage ranking while preserving the public service contract."""

    def __init__(
        self,
        *,
        top_k: int = 10,
        sequence_weight: float = 0.15,
        model_mode: str = "off",
        model_path: str | Path = "",
        model_sha256: str = "",
        model_top_n: int = 20,
        model_blend_weight: float = 0.5,
        model_timeout_ms: float = 100.0,
        checkpoint_loader: Callable[..., object] | None = None,
    ):
        self.top_k = max(1, int(top_k))
        self.sequence_weight = self._clamp(sequence_weight, 0.0, 1.0)
        normalized_mode = str(model_mode).strip().lower()
        self.model_mode = (
            normalized_mode if normalized_mode in {"off", "shadow", "rerank"} else "off"
        )
        self.model_path = model_path
        self.model_sha256 = model_sha256
        self.model_top_n = max(1, int(model_top_n))
        try:
            blend_weight = float(model_blend_weight)
        except (TypeError, ValueError):
            blend_weight = 0.5
        if not math.isfinite(blend_weight):
            blend_weight = 0.5
        self.model_blend_weight = self._clamp(blend_weight, 0.0, 1.0)
        try:
            timeout_ms = float(model_timeout_ms)
        except (TypeError, ValueError):
            timeout_ms = 100.0
        self.model_timeout_ms = timeout_ms if math.isfinite(timeout_ms) else 100.0
        self.model_timeout_ms = max(0.0, self.model_timeout_ms)
        self.checkpoint_loader = checkpoint_loader
        self._runtime_lock = threading.Lock()
        self._checkpoint_load_attempted = False
        self._checkpoint_result: object | None = None
        self._learned_reranker: object | None = None

    def rank_candidates(self, request: RankingRequest) -> RankingResult:
        candidate_pool = request.candidate_pool
        algorithm_input = candidate_pool.algorithm_input

        ranked_spot_candidates = self._rank_group(
            candidate_pool.spot_candidates,
            algorithm_input,
            poi_type="spot",
        )
        ranked_food_candidates = self._rank_group(
            candidate_pool.food_candidates,
            algorithm_input,
            poi_type="food",
        )
        ranked_hotel_candidates = self._rank_group(
            candidate_pool.hotel_candidates,
            algorithm_input,
            poi_type="hotel",
        )

        rule_result = RankingResult(
            source_candidate_pool_path=candidate_pool.result_file_path,
            ranked_spot_candidates=ranked_spot_candidates,
            ranked_food_candidates=ranked_food_candidates,
            ranked_hotel_candidates=ranked_hotel_candidates,
            debug_meta={
                "status": "ok",
                "message": "Phase 1/2 ranking is active.",
                "spot_candidate_count": len(candidate_pool.spot_candidates),
                "food_candidate_count": len(candidate_pool.food_candidates),
                "hotel_candidate_count": len(candidate_pool.hotel_candidates),
                "alpha": algorithm_input.fusion_config.alpha,
                "top_k": self.top_k,
                "sequence_weight": self.sequence_weight,
                "sequence_active": self._has_usable_history(algorithm_input),
            },
        )
        if self.model_mode == "off":
            return rule_result
        return self._apply_learned_ranking(rule_result, algorithm_input)

    def _apply_learned_ranking(
        self, rule_result: RankingResult, algorithm_input: AlgorithmInput
    ) -> RankingResult:
        from backend.app.services.ranking.learned.checkpoint import load_learned_checkpoint
        from backend.app.services.ranking.learned.reranker import LearnedReranker

        with self._runtime_lock:
            if not self._checkpoint_load_attempted:
                loader = self.checkpoint_loader or load_learned_checkpoint
                try:
                    self._checkpoint_result = loader(
                        self.model_path, self.model_sha256
                    )
                except Exception:
                    self._checkpoint_result = None
                self._checkpoint_load_attempted = True
                loaded_checkpoint = self._checkpoint_result
                if (
                    loaded_checkpoint is not None
                    and getattr(loaded_checkpoint, "loaded", False)
                    and getattr(loaded_checkpoint, "model", None) is not None
                    and getattr(loaded_checkpoint, "metadata", None) is not None
                ):
                    try:
                        self._learned_reranker = LearnedReranker(
                            loaded_checkpoint.model,
                            loaded_checkpoint.metadata,
                            timeout_ms=self.model_timeout_ms,
                        )
                    except Exception:
                        self._learned_reranker = None
            loaded = self._checkpoint_result
        groups = [
            rule_result.ranked_spot_candidates,
            rule_result.ranked_food_candidates,
            rule_result.ranked_hotel_candidates,
        ]
        diagnostics = {
            "model_mode": self.model_mode,
            "model_version": getattr(loaded, "version", None),
            "checkpoint_hash": getattr(loaded, "sha256", None),
            # Total candidates submitted across the three rule top-N prefixes.
            "rerank_candidate_count": 0,
            "fallback_reason": None,
            "inference_ms": 0.0,
            "ranking_changed": False,
        }
        if loaded is None or not getattr(loaded, "loaded", False):
            diagnostics["fallback_reason"] = getattr(
                loaded, "fallback_reason", "loader_error"
            )
            rule_result.debug_meta.update(diagnostics)
            return rule_result

        if (
            getattr(loaded, "model", None) is None
            or getattr(loaded, "metadata", None) is None
        ):
            diagnostics["fallback_reason"] = "invalid_checkpoint"
            rule_result.debug_meta.update(diagnostics)
            return rule_result
        reranker = self._learned_reranker
        if not isinstance(reranker, LearnedReranker):
            diagnostics["fallback_reason"] = "invalid_checkpoint"
            rule_result.debug_meta.update(diagnostics)
            return rule_result
        scored_groups: list[tuple[list[RankedCandidate], list[float]]] = []
        for group in groups:
            prefix = group[: min(self.model_top_n, len(group))]
            scored = reranker.score(prefix, algorithm_input)
            diagnostics["inference_ms"] += scored.inference_ms
            diagnostics["rerank_candidate_count"] += len(prefix)
            if scored.fallback_reason is not None or scored.scores is None:
                diagnostics["fallback_reason"] = scored.fallback_reason
                rule_result.debug_meta.update(diagnostics)
                return rule_result
            scored_groups.append((group, scored.scores))

        diagnostics["ranking_changed"] = any(
            self._counterfactual_order(group, scores)
            != [item.poi_id for item in group]
            for group, scores in scored_groups
        )

        # Shadow deliberately returns the exact rule candidate structures.
        if self.model_mode == "shadow":
            rule_result.debug_meta.update(diagnostics)
            return rule_result

        result = rule_result.model_copy(deep=True)
        output_groups = [
            result.ranked_spot_candidates,
            result.ranked_food_candidates,
            result.ranked_hotel_candidates,
        ]
        for (rule_group, learned_scores), output_group in zip(
            scored_groups, output_groups
        ):
            count = len(learned_scores)
            blended: list[tuple[int, RankedCandidate]] = []
            for index, (item, learned_score) in enumerate(
                zip(output_group[:count], learned_scores)
            ):
                rule_score = float(rule_group[index].score)
                final_score = (
                    (1.0 - self.model_blend_weight) * rule_score
                    + self.model_blend_weight * learned_score
                )
                if not math.isfinite(final_score):
                    diagnostics["fallback_reason"] = "nonfinite_score"
                    rule_result.debug_meta.update(diagnostics)
                    return rule_result
                item.score_breakdown = {
                    **item.score_breakdown,
                    "rule": rule_score,
                    "learned": learned_score,
                    "learned_weight": self.model_blend_weight,
                }
                item.score = round(final_score, 4)
                blended.append((index, item))
            blended.sort(key=lambda pair: (-pair[1].score, pair[0]))
            output_group[:] = [item for _, item in blended] + output_group[count:]
            for rank, item in enumerate(output_group, 1):
                item.rank = rank

        result.debug_meta.update(diagnostics)
        return result

    def _counterfactual_order(
        self, rule_group: list[RankedCandidate], learned_scores: list[float]
    ) -> list[str]:
        """Return the blended top-N order without mutating rule candidates."""

        prefix = [
            (
                index,
                item.poi_id,
                round(
                    (1.0 - self.model_blend_weight) * float(item.score)
                    + self.model_blend_weight * learned_score,
                    4,
                ),
            )
            for index, (item, learned_score) in enumerate(
                zip(rule_group, learned_scores)
            )
        ]
        prefix.sort(key=lambda entry: (-entry[2], entry[0]))
        return [poi_id for _, poi_id, _ in prefix] + [
            item.poi_id for item in rule_group[len(learned_scores) :]
        ]

    def _rank_group(
        self,
        candidates: list[CandidatePoi],
        algorithm_input: AlgorithmInput,
        *,
        poi_type: str,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        objective_scores = self._calculate_objective_scores(candidates, algorithm_input)
        subjective_scores = self._calculate_subjective_scores(
            candidates,
            algorithm_input,
            poi_type=poi_type,
        )
        sequence_scores = self._calculate_sequence_scores(candidates, algorithm_input)
        alpha = self._clamp(algorithm_input.fusion_config.alpha, 0.0, 1.0)
        sequence_weight = self.sequence_weight if sequence_scores is not None else 0.0
        base_weight = 1.0 - sequence_weight

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            objective_score = objective_scores.get(candidate.poi_id, 0.5)
            subjective_score = subjective_scores.get(candidate.poi_id, 0.5)
            sequence_score = (
                sequence_scores.get(candidate.poi_id, 1.0)
                if sequence_scores is not None
                else None
            )
            base_score = alpha * objective_score + (1 - alpha) * subjective_score
            final_score = self._clamp(
                base_weight * base_score
                + sequence_weight * (sequence_score if sequence_score is not None else 1.0),
                0.0,
                1.0,
            )

            candidate.objective_features = {
                **candidate.objective_features,
                "objective_score": round(objective_score, 4),
                "subjective_score": round(subjective_score, 4),
            }
            ranked.append(
                RankedCandidate(
                    poi_id=candidate.poi_id,
                    poi_type=candidate.poi_type,
                    score=round(final_score, 4),
                    candidate=candidate,
                    score_breakdown={
                        "objective": round(objective_score, 4),
                        "subjective": round(subjective_score, 4),
                        "alpha": round(alpha, 4),
                        "sequence": round(sequence_score, 4) if sequence_score is not None else 0.0,
                        "sequence_weight": round(sequence_weight, 4),
                    },
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                item.candidate.center_distance_m is None,
                item.candidate.center_distance_m
                if item.candidate.center_distance_m is not None
                else 10**9,
                -(item.candidate.rating or 0),
                item.candidate.name,
                item.poi_id,
            )
        )
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[: self.top_k]

    def _calculate_sequence_scores(
        self,
        candidates: list[CandidatePoi],
        algorithm_input: AlgorithmInput,
    ) -> dict[str, float] | None:
        """Apply a small repeat-visit penalty when at least three history IDs exist.

        The current input contract has POI IDs but no historical tags, so this is
        intentionally not presented as a learned sequence model.
        """
        if not self._has_usable_history(algorithm_input):
            return None
        recent_ids = set(algorithm_input.sequence_model_input.historical_poi_ids[-3:])
        return {
            candidate.poi_id: (0.0 if candidate.poi_id in recent_ids else 1.0)
            for candidate in candidates
        }

    @staticmethod
    def _has_usable_history(algorithm_input: AlgorithmInput) -> bool:
        sequence_input = algorithm_input.sequence_model_input
        return sequence_input.has_history and len(sequence_input.historical_poi_ids) >= 3

    def _calculate_objective_scores(
        self,
        candidates: list[CandidatePoi],
        algorithm_input: AlgorithmInput,
    ) -> dict[str, float]:
        ratings = [candidate.rating for candidate in candidates if candidate.rating is not None]
        distances = [
            candidate.center_distance_m
            for candidate in candidates
            if candidate.center_distance_m is not None
        ]
        popularities = [
            candidate.popularity for candidate in candidates if candidate.popularity is not None
        ]

        weights = algorithm_input.objective_weights
        raw_weight_sum = (
            max(weights.rating_weight, 0)
            + max(weights.distance_weight, 0)
            + max(weights.popularity_weight, 0)
        )
        if raw_weight_sum <= 0:
            rating_weight = distance_weight = popularity_weight = 1 / 3
        else:
            rating_weight = max(weights.rating_weight, 0) / raw_weight_sum
            distance_weight = max(weights.distance_weight, 0) / raw_weight_sum
            popularity_weight = max(weights.popularity_weight, 0) / raw_weight_sum

        scores: dict[str, float] = {}
        for candidate in candidates:
            normalized_rating = self._normalize(candidate.rating, ratings)
            normalized_distance = self._normalize_reverse(
                candidate.center_distance_m,
                distances,
            )
            normalized_popularity = self._normalize(candidate.popularity, popularities)
            score = (
                rating_weight * normalized_rating
                + distance_weight * normalized_distance
                + popularity_weight * normalized_popularity
            )
            scores[candidate.poi_id] = self._clamp(score, 0.0, 1.0)
        return scores

    def _calculate_subjective_scores(
        self,
        candidates: list[CandidatePoi],
        algorithm_input: AlgorithmInput,
        *,
        poi_type: str,
    ) -> dict[str, float]:
        user_counter = self._build_user_preference_counter(algorithm_input, poi_type=poi_type)
        budget_min = algorithm_input.subjective_preference.budget_min_cny
        budget_max = algorithm_input.subjective_preference.budget_max_cny

        scores: dict[str, float] = {}
        for candidate in candidates:
            candidate_counter = self._build_candidate_counter(candidate)
            text_similarity = (
                self._cosine_similarity(user_counter, candidate_counter)
                if user_counter
                else 0.5
            )
            budget_fit = self._budget_fit_score(
                candidate.price_value_cny,
                budget_min,
                budget_max,
            )
            if budget_fit is None:
                subjective_score = text_similarity
            elif user_counter:
                subjective_score = 0.75 * text_similarity + 0.25 * budget_fit
            else:
                subjective_score = budget_fit
            scores[candidate.poi_id] = self._clamp(subjective_score, 0.0, 1.0)
        return scores

    def _build_user_preference_counter(
        self,
        algorithm_input: AlgorithmInput,
        *,
        poi_type: str,
    ) -> Counter[str]:
        preference = algorithm_input.subjective_preference
        terms: list[str] = []
        terms.extend(preference.preference_terms)
        terms.extend(preference.travel_styles)
        if poi_type == "spot":
            terms.extend(preference.spot_keywords)
        elif poi_type == "food":
            terms.extend(preference.food_keywords)
        elif poi_type == "hotel":
            terms.extend(preference.hotel_keywords)
        return Counter(self._normalize_terms(terms))

    def _build_candidate_counter(self, candidate: CandidatePoi) -> Counter[str]:
        terms: list[str] = []
        terms.extend(candidate.tags)
        terms.extend(
            filter(
                None,
                [
                    candidate.name,
                    candidate.address,
                    candidate.source_dataset,
                ],
            )
        )
        return Counter(self._normalize_terms(terms))

    def _normalize_terms(self, values: Iterable[str]) -> list[str]:
        tokens: list[str] = []
        for value in values:
            text = value.strip().lower()
            if not text:
                continue
            parts = re.split(r"[\s,\uFF0C\u3001/|;\uFF1B]+", text)
            for part in parts:
                token = part.strip()
                if token:
                    tokens.append(token)
        return tokens

    def _budget_fit_score(
        self,
        price_value_cny: float | None,
        budget_min_cny: int | None,
        budget_max_cny: int | None,
    ) -> float | None:
        if price_value_cny is None:
            return None
        if budget_min_cny is None and budget_max_cny is None:
            return None

        if budget_min_cny is not None and price_value_cny < budget_min_cny:
            gap = budget_min_cny - price_value_cny
            return self._clamp(1.0 - gap / max(budget_min_cny, 1), 0.0, 1.0)

        if budget_max_cny is not None and price_value_cny > budget_max_cny:
            gap = price_value_cny - budget_max_cny
            return self._clamp(1.0 - gap / max(budget_max_cny, 1), 0.0, 1.0)

        return 1.0

    @staticmethod
    def _normalize(value: float | None, values: list[float]) -> float:
        if value is None or not values:
            return 0.5
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum):
            return 0.5
        return (value - minimum) / (maximum - minimum)

    @staticmethod
    def _normalize_reverse(value: int | None, values: list[int]) -> float:
        if value is None or not values:
            return 0.5
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return 0.5
        return 1 - (value - minimum) / (maximum - minimum)

    @staticmethod
    def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot_product = 0.0
        for token, count in left.items():
            dot_product += count * right.get(token, 0)
        left_norm = math.sqrt(sum(count * count for count in left.values()))
        right_norm = math.sqrt(sum(count * count for count in right.values()))
        if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
            return 0.0
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def build_ranking_service() -> RankingService:
    from backend.app.core.config import get_settings

    settings = get_settings()
    return RankingService(
        model_mode=settings.ranking_model_mode,
        model_path=settings.ranking_model_path,
        model_sha256=settings.ranking_model_sha256,
        model_top_n=settings.ranking_model_top_n,
        model_blend_weight=settings.ranking_model_blend_weight,
    )
