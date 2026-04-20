from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from backend.app.models.adapter import CandidatePoi
from backend.app.models.extract import AlgorithmInput
from backend.app.models.ranking import RankedCandidate, RankingRequest, RankingResult


class RankingService:
    """Phase 1/2 ranking service wired to the main backend workflow."""

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

        return RankingResult(
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
            },
        )

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
        alpha = self._clamp(algorithm_input.fusion_config.alpha, 0.0, 1.0)

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            objective_score = objective_scores.get(candidate.poi_id, 0.5)
            subjective_score = subjective_scores.get(candidate.poi_id, 0.5)
            final_score = self._clamp(
                alpha * objective_score + (1 - alpha) * subjective_score,
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
            )
        )
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked

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
    return RankingService()
