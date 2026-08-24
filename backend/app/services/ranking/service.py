from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.models.adapter import CandidatePoi, CandidatePoolResult
from backend.app.models.extract import AlgorithmInput
from backend.app.models.ranking import RankedCandidate, RankingRequest, RankingResult


class POIRecommendationService:
    DEFAULT_THETA = {"theta1": 0.4, "theta2": 0.3, "theta3": 0.3}
    DEFAULT_ALPHA = 0.6
    DEFAULT_LAMBDA_SEQ = 0.2
    DEFAULT_TOP_K = 20

    # 模型checkpoint路径
    _CKPT_DIR = Path(__file__).parent / "checkpoints"
    _CKPT_PATH = _CKPT_DIR / "best.pt"

    def __init__(self) -> None:
        self.theta1 = self.DEFAULT_THETA["theta1"]
        self.theta2 = self.DEFAULT_THETA["theta2"]
        self.theta3 = self.DEFAULT_THETA["theta3"]
        self.alpha = self.DEFAULT_ALPHA
        self.lambda_seq = self.DEFAULT_LAMBDA_SEQ
        self.top_k = self.DEFAULT_TOP_K

        # 尝试加载训练好的深度学习模型
        self._model = None
        self._tag_vocab: dict[str, int] | None = None
        self._poi_lookup: dict[str, dict] = {}
        self._embed_dim = 64
        self._load_model()

    def _load_model(self):
        """加载训练好的模型checkpoint（失败时静默回退到规则算法）"""
        try:
            import torch
            from backend.app.services.ranking.model.poi_recommender import POIDeepRecommender

            if not self._CKPT_PATH.exists():
                return

            ckpt = torch.load(str(self._CKPT_PATH), map_location="cpu", weights_only=False)
            self._tag_vocab = ckpt["tag_vocab"]
            self._poi_lookup = {p["poi_id"]: p for p in ckpt.get("poi_list", [])}
            self._embed_dim = ckpt.get("embed_dim", 64)

            self._model = POIDeepRecommender(
                num_tags=ckpt["num_tags"],
                embed_dim=self._embed_dim,
                window_size=10,
                rho=0.5,
            )
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
        except Exception as e:
            self._model = None
            import sys
            print(f"[RankingService] 模型未加载，使用规则算法: {e}", file=sys.stderr)

    def _tags_to_indices(self, tags: list[str]) -> list[int]:
        if not self._tag_vocab or not tags:
            return [0]
        return [self._tag_vocab[t] for t in tags if t in self._tag_vocab] or [0]

    # ==================== 主入口：模型优先，规则fallback ====================

    def rank_candidates(self, request: RankingRequest) -> RankingResult:
        if self._model is not None:
            return self._rank_with_model(request)
        return self._rank_with_rules(request)

    # ==================== 深度学习模型排序 ====================

    def _rank_with_model(self, request: RankingRequest) -> RankingResult:
        import torch

        candidate_pool = request.candidate_pool
        algorithm_input = candidate_pool.algorithm_input

        # Stage 1: 客观打分（复用现有逻辑）
        spot_candidates = self._stage1_objective_scoring(candidate_pool.spot_candidates)
        food_candidates = self._stage1_objective_scoring(candidate_pool.food_candidates)
        hotel_candidates = self._stage1_objective_scoring(candidate_pool.hotel_candidates)
        all_candidates = spot_candidates + food_candidates + hotel_candidates

        # 提取用户偏好 -> 嵌入
        user_preferences = self._extract_preferences(algorithm_input)
        user_tag_idx = self._tags_to_indices(user_preferences)
        user_tensor = torch.tensor(user_tag_idx, dtype=torch.long)
        user_vec = self._model.get_user_embedding(user_tensor)

        # 提取历史轨迹
        user_history = self._extract_history(algorithm_input)

        # 为每个候选POI计算模型得分
        final_scores: dict[str, float] = {}
        breakdowns: dict[str, dict] = {}

        for poi in all_candidates:
            # POI嵌入
            poi_tag_idx = self._tags_to_indices(poi.tags)
            poi_tensor = torch.tensor(poi_tag_idx, dtype=torch.long)
            poi_vec = self._model.get_poi_embedding(poi_tensor)

            # 客观分
            s_obj = (poi.objective_features or {}).get("objective_score", 0.5)
            s_obj_t = torch.tensor([s_obj], dtype=torch.float32)

            # 历史序列构建
            history_seq = None
            history_locs = None
            current_loc = None

            if user_history and len(user_history) >= 3 and poi.latitude and poi.longitude:
                hist_vecs = []
                hist_locs_list = []
                for h in user_history:
                    h_pid = h.get("poi_id")
                    if h_pid and h_pid in self._poi_lookup:
                        hp = self._poi_lookup[h_pid]
                        h_idx = self._tags_to_indices(hp.get("tags", []))
                        h_t = torch.tensor(h_idx, dtype=torch.long)
                        hist_vecs.append(self._model.get_poi_embedding(h_t))
                        hist_locs_list.append(torch.tensor(
                            [hp.get("lat", 0.0), hp.get("lng", 0.0)], dtype=torch.float32))

                if len(hist_vecs) >= 3:
                    history_seq = torch.stack(hist_vecs)
                    history_locs = torch.stack(hist_locs_list)
                    current_loc = torch.tensor(
                        [poi.latitude, poi.longitude], dtype=torch.float32)

            # 模型推理
            s_total, s_bilinear, s_seq = self._model.inference_single(
                user_vec, poi_vec, s_obj_t,
                history_seq=history_seq,
                history_locs=history_locs,
                current_loc=current_loc,
            )

            final_scores[poi.poi_id] = max(0.0, min(1.0, s_total))
            breakdowns[poi.poi_id] = {
                "objective": round(s_obj, 4),
                "subjective": round(s_bilinear, 4),
                "sequence": round(s_seq, 4),
            }

        ranked_spots = self._rank_group_dl(spot_candidates, final_scores, breakdowns)
        ranked_food = self._rank_group_dl(food_candidates, final_scores, breakdowns)
        ranked_hotel = self._rank_group_dl(hotel_candidates, final_scores, breakdowns)

        return RankingResult(
            source_candidate_pool_path=candidate_pool.result_file_path,
            ranked_spot_candidates=ranked_spots,
            ranked_food_candidates=ranked_food,
            ranked_hotel_candidates=ranked_hotel,
            debug_meta={
                "status": "ok",
                "message": "深度学习四阶段推荐算法 (v0.1.0-synthetic)",
                "model": "POIDeepRecommender",
                "embed_dim": self._embed_dim,
                "spot_candidate_count": len(spot_candidates),
                "food_candidate_count": len(food_candidates),
                "hotel_candidate_count": len(hotel_candidates),
            },
        )

    def _rank_group_dl(self, candidates: list[CandidatePoi],
                       scores: dict[str, float],
                       breakdowns: dict[str, dict]) -> list[RankedCandidate]:
        ranked = []
        for candidate in candidates:
            score = scores.get(candidate.poi_id, 0.0)
            ranked.append(RankedCandidate(
                poi_id=candidate.poi_id,
                poi_type=candidate.poi_type,
                score=round(score, 4),
                candidate=candidate,
                score_breakdown=breakdowns.get(candidate.poi_id, {}),
            ))
        ranked.sort(key=lambda item: (
            -item.score,
            item.candidate.center_distance_m is None,
            item.candidate.center_distance_m if item.candidate.center_distance_m else 10**9,
            -(item.candidate.rating or 0),
        ))
        for index, item in enumerate(ranked, start=1):
            item.rank = index
        return ranked[: self.top_k]

    # ==================== 规则算法（fallback） ====================

    def _rank_with_rules(self, request: RankingRequest) -> RankingResult:
        candidate_pool = request.candidate_pool
        algorithm_input = candidate_pool.algorithm_input

        spot_candidates = self._stage1_objective_scoring(candidate_pool.spot_candidates)
        food_candidates = self._stage1_objective_scoring(candidate_pool.food_candidates)
        hotel_candidates = self._stage1_objective_scoring(candidate_pool.hotel_candidates)

        user_preferences = self._extract_preferences(algorithm_input)
        all_candidates = spot_candidates + food_candidates + hotel_candidates
        subjective_scores = self._stage2_subjective_weighting(all_candidates, user_preferences)

        user_history = self._extract_history(algorithm_input)
        sequence_scores = self._stage3_sequence_prediction(all_candidates, user_history)

        final_scores = self._stage4_fuse_scores(all_candidates, subjective_scores, sequence_scores)

        ranked_spots = self._rank_group(spot_candidates, final_scores, "spot")
        ranked_food = self._rank_group(food_candidates, final_scores, "food")
        ranked_hotel = self._rank_group(hotel_candidates, final_scores, "hotel")

        return RankingResult(
            source_candidate_pool_path=candidate_pool.result_file_path,
            ranked_spot_candidates=ranked_spots,
            ranked_food_candidates=ranked_food,
            ranked_hotel_candidates=ranked_hotel,
            debug_meta={
                "status": "ok",
                "message": "四阶段推荐算法已启用 (规则版)",
                "spot_candidate_count": len(spot_candidates),
                "food_candidate_count": len(food_candidates),
                "hotel_candidate_count": len(hotel_candidates),
                "alpha": self.alpha,
                "lambda_seq": self.lambda_seq,
            },
        )

    def _stage1_objective_scoring(self, candidates: list[CandidatePoi]) -> list[CandidatePoi]:
        if not candidates:
            return candidates

        ratings = [c.rating for c in candidates if c.rating is not None]
        distances = [c.center_distance_m for c in candidates if c.center_distance_m is not None]
        popularities = [c.popularity for c in candidates if c.popularity is not None]

        def normalize(value: float | None, values: list[float]) -> float:
            if value is None or not values:
                return 0.5
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return 0.5
            return (value - min_val) / (max_val - min_val)

        def normalize_distance(value: int | None, distances: list[int]) -> float:
            if value is None or not distances:
                return 0.5
            min_val, max_val = min(distances), max(distances)
            if max_val == min_val:
                return 0.5
            return 1 - (value - min_val) / (max_val - min_val)

        updated = []
        for poi in candidates:
            norm_rating = normalize(poi.rating, ratings)
            norm_distance = normalize_distance(poi.center_distance_m, distances)
            norm_popularity = normalize(poi.popularity, popularities)

            objective_score = (
                self.theta1 * norm_rating
                + self.theta2 * norm_distance
                + self.theta3 * norm_popularity
            )

            obj_features = dict(poi.objective_features or {})
            obj_features["objective_score"] = max(0.0, min(1.0, objective_score))
            poi.objective_features = obj_features
            updated.append(poi)

        return updated

    def _stage2_subjective_weighting(
        self, candidates: list[CandidatePoi], preferences: list[str]
    ) -> dict[str, float]:
        if not preferences:
            return {c.poi_id: 0.5 for c in candidates}

        all_features = [
            "美食", "景点", "酒店", "购物", "娱乐", "文化", "历史",
            "自然", "休闲", "浪漫", "亲子", "商务", "性价比", "高端",
            "川菜", "粤菜", "西餐", "日料", "火锅", "烧烤", "小吃",
            "博物馆", "公园", "寺庙", "古迹", "海滩", "山脉", "湖泊"
        ]

        vectorizer = TfidfVectorizer(vocabulary=all_features, lowercase=False)

        try:
            user_vector = vectorizer.fit_transform([" ".join(preferences)]).toarray()[0]
        except Exception:
            return {c.poi_id: 0.5 for c in candidates}

        scores = {}
        for poi in candidates:
            if not poi.tags:
                scores[poi.poi_id] = 0.5
                continue

            try:
                poi_vector = vectorizer.fit_transform([" ".join(poi.tags)]).toarray()[0]
                norm1, norm2 = np.linalg.norm(user_vector), np.linalg.norm(poi_vector)
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(user_vector, poi_vector) / (norm1 * norm2))
                scores[poi.poi_id] = max(0, similarity)
            except Exception:
                scores[poi.poi_id] = 0.5

        return scores

    def _stage3_sequence_prediction(
        self, candidates: list[CandidatePoi], user_history: list[dict] | None
    ) -> dict[str, float] | None:
        if not user_history or len(user_history) < 3:
            return None

        history_poi_ids = [h.get("poi_id") for h in user_history if h.get("poi_id")]
        history_tags = []
        for h in user_history:
            if "tags" in h:
                history_tags.extend(h["tags"] if isinstance(h["tags"], list) else [h["tags"]])

        scores = {}
        for poi in candidates:
            score = 0.5

            if poi.tags:
                overlap = len(set(poi.tags) & set(history_tags))
                if overlap > 0:
                    score += 0.2 * overlap

            if poi.poi_id in history_poi_ids[-3:]:
                score *= 0.3

            scores[poi.poi_id] = min(1.0, max(0.0, score))

        return scores

    def _stage4_fuse_scores(
        self,
        candidates: list[CandidatePoi],
        subjective_scores: dict[str, float],
        sequence_scores: dict[str, float] | None,
    ) -> dict[str, float]:
        final_scores = {}
        lambda_seq = self.lambda_seq if sequence_scores else 0.0

        for poi in candidates:
            s_obj = poi.objective_features.get("objective_score", 0.5) if poi.objective_features else 0.5
            s_sub = subjective_scores.get(poi.poi_id, 0.5)
            s_seq = sequence_scores.get(poi.poi_id, 0.5) if sequence_scores else 0.5

            final_score = (
                self.alpha * s_obj
                + (1 - self.alpha) * s_sub
                + lambda_seq * s_seq
            )
            final_scores[poi.poi_id] = min(1.0, max(0.0, final_score))

        return final_scores

    def _rank_group(
        self, candidates: list[CandidatePoi], scores: dict[str, float], poi_type: str
    ) -> list[RankedCandidate]:
        ranked = []
        for candidate in candidates:
            score = scores.get(candidate.poi_id, 0.0)
            ranked.append(
                RankedCandidate(
                    poi_id=candidate.poi_id,
                    poi_type=candidate.poi_type,
                    score=round(score, 4),
                    candidate=candidate,
                    score_breakdown={
                        "objective": round(
                            candidate.objective_features.get("objective_score", 0), 4
                        ) if candidate.objective_features else 0,
                        "subjective": round(scores.get(candidate.poi_id, 0.5) * 0.5, 4),
                    },
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.score,
                item.candidate.center_distance_m is None,
                item.candidate.center_distance_m if item.candidate.center_distance_m else 10**9,
                -(item.candidate.rating or 0),
                item.candidate.name,
                item.poi_id,
            )
        )
        for index, item in enumerate(ranked, start=1):
            item.rank = index

        return ranked[: self.top_k]

    def _extract_preferences(self, algorithm_input: AlgorithmInput) -> list[str]:
        if not algorithm_input:
            return []

        preference = algorithm_input.subjective_preference
        terms = []
        terms.extend(preference.preference_terms)
        terms.extend(preference.travel_styles)
        terms.extend(preference.spot_keywords)
        terms.extend(preference.food_keywords)
        terms.extend(preference.hotel_keywords)

        return list(set(filter(None, terms)))

    def _extract_history(self, algorithm_input: AlgorithmInput) -> list[dict] | None:
        if not algorithm_input:
            return None

        sequence_input = algorithm_input.sequence_model_input
        if not sequence_input or not sequence_input.has_history:
            return None

        history = []
        for poi_id in sequence_input.historical_poi_ids:
            history.append({"poi_id": poi_id})
        return history


def build_ranking_service() -> POIRecommendationService:
    return POIRecommendationService()
