from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.models.adapter import CandidatePoi, CandidatePoolResult


class RankedCandidate(BaseModel):
    poi_id: str
    poi_type: str
    score: float = 0.0
    rank: int = 0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    candidate: CandidatePoi


class RankingResult(BaseModel):
    source_candidate_pool_path: str | None = None
    ranked_spot_candidates: list[RankedCandidate] = Field(default_factory=list)
    ranked_food_candidates: list[RankedCandidate] = Field(default_factory=list)
    ranked_hotel_candidates: list[RankedCandidate] = Field(default_factory=list)
    debug_meta: dict[str, str | int | float | None] = Field(default_factory=dict)


class RankingRequest(BaseModel):
    candidate_pool: CandidatePoolResult
