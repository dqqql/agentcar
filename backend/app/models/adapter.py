from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.models.extract import AlgorithmInput, GeoPoint


class AdapterRequest(BaseModel):
    extract_result_path: str | None = None
    place_detail_path: str | None = None
    food_detail_path: str | None = None
    hotel_detail_path: str | None = None


class CandidatePoi(BaseModel):
    poi_id: str
    poi_type: str
    source_dataset: str
    name: str
    address: str | None = None
    longitude: float | None = None
    latitude: float | None = None
    center_distance_m: int | None = None
    rating: float | None = None
    popularity: float | None = None
    price_value_cny: float | None = None
    review_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    objective_features: dict[str, float | int | str | None] = Field(default_factory=dict)
    source_provider: str | None = None


class CandidatePoolMeta(BaseModel):
    destination_text: str | None = None
    center_location: GeoPoint = Field(default_factory=GeoPoint)
    search_radius_m: int
    source_extract_result_path: str | None = None
    source_detail_paths: dict[str, str] = Field(default_factory=dict)
    kept_counts: dict[str, int] = Field(default_factory=dict)


class CandidatePoolResult(BaseModel):
    result_file_path: str | None = None
    algorithm_input: AlgorithmInput
    meta: CandidatePoolMeta
    spot_candidates: list[CandidatePoi] = Field(default_factory=list)
    food_candidates: list[CandidatePoi] = Field(default_factory=list)
    hotel_candidates: list[CandidatePoi] = Field(default_factory=list)
