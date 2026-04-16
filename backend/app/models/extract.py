from __future__ import annotations

from pydantic import BaseModel, Field


class GeoPoint(BaseModel):
    lng: float | None = None
    lat: float | None = None


class SearchContext(BaseModel):
    destination_text: str | None = None
    center_location: GeoPoint = Field(default_factory=GeoPoint)
    search_radius_m: int = 5000
    candidate_types: list[str] = Field(default_factory=lambda: ["spot", "food", "hotel"])
    date_texts: list[str] = Field(default_factory=list)
    people_count: int | None = None


class ObjectiveWeightConfig(BaseModel):
    rating_weight: float = 0.45
    distance_weight: float = 0.35
    popularity_weight: float = 0.20


class FusionConfig(BaseModel):
    alpha: float = 0.60


class SubjectivePreferenceInput(BaseModel):
    destination: str | None = None
    budget_text: str | None = None
    budget_min_cny: int | None = None
    budget_max_cny: int | None = None
    people_count: int | None = None
    spot_keywords: list[str] = Field(default_factory=list)
    food_keywords: list[str] = Field(default_factory=list)
    hotel_keywords: list[str] = Field(default_factory=list)
    travel_styles: list[str] = Field(default_factory=list)
    preference_terms: list[str] = Field(default_factory=list)


class SequenceModelInput(BaseModel):
    has_history: bool = False
    historical_poi_ids: list[str] = Field(default_factory=list)
    time_context: list[str] = Field(default_factory=list)


class AlgorithmInput(BaseModel):
    search_context: SearchContext
    objective_weights: ObjectiveWeightConfig = Field(default_factory=ObjectiveWeightConfig)
    subjective_preference: SubjectivePreferenceInput
    fusion_config: FusionConfig = Field(default_factory=FusionConfig)
    sequence_model_input: SequenceModelInput = Field(default_factory=SequenceModelInput)


class ExtractRequest(BaseModel):
    text: str | None = None
    text_file_path: str | None = None


class ExtractResult(BaseModel):
    source_type: str
    source_file_path: str | None = None
    result_file_path: str | None = None
    raw_text: str
    destination: str | None = None
    dates: list[str] = Field(default_factory=list)
    budget_text: str | None = None
    budget_min_cny: int | None = None
    budget_max_cny: int | None = None
    people_count: int | None = None
    spot_keywords: list[str] = Field(default_factory=list)
    food_keywords: list[str] = Field(default_factory=list)
    hotel_keywords: list[str] = Field(default_factory=list)
    travel_styles: list[str] = Field(default_factory=list)
    detected_keywords: list[str] = Field(default_factory=list)
    algorithm_input: AlgorithmInput | None = None
