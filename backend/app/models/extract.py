from __future__ import annotations

from pydantic import BaseModel, Field


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
