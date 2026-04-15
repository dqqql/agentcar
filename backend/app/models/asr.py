from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ASRSegment(BaseModel):
    text: str
    start_ms: int | None = None
    end_ms: int | None = None


class ASRResult(BaseModel):
    provider: str
    filename: str
    text: str
    text_file_path: str | None = None
    language: str = "zh"
    duration_ms: int | None = None
    segments: list[ASRSegment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
