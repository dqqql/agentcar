from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    code: int = Field(default=200)
    message: str = Field(default="success")
    data: dict[str, Any]
