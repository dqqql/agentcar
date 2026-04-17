from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.models.common import ApiResponse
from backend.app.services.pipeline.service import build_pipeline_service

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
pipeline_service = build_pipeline_service()


def _resolve_destination(request_destination: str | None, extract_data: dict) -> str:
    destination = (
        (request_destination or "").strip()
        or (extract_data.get("destination") or "").strip()
        or (
            extract_data.get("algorithm_input", {})
            .get("subjective_preference", {})
            .get("destination", "")
            .strip()
        )
    )
    if not destination:
        raise ValueError("未识别到目的地，请在输入中明确城市或地点。")
    return destination

class GatherRequest(BaseModel):
    extract_result_path: str
    destination: str | None = None

@router.post("/gather-candidates", response_model=ApiResponse)
async def gather_candidates(request: GatherRequest) -> ApiResponse:
    try:
        extract_path = Path(request.extract_result_path)
        if not extract_path.exists():
            raise ValueError(f"Extract result path not found: {request.extract_result_path}")
            
        with open(extract_path, "r", encoding="utf-8") as f:
            extract_data = json.load(f)

        destination = _resolve_destination(request.destination, extract_data)
        spot_keywords = extract_data.get("spot_keywords", [])
        spot_keyword = spot_keywords[0] if spot_keywords else "景点"

        result = pipeline_service.gather_and_adapt(str(extract_path), destination, spot_keyword)
        return ApiResponse(data=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
