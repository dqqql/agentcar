from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models.common import ApiResponse
from backend.app.models.extract import ExtractRequest, ExtractResult
from backend.app.services.extract import build_extract_service

router = APIRouter(prefix="/api/extract", tags=["extract"])
extract_service = build_extract_service()


@router.post("/keywords", response_model=ApiResponse)
async def extract_keywords(request: ExtractRequest) -> ApiResponse:
    try:
        result: ExtractResult = extract_service.extract(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(data=result.model_dump())
