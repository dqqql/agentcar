from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models.adapter import AdapterRequest, CandidatePoolResult
from backend.app.models.common import ApiResponse
from backend.app.services.adapter import build_candidate_adapter_service

router = APIRouter(prefix="/api/adapter", tags=["adapter"])
adapter_service = build_candidate_adapter_service()


@router.post("/candidate-pool", response_model=ApiResponse)
async def build_candidate_pool(request: AdapterRequest) -> ApiResponse:
    try:
        result: CandidatePoolResult = adapter_service.build_candidate_pool(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(data=result.model_dump())
