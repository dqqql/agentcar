from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.models.asr import ASRResult
from backend.app.models.common import ApiResponse
from backend.app.services.asr.service import build_asr_service

router = APIRouter(prefix="/api/asr", tags=["asr"])
asr_service = build_asr_service()


@router.post("/transcribe", response_model=ApiResponse)
async def transcribe_audio(file: UploadFile = File(...)) -> ApiResponse:
    try:
        result: ASRResult = await asr_service.transcribe_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ApiResponse(data=result.model_dump())
