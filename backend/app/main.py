from __future__ import annotations

from fastapi import FastAPI

from backend.app.api.asr import router as asr_router
from backend.app.core.config import get_settings
from backend.app.models.common import ApiResponse

settings = get_settings()
app = FastAPI(title=settings.app_name)
app.include_router(asr_router)


def build_response(message: str, data: dict) -> ApiResponse:
    return ApiResponse(message=message, data=data)


@app.get("/", response_model=ApiResponse)
async def root() -> ApiResponse:
    return build_response(
        "success",
        {
            "project": "intelligent-cockpit-travel",
            "stage": settings.app_stage,
            "status": "running",
            "modules": {
                "asr": "ready",
                "keyword_extraction": "planned",
                "ranking_model": "planned",
                "frontend_output": "planned",
            },
        },
    )


@app.get("/health", response_model=ApiResponse)
async def health() -> ApiResponse:
    return build_response(
        "success",
        {
            "service": "backend",
            "status": "ok",
            "asr_provider": settings.asr_provider,
        },
    )
