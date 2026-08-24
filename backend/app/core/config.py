from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _ranking_mode() -> str:
    value = os.getenv("RANKING_MODEL_MODE", "off").strip().lower()
    return value if value in {"off", "shadow", "rerank"} else "off"


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not minimum <= value <= maximum:
        return default
    return value


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    app_name: str = "Intelligent Cockpit Travel Agent"
    app_stage: str = "Layer 1 input processing bootstrap"
    asr_provider: str = os.getenv("ASR_PROVIDER", "funasr")
    asr_temp_dir: Path = Path(os.getenv("ASR_TEMP_DIR", "backend/.tmp/asr"))
    asr_text_output_dir: Path = Path(os.getenv("ASR_TEXT_OUTPUT_DIR", "data/asr_text"))
    asr_max_file_size_mb: int = int(os.getenv("ASR_MAX_FILE_SIZE_MB", "25"))
    extract_output_dir: Path = Path(os.getenv("EXTRACT_OUTPUT_DIR", "data/extract_result"))
    adapter_output_dir: Path = Path(os.getenv("ADAPTER_OUTPUT_DIR", "data/candidate_pool"))
    funasr_model: str = os.getenv("FUNASR_MODEL", "paraformer-zh")
    funasr_vad_model: str = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
    funasr_punc_model: str = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
    funasr_device: str = os.getenv("FUNASR_DEVICE", "cpu")
    ranking_model_mode: str = field(default_factory=_ranking_mode)
    ranking_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "RANKING_MODEL_PATH",
                "backend/app/services/ranking/checkpoints/reranker-v1.pt",
            )
        )
    )
    ranking_model_sha256: str = field(
        default_factory=lambda: os.getenv("RANKING_MODEL_SHA256", "").strip().lower()
    )
    ranking_model_top_n: int = field(
        default_factory=lambda: _bounded_int("RANKING_MODEL_TOP_N", 20, 1, 100)
    )
    ranking_model_blend_weight: float = field(
        default_factory=lambda: _bounded_float(
            "RANKING_MODEL_BLEND_WEIGHT", 0.5, 0.0, 1.0
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
