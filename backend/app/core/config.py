from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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
    funasr_model: str = os.getenv("FUNASR_MODEL", "paraformer-zh")
    funasr_vad_model: str = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
    funasr_punc_model: str = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
    funasr_device: str = os.getenv("FUNASR_DEVICE", "cpu")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
