from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.models.asr import ASRResult


class ASRProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, filename: str) -> ASRResult:
        raise NotImplementedError
