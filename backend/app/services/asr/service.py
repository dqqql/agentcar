from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import uuid

from fastapi import UploadFile

from backend.app.core.config import Settings, get_settings
from backend.app.models.asr import ASRResult
from backend.app.services.asr.base import ASRProvider
from backend.app.services.asr.funasr_provider import FunASRProvider

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}


class ASRService:
    def __init__(self, provider: ASRProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    async def transcribe_upload(self, upload_file: UploadFile) -> ASRResult:
        filename = upload_file.filename or "audio"
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"暂不支持该音频格式：{suffix or 'unknown'}。"
                f"当前支持：{', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

        self._settings.asr_temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._settings.asr_temp_dir / f"{uuid.uuid4().hex}{suffix}"

        try:
            file_size = await self._save_upload_file(upload_file, temp_path)
            self._validate_size(file_size)
            result = self._provider.transcribe(temp_path, filename)
            result.text_file_path = str(self._save_text_result(filename, result.text))
            return result
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def _save_upload_file(self, upload_file: UploadFile, destination: Path) -> int:
        await upload_file.seek(0)
        total_size = 0
        with destination.open("wb") as file_obj:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                file_obj.write(chunk)
        await upload_file.close()
        return total_size

    def _validate_size(self, file_size: int) -> None:
        max_bytes = self._settings.asr_max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(
                f"音频文件过大：{file_size / 1024 / 1024:.2f} MB，"
                f"当前限制为 {self._settings.asr_max_file_size_mb} MB。"
            )

    def _save_text_result(self, filename: str, text: str) -> Path:
        output_dir = self._settings.asr_text_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        safe_stem = re.sub(r'[<>:"/\\|?*\s]+', "_", stem).strip("._") or "audio"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{safe_stem}.txt"
        output_path.write_text(text.strip(), encoding="utf-8")
        return output_path.resolve()


def build_asr_service(settings: Settings | None = None) -> ASRService:
    settings = settings or get_settings()
    provider_name = settings.asr_provider.lower().strip()

    if provider_name != "funasr":
        raise RuntimeError(f"当前未实现的 ASR Provider: {provider_name}")

    return ASRService(provider=FunASRProvider(settings), settings=settings)
