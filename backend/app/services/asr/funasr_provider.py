from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.models.asr import ASRResult, ASRSegment
from backend.app.services.asr.base import ASRProvider


class FunASRProvider(ASRProvider):
    name = "funasr"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "FunASR 未安装，请先安装 requirements.txt 中的依赖后再启动 ASR 服务。"
                ) from exc

            self._model = AutoModel(
                model=self._settings.funasr_model,
                vad_model=self._settings.funasr_vad_model,
                punc_model=self._settings.funasr_punc_model,
                device=self._settings.funasr_device,
            )
            return self._model

    def transcribe(self, audio_path: Path, filename: str) -> ASRResult:
        model = self._get_model()

        try:
            result = model.generate(input=[str(audio_path)], cache={}, batch_size_s=0)
        except Exception as exc:
            raise RuntimeError(f"FunASR 转写失败: {exc}") from exc

        first_result = result[0] if isinstance(result, list) and result else {}
        text = str(first_result.get("text", "")).strip()
        sentence_info = first_result.get("sentence_info") or []
        segments = self._extract_segments(sentence_info, text)

        metadata = {
            "model": self._settings.funasr_model,
            "vad_model": self._settings.funasr_vad_model,
            "punc_model": self._settings.funasr_punc_model,
            "device": self._settings.funasr_device,
            "raw_keys": sorted(first_result.keys()),
        }

        return ASRResult(
            provider=self.name,
            filename=filename,
            text=text,
            language="zh",
            segments=segments,
            metadata=metadata,
        )

    def _extract_segments(self, sentence_info: list[dict[str, Any]], fallback_text: str) -> list[ASRSegment]:
        segments: list[ASRSegment] = []

        for sentence in sentence_info:
            if not isinstance(sentence, dict):
                continue

            segment_text = str(sentence.get("text", "")).strip()
            if not segment_text:
                continue

            start_ms = self._to_milliseconds(
                sentence.get("start", sentence.get("start_time"))
            )
            end_ms = self._to_milliseconds(
                sentence.get("end", sentence.get("end_time"))
            )
            segments.append(
                ASRSegment(
                    text=segment_text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )

        if segments:
            return segments

        if fallback_text:
            return [ASRSegment(text=fallback_text)]

        return []

    @staticmethod
    def _to_milliseconds(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if number > 1000:
            return int(number)
        return int(number * 1000)
