from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from backend.app.core.config import Settings, get_settings
from backend.app.models.extract import ExtractRequest, ExtractResult
from backend.app.services.extract.rule_extractor import RuleExtractor


class ExtractService:
    def __init__(self, extractor: RuleExtractor, settings: Settings) -> None:
        self._extractor = extractor
        self._settings = settings

    def extract(self, request: ExtractRequest) -> ExtractResult:
        text, source_type, source_file_path, output_name_hint = self._resolve_input(request)
        result = self._extractor.extract(
            text,
            source_type=source_type,
            source_file_path=source_file_path,
        )
        result.result_file_path = str(self._save_result(result, output_name_hint))
        return result

    def _resolve_input(self, request: ExtractRequest) -> tuple[str, str, str | None, str]:
        if request.text and request.text.strip():
            return request.text.strip(), "text", None, "text_input"

        if request.text_file_path and request.text_file_path.strip():
            file_path = self._resolve_file_path(request.text_file_path.strip())
            if not file_path.exists():
                raise ValueError(f"文本文件不存在：{file_path}")
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError(f"文本文件为空：{file_path}")
            return text, "text_file", str(file_path), file_path.stem

        raise ValueError("请提供 text 或 text_file_path。")

    def _resolve_file_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._settings.project_root / path
        resolved = path.resolve()
        project_root = self._settings.project_root.resolve()
        if project_root not in resolved.parents and resolved != project_root:
            raise ValueError("只允许读取项目目录内的文本文件。")
        return resolved

    def _save_result(self, result: ExtractResult, name_hint: str) -> Path:
        output_dir = (self._settings.project_root / self._settings.extract_output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_hint = re.sub(r'[<>:"/\\|?*\s]+', "_", name_hint).strip("._") or "extract"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{safe_hint}.json"
        output_path.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path


def build_extract_service(settings: Settings | None = None) -> ExtractService:
    settings = settings or get_settings()
    return ExtractService(extractor=RuleExtractor(), settings=settings)
