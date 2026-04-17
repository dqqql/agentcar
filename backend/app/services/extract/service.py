from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from backend.app.core.config import Settings, get_settings
from backend.app.models.extract import (
    AlgorithmInput,
    ExtractRequest,
    ExtractResult,
    FusionConfig,
    ObjectiveWeightConfig,
    SearchContext,
    SequenceModelInput,
    SubjectivePreferenceInput,
)
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
        result.algorithm_input = self._build_algorithm_input(result)
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"{timestamp}_{safe_hint}.json"
        temp_path = output_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(output_path)
        return output_path

    def _build_algorithm_input(self, result: ExtractResult) -> AlgorithmInput:
        return AlgorithmInput(
            search_context=SearchContext(
                destination_text=result.destination,
                search_radius_m=self._infer_search_radius(result),
                candidate_types=["spot", "food", "hotel"],
                date_texts=result.dates,
                people_count=result.people_count,
            ),
            objective_weights=ObjectiveWeightConfig(),
            subjective_preference=SubjectivePreferenceInput(
                destination=result.destination,
                budget_text=result.budget_text,
                budget_min_cny=result.budget_min_cny,
                budget_max_cny=result.budget_max_cny,
                people_count=result.people_count,
                spot_keywords=result.spot_keywords,
                food_keywords=result.food_keywords,
                hotel_keywords=result.hotel_keywords,
                travel_styles=result.travel_styles,
                preference_terms=result.detected_keywords,
            ),
            fusion_config=FusionConfig(),
            sequence_model_input=SequenceModelInput(
                has_history=False,
                historical_poi_ids=[],
                time_context=result.dates,
            ),
        )

    @staticmethod
    def _infer_search_radius(result: ExtractResult) -> int:
        if "自驾" in result.travel_styles:
            return 12000
        if result.destination and any(keyword in result.destination for keyword in ("大学", "机场")):
            return 3000
        return 5000


def build_extract_service(settings: Settings | None = None) -> ExtractService:
    settings = settings or get_settings()
    return ExtractService(extractor=RuleExtractor(), settings=settings)
