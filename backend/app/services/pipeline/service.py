from __future__ import annotations

import locale
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backend.app.models.adapter import AdapterRequest, CandidatePoolResult, CandidatePoi
from backend.app.services.adapter.service import build_candidate_adapter_service

PROJECT_ROOT = Path(__file__).resolve().parents[4]
GETDATA_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "getdata"
HOTEL_LOCATION_MAP = {
    "北京市": "Beijing, China",
    "天津市": "Tianjin, China",
    "杭州市": "Hangzhou, China",
}

class PipelineService:
    def __init__(self):
        self.adapter_service = build_candidate_adapter_service()

    def tail_text(self, value: str | None, max_length: int = 3000) -> str:
        text = value or ""
        return text[-max_length:]

    def decode_process_output(self, value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value

        encodings = [locale.getpreferredencoding(False), "utf-8", "gbk", "cp936"]
        for encoding in encodings:
            if not encoding: continue
            try: return value.decode(encoding)
            except UnicodeDecodeError: continue
        return value.decode("utf-8", errors="replace")

    def resolve_new_detail_path(self, output_root: Path, existing_dirs: set[Path]) -> Path:
        current_dirs = [path.resolve() for path in output_root.iterdir() if path.is_dir()]
        new_dirs = [path for path in current_dirs if path not in existing_dirs]
        if not new_dirs:
            raise RuntimeError(f"本次运行后未在 {output_root} 下生成新的 output 目录。")

        latest_dir = max(new_dirs, key=lambda path: path.stat().st_mtime)
        detail_path = latest_dir / "detail.json"
        if not detail_path.exists():
            raise RuntimeError(f"输出目录中缺少 detail.json：{detail_path}")
        return detail_path

    def run_interactive_script(self, dataset: str, script_path: Path, input_lines: list[str], timeout_seconds: int = 300) -> Path:
        output_root = GETDATA_OUTPUT_ROOT / dataset / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        existing_dirs = {path.resolve() for path in output_root.iterdir() if path.is_dir()}

        stdin_encoding = locale.getpreferredencoding(False) or "gbk"
        raw_input = ("\n".join(input_lines) + "\n").encode(stdin_encoding, errors="replace")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            input=raw_input, cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=timeout_seconds, check=False,
        )
        stdout_text = self.decode_process_output(completed.stdout)
        stderr_text = self.decode_process_output(completed.stderr)

        if completed.returncode != 0:
            raise RuntimeError(f"{dataset} 脚本执行失败，返回码 {completed.returncode}\nstdout:\n{self.tail_text(stdout_text)}\nstderr:\n{self.tail_text(stderr_text)}")

        return self.resolve_new_detail_path(output_root, existing_dirs)

    def build_food_inputs(self, destination: str) -> list[str]:
        return [destination, "3000", "30"]

    def build_hotel_inputs(self, destination: str) -> list[str]:
        check_in_date = date.today() + timedelta(days=7)
        hotel_location = HOTEL_LOCATION_MAP.get(destination, destination)
        return [hotel_location, "4000", "20", check_in_date.isoformat(), "1", "pipeline_api"]

    def build_place_inputs(self, destination: str, spot_keyword: str) -> list[str]:
        return [destination, spot_keyword]

    def flatten_candidates(self, result: CandidatePoolResult) -> list[dict[str, Any]]:
        all_candidates: list[CandidatePoi] = result.spot_candidates + result.food_candidates + result.hotel_candidates
        flattened = [
            {
                "poi_type": candidate.poi_type, "poi_id": candidate.poi_id, "name": candidate.name,
                "source_dataset": candidate.source_dataset, "center_distance_m": candidate.center_distance_m,
                "rating": candidate.rating, "popularity": candidate.popularity, "price_value_cny": candidate.price_value_cny,
                "review_count": candidate.review_count, "tags": candidate.tags, "objective_features": candidate.objective_features,
                "longitude": candidate.longitude, "latitude": candidate.latitude, "address": candidate.address,
            } for candidate in all_candidates
        ]
        flattened.sort(key=lambda item: (item["poi_type"], item["center_distance_m"] is None, item["center_distance_m"] if item["center_distance_m"] is not None else 10**9, -(item["rating"] or 0)))
        return flattened

    def gather_and_adapt(self, extract_result_path: str, destination: str, spot_keyword: str = "景点") -> dict[str, Any]:
        place_detail_path = self.run_interactive_script("place", PROJECT_ROOT / "scripts" / "getdata" / "place" / "main.py", self.build_place_inputs(destination, spot_keyword))
        food_detail_path = self.run_interactive_script("food", PROJECT_ROOT / "scripts" / "getdata" / "food" / "main.py", self.build_food_inputs(destination))
        hotel_detail_path = self.run_interactive_script("hotel", PROJECT_ROOT / "scripts" / "getdata" / "hotel" / "main.py", self.build_hotel_inputs(destination))

        candidate_result = self.adapter_service.build_candidate_pool(
            AdapterRequest(
                extract_result_path=extract_result_path,
                place_detail_path=str(place_detail_path),
                food_detail_path=str(food_detail_path),
                hotel_detail_path=str(hotel_detail_path),
            )
        )
        return {
            "flattened_candidates": self.flatten_candidates(candidate_result),
            "candidate_counts": {
                "spot": len(candidate_result.spot_candidates),
                "food": len(candidate_result.food_candidates),
                "hotel": len(candidate_result.hotel_candidates),
            }
        }

def build_pipeline_service() -> PipelineService:
    return PipelineService()
