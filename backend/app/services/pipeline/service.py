from __future__ import annotations

import json
import locale
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backend.app.models.adapter import AdapterRequest, CandidatePoi, CandidatePoolResult
from backend.app.models.ranking import RankedCandidate, RankingRequest, RankingResult
from backend.app.services.adapter.service import build_candidate_adapter_service
from backend.app.services.ranking import build_ranking_service

PROJECT_ROOT = Path(__file__).resolve().parents[4]
GETDATA_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "getdata"
HOTEL_LOCATION_MAP = {
    "\u5317\u4eac\u5e02": "Beijing, China",
    "\u5929\u6d25\u5e02": "Tianjin, China",
    "\u676d\u5dde\u5e02": "Hangzhou, China",
}


class PipelineService:
    def __init__(self):
        self.adapter_service = build_candidate_adapter_service()
        self.ranking_service = build_ranking_service()

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
            if not encoding:
                continue
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")

    def resolve_new_detail_path(self, output_root: Path, existing_dirs: set[Path]) -> Path:
        current_dirs = [path.resolve() for path in output_root.iterdir() if path.is_dir()]
        new_dirs = [path for path in current_dirs if path not in existing_dirs]
        if not new_dirs:
            raise RuntimeError(f"No new output directory was created under {output_root}.")

        latest_dir = max(new_dirs, key=lambda path: path.stat().st_mtime)
        detail_path = latest_dir / "detail.json"
        if not detail_path.exists():
            raise RuntimeError(f"Missing detail.json in output directory: {detail_path}")
        return detail_path

    def run_interactive_script(
        self,
        dataset: str,
        script_path: Path,
        input_lines: list[str],
        timeout_seconds: int = 300,
    ) -> Path:
        output_root = GETDATA_OUTPUT_ROOT / dataset / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        existing_dirs = {path.resolve() for path in output_root.iterdir() if path.is_dir()}

        stdin_encoding = locale.getpreferredencoding(False) or "gbk"
        raw_input = ("\n".join(input_lines) + "\n").encode(stdin_encoding, errors="replace")

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            input=raw_input,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_text = self.decode_process_output(completed.stdout)
        stderr_text = self.decode_process_output(completed.stderr)

        if completed.returncode != 0:
            raise RuntimeError(
                f"{dataset} script failed with exit code {completed.returncode}\n"
                f"stdout:\n{self.tail_text(stdout_text)}\n"
                f"stderr:\n{self.tail_text(stderr_text)}"
            )

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
        all_candidates: list[CandidatePoi] = (
            result.spot_candidates + result.food_candidates + result.hotel_candidates
        )
        flattened = [
            {
                "poi_type": candidate.poi_type,
                "poi_id": candidate.poi_id,
                "name": candidate.name,
                "source_dataset": candidate.source_dataset,
                "center_distance_m": candidate.center_distance_m,
                "rating": candidate.rating,
                "popularity": candidate.popularity,
                "price_value_cny": candidate.price_value_cny,
                "review_count": candidate.review_count,
                "tags": candidate.tags,
                "objective_features": candidate.objective_features,
                "longitude": candidate.longitude,
                "latitude": candidate.latitude,
                "address": candidate.address,
            }
            for candidate in all_candidates
        ]
        flattened.sort(
            key=lambda item: (
                item["poi_type"],
                item["center_distance_m"] is None,
                item["center_distance_m"] if item["center_distance_m"] is not None else 10**9,
                -(item["rating"] or 0),
            )
        )
        return flattened

    def flatten_ranked_candidates(self, result: RankingResult) -> list[dict[str, Any]]:
        ranked_groups: list[RankedCandidate] = (
            result.ranked_spot_candidates
            + result.ranked_food_candidates
            + result.ranked_hotel_candidates
        )
        flattened: list[dict[str, Any]] = []
        for ranked in ranked_groups:
            candidate = ranked.candidate
            flattened.append(
                {
                    "poi_type": candidate.poi_type,
                    "poi_id": candidate.poi_id,
                    "name": candidate.name,
                    "source_dataset": candidate.source_dataset,
                    "center_distance_m": candidate.center_distance_m,
                    "rating": candidate.rating,
                    "popularity": candidate.popularity,
                    "price_value_cny": candidate.price_value_cny,
                    "review_count": candidate.review_count,
                    "tags": candidate.tags,
                    "objective_features": candidate.objective_features,
                    "longitude": candidate.longitude,
                    "latitude": candidate.latitude,
                    "address": candidate.address,
                    "final_score": ranked.score,
                    "rank": ranked.rank,
                    "score_breakdown": ranked.score_breakdown,
                }
            )
        return flattened

    def gather_and_adapt(
        self,
        extract_result_path: str,
        destination: str,
        spot_keyword: str = "\u666f\u70b9",
    ) -> dict[str, Any]:
        place_detail_path = self.run_interactive_script(
            "place",
            PROJECT_ROOT / "scripts" / "getdata" / "place" / "main.py",
            self.build_place_inputs(destination, spot_keyword),
        )
        food_detail_path = self.run_interactive_script(
            "food",
            PROJECT_ROOT / "scripts" / "getdata" / "food" / "main.py",
            self.build_food_inputs(destination),
        )
        hotel_detail_path = self.run_interactive_script(
            "hotel",
            PROJECT_ROOT / "scripts" / "getdata" / "hotel" / "main.py",
            self.build_hotel_inputs(destination),
        )

        candidate_result = self.adapter_service.build_candidate_pool(
            AdapterRequest(
                extract_result_path=extract_result_path,
                place_detail_path=str(place_detail_path),
                food_detail_path=str(food_detail_path),
                hotel_detail_path=str(hotel_detail_path),
            )
        )
        ranking_result = self.ranking_service.rank_candidates(
            RankingRequest(candidate_pool=candidate_result)
        )
        ranked_candidates = self.flatten_ranked_candidates(ranking_result)

        return {
            "flattened_candidates": ranked_candidates,
            "ranked_candidates": ranked_candidates,
            "unranked_candidates": self.flatten_candidates(candidate_result),
            "candidate_counts": {
                "spot": len(candidate_result.spot_candidates),
                "food": len(candidate_result.food_candidates),
                "hotel": len(candidate_result.hotel_candidates),
            },
            "ranking_meta": ranking_result.debug_meta,
        }


def build_pipeline_service() -> PipelineService:
    return PipelineService()
