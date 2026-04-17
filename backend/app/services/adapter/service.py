from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend.app.core.config import Settings, get_settings
from backend.app.models.adapter import (
    AdapterRequest,
    CandidatePoi,
    CandidatePoolMeta,
    CandidatePoolResult,
)
from backend.app.models.extract import AlgorithmInput, ExtractResult, GeoPoint


class CandidateAdapterService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_candidate_pool(self, request: AdapterRequest) -> CandidatePoolResult:
        extract_result, extract_path = self._load_extract_result(request.extract_result_path)
        detail_paths = self._resolve_detail_paths(request)
        detail_payloads = {name: self._load_detail_payload(path) for name, path in detail_paths.items()}
        center_location = self._resolve_center_location(extract_result, detail_payloads)
        algorithm_input = extract_result.algorithm_input or AlgorithmInput.model_validate({})
        algorithm_input.search_context.center_location = center_location

        place_records = detail_payloads["place"].get("records", [])
        food_records = detail_payloads["food"].get("records", [])
        hotel_records = detail_payloads["hotel"].get("records", [])

        spot_candidates = self._adapt_place_candidates(place_records, center_location, algorithm_input.search_context.search_radius_m)
        food_candidates = self._adapt_food_candidates(food_records, center_location, algorithm_input.search_context.search_radius_m)
        hotel_candidates = self._adapt_hotel_candidates(hotel_records, center_location, algorithm_input.search_context.search_radius_m)

        result = CandidatePoolResult(
            algorithm_input=algorithm_input,
            meta=CandidatePoolMeta(
                destination_text=extract_result.destination,
                center_location=center_location,
                search_radius_m=algorithm_input.search_context.search_radius_m,
                source_extract_result_path=str(extract_path) if extract_path else None,
                source_detail_paths={k: str(v) for k, v in detail_paths.items()},
                kept_counts={
                    "spot": len(spot_candidates),
                    "food": len(food_candidates),
                    "hotel": len(hotel_candidates),
                },
            ),
            spot_candidates=spot_candidates,
            food_candidates=food_candidates,
            hotel_candidates=hotel_candidates,
        )
        result.result_file_path = str(self._save_result(result, extract_path))
        return result

    def _load_extract_result(self, raw_path: str | None) -> tuple[ExtractResult, Path]:
        path = self._resolve_project_path(raw_path) if raw_path else self._latest_json_file(self._settings.extract_output_dir)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ExtractResult.model_validate(payload), path

    def _resolve_detail_paths(self, request: AdapterRequest) -> dict[str, Path]:
        return {
            "place": self._resolve_dataset_detail_path("place", request.place_detail_path),
            "food": self._resolve_dataset_detail_path("food", request.food_detail_path),
            "hotel": self._resolve_dataset_detail_path("hotel", request.hotel_detail_path),
        }

    def _resolve_dataset_detail_path(self, dataset: str, raw_path: str | None) -> Path:
        if raw_path:
            return self._resolve_project_path(raw_path)

        base = self._settings.project_root / "scripts" / "getdata" / dataset / "output"
        latest_dir = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if not latest_dir:
            raise ValueError(f"未找到 {dataset} 的 output 目录，请先运行对应脚本。")
        return latest_dir[0] / "detail.json"

    def _resolve_project_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._settings.project_root / path
        resolved = path.resolve()
        project_root = self._settings.project_root.resolve()
        if project_root not in resolved.parents and resolved != project_root:
            raise ValueError("只允许读取项目目录内的文件。")
        if not resolved.exists():
            raise ValueError(f"文件不存在：{resolved}")
        return resolved

    def _latest_json_file(self, relative_dir: Path) -> Path:
        directory = (self._settings.project_root / relative_dir).resolve()
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise ValueError(f"未找到目录中的结果文件：{directory}")
        return files[0]

    @staticmethod
    def _load_detail_payload(detail_path: Path) -> dict[str, Any]:
        return json.loads(detail_path.read_text(encoding="utf-8"))

    def _resolve_center_location(self, extract_result: ExtractResult, detail_payloads: dict[str, dict[str, Any]]) -> GeoPoint:
        if extract_result.algorithm_input and extract_result.algorithm_input.search_context.center_location.lng is not None:
            return extract_result.algorithm_input.search_context.center_location

        destination = extract_result.destination
        if destination:
            geocoded = self._geocode_destination(destination)
            if geocoded.lng is not None and geocoded.lat is not None:
                return geocoded

        fallback_center = self._resolve_center_from_detail_payloads(detail_payloads)
        if fallback_center.lng is not None and fallback_center.lat is not None:
            return fallback_center

        return GeoPoint()

    def _geocode_destination(self, destination: str) -> GeoPoint:
        cleaned_destination = re.sub(r"(附近|周边|旁边|一带)$", "", destination).strip()
        if not cleaned_destination:
            return GeoPoint()

        params = urlencode(
            {
                "q": cleaned_destination,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            }
        )
        url = f"https://nominatim.openstreetmap.org/search?{params}"
        request = Request(url, headers={"User-Agent": "agentcar-candidate-adapter/1.0"})
        try:
            with urlopen(request, timeout=20) as response:
                results = json.load(response)
        except (HTTPError, URLError):
            return GeoPoint()

        if not results:
            return GeoPoint()

        best = results[0]
        return GeoPoint(
            lng=float(best["lon"]),
            lat=float(best["lat"]),
        )

    def _resolve_center_from_detail_payloads(self, detail_payloads: dict[str, dict[str, Any]]) -> GeoPoint:
        food_query = detail_payloads.get("food", {}).get("query", {})
        center_location = food_query.get("center_location")
        if center_location:
            lng, lat = self._parse_location_pair(center_location)
            if lng is not None and lat is not None:
                return GeoPoint(lng=lng, lat=lat)

        hotel_query = detail_payloads.get("hotel", {}).get("query", {})
        location_input = hotel_query.get("location_input")
        if location_input:
            geocoded = self._geocode_destination(str(location_input))
            if geocoded.lng is not None and geocoded.lat is not None:
                return geocoded

        place_query = detail_payloads.get("place", {}).get("query", {})
        city = place_query.get("city")
        if city:
            geocoded = self._geocode_destination(str(city))
            if geocoded.lng is not None and geocoded.lat is not None:
                return geocoded

        return GeoPoint()

    def _adapt_place_candidates(self, records: list[dict[str, Any]], center: GeoPoint, radius_m: int) -> list[CandidatePoi]:
        candidates: list[CandidatePoi] = []
        for record in records:
            longitude = self._to_float(record.get("lon"))
            latitude = self._to_float(record.get("lat"))
            distance = self._calc_distance(center, latitude, longitude)
            if not self._within_radius(distance, radius_m):
                continue

            rating = self._to_float(record.get("rating"))
            popularity = (
                (self._to_float(record.get("photo_count")) or 0) * 5
                + (self._to_float(record.get("groupbuy_num")) or 0) * 2
                + (self._to_float(record.get("discount_num")) or 0)
            )
            price = self._to_float(record.get("cost"))
            tags = self._split_multi_text(record.get("type")) + self._split_multi_text(record.get("tag"))

            candidates.append(
                CandidatePoi(
                    poi_id=str(record.get("id") or ""),
                    poi_type="spot",
                    source_dataset="place",
                    name=str(record.get("name") or ""),
                    address=self._to_text(record.get("address")),
                    longitude=longitude,
                    latitude=latitude,
                    center_distance_m=distance,
                    rating=rating,
                    popularity=popularity if popularity > 0 else None,
                    price_value_cny=price,
                    review_count=None,
                    tags=self._unique(tags),
                    objective_features={
                        "rating": rating,
                        "distance_m": distance,
                        "popularity_proxy": popularity,
                        "price_cny": price,
                        "photo_count": self._to_int(record.get("photo_count")),
                        "groupbuy_num": self._to_int(record.get("groupbuy_num")),
                        "discount_num": self._to_int(record.get("discount_num")),
                    },
                    source_provider=self._to_text(record.get("source_provider")),
                )
            )
        return candidates

    def _adapt_food_candidates(self, records: list[dict[str, Any]], center: GeoPoint, radius_m: int) -> list[CandidatePoi]:
        candidates: list[CandidatePoi] = []
        for record in records:
            longitude = self._to_float(record.get("longitude"))
            latitude = self._to_float(record.get("latitude"))
            distance = self._calc_distance(center, latitude, longitude) or self._to_int(record.get("distance_m"))
            if not self._within_radius(distance, radius_m):
                continue

            rating = self._to_float(record.get("rating"))
            price = self._to_float(record.get("price_avg_cny"))
            photo_count = self._to_int(record.get("photo_count")) or 0
            popularity = (rating or 0) * 20 + photo_count * 5
            tags = self._split_multi_text(record.get("tags")) + self._split_multi_text(record.get("category"))

            candidates.append(
                CandidatePoi(
                    poi_id=str(record.get("poi_id") or ""),
                    poi_type="food",
                    source_dataset="food",
                    name=str(record.get("name") or ""),
                    address=self._to_text(record.get("address")),
                    longitude=longitude,
                    latitude=latitude,
                    center_distance_m=distance,
                    rating=rating,
                    popularity=popularity if popularity > 0 else None,
                    price_value_cny=price,
                    review_count=None,
                    tags=self._unique(tags),
                    objective_features={
                        "rating": rating,
                        "distance_m": distance,
                        "popularity_proxy": popularity,
                        "price_cny": price,
                        "photo_count": photo_count,
                    },
                    source_provider=self._to_text(record.get("source_provider")),
                )
            )
        return candidates

    def _adapt_hotel_candidates(self, records: list[dict[str, Any]], center: GeoPoint, radius_m: int) -> list[CandidatePoi]:
        candidates: list[CandidatePoi] = []
        for record in records:
            longitude = self._to_float(record.get("longitude"))
            latitude = self._to_float(record.get("latitude"))
            distance = self._calc_distance(center, latitude, longitude) or self._to_int(record.get("distance_m"))
            if not self._within_radius(distance, radius_m):
                continue

            rating = self._to_float(record.get("rating"))
            review_count = self._to_int(record.get("review_count"))
            popularity = float(review_count) if review_count is not None else None
            price_min = self._to_float(record.get("price_min_cny"))
            price_max = self._to_float(record.get("price_max_cny"))
            if price_min is not None and price_max is not None:
                price = round((price_min + price_max) / 2, 2)
            else:
                price = price_min or price_max
            tags = self._split_multi_list(record.get("amenities")) + self._split_multi_text(record.get("tourism_type"))

            candidates.append(
                CandidatePoi(
                    poi_id=str(record.get("hotel_id") or ""),
                    poi_type="hotel",
                    source_dataset="hotel",
                    name=str(record.get("name") or ""),
                    address=self._to_text(record.get("address")),
                    longitude=longitude,
                    latitude=latitude,
                    center_distance_m=distance,
                    rating=rating,
                    popularity=popularity,
                    price_value_cny=price,
                    review_count=review_count,
                    tags=self._unique(tags),
                    objective_features={
                        "rating": rating,
                        "distance_m": distance,
                        "popularity_proxy": popularity,
                        "price_cny": price,
                        "review_count": review_count,
                        "stars": self._to_float(record.get("stars")),
                    },
                    source_provider=self._to_text((record.get("source") or {}).get("provider")),
                )
            )
        return candidates

    def _save_result(self, result: CandidatePoolResult, extract_path: Path | None) -> Path:
        output_dir = (self._settings.project_root / self._settings.adapter_output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        name_hint = extract_path.stem if extract_path else "candidate_pool"
        safe_name = re.sub(r'[<>:"/\\|?*\s]+', "_", name_hint).strip("._") or "candidate_pool"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{safe_name}.json"
        output_path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path.resolve()

    @staticmethod
    def _calc_distance(center: GeoPoint, lat: float | None, lng: float | None) -> int | None:
        if center.lat is None or center.lng is None or lat is None or lng is None:
            return None
        radius = 6371000
        phi1 = math.radians(center.lat)
        phi2 = math.radians(lat)
        d_phi = math.radians(lat - center.lat)
        d_lambda = math.radians(lng - center.lng)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        return int(2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    @staticmethod
    def _within_radius(distance: int | None, radius_m: int) -> bool:
        if distance is None:
            return True
        return distance <= radius_m

    @staticmethod
    def _split_multi_text(value: Any) -> list[str]:
        text = CandidateAdapterService._to_text(value)
        if not text:
            return []
        return [item.strip() for item in re.split(r"[;|,，、]+", text) if item.strip()]

    @staticmethod
    def _parse_location_pair(value: Any) -> tuple[float | None, float | None]:
        text = CandidateAdapterService._to_text(value)
        if not text:
            return None, None
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            return None, None
        lng = CandidateAdapterService._to_float(parts[0])
        lat = CandidateAdapterService._to_float(parts[1])
        return lng, lat

    @staticmethod
    def _split_multi_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return CandidateAdapterService._split_multi_text(value)

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    @staticmethod
    def _to_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            return " | ".join(values) if values else None
        text = str(value).strip()
        if text in {"", "[]", "None", "null"}:
            return None
        return text

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, "", [], "[]", "None", "null"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        number = CandidateAdapterService._to_float(value)
        return int(number) if number is not None else None


def build_candidate_adapter_service(settings: Settings | None = None) -> CandidateAdapterService:
    settings = settings or get_settings()
    return CandidateAdapterService(settings=settings)
