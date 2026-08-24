from __future__ import annotations

from datetime import date

import pytest
import requests

from backend.app.models.extract import GeoPoint
from backend.app.services.adapter.service import build_candidate_adapter_service
from scripts.getdata.hotel import main as hotel


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_amap_key_is_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_get(url: str, *, params: dict, **_: object) -> FakeResponse:
        calls.append((url, params))
        if url == hotel.AMAP_GEOCODE_URL:
            return FakeResponse(
                {
                    "status": "1",
                    "geocodes": [
                        {"location": "116.4074,39.9042", "formatted_address": "北京市"}
                    ],
                }
            )
        return FakeResponse({"status": "1", "pois": []})

    monkeypatch.setenv(hotel.AMAP_API_KEY_ENV, "test-key")
    monkeypatch.setattr(hotel.requests, "get", fake_get)

    lat, lon, _ = hotel.geocode_location("北京市")
    assert (lat, lon) == (39.9042, 116.4074)
    hotel.fetch_amap_hotels(lat, lon, 4000, 2)
    assert calls[0][1]["key"] == "test-key"
    assert calls[1][1]["key"] == "test-key"


def test_missing_key_does_not_request_unknown_location(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(hotel.AMAP_API_KEY_ENV, raising=False)

    def unexpected_request(*_: object, **__: object) -> None:
        raise AssertionError("network request must not be made without a key")

    monkeypatch.setattr(hotel.requests, "get", unexpected_request)
    with pytest.raises(RuntimeError, match=hotel.AMAP_API_KEY_ENV):
        hotel.geocode_location("未配置城市")


def test_amap_response_normalizes_coordinates_and_contract() -> None:
    normalized = hotel.normalize_amap_hotels(
        [
            {
                "id": "poi-1",
                "name": "测试酒店",
                "location": "116.4100,39.9100",
                "type": "住宿服务;宾馆酒店",
                "typecode": "100100",
                "pname": "北京市",
                "cityname": "北京市",
                "adname": "东城区",
                "address": "测试路1号",
                "business": {"tel": "010-12345678", "tag": "四星级"},
            }
        ],
        center_lat=39.9042,
        center_lon=116.4074,
        max_results=5,
    )
    assert len(normalized) == 1
    assert normalized[0]["latitude"] == 39.91
    assert normalized[0]["longitude"] == 116.41
    assert normalized[0]["source_provider"] == "amap_place_around"


def test_amap_exception_produces_nonempty_adapter_compatible_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hotel.AMAP_API_KEY_ENV, "test-key")

    def timeout(*_: object, **__: object) -> None:
        raise requests.Timeout("mock timeout")

    monkeypatch.setattr(hotel.requests, "get", timeout)
    records, _ = hotel.build_hotel_dataset(
        location_text="116.4074,39.9042",
        radius=4000,
        max_results=3,
        check_in_date=date(2026, 9, 1),
        nights=1,
    )
    assert len(records) == 3
    assert all(record["source"]["provider"] == "local_fallback" for record in records)

    adapter = build_candidate_adapter_service()
    candidates = adapter._adapt_hotel_candidates(
        records,
        GeoPoint(lng=116.4074, lat=39.9042),
        10_000,
    )
    assert len(candidates) == 3
    assert all(candidate.name for candidate in candidates)
