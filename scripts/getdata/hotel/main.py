from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
GETDATA_DIR = SCRIPT_DIR.parent
if str(GETDATA_DIR) not in sys.path:
    sys.path.insert(0, str(GETDATA_DIR))

from output_utils import build_output_bundle, write_detail_json, write_summary_csv

DEFAULT_LOCATION = "Wangfujing, Beijing"
DEFAULT_RADIUS = 4000
DEFAULT_MAX_RESULTS = 20
DEFAULT_NIGHTS = 1
DEFAULT_TOURISM_TYPES = ("hotel", "hostel", "guest_house", "motel", "resort")
USER_AGENT = "agentcar-hotel-prototype/1.0"
DEFAULT_OUTPUT_LABEL = "hotel_candidates"

TOURISM_TYPE_LABELS = {
    "hotel": "酒店",
    "hostel": "青年旅舍",
    "guest_house": "民宿/宾馆",
    "motel": "汽车旅馆",
    "resort": "度假酒店",
}
AVAILABILITY_STATUS_LABELS = {
    "available": "可预订",
    "low_stock": "库存紧张",
    "sold_out": "已售罄",
}
AMENITY_LABELS = {
    "front_desk_24h": "24小时前台",
    "hot_water": "热水",
    "wifi": "Wi-Fi",
    "parking": "停车场",
    "accessible": "无障碍设施",
    "air_conditioning": "空调",
    "breakfast_option": "可提供早餐",
    "gym": "健身房",
    "laundry": "洗衣服务",
    "business_area": "商务区",
    "pool": "泳池",
    "family_friendly": "亲子友好",
    "metro_access": "地铁便利",
}
SUMMARY_COLUMNS = [
    ("dataset_type", "数据类型"),
    ("hotel_id", "酒店ID"),
    ("name", "酒店名称"),
    ("tourism_type", "酒店类型"),
    ("location_label", "目的地"),
    ("latitude", "纬度"),
    ("longitude", "经度"),
    ("distance_m", "距中心点距离(米)"),
    ("address", "地址"),
    ("phone", "电话"),
    ("website", "网站"),
    ("stars", "星级"),
    ("rating", "评分"),
    ("review_count", "评论数"),
    ("amenities_text", "设施列表"),
    ("check_in_date", "入住日期"),
    ("check_out_date", "离店日期"),
    ("nights", "入住晚数"),
    ("total_rooms", "总房量"),
    ("remaining_rooms", "剩余房量"),
    ("availability_status", "库存状态"),
    ("price_min_cny", "最低价(元)"),
    ("price_max_cny", "最高价(元)"),
    ("room_types_count", "房型数量"),
    ("room_types_summary", "房型概览"),
    ("room_types_json", "房型明细JSON"),
    ("source_provider", "数据来源"),
    ("source_osm_id", "来源OSM_ID"),
    ("source_fetched_at", "抓取时间"),
    ("source_json", "来源信息JSON"),
    ("full_hotel_json", "完整详情JSON"),
]


def prompt_with_default(prompt_text: str, default_value: Any) -> str:
    value = input(f"{prompt_text}（直接回车则使用默认值：{default_value}）：").strip()
    return value or str(default_value)


def is_coordinate(value: str) -> bool:
    return bool(re.fullmatch(r"\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*", value))


def parse_date_input(raw_value: str) -> date:
    return datetime.strptime(raw_value, "%Y-%m-%d").date()


def parse_stars(raw_value: str | None) -> float | None:
    if not raw_value:
        return None
    match = re.search(r"\d+(\.\d+)?", str(raw_value))
    if not match:
        return None
    return float(match.group(0))


def geocode_location(location_text: str) -> tuple[float, float, dict[str, Any]]:
    params = urlencode(
        {
            "q": location_text,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }
    )
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request, timeout=20) as response:
            results = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to geocode location '{location_text}': {exc}") from exc

    if not results:
        raise RuntimeError(f"Could not geocode location '{location_text}'")

    best = results[0]
    return float(best["lat"]), float(best["lon"]), best


def normalize_location(location_text: str) -> tuple[float, float, dict[str, Any] | None]:
    if is_coordinate(location_text):
        lon_text, lat_text = [part.strip() for part in location_text.split(",")]
        return float(lat_text), float(lon_text), None
    return geocode_location(location_text)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    radius = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return int(2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def build_overpass_query(lat: float, lon: float, radius: int) -> str:
    tourism_pattern = "|".join(DEFAULT_TOURISM_TYPES)
    return f"""
[out:json][timeout:25];
(
  node["tourism"~"^({tourism_pattern})$"](around:{radius},{lat},{lon});
  way["tourism"~"^({tourism_pattern})$"](around:{radius},{lat},{lon});
  relation["tourism"~"^({tourism_pattern})$"](around:{radius},{lat},{lon});
);
out center tags;
""".strip()


def fetch_osm_hotels(lat: float, lon: float, radius: int) -> list[dict[str, Any]]:
    query = build_overpass_query(lat, lon, radius)
    request = Request(
        "https://overpass-api.de/api/interpreter",
        data=query.encode("utf-8"),
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urlopen(request, timeout=40) as response:
            payload = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to query OpenStreetMap hotel data: {exc}") from exc

    return payload.get("elements", [])


def fallback_hotels(lat: float, lon: float, location_text: str, max_results: int) -> list[dict[str, Any]]:
    seeds = [
        "Central Hotel",
        "Riverside Hotel",
        "Garden Hotel",
        "City View Hotel",
        "Parkside Hotel",
        "Metro Hotel",
        "Harbor Hotel",
        "Skyline Hotel",
    ]
    area_label = location_text.split(",")[0].strip() or "Demo"
    hotels: list[dict[str, Any]] = []

    for index in range(min(max_results, len(seeds))):
        offset = 0.006 * (index + 1)
        hotels.append(
            {
                "type": "fallback",
                "id": 900000 + index,
                "lat": lat + offset / 2,
                "lon": lon - offset / 3,
                "tags": {
                    "name": f"{area_label} {seeds[index]}",
                    "tourism": "hotel",
                    "stars": str(3 + index % 3),
                    "addr:full": area_label,
                },
            }
        )

    return hotels


def extract_point(element: dict[str, Any]) -> tuple[float | None, float | None]:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None, None


def normalize_osm_hotels(
    elements: list[dict[str, Any]],
    *,
    center_lat: float,
    center_lon: float,
    max_results: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()

    for element in elements:
        lat, lon = extract_point(element)
        if lat is None or lon is None:
            continue

        tags = element.get("tags") or {}
        name = (tags.get("name") or tags.get("name:en") or "").strip()
        if not name:
            continue

        tourism_type = (tags.get("tourism") or "hotel").strip()
        distance_m = haversine_meters(center_lat, center_lon, lat, lon)
        unique_key = (name.lower(), tourism_type.lower(), distance_m // 20)
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        normalized.append(
            {
                "osm_id": f"{element.get('type', 'node')}/{element.get('id')}",
                "name": name,
                "tourism_type": tourism_type,
                "latitude": lat,
                "longitude": lon,
                "distance_m": distance_m,
                "stars": parse_stars(tags.get("stars")),
                "address": (
                    tags.get("addr:full")
                    or ", ".join(
                        value
                        for value in (
                            tags.get("addr:street"),
                            tags.get("addr:housenumber"),
                            tags.get("addr:city"),
                        )
                        if value
                    )
                ),
                "phone": tags.get("phone") or tags.get("contact:phone") or "",
                "website": tags.get("website") or tags.get("contact:website") or "",
                "source_tags": tags,
            }
        )

    normalized.sort(key=lambda item: (item["distance_m"], item["name"]))
    return normalized[:max_results]


def build_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def build_amenities(source_tags: dict[str, Any], star_level: float, rng: random.Random) -> list[str]:
    amenities = {
        "front_desk_24h",
        "hot_water",
        "wifi",
    }
    if source_tags.get("parking") in {"yes", "private", "public"}:
        amenities.add("parking")
    if source_tags.get("wheelchair") == "yes":
        amenities.add("accessible")
    if star_level >= 3:
        amenities.update({"air_conditioning", "breakfast_option"})
    if star_level >= 4:
        amenities.update({"gym", "laundry", "business_area"})
    if star_level >= 4.5:
        amenities.update({"pool", "family_friendly"})
    if rng.random() > 0.55:
        amenities.add("metro_access")
    return sorted(amenities)


def room_templates_for_star(star_level: float) -> list[dict[str, Any]]:
    templates = [
        {
            "name": "Standard Queen Room",
            "capacity": 2,
            "bed_type": "1 queen bed",
            "window": "city_view",
            "price_factor": 1.00,
        },
        {
            "name": "Standard Twin Room",
            "capacity": 2,
            "bed_type": "2 single beds",
            "window": "city_view",
            "price_factor": 1.05,
        },
        {
            "name": "Superior King Room",
            "capacity": 2,
            "bed_type": "1 king bed",
            "window": "city_view",
            "price_factor": 1.18,
        },
        {
            "name": "Deluxe Twin Room",
            "capacity": 2,
            "bed_type": "2 double beds",
            "window": "open_view",
            "price_factor": 1.28,
        },
        {
            "name": "Family Room",
            "capacity": 3,
            "bed_type": "1 queen bed + 1 single bed",
            "window": "family_view",
            "price_factor": 1.36,
        },
        {
            "name": "Executive Room",
            "capacity": 2,
            "bed_type": "1 king bed",
            "window": "high_floor",
            "price_factor": 1.52,
        },
        {
            "name": "Junior Suite",
            "capacity": 3,
            "bed_type": "1 king bed",
            "window": "landmark_view",
            "price_factor": 1.86,
        },
    ]

    if star_level < 3:
        return templates[:3]
    if star_level < 4:
        return templates[:5]
    if star_level < 4.5:
        return templates[:6]
    return templates


def nightly_price(base_price: int, current_date: date, rng: random.Random) -> int:
    month_factor = {
        1: 0.92,
        2: 0.96,
        3: 0.98,
        4: 1.02,
        5: 1.10,
        6: 1.06,
        7: 1.08,
        8: 1.12,
        9: 1.04,
        10: 1.16,
        11: 0.97,
        12: 1.08,
    }[current_date.month]
    weekday_factor = 1.10 if current_date.weekday() in (4, 5) else 1.0
    random_factor = 1 + rng.uniform(-0.05, 0.07)
    return int(round(base_price * month_factor * weekday_factor * random_factor))


def build_room_types(
    *,
    hotel_id: str,
    star_level: float,
    total_rooms: int,
    check_in_date: date,
    nights: int,
    base_price: int,
) -> tuple[list[dict[str, Any]], int]:
    templates = room_templates_for_star(star_level)
    rng = random.Random(build_seed(hotel_id, check_in_date.isoformat(), nights, "room_types"))
    template_count = min(len(templates), max(3, min(len(templates), total_rooms // 30 + 2)))
    selected_templates = templates[:template_count]

    weights = [rng.uniform(0.8, 1.4) for _ in selected_templates]
    weight_total = sum(weights)
    allocations = [max(4, int(total_rooms * weight / weight_total)) for weight in weights]

    diff = total_rooms - sum(allocations)
    index = 0
    while diff != 0 and allocations:
        slot = index % len(allocations)
        if diff > 0:
            allocations[slot] += 1
            diff -= 1
        elif allocations[slot] > 4:
            allocations[slot] -= 1
            diff += 1
        index += 1

    room_types: list[dict[str, Any]] = []
    total_remaining_rooms = 0

    for template, room_total in zip(selected_templates, allocations):
        room_rng = random.Random(build_seed(hotel_id, template["name"], check_in_date.isoformat(), nights))
        occupancy_ratio = clamp(
            0.52
            + (0.08 if check_in_date.weekday() in (4, 5) else 0.0)
            + room_rng.uniform(-0.14, 0.22),
            0.18,
            0.98,
        )
        remaining_rooms = max(0, room_total - int(round(room_total * occupancy_ratio)))
        total_remaining_rooms += remaining_rooms

        nightly_quotes: list[dict[str, Any]] = []
        room_base_price = int(round(base_price * template["price_factor"]))
        for offset in range(nights):
            current_date = check_in_date + timedelta(days=offset)
            quote_rng = random.Random(build_seed(hotel_id, template["name"], current_date.isoformat(), "nightly"))
            nightly_quotes.append(
                {
                    "date": current_date.isoformat(),
                    "price_cny": nightly_price(room_base_price, current_date, quote_rng),
                    "remaining_rooms": max(0, remaining_rooms - offset * room_rng.randint(0, 2)),
                }
            )

        room_types.append(
            {
                "room_type_id": hashlib.md5(f"{hotel_id}|{template['name']}".encode("utf-8")).hexdigest()[:12],
                "name": template["name"],
                "capacity": template["capacity"],
                "bed_type": template["bed_type"],
                "window": template["window"],
                "breakfast_included": room_rng.random() > 0.45,
                "free_cancellation": room_rng.random() > 0.35,
                "total_rooms": room_total,
                "remaining_rooms": remaining_rooms,
                "nightly_quotes": nightly_quotes,
                "average_nightly_price_cny": int(
                    round(sum(item["price_cny"] for item in nightly_quotes) / len(nightly_quotes))
                ),
            }
        )

    return room_types, total_remaining_rooms


def enrich_hotels(
    hotels: list[dict[str, Any]],
    *,
    check_in_date: date,
    nights: int,
    location_label: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []

    for index, hotel in enumerate(hotels, start=1):
        star_level = hotel["stars"] or random.Random(build_seed(hotel["osm_id"], "stars")).choice([2.5, 3.0, 3.5, 4.0, 4.5])
        seed = build_seed(hotel["osm_id"], check_in_date.isoformat(), nights)
        rng = random.Random(seed)

        if star_level >= 4.5:
            total_rooms = rng.randint(180, 360)
            base_price = rng.randint(720, 1380)
        elif star_level >= 4.0:
            total_rooms = rng.randint(120, 260)
            base_price = rng.randint(480, 980)
        elif star_level >= 3.0:
            total_rooms = rng.randint(70, 180)
            base_price = rng.randint(280, 580)
        else:
            total_rooms = rng.randint(28, 88)
            base_price = rng.randint(160, 360)

        rating = round(clamp(3.7 + (star_level - 3) * 0.28 + rng.uniform(-0.25, 0.32), 3.5, 4.9), 1)
        review_count = int(clamp(total_rooms * rng.uniform(6, 20), 40, 6800))
        amenities = build_amenities(hotel["source_tags"], star_level, rng)
        room_types, remaining_rooms = build_room_types(
            hotel_id=hotel["osm_id"],
            star_level=star_level,
            total_rooms=total_rooms,
            check_in_date=check_in_date,
            nights=nights,
            base_price=base_price,
        )

        average_prices = [room["average_nightly_price_cny"] for room in room_types]
        price_min = min(average_prices)
        price_max = max(average_prices)
        if remaining_rooms == 0:
            availability_status = "sold_out"
        elif remaining_rooms <= max(4, int(total_rooms * 0.1)):
            availability_status = "low_stock"
        else:
            availability_status = "available"

        enriched.append(
            {
                "hotel_id": f"proto_hotel_{index:03d}",
                "name": hotel["name"],
                "tourism_type": hotel["tourism_type"],
                "location_label": location_label,
                "latitude": round(hotel["latitude"], 6),
                "longitude": round(hotel["longitude"], 6),
                "distance_m": hotel["distance_m"],
                "address": hotel["address"],
                "phone": hotel["phone"],
                "website": hotel["website"],
                "stars": star_level,
                "rating": rating,
                "review_count": review_count,
                "amenities": amenities,
                "check_in_date": check_in_date.isoformat(),
                "check_out_date": (check_in_date + timedelta(days=nights)).isoformat(),
                "nights": nights,
                "total_rooms": total_rooms,
                "remaining_rooms": remaining_rooms,
                "availability_status": availability_status,
                "price_min_cny": price_min,
                "price_max_cny": price_max,
                "room_types": room_types,
                "source": {
                    "provider": "OpenStreetMap Overpass API",
                    "osm_id": hotel["osm_id"],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )

    return enriched


def translate_label(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def format_amenities_for_csv(amenities: list[str]) -> str:
    return "、".join(translate_label(item, AMENITY_LABELS) for item in amenities)


def format_room_types_for_csv(room_types: list[dict[str, Any]]) -> str:
    formatted_parts: list[str] = []
    for room in room_types:
        breakfast_text = "含早" if room.get("breakfast_included") else "不含早"
        cancellation_text = "可免费取消" if room.get("free_cancellation") else "不可免费取消"
        formatted_parts.append(
            f"{room.get('name', '')}(可住{room.get('capacity', '')}人，{room.get('bed_type', '')}，"
            f"均价¥{room.get('average_nightly_price_cny', '')}，余量{room.get('remaining_rooms', '')}/{room.get('total_rooms', '')}，"
            f"{breakfast_text}，{cancellation_text})"
        )
    return "；".join(formatted_parts)


def build_hotel_summary_rows(hotels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hotel in hotels:
        source = hotel.get("source") or {}
        room_types = hotel.get("room_types", [])
        rows.append(
            {
                "dataset_type": "hotel",
                "hotel_id": hotel.get("hotel_id", ""),
                "name": hotel.get("name", ""),
                "tourism_type": translate_label(str(hotel.get("tourism_type", "")), TOURISM_TYPE_LABELS),
                "location_label": hotel.get("location_label", ""),
                "latitude": hotel.get("latitude", ""),
                "longitude": hotel.get("longitude", ""),
                "distance_m": hotel.get("distance_m", ""),
                "address": hotel.get("address", ""),
                "phone": hotel.get("phone", ""),
                "website": hotel.get("website", ""),
                "stars": hotel.get("stars", ""),
                "rating": hotel.get("rating", ""),
                "review_count": hotel.get("review_count", ""),
                "amenities_text": format_amenities_for_csv(hotel.get("amenities", [])),
                "check_in_date": hotel.get("check_in_date", ""),
                "check_out_date": hotel.get("check_out_date", ""),
                "nights": hotel.get("nights", ""),
                "total_rooms": hotel.get("total_rooms", ""),
                "remaining_rooms": hotel.get("remaining_rooms", ""),
                "availability_status": translate_label(
                    str(hotel.get("availability_status", "")), AVAILABILITY_STATUS_LABELS
                ),
                "price_min_cny": hotel.get("price_min_cny", ""),
                "price_max_cny": hotel.get("price_max_cny", ""),
                "room_types_count": len(room_types),
                "room_types_summary": format_room_types_for_csv(room_types),
                "room_types_json": json.dumps(room_types, ensure_ascii=False),
                "source_provider": source.get("provider", ""),
                "source_osm_id": source.get("osm_id", ""),
                "source_fetched_at": source.get("fetched_at", ""),
                "source_json": json.dumps(source, ensure_ascii=False),
                "full_hotel_json": json.dumps(hotel, ensure_ascii=False),
            }
        )
    return rows


def save_outputs(
    hotels: list[dict[str, Any]],
    *,
    output_label: str,
    query: dict[str, Any],
) -> tuple[Path, Path]:
    bundle_dir, bundle_name = build_output_bundle(
        SCRIPT_DIR,
        "hotel",
        output_label,
        query.get("check_in_date", ""),
    )
    summary_rows = build_hotel_summary_rows(hotels)
    detail_payload = {
        "dataset_type": "hotel",
        "bundle_name": bundle_name,
        "generated_at": datetime.now().isoformat(),
        "query": query,
        "record_count": len(hotels),
        "records": hotels,
    }
    summary_path = write_summary_csv(summary_rows, SUMMARY_COLUMNS, bundle_dir)
    detail_path = write_detail_json(detail_payload, bundle_dir)
    return summary_path, detail_path


def prompt_user_inputs() -> dict[str, Any]:
    today = date.today()
    default_check_in = today + timedelta(days=14)
    location = prompt_with_default("请输入目的地、地址或坐标", DEFAULT_LOCATION)
    radius_text = prompt_with_default("请输入搜索半径（米）", DEFAULT_RADIUS)
    max_results_text = prompt_with_default("请输入最多生成多少家酒店", DEFAULT_MAX_RESULTS)
    check_in_text = prompt_with_default("请输入入住日期（YYYY-MM-DD）", default_check_in.isoformat())
    nights_text = prompt_with_default("请输入入住晚数", DEFAULT_NIGHTS)
    output_text = prompt_with_default("请输入输出目录标识", DEFAULT_OUTPUT_LABEL)

    try:
        radius = max(500, int(radius_text))
    except ValueError:
        print(f"搜索半径无效，已回退为默认值 {DEFAULT_RADIUS}")
        radius = DEFAULT_RADIUS

    try:
        max_results = max(1, int(max_results_text))
    except ValueError:
        print(f"酒店数量无效，已回退为默认值 {DEFAULT_MAX_RESULTS}")
        max_results = DEFAULT_MAX_RESULTS

    try:
        check_in_date = parse_date_input(check_in_text)
    except ValueError:
        print(f"日期格式无效，已回退为默认值 {default_check_in.isoformat()}")
        check_in_date = default_check_in

    try:
        nights = max(1, int(nights_text))
    except ValueError:
        print(f"入住晚数无效，已回退为默认值 {DEFAULT_NIGHTS}")
        nights = DEFAULT_NIGHTS

    return {
        "location": location,
        "radius": radius,
        "max_results": max_results,
        "check_in_date": check_in_date,
        "nights": nights,
        "output_label": output_text,
    }


def build_hotel_dataset(
    *,
    location_text: str,
    radius: int,
    max_results: int,
    check_in_date: date,
    nights: int,
) -> tuple[list[dict[str, Any]], str]:
    center_lat, center_lon, geocode_info = normalize_location(location_text)
    location_label = geocode_info.get("display_name", location_text) if geocode_info else location_text
    print(f"已解析目的地：{location_label}")
    print(f"中心坐标：{center_lat:.6f}, {center_lon:.6f}")

    raw_hotels = fetch_osm_hotels(center_lat, center_lon, radius)
    if not raw_hotels:
        print("Overpass 没有返回公开酒店数据，已自动回退为附近模拟酒店数据。")
        raw_hotels = fallback_hotels(center_lat, center_lon, location_text, max_results)

    normalized_hotels = normalize_osm_hotels(
        raw_hotels,
        center_lat=center_lat,
        center_lon=center_lon,
        max_results=max_results,
    )
    if not normalized_hotels:
        normalized_hotels = normalize_osm_hotels(
            fallback_hotels(center_lat, center_lon, location_text, max_results),
            center_lat=center_lat,
            center_lon=center_lon,
            max_results=max_results,
        )

    hotels = enrich_hotels(
        normalized_hotels,
        check_in_date=check_in_date,
        nights=nights,
        location_label=location_label,
    )
    return hotels, location_label


def main() -> None:
    user_inputs = prompt_user_inputs()

    try:
        hotels, location_label = build_hotel_dataset(
            location_text=user_inputs["location"],
            radius=user_inputs["radius"],
            max_results=user_inputs["max_results"],
            check_in_date=user_inputs["check_in_date"],
            nights=user_inputs["nights"],
        )
    except RuntimeError as exc:
        print(exc)
        return

    if not hotels:
        print("未能生成酒店数据集。")
        return

    summary_path, detail_path = save_outputs(
        hotels,
        output_label=user_inputs["output_label"],
        query={
            "location_input": user_inputs["location"],
            "radius_m": user_inputs["radius"],
            "max_results": user_inputs["max_results"],
            "check_in_date": user_inputs["check_in_date"].isoformat(),
            "nights": user_inputs["nights"],
            "location_label": location_label,
        },
    )
    print(
        f"已在 {location_label} 周边生成 {len(hotels)} 家酒店候选数据。"
        f"汇总文件已保存到 {summary_path}，详细文件已保存到 {detail_path}。"
    )


if __name__ == "__main__":
    main()
