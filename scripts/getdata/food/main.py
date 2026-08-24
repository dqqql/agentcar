from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
import re

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
GETDATA_DIR = SCRIPT_DIR.parent
if str(GETDATA_DIR) not in sys.path:
    sys.path.insert(0, str(GETDATA_DIR))

from output_utils import build_output_bundle, write_detail_json, write_summary_csv

# 高德地图 API 配置
AMAP_KEY = "772d9db2668f6bfb9c3238702c9b9b9e"
AROUND_URL = "https://restapi.amap.com/v5/place/around"
GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
CATEGORIES = "050000"  # 餐饮服务分类代码
SHOW_FIELDS = "business,navi,photos,indoor"

DEFAULT_LOCATION = "南开大学"
DEFAULT_RADIUS = 3000
DEFAULT_MAX_RESULTS = 100
API_PAGE_SIZE = 25
SUMMARY_COLUMNS = [
    ("dataset_type", "数据类型"),
    ("poi_id", "POI ID"),
    ("name", "名称"),
    ("category", "类型"),
    ("category_code", "类型编码"),
    ("address", "地址"),
    ("province", "省"),
    ("city", "市"),
    ("district", "区"),
    ("longitude", "经度"),
    ("latitude", "纬度"),
    ("distance_m", "距离(米)"),
    ("price_avg_cny", "人均消费"),
    ("rating", "评分"),
    ("phone", "电话"),
    ("business_area", "商圈"),
    ("open_today", "今日营业时间"),
    ("open_week", "营业时间"),
    ("tags", "标签"),
    ("photo_count", "图片数量"),
    ("first_photo_url", "首图"),
    ("source_provider", "数据来源"),
]


def is_coordinate(value):
    return bool(re.fullmatch(r"\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*", value))


def prompt_with_default(prompt_text, default_value):
    value = input(f"{prompt_text}（直接回车使用默认值：{default_value}）: ").strip()
    return value or str(default_value)


def geocode_address(address):
    params = {
        "key": AMAP_KEY,
        "address": address,
    }

    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=10)
        result = response.json()
    except Exception as exc:
        raise RuntimeError(f"地点解析失败: {exc}") from exc

    if result.get("status") != "1":
        raise RuntimeError(f"地点解析失败: {result.get('info', '未知错误')}")

    geocodes = result.get("geocodes", [])
    if not geocodes:
        raise RuntimeError(f"未找到地点: {address}")

    return geocodes[0]["location"], geocodes[0]


def normalize_location(location_input):
    if is_coordinate(location_input):
        return location_input.strip(), None
    return geocode_address(location_input)


def split_location(location_text):
    parts = [part.strip() for part in str(location_text or "").split(",")]
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def get_pois_around(location, radius=DEFAULT_RADIUS, max_results=DEFAULT_MAX_RESULTS):
    pois = []
    page_num = 1
    max_results = max(1, int(max_results))

    while len(pois) < max_results:
        page_size = min(API_PAGE_SIZE, max_results - len(pois))
        params = {
            "key": AMAP_KEY,
            "location": location,
            "radius": radius,
            "types": CATEGORIES,
            "show_fields": SHOW_FIELDS,
            "page_size": page_size,
            "page_num": page_num,
        }

        try:
            response = requests.get(AROUND_URL, params=params, timeout=10)
            result = response.json()

            if result.get("status") != "1":
                print(f"请求失败: {result.get('info', '未知错误')}")
                break

            current_pois = result.get("pois", [])
            if not current_pois:
                break

            pois.extend(current_pois)

            if len(current_pois) < page_size:
                break

            page_num += 1
            time.sleep(0.2)
        except Exception as exc:
            print(f"请求发生异常: {exc}")
            break

    return pois[:max_results]


def join_photo_urls(photos):
    return " | ".join(photo.get("url", "") for photo in photos if photo.get("url"))


def first_photo_url(photos):
    if not photos:
        return ""
    return str(photos[0].get("url", ""))


def build_food_record(poi):
    business = poi.get("business") or {}
    navi = poi.get("navi") or {}
    indoor = poi.get("indoor") or {}
    photos = poi.get("photos") or []
    longitude, latitude = split_location(poi.get("location", ""))
    tags = [
        tag
        for tag in (
            business.get("tag"),
            business.get("keytag"),
            business.get("rectag"),
        )
        if tag
    ]

    return {
        "dataset_type": "food",
        "poi_id": str(poi.get("id", "")),
        "name": str(poi.get("name", "")),
        "category": str(poi.get("type", "")),
        "category_code": str(poi.get("typecode", "")),
        "address": str(poi.get("address", "")),
        "province": str(poi.get("pname", "")),
        "city": str(poi.get("cityname", "")),
        "district": str(poi.get("adname", "")),
        "province_code": str(poi.get("pcode", "")),
        "city_code": str(poi.get("citycode", "")),
        "district_code": str(poi.get("adcode", "")),
        "longitude": longitude,
        "latitude": latitude,
        "location_raw": str(poi.get("location", "")),
        "distance_m": str(poi.get("distance", "")),
        "price_avg_cny": str(business.get("cost", "")),
        "rating": str(business.get("rating", "")),
        "phone": str(business.get("tel", "")),
        "open_today": str(business.get("opentime_today", "")),
        "open_week": str(business.get("opentime_week", "")),
        "business_area": str(business.get("business_area", "")),
        "alias": str(business.get("alias", "")),
        "tags": " | ".join(str(tag) for tag in tags),
        "entrance_location": str(navi.get("entr_location", "")),
        "gridcode": str(navi.get("gridcode", "")),
        "indoor_map": str(indoor.get("indoor_map", "")),
        "photo_count": len(photos),
        "first_photo_url": first_photo_url(photos),
        "photo_urls": join_photo_urls(photos),
        "source_provider": "Amap Place Around API",
        "source_raw": poi,
    }


def save_outputs(records, query):
    bundle_dir, bundle_name = build_output_bundle(
        SCRIPT_DIR,
        "food",
        query.get("location_label") or query.get("location_input"),
    )
    detail_payload = {
        "dataset_type": "food",
        "bundle_name": bundle_name,
        "generated_at": datetime.now().isoformat(),
        "query": query,
        "record_count": len(records),
        "records": records,
    }
    summary_path = write_summary_csv(records, SUMMARY_COLUMNS, bundle_dir)
    detail_path = write_detail_json(detail_payload, bundle_dir)
    return summary_path, detail_path


def prompt_user_inputs():
    location = prompt_with_default(
        "请输入要查询的地点名称、地址或经纬度",
        DEFAULT_LOCATION,
    )

    radius_text = prompt_with_default("请输入搜索半径（米）", DEFAULT_RADIUS)
    max_results_text = prompt_with_default("请输入最多返回多少条", DEFAULT_MAX_RESULTS)

    try:
        radius = int(radius_text)
    except ValueError:
        print(f"半径输入无效，已改用默认值 {DEFAULT_RADIUS}")
        radius = DEFAULT_RADIUS

    try:
        max_results = int(max_results_text)
    except ValueError:
        print(f"条数输入无效，已改用默认值 {DEFAULT_MAX_RESULTS}")
        max_results = DEFAULT_MAX_RESULTS

    return {
        "location": location,
        "radius": radius,
        "max_results": max_results,
    }


def main():
    user_inputs = prompt_user_inputs()

    try:
        center_location, geocode_info = normalize_location(user_inputs["location"])
    except RuntimeError as exc:
        print(exc)
        return

    if geocode_info:
        location_label = geocode_info.get("formatted_address", user_inputs["location"])
        print(f"已解析地点: {location_label} -> {center_location}")
    else:
        location_label = user_inputs["location"]
        print(f"使用坐标查询: {center_location}")

    print(
        f"开始获取 {user_inputs['location']} 附近 {user_inputs['radius']} 米内的餐饮场所数据，"
        f"最多返回 {user_inputs['max_results']} 条..."
    )
    restaurants = get_pois_around(
        center_location,
        radius=user_inputs["radius"],
        max_results=user_inputs["max_results"],
    )

    if not restaurants:
        print("未获取到任何餐饮场所数据")
        return

    records = [build_food_record(poi) for poi in restaurants]
    summary_path, detail_path = save_outputs(
        records,
        {
            "location_input": user_inputs["location"],
            "location_label": location_label,
            "center_location": center_location,
            "radius_m": user_inputs["radius"],
            "max_results": user_inputs["max_results"],
        },
    )
    print(
        f"共获取到 {len(records)} 个餐饮场所。"
        f"汇总文件已保存到 {summary_path}，详细文件已保存到 {detail_path}。"
    )


if __name__ == "__main__":
    main()
