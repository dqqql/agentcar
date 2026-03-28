from pathlib import Path
import re
import time

import pandas as pd
import requests

# 高德地图 API 配置
AMAP_KEY = "772d9db2668f6bfb9c3238702c9b9b9e"
AROUND_URL = "https://restapi.amap.com/v5/place/around"
GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
CATEGORIES = "050000"  # 餐饮服务分类代码
SHOW_FIELDS = "business,navi,photos,indoor"

DEFAULT_LOCATION = "南开大学"
DEFAULT_RADIUS = 3000
DEFAULT_MAX_RESULTS = 100  # 测试阶段默认查 100 条，正式环境可直接改这里
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "restaurants.xlsx"
API_PAGE_SIZE = 25  # 高德 v5 page_size 最大值为 25


def is_coordinate(value):
    """判断输入是否为 '经度,纬度' 格式。"""
    return bool(re.fullmatch(r"\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*", value))


def geocode_address(address):
    """将地点名称或地址解析为高德经纬度坐标。"""
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
    """支持直接传坐标，或传地点名/地址。"""
    if is_coordinate(location_input):
        return location_input.strip(), None
    return geocode_address(location_input)


def get_pois_around(location, radius=DEFAULT_RADIUS, max_results=DEFAULT_MAX_RESULTS):
    """获取指定位置周围的餐饮 POI，并按 max_results 截断。"""
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


def resolve_output_path(output_text):
    """把用户输入的文件名解析成最终输出路径，并自动补全 xlsx 后缀。"""
    output_path = Path(output_text).expanduser()

    if not output_path.suffix:
        output_path = output_path.with_suffix(".xlsx")

    if not output_path.is_absolute():
        output_path = DEFAULT_OUTPUT_PATH.parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_to_excel(pois, filename="restaurants.xlsx"):
    """将 POI 数据保存到 Excel 文件。"""
    if not pois:
        print("没有数据可保存")
        return

    data = []
    for poi in pois:
        business = poi.get("business") or {}
        navi = poi.get("navi") or {}
        indoor = poi.get("indoor") or {}
        photos = poi.get("photos") or []

        data.append(
            {
                "名称": str(poi.get("name", "")),
                "地址": str(poi.get("address", "")),
                "类型": str(poi.get("type", "")),
                "类型编码": str(poi.get("typecode", "")),
                "距离(米)": str(poi.get("distance", "")),
                "人均消费": str(business.get("cost", "")),
                "评分": str(business.get("rating", "")),
                "电话": str(business.get("tel", "")),
                "今日营业时间": str(business.get("opentime_today", "")),
                "营业时间": str(business.get("opentime_week", "")),
                "特色内容": str(business.get("tag", "")),
                "主标签": str(business.get("keytag", "")),
                "补充标签": str(business.get("rectag", "")),
                "商圈": str(business.get("business_area", "")),
                "别名": str(business.get("alias", "")),
                "省": str(poi.get("pname", "")),
                "市": str(poi.get("cityname", "")),
                "区": str(poi.get("adname", "")),
                "省编码": str(poi.get("pcode", "")),
                "市编码": str(poi.get("citycode", "")),
                "区编码": str(poi.get("adcode", "")),
                "坐标": str(poi.get("location", "")),
                "入口坐标": str(navi.get("entr_location", "")),
                "导航网格": str(navi.get("gridcode", "")),
                "室内地图": str(indoor.get("indoor_map", "")),
                "首图": first_photo_url(photos),
                "图片数量": str(len(photos)),
                "图片链接": join_photo_urls(photos),
                "POI ID": str(poi.get("id", "")),
            }
        )

    df = pd.DataFrame(data)

    try:
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"数据已保存到 {filename}")
    except Exception as exc:
        print(f"保存文件时出错: {exc}")
        csv_filename = filename.replace(".xlsx", ".csv")
        df.to_csv(csv_filename, index=False, encoding="utf_8_sig")
        print(f"已将数据保存为 CSV 格式: {csv_filename}")


def prompt_with_default(prompt_text, default_value):
    value = input(f"{prompt_text}（直接回车使用默认值：{default_value}）: ").strip()
    return value or str(default_value)


def prompt_user_inputs():
    """运行时询问用户查询参数。"""
    location = prompt_with_default(
        "请输入要查询的地点名称、地址或经纬度",
        DEFAULT_LOCATION,
    )

    radius_text = prompt_with_default("请输入搜索半径（米）", DEFAULT_RADIUS)
    max_results_text = prompt_with_default("请输入最多返回多少条", DEFAULT_MAX_RESULTS)
    output = prompt_with_default(
        "请输入输出文件名或完整路径",
        str(DEFAULT_OUTPUT_PATH),
    )

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
        "output": str(resolve_output_path(output)),
    }


def main():
    user_inputs = prompt_user_inputs()

    try:
        center_location, geocode_info = normalize_location(user_inputs["location"])
    except RuntimeError as exc:
        print(exc)
        return

    if geocode_info:
        print(
            f"已解析地点: {geocode_info.get('formatted_address', user_inputs['location'])} "
            f"-> {center_location}"
        )
    else:
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

    if restaurants:
        print(f"共获取到 {len(restaurants)} 个餐饮场所")
        save_to_excel(restaurants, user_inputs["output"])
    else:
        print("未获取到任何餐饮场所数据")


if __name__ == "__main__":
    main()
