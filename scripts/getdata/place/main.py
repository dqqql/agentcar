import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import request
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
GETDATA_DIR = SCRIPT_DIR.parent
if str(GETDATA_DIR) not in sys.path:
    sys.path.insert(0, str(GETDATA_DIR))

from output_utils import build_output_bundle, write_detail_json, write_summary_csv
from coordTransform_utils import gcj02_to_wgs84, gcj02_to_bd09
#from shp import trans_point_to_shp

amap_web_key = '3ff0b1a041f41b85422d9bc00e68f7a7' # 请在此处填入您的高德API Key
keyword = ['公园']
city = ['北京市']
# 输出数据坐标系,1为高德GCJ20坐标系，2WGS84坐标系，3百度BD09坐标系
coord = 2

poi_search_url = "http://restapi.amap.com/v3/place/text"
poi_boundary_url = "https://ditu.amap.com/detail/get/detail"
PAGE_SIZE = 25
# 测试阶段临时限制总抓取量，测试结束后可改为 None 取消限制。
MAX_TOTAL_RESULTS = 100
EXPORT_COLUMNS = [
    'lon', 'lat', 'location_raw', 'id', 'parent', 'name', 'type', 'typecode',
    'biz_type', 'address', 'pname', 'pcode', 'cityname', 'citycode', 'adname',
    'adcode', 'business_area', 'distance', 'tel', 'postcode', 'website',
    'email', 'entr_location', 'exit_location', 'navi_poiid', 'gridcode',
    'alias', 'parking_type', 'tag', 'indoor_map', 'indoor_data', 'cpid',
    'floor', 'truefloor', 'groupbuy_num', 'discount_num', 'rating', 'cost',
    'meal_ordering', 'seat_ordering', 'ticket_ordering', 'hotel_ordering',
    'photo_count', 'photo_titles', 'photo_urls', 'children'
]
EXPORT_COLUMN_LABELS = {
    'lon': '经度',
    'lat': '纬度',
    'location_raw': '原始坐标',
    'id': 'POI ID',
    'parent': '父级POI ID',
    'name': '名称',
    'type': '类型',
    'typecode': '类型编码',
    'biz_type': '业务类型',
    'address': '地址',
    'pname': '省份',
    'pcode': '省份编码',
    'cityname': '城市',
    'citycode': '城市编码',
    'adname': '区县',
    'adcode': '区县编码',
    'business_area': '商圈',
    'distance': '距离',
    'tel': '电话',
    'postcode': '邮编',
    'website': '网站',
    'email': '邮箱',
    'entr_location': '入口坐标',
    'exit_location': '出口坐标',
    'navi_poiid': '导航POI ID',
    'gridcode': '网格编码',
    'alias': '别名',
    'parking_type': '停车场类型',
    'tag': '标签',
    'indoor_map': '室内地图标记',
    'indoor_data': '室内信息',
    'cpid': 'CPID',
    'floor': '楼层',
    'truefloor': '真实楼层',
    'groupbuy_num': '团购数量',
    'discount_num': '优惠数量',
    'rating': '评分',
    'cost': '人均消费',
    'meal_ordering': '是否支持订餐',
    'seat_ordering': '是否支持订座',
    'ticket_ordering': '是否支持订票',
    'hotel_ordering': '是否支持订房',
    'photo_count': '图片数量',
    'photo_titles': '图片标题',
    'photo_urls': '图片链接',
    'children': '子POI列表',
}
SUMMARY_COLUMNS = [
    ('dataset_type', '数据类型'),
    *[(column_name, EXPORT_COLUMN_LABELS.get(column_name, column_name)) for column_name in EXPORT_COLUMNS],
    ('source_provider', '数据来源'),
]


def parse_input_list(raw_value):
    return [item.strip() for item in raw_value.replace('，', ',').split(',') if item.strip()]


def prompt_search_inputs(default_cities, default_keywords):
    print("请输入地区和关键词，多个值可用逗号分隔；直接回车将使用默认值。")
    city_input = input(f"请输入地区（默认：{', '.join(default_cities)}）：").strip()
    keyword_input = input(f"请输入关键词（默认：{', '.join(default_keywords)}）：").strip()

    selected_cities = parse_input_list(city_input) if city_input else list(default_cities)
    selected_keywords = parse_input_list(keyword_input) if keyword_input else list(default_keywords)

    if not selected_cities:
        selected_cities = list(default_cities)
    if not selected_keywords:
        selected_keywords = list(default_keywords)

    print(f"本次查询地区：{', '.join(selected_cities)}")
    print(f"本次查询关键词：{', '.join(selected_keywords)}")
    return selected_cities, selected_keywords


# 根据城市名称和分类关键字获取poi数据
def getpois(cityname, keywords, max_results=None):
    i = 1
    poilist = []
    while True:  # 使用while循环不断分页获取数据
        remaining = None if max_results is None else max_results - len(poilist)
        if remaining is not None and remaining <= 0:
            print(f"已达到当前查询临时上限：{max_results} 条")
            break

        page_size = PAGE_SIZE if remaining is None else min(PAGE_SIZE, remaining)
        result_str = getpoi_page(cityname, keywords, i, page_size)
        print(f"DEBUG: Page {i} Response: {result_str}")
        result = json.loads(result_str)  # 将字符串转换为json
        
        if result['status'] != '1':
            print(f"API请求失败: {result.get('info', '未知错误')} (代码: {result.get('infocode', 'N/A')})")
            break

        if result['count'] == '0':
            break

        added_count = hand(poilist, result, remaining)
        if added_count == 0:
            break
        if max_results is not None and len(poilist) >= max_results:
            print(f"已达到当前查询临时上限：{max_results} 条")
            break

        i = i + 1
        time.sleep(0.5) # 防止 QPS 过高导致限制
    return poilist


def stringify_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def normalize_business_area(value):
    if isinstance(value, list):
        return ','.join(str(item) for item in value if item)
    return value or ''


def convert_location(location):
    if not location:
        return '', ''

    parts = str(location).split(",")
    if len(parts) != 2:
        return '', ''

    lng, lat = parts
    try:
        lng = float(lng)
        lat = float(lat)
    except (TypeError, ValueError):
        return '', ''

    if coord == 2:
        lng, lat = gcj02_to_wgs84(lng, lat)
    elif coord == 3:
        lng, lat = gcj02_to_bd09(lng, lat)

    return lng, lat


def normalize_photos(photos):
    if not isinstance(photos, list):
        return 0, '', ''

    titles = []
    urls = []
    for photo in photos:
        if not isinstance(photo, dict):
            continue
        title = photo.get('title')
        url = photo.get('url')
        if title:
            titles.append(str(title))
        if url:
            urls.append(str(url))

    return len(urls), '|'.join(titles), '|'.join(urls)


def build_export_row(poi):
    biz_ext = poi.get('biz_ext') or {}
    if not isinstance(biz_ext, dict):
        biz_ext = {}

    photos = poi.get('photos') or []
    photo_count, photo_titles, photo_urls = normalize_photos(photos)
    lon, lat = convert_location(poi.get('location'))

    return {
        'lon': lon,
        'lat': lat,
        'location_raw': poi.get('location', ''),
        'id': poi.get('id', ''),
        'parent': poi.get('parent', ''),
        'name': poi.get('name', ''),
        'type': poi.get('type', ''),
        'typecode': poi.get('typecode', ''),
        'biz_type': poi.get('biz_type', ''),
        'address': poi.get('address', ''),
        'pname': poi.get('pname', ''),
        'pcode': poi.get('pcode', ''),
        'cityname': poi.get('cityname', ''),
        'citycode': poi.get('citycode', ''),
        'adname': poi.get('adname', ''),
        'adcode': poi.get('adcode', ''),
        'business_area': normalize_business_area(poi.get('business_area')),
        'distance': poi.get('distance', ''),
        'tel': poi.get('tel', ''),
        'postcode': poi.get('postcode', ''),
        'website': poi.get('website', ''),
        'email': poi.get('email', ''),
        'entr_location': poi.get('entr_location', ''),
        'exit_location': poi.get('exit_location', ''),
        'navi_poiid': poi.get('navi_poiid', ''),
        'gridcode': poi.get('gridcode', ''),
        'alias': poi.get('alias', ''),
        'parking_type': poi.get('parking_type', ''),
        'tag': poi.get('tag', ''),
        'indoor_map': poi.get('indoor_map', ''),
        'indoor_data': stringify_value(poi.get('indoor_data')),
        'cpid': poi.get('cpid', ''),
        'floor': poi.get('floor', ''),
        'truefloor': poi.get('truefloor', ''),
        'groupbuy_num': poi.get('groupbuy_num', ''),
        'discount_num': poi.get('discount_num', ''),
        'rating': biz_ext.get('rating', ''),
        'cost': biz_ext.get('cost', ''),
        'meal_ordering': biz_ext.get('meal_ordering', ''),
        'seat_ordering': biz_ext.get('seat_ordering', ''),
        'ticket_ordering': biz_ext.get('ticket_ordering', ''),
        'hotel_ordering': biz_ext.get('hotel_ordering', ''),
        'photo_count': photo_count,
        'photo_titles': photo_titles,
        'photo_urls': photo_urls,
        'children': stringify_value(poi.get('children')),
    }


def build_export_rows(poilist):
    return [build_export_row(poi) for poi in poilist]


def build_detail_record(poi):
    row = build_export_row(poi)
    row['dataset_type'] = 'place'
    row['source_provider'] = 'Amap Place Text API'
    row['source_raw'] = poi
    return row


def save_outputs(records, query):
    bundle_dir, bundle_name = build_output_bundle(
        SCRIPT_DIR,
        'place',
        query.get('city'),
        query.get('keyword'),
    )
    detail_payload = {
        'dataset_type': 'place',
        'bundle_name': bundle_name,
        'generated_at': datetime.now().isoformat(),
        'query': query,
        'record_count': len(records),
        'records': records,
    }
    summary_path = write_summary_csv(records, SUMMARY_COLUMNS, bundle_dir)
    detail_path = write_detail_json(detail_payload, bundle_dir)
    return summary_path, detail_path


# 将返回的poi数据装入集合返回
def hand(poilist, result, remaining=None):
    # result = json.loads(result)  # 将字符串转换为json
    pois = result.get('pois', [])
    if remaining is not None:
        pois = pois[:remaining]

    for poi in pois:
        poilist.append(poi)

    return len(pois)


# 单页获取pois
def getpoi_page(cityname, keywords, page, offset):
    req_url = poi_search_url + "?key=" + amap_web_key + '&extensions=all&keywords=' + quote(
        keywords) + '&city=' + quote(cityname) + '&citylimit=true' + '&offset=' + str(offset) + '&page=' + str(
        page) + '&output=json'
    data = ''
    print('============请求url:' + req_url)
    with request.urlopen(req_url) as f:
        data = f.read()
        data = data.decode('utf-8')
    return data


def resolve_area_code_from_keyword(keyword):
    '''
    当输入不是标准行政区名称时，尝试按地点名称反查所属行政区 adcode
    :param keyword:
    :return:
    '''

    keyword = str(keyword).strip()
    if not keyword:
        return ""

    req_url = poi_search_url + "?key=" + amap_web_key + '&extensions=all&keywords=' + quote(
        keyword) + '&offset=1&page=1&output=json'
    print('尝试按地点名称解析所属行政区：' + keyword)
    print(req_url)

    with request.urlopen(req_url) as f:
        data = f.read()
        data = data.decode('utf-8')

    result = json.loads(data)
    pois = result.get('pois') or []
    if not pois:
        return ""

    poi = pois[0]
    pname = str(poi.get('pname') or '').strip()
    cityname = str(poi.get('cityname') or '').strip()
    adname = str(poi.get('adname') or '').strip()
    adcode = str(poi.get('adcode') or '').strip()
    resolved_parts = [part for part in [pname, cityname, adname] if part and part != '[]']
    resolved_name = ''.join(resolved_parts) if resolved_parts else keyword

    if adcode:
        print(f"未直接匹配到行政区，已自动定位到：{resolved_name}（adcode: {adcode}）")
    return adcode


def get_areas(code):
    '''
    获取城市的所有区域
    :param code:
    :return:
    '''

    code_str = str(code).strip()
    print('获取城市的所有区域：code: ' + code_str)
    data_str = get_distrinctNoCache(code_str)

    print('get_distrinct result:' + data_str)

    data = json.loads(data_str)

    if data.get('status') != '1':
        print(f"获取行政区划失败: {data.get('info', '未知错误')}")
        return ""

    district_list = data.get('districts') or []
    if not district_list:
        fallback_area = resolve_area_code_from_keyword(code_str)
        if fallback_area:
            return fallback_area
        print(f"未找到“{code_str}”对应的行政区划，将直接按原输入检索。")
        return ""

    current_district = district_list[0]
    districts = current_district.get('districts') or []

    # 判断是否是直辖市
    # 北京市、上海市、天津市、重庆市。
    if (code_str.startswith('重庆') or code_str.startswith('上海') or code_str.startswith('北京') or code_str.startswith('天津')):
        if districts and districts[0].get('districts'):
            districts = districts[0].get('districts') or []

    if not districts:
        adcode = str(current_district.get('adcode') or '').strip()
        if adcode:
            print(f"未获取到下级区域，改为使用当前区域 adcode：{adcode}")
        return adcode

    area_codes = []
    for district in districts:
        adcode = str(district.get('adcode') or '').strip()
        if adcode:
            area_codes.append(adcode)

    area = ','.join(area_codes)
    print(area)
    return area


def get_data(city, keyword):
    '''
    根据城市名以及POI类型爬取数据
    :param city:
    :param keyword:
    :return:
    '''
    isNeedAreas = True
    area = ""
    if isNeedAreas:
        area = get_areas(city)
    all_pois = []
    if area != None and area != "":
        area_list = [item.strip() for item in str(area).replace('，', ',').split(',') if item.strip()]

        for area in area_list:
            remaining_limit = None if MAX_TOTAL_RESULTS is None else MAX_TOTAL_RESULTS - len(all_pois)
            if remaining_limit is not None and remaining_limit <= 0:
                print(f"已达到当前任务临时上限：{MAX_TOTAL_RESULTS} 条")
                break

            pois_area = getpois(area, keyword, remaining_limit)
            print('当前城区：' + str(area) + ', 分类：' + str(keyword) + ", 总的有" + str(len(pois_area)) + "条数据")
            all_pois.extend(pois_area)
        if MAX_TOTAL_RESULTS is not None:
            all_pois = all_pois[:MAX_TOTAL_RESULTS]
        print("所有城区的数据汇总，总数为：" + str(len(all_pois)))
        records = [build_detail_record(poi) for poi in all_pois]
        return save_outputs(
            records,
            {
                'city': city,
                'keyword': keyword,
                'area_codes': area_list,
                'max_total_results': MAX_TOTAL_RESULTS,
                'coordinate_system': coord,
            },
        )
    else:
        pois_area = getpois(city, keyword, MAX_TOTAL_RESULTS)
        records = [build_detail_record(poi) for poi in pois_area]
        return save_outputs(
            records,
            {
                'city': city,
                'keyword': keyword,
                'area_codes': [],
                'max_total_results': MAX_TOTAL_RESULTS,
                'coordinate_system': coord,
            },
        )

    return None


def get_distrinctNoCache(code):
    '''
    获取中国城市行政区划
    :return:
    '''

    url = "https://restapi.amap.com/v3/config/district?subdistrict=2&extensions=all&key=" + amap_web_key

    req_url = url + "&keywords=" + quote(code)

    print(req_url)

    with request.urlopen(req_url) as f:
        data = f.read()
        data = data.decode('utf-8')
    print(code, data)
    return data


if __name__ == '__main__':
    selected_cities, selected_keywords = prompt_search_inputs(city, keyword)

    for ct in selected_cities:
        for type in selected_keywords:
            summary_path, detail_path = get_data(ct, type)
            print(f"汇总文件已保存到 {summary_path}")
            print(f"详细文件已保存到 {detail_path}")
# 致谢
# 在学习过程中特别感谢其他开发者提供的代码和帮助。为共
