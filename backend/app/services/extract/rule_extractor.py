from __future__ import annotations

import re
from typing import Iterable

from backend.app.models.extract import ExtractResult

DESTINATION_PATTERNS = [
    r"(?:去|到|前往|想去|打算去|准备去)([\u4e00-\u9fa5]{2,12})",
    r"(?:在|到达|抵达)([\u4e00-\u9fa5]{2,12})(?:玩|旅游|旅行|出差|住|待)",
]
DESTINATION_SUFFIXES = (
    "市",
    "省",
    "区",
    "县",
    "镇",
    "州",
    "岛",
    "湾",
    "山",
    "大学",
    "机场",
)
COMMON_DESTINATIONS = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "重庆",
    "西安",
    "南京",
    "苏州",
    "武汉",
    "长沙",
    "天津",
    "青岛",
    "三亚",
    "昆明",
    "厦门",
    "哈尔滨",
]
DATE_KEYWORDS = [
    "今天",
    "明天",
    "后天",
    "这周末",
    "周末",
    "五一",
    "国庆",
    "端午",
    "中秋",
    "春节",
]
SPOT_KEYWORDS = [
    "博物馆",
    "美术馆",
    "古镇",
    "乐园",
    "动物园",
    "植物园",
    "公园",
    "景区",
    "海边",
    "沙滩",
    "寺庙",
    "雪山",
    "湖",
    "温泉",
    "步行街",
    "夜景",
    "迪士尼",
]
FOOD_KEYWORDS = [
    "火锅",
    "烧烤",
    "海鲜",
    "咖啡",
    "小吃",
    "粤菜",
    "川菜",
    "湘菜",
    "日料",
    "西餐",
    "早餐",
    "夜宵",
    "甜品",
    "奶茶",
]
HOTEL_KEYWORDS = [
    "酒店",
    "民宿",
    "青旅",
    "度假酒店",
    "商务酒店",
    "高档酒店",
    "经济型酒店",
    "亲子酒店",
]
TRAVEL_STYLE_KEYWORDS = [
    "亲子",
    "情侣",
    "家庭",
    "自驾",
    "休闲",
    "深度游",
    "特种兵",
    "文艺",
    "轻松",
    "紧凑",
]
CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


class RuleExtractor:
    def extract(self, raw_text: str, *, source_type: str, source_file_path: str | None = None) -> ExtractResult:
        normalized_text = self._normalize_text(raw_text)
        destination = self._extract_destination(normalized_text)
        dates = self._extract_dates(normalized_text)
        budget_text, budget_min, budget_max = self._extract_budget(normalized_text)
        people_count = self._extract_people_count(normalized_text)
        spot_keywords = self._extract_keywords(normalized_text, SPOT_KEYWORDS)
        food_keywords = self._extract_keywords(normalized_text, FOOD_KEYWORDS)
        hotel_keywords = self._extract_keywords(normalized_text, HOTEL_KEYWORDS)
        travel_styles = self._extract_keywords(normalized_text, TRAVEL_STYLE_KEYWORDS)

        detected_keywords = self._merge_keywords(
            [destination] if destination else [],
            dates,
            [budget_text] if budget_text else [],
            spot_keywords,
            food_keywords,
            hotel_keywords,
            travel_styles,
        )

        return ExtractResult(
            source_type=source_type,
            source_file_path=source_file_path,
            raw_text=normalized_text,
            destination=destination,
            dates=dates,
            budget_text=budget_text,
            budget_min_cny=budget_min,
            budget_max_cny=budget_max,
            people_count=people_count,
            spot_keywords=spot_keywords,
            food_keywords=food_keywords,
            hotel_keywords=hotel_keywords,
            travel_styles=travel_styles,
            detected_keywords=detected_keywords,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    def _extract_destination(self, text: str) -> str | None:
        for pattern in DESTINATION_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            candidate = self._clean_destination(match.group(1))
            if candidate:
                return candidate

        for city in COMMON_DESTINATIONS:
            if city in text:
                return city
        return None

    def _clean_destination(self, candidate: str) -> str | None:
        cleaned = re.sub(r"(旅游|旅行|出差|玩|住|待|看看)$", "", candidate).strip("，。；、 ")
        if len(cleaned) < 2:
            return None
        if cleaned in COMMON_DESTINATIONS:
            return cleaned
        if cleaned.endswith(DESTINATION_SUFFIXES):
            return cleaned
        return cleaned

    def _extract_dates(self, text: str) -> list[str]:
        dates = list(self._extract_keywords(text, DATE_KEYWORDS))
        dates.extend(re.findall(r"\d{1,2}月\d{1,2}日", text))
        dates.extend(re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text))
        return self._unique_keep_order(dates)

    def _extract_budget(self, text: str) -> tuple[str | None, int | None, int | None]:
        range_match = re.search(r"(\d{2,6})\s*[-到至~]\s*(\d{2,6})\s*(?:元|块|人民币)", text)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            return range_match.group(0), min(start, end), max(start, end)

        under_match = re.search(r"(?:预算|控制在|不超过|最多)\s*(\d{2,6})\s*(?:元|块|人民币)", text)
        if under_match:
            value = int(under_match.group(1))
            return under_match.group(0), 0, value

        single_match = re.search(r"(\d{2,6})\s*(?:元|块|人民币)(?:以内|左右)?", text)
        if single_match:
            value = int(single_match.group(1))
            return single_match.group(0), value, value

        return None, None, None

    def _extract_people_count(self, text: str) -> int | None:
        numeric_match = re.search(r"(\d{1,2})\s*个?\s*人", text)
        if numeric_match:
            return int(numeric_match.group(1))

        chinese_match = re.search(r"([一二两三四五六七八九十])\s*个?\s*人", text)
        if chinese_match:
            return CHINESE_NUMBER_MAP.get(chinese_match.group(1))

        family_match = re.search(r"一家([一二两三四五六七八九十])口", text)
        if family_match:
            return CHINESE_NUMBER_MAP.get(family_match.group(1))

        return None

    def _extract_keywords(self, text: str, keywords: Iterable[str]) -> list[str]:
        return self._unique_keep_order([keyword for keyword in keywords if keyword in text])

    def _merge_keywords(self, *groups: Iterable[str]) -> list[str]:
        merged: list[str] = []
        for group in groups:
            merged.extend(item for item in group if item)
        return self._unique_keep_order(merged)

    @staticmethod
    def _unique_keep_order(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
