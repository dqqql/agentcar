from __future__ import annotations

import argparse
import locale
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.adapter import AdapterRequest, CandidatePoi, CandidatePoolResult
from backend.app.models.extract import ExtractRequest, ExtractResult
from backend.app.services.adapter.service import build_candidate_adapter_service
from backend.app.services.extract.service import build_extract_service


TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "output"
GETDATA_OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "getdata"
DEFAULT_SEED = None
HOTEL_LOCATION_MAP = {
    "北京市": "Beijing, China",
    "天津市": "Tianjin, China",
    "杭州市": "Hangzhou, China",
}


@dataclass(frozen=True)
class Scenario:
    destination: str
    date_phrases: tuple[str, ...]
    spot_keywords: tuple[str, ...]
    food_keywords: tuple[str, ...]
    hotel_keywords: tuple[str, ...]
    travel_styles: tuple[str, ...]
    budget_ranges: tuple[tuple[int, int], ...]
    people_counts: tuple[int, ...]


SCENARIOS = (
    Scenario(
        destination="天津市",
        date_phrases=("这周末", "五一"),
        spot_keywords=("公园", "博物馆", "古镇"),
        food_keywords=("火锅", "小吃", "烧烤"),
        hotel_keywords=("民宿", "酒店"),
        travel_styles=("轻松", "美食"),
        budget_ranges=((2000, 3500), (3000, 5000)),
        people_counts=(2, 3),
    ),
    Scenario(
        destination="北京市",
        date_phrases=("下周末", "五一"),
        spot_keywords=("博物馆", "公园", "景区"),
        food_keywords=("烤鸭", "火锅", "小吃"),
        hotel_keywords=("酒店", "经济型酒店"),
        travel_styles=("轻松", "文化"),
        budget_ranges=((2500, 4000), (4000, 6000)),
        people_counts=(2, 4),
    ),
    Scenario(
        destination="杭州市",
        date_phrases=("这周末", "端午"),
        spot_keywords=("古镇", "公园", "博物馆"),
        food_keywords=("小吃", "面食", "杭帮菜"),
        hotel_keywords=("民宿", "酒店"),
        travel_styles=("轻松", "拍照"),
        budget_ranges=((2200, 3800), (3500, 5500)),
        people_counts=(2, 3),
    ),
)


def build_random_request_text(rng: random.Random) -> tuple[str, dict[str, Any]]:
    scenario = rng.choice(SCENARIOS)
    date_phrase = rng.choice(scenario.date_phrases)
    people_count = rng.choice(scenario.people_counts)
    budget_min, budget_max = rng.choice(scenario.budget_ranges)
    selected_spots = rng.sample(list(scenario.spot_keywords), k=min(2, len(scenario.spot_keywords)))
    selected_foods = rng.sample(list(scenario.food_keywords), k=min(2, len(scenario.food_keywords)))
    hotel_keyword = rng.choice(scenario.hotel_keywords)
    style = rng.choice(scenario.travel_styles)

    text = (
        f"我想{date_phrase}去{scenario.destination}玩，预算{budget_min}到{budget_max}元，"
        f"{people_count}个人，想逛{'和'.join(selected_spots)}，想吃{'和'.join(selected_foods)}，"
        f"住宿希望是{hotel_keyword}，整体行程{style}一点。"
    )
    return text, {
        "scenario_destination": scenario.destination,
        "date_phrase": date_phrase,
        "spot_keywords": selected_spots,
        "food_keywords": selected_foods,
        "hotel_keyword": hotel_keyword,
        "travel_style": style,
        "budget_min_cny": budget_min,
        "budget_max_cny": budget_max,
        "people_count": people_count,
    }


def run_interactive_script(
    dataset: str,
    script_path: Path,
    input_lines: list[str],
    timeout_seconds: int = 300,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
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
    stdout_text = decode_process_output(completed.stdout)
    stderr_text = decode_process_output(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"{dataset} 脚本执行失败，返回码 {completed.returncode}\n"
            f"stdout:\n{tail_text(stdout_text)}\n"
            f"stderr:\n{tail_text(stderr_text)}"
        )

    latest_detail_path = resolve_new_detail_path(output_root, existing_dirs)
    return latest_detail_path, completed


def tail_text(value: str | None, max_length: int = 3000) -> str:
    text = value or ""
    return text[-max_length:]


def decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    encodings = [
        locale.getpreferredencoding(False),
        "utf-8",
        "gbk",
        "cp936",
    ]
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def resolve_new_detail_path(output_root: Path, existing_dirs: set[Path]) -> Path:
    current_dirs = [path.resolve() for path in output_root.iterdir() if path.is_dir()]
    new_dirs = [path for path in current_dirs if path not in existing_dirs]
    if not new_dirs:
        raise RuntimeError(f"本次运行后未在 {output_root} 下生成新的 output 目录。")

    latest_dir = max(new_dirs, key=lambda path: path.stat().st_mtime)
    detail_path = latest_dir / "detail.json"
    if not detail_path.exists():
        raise RuntimeError(f"输出目录中缺少 detail.json：{detail_path}")
    return detail_path


def build_food_inputs(destination: str) -> list[str]:
    return [destination, "3000", "30"]


def build_hotel_inputs(destination: str) -> list[str]:
    check_in_date = date.today() + timedelta(days=7)
    hotel_location = HOTEL_LOCATION_MAP.get(destination, destination)
    return [
        hotel_location,
        "4000",
        "20",
        check_in_date.isoformat(),
        "1",
        "pipeline_demo",
    ]


def build_place_inputs(destination: str, spot_keyword: str) -> list[str]:
    return [destination, spot_keyword]


def flatten_candidates(result: CandidatePoolResult) -> list[dict[str, Any]]:
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


def save_demo_result(payload: dict[str, Any]) -> Path:
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = TEST_OUTPUT_DIR / f"pipeline_demo_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def print_summary(
    *,
    generated_text: str,
    extract_result: ExtractResult,
    detail_paths: dict[str, Path],
    candidate_result: CandidatePoolResult,
    flattened_candidates: list[dict[str, Any]],
    saved_path: Path,
    seed: int | None,
) -> None:
    print("\n===== 联调完成 =====")
    print(f"随机种子: {seed if seed is not None else '系统随机'}")
    print(f"生成文本: {generated_text}")
    print(f"提取目的地: {extract_result.destination}")
    print(f"景点偏好: {extract_result.spot_keywords}")
    print(f"餐饮偏好: {extract_result.food_keywords}")
    print(f"酒店偏好: {extract_result.hotel_keywords}")
    print(f"出行风格: {extract_result.travel_styles}")
    print(f"提取结果文件: {extract_result.result_file_path}")
    print(f"place detail: {detail_paths['place']}")
    print(f"food detail: {detail_paths['food']}")
    print(f"hotel detail: {detail_paths['hotel']}")
    print(f"候选池结果文件: {candidate_result.result_file_path}")
    print(f"联调汇总文件: {saved_path}")
    print(
        "候选数量: "
        f"spot={len(candidate_result.spot_candidates)}, "
        f"food={len(candidate_result.food_candidates)}, "
        f"hotel={len(candidate_result.hotel_candidates)}, "
        f"total={len(flattened_candidates)}"
    )
    print("可输入算法的候选示例（前 10 条）:")
    for item in flattened_candidates[:10]:
        print(
            f"- [{item['poi_type']}] {item['name']} | "
            f"distance={item['center_distance_m']}m | "
            f"rating={item['rating']} | popularity={item['popularity']} | "
            f"price={item['price_value_cny']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="随机生成文本并联调提取、采集、适配流程。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="可选随机种子，便于复现。")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    generated_text, scenario_meta = build_random_request_text(rng)

    extract_service = build_extract_service()
    extract_result = extract_service.extract(ExtractRequest(text=generated_text))

    destination_for_scripts = extract_result.destination or scenario_meta["scenario_destination"]
    spot_keyword_for_place = (
        extract_result.spot_keywords[0]
        if extract_result.spot_keywords
        else scenario_meta["spot_keywords"][0]
    )

    place_detail_path, _ = run_interactive_script(
        "place",
        PROJECT_ROOT / "scripts" / "getdata" / "place" / "main.py",
        build_place_inputs(destination_for_scripts, spot_keyword_for_place),
    )
    food_detail_path, _ = run_interactive_script(
        "food",
        PROJECT_ROOT / "scripts" / "getdata" / "food" / "main.py",
        build_food_inputs(destination_for_scripts),
    )
    hotel_detail_path, _ = run_interactive_script(
        "hotel",
        PROJECT_ROOT / "scripts" / "getdata" / "hotel" / "main.py",
        build_hotel_inputs(destination_for_scripts),
    )

    adapter_service = build_candidate_adapter_service()
    candidate_result = adapter_service.build_candidate_pool(
        AdapterRequest(
            extract_result_path=extract_result.result_file_path,
            place_detail_path=str(place_detail_path),
            food_detail_path=str(food_detail_path),
            hotel_detail_path=str(hotel_detail_path),
        )
    )

    flattened_candidates = flatten_candidates(candidate_result)
    payload = {
        "generated_text": generated_text,
        "scenario_meta": scenario_meta,
        "extract_result_path": extract_result.result_file_path,
        "extract_summary": {
            "destination": extract_result.destination,
            "dates": extract_result.dates,
            "budget_text": extract_result.budget_text,
            "budget_min_cny": extract_result.budget_min_cny,
            "budget_max_cny": extract_result.budget_max_cny,
            "people_count": extract_result.people_count,
            "spot_keywords": extract_result.spot_keywords,
            "food_keywords": extract_result.food_keywords,
            "hotel_keywords": extract_result.hotel_keywords,
            "travel_styles": extract_result.travel_styles,
            "algorithm_input": (
                extract_result.algorithm_input.model_dump()
                if extract_result.algorithm_input
                else None
            ),
        },
        "detail_paths": {
            "place": str(place_detail_path),
            "food": str(food_detail_path),
            "hotel": str(hotel_detail_path),
        },
        "candidate_pool_result_path": candidate_result.result_file_path,
        "candidate_counts": {
            "spot": len(candidate_result.spot_candidates),
            "food": len(candidate_result.food_candidates),
            "hotel": len(candidate_result.hotel_candidates),
            "total": len(flattened_candidates),
        },
        "algorithm_ready_candidates": flattened_candidates,
    }
    saved_path = save_demo_result(payload)
    print_summary(
        generated_text=generated_text,
        extract_result=extract_result,
        detail_paths={
            "place": place_detail_path,
            "food": food_detail_path,
            "hotel": hotel_detail_path,
        },
        candidate_result=candidate_result,
        flattened_candidates=flattened_candidates,
        saved_path=saved_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
