from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def sanitize_path_component(value: Any, fallback: str = "dataset") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback

    text = re.sub(r'[<>:"/\\|?*\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or fallback


def build_output_bundle(script_dir: Path, dataset_type: str, *name_parts: Any) -> tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_parts = [sanitize_path_component(dataset_type, "dataset")]
    for part in name_parts:
        text = str(part or "").strip()
        if text:
            safe_parts.append(sanitize_path_component(text, "item"))

    bundle_name = "_".join(safe_parts + [timestamp])
    bundle_dir = script_dir / "output" / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir, bundle_name


def write_summary_csv(
    rows: Iterable[dict[str, Any]],
    columns: list[tuple[str, str]],
    output_dir: Path,
) -> Path:
    output_path = output_dir / "summary.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([label for _, label in columns])
        for row in rows:
            writer.writerow([normalize_value(row.get(key, "")) for key, _ in columns])
    return output_path


def write_detail_json(payload: dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "detail.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
