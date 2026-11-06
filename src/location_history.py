# -*- coding: utf-8 -*-
"""地理位置地址历史：查过的地点与坐标保存在本机，下次直接选用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_MAX_ENTRIES = 40


def default_history_path() -> Path:
    return Path(__file__).resolve().parent / "location_history.json"


def _clean_name(name: str) -> str:
    return (name or "").strip().strip('"').strip("'")


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_coords(lat: Optional[float], lon: Optional[float]) -> bool:
    return (
        lat is not None
        and lon is not None
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def load_location_history(path: Optional[Path] = None) -> List[Dict]:
    """最近使用的地址在前。文件缺失或损坏时返回空列表。"""
    src = Path(path) if path is not None else default_history_path()
    if not src.is_file():
        return []
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: List[Dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _clean_name(str(item.get("name") or ""))
        lat = _as_float(item.get("lat"))
        lon = _as_float(item.get("lon"))
        if not name or not _valid_coords(lat, lon):
            continue
        out.append(
            {
                "name": name,
                "lat": lat,
                "lon": lon,
                "province": str(item.get("province") or "").strip(),
                "city": str(item.get("city") or "").strip(),
            }
        )
    return out


def lookup_location(name: str, path: Optional[Path] = None) -> Optional[Dict]:
    key = _clean_name(name)
    if not key:
        return None
    for item in load_location_history(path):
        if item["name"] == key:
            return item
    return None


def remember_location(
    name: str,
    lat: float,
    lon: float,
    province: str = "",
    city: str = "",
    path: Optional[Path] = None,
) -> List[Dict]:
    """把地址记到历史最前。同名只保留一条，并更新坐标。"""
    key = _clean_name(name)
    lat_f = _as_float(lat)
    lon_f = _as_float(lon)
    if not key or not _valid_coords(lat_f, lon_f):
        return load_location_history(path)

    history = load_location_history(path)
    previous = next((item for item in history if item["name"] == key), None)
    prov = (province or "").strip() or (previous or {}).get("province") or ""
    city_name = (city or "").strip() or (previous or {}).get("city") or ""
    kept = [item for item in history if item["name"] != key]
    kept.insert(
        0,
        {
            "name": key,
            "lat": lat_f,
            "lon": lon_f,
            "province": prov,
            "city": city_name,
        },
    )
    kept = kept[:_MAX_ENTRIES]
    dest = Path(path) if path is not None else default_history_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return kept
