# -*- coding: utf-8 -*-
"""GPX 读取与合并（标准库 XML，无额外依赖）。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


def resolve_gpx_path_list(
    gpx_path: Optional[str] = None,
    gpx_paths: Optional[Sequence[str]] = None,
) -> List[str]:
    """合并单路径与路径列表，去重并仅保留存在的文件。"""
    out: List[str] = []
    if gpx_paths:
        for raw in gpx_paths:
            p = (raw or "").strip()
            if not p or p in out:
                continue
            if Path(p).expanduser().is_file():
                out.append(str(Path(p).expanduser().resolve()))
    single = (gpx_path or "").strip()
    if single:
        p = Path(single).expanduser()
        if p.is_file():
            resolved = str(p.resolve())
            if resolved not in out:
                out.insert(0, resolved)
    return out


@dataclass
class GpxPoint:
    time: Optional[datetime]
    lat: float
    lon: float
    ele: Optional[float] = None


def _parse_iso_time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_trkpt(elem: ET.Element) -> Optional[GpxPoint]:
    lat_s = elem.attrib.get("lat")
    lon_s = elem.attrib.get("lon")
    if lat_s is None or lon_s is None:
        return None
    try:
        lat, lon = float(lat_s), float(lon_s)
    except ValueError:
        return None
    ele: Optional[float] = None
    t: Optional[datetime] = None
    for child in elem:
        ln = _local_name(child.tag)
        if ln == "ele" and child.text:
            try:
                ele = float(child.text.strip())
            except ValueError:
                pass
        elif ln == "time" and child.text:
            t = _parse_iso_time(child.text.strip())
    return GpxPoint(time=t, lat=lat, lon=lon, ele=ele)


def gpx_times_look_utc(path: str, *, sample: int = 30) -> bool:
    """GPX 中多数 <time> 带 Z/时区时视为 UTC；否则多为本地 naive 时间。"""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return False
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    import re

    times = re.findall(r"<time>([^<]+)</time>", text, flags=re.I)[:sample]
    if not times:
        return False
    utc_like = sum(
        1 for t in times if "Z" in t.upper() or "+" in t or "-" in t[10:]
    )
    return utc_like >= max(1, len(times) // 2)


def load_gpx(path: str) -> List[GpxPoint]:
    """加载单个 GPX 文件中的 trkpt / rtept / wpt 点（按文件顺序）。"""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"GPX 不存在: {p}")
    tree = ET.parse(str(p))
    root = tree.getroot()
    points: List[GpxPoint] = []
    for elem in root.iter():
        ln = _local_name(elem.tag)
        if ln in ("trkpt", "rtept", "wpt"):
            pt = _parse_trkpt(elem)
            if pt is not None:
                points.append(pt)
    return _sort_points(points)


def _sort_points(points: List[GpxPoint]) -> List[GpxPoint]:
    with_t = [p for p in points if p.time is not None]
    without_t = [p for p in points if p.time is None]
    with_t.sort(key=lambda x: x.time)  # type: ignore[arg-type]
    return with_t + without_t


def load_gpx_many(paths: Sequence[str]) -> List[GpxPoint]:
    """加载多个 GPX 并按时间合并为一条轨迹。"""
    merged: List[GpxPoint] = []
    valid = resolve_gpx_path_list(gpx_paths=paths)
    if not valid:
        raise FileNotFoundError("未找到有效的 GPX 文件")
    for gp in valid:
        merged.extend(load_gpx(gp))
    merged = _sort_points(merged)
    if not merged:
        raise ValueError("GPX 合并结果为空，请检查是否含 trkpt/rtept/wpt")
    return merged


def merge_gpx_files(paths: Sequence[str], out_path: str) -> str:
    """合并多个 GPX 为一条轨迹并写入 out_path。"""
    merged = load_gpx_many(paths)

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="birdy-gpx-merge" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk><name>Merged</name><trkseg>",
    ]
    for pt in merged:
        t_attr = ""
        if pt.time is not None:
            t_attr = f"<time>{pt.time.isoformat()}Z</time>"
        ele_attr = ""
        if pt.ele is not None:
            ele_attr = f"<ele>{pt.ele:.2f}</ele>"
        lines.append(
            f'    <trkpt lat="{pt.lat:.7f}" lon="{pt.lon:.7f}">'
            f"{ele_attr}{t_attr}</trkpt>"
        )
    lines.extend(["  </trkseg></trk>", "</gpx>"])
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
