# -*- coding: utf-8 -*-
"""观鸟记录导出文件名：观测日期/时间 + 英文地理标识 + 导出时刻，避免覆盖未上传文件。"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Set

from .scan import ChecklistBucket

_ASCII_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _ascii_slug(part: str, *, max_len: int = 40, fallback: str = "unknown") -> str:
    s = (part or "").strip()
    if not s:
        return fallback
    s = _ASCII_SLUG_RE.sub("_", s).strip("_")
    return (s[:max_len] if s else fallback) or fallback


def coords_geo_slug(
    lat: Optional[float],
    lon: Optional[float],
    *,
    region_code: str = "",
) -> str:
    """
    英文地理片段：``lat24p5919_lon117p9492``；无坐标时用 eBird 区域码或 ``no_coords``。
    """
    if lat is not None and lon is not None:
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        la = f"{ns}{abs(lat):.4f}".replace(".", "p")
        lo = f"{ew}{abs(lon):.4f}".replace(".", "p")
        return f"{la}_{lo}"
    rc = _ascii_slug(region_code, max_len=24, fallback="")
    if rc:
        return f"region_{rc}"
    return "no_coords"


def checklist_export_slug(
    bucket: ChecklistBucket,
    *,
    region_code: str = "",
    export_moment: Optional[datetime] = None,
    seq: int = 0,
) -> str:
    """
    例如 ``20260504_0800_lat24p5919_lon117p9492_exp153045``。
    ``seq>0`` 时追加 ``_b02``，避免同秒同地点多 checklist 重名。
    """
    if bucket.day_end is not None and bucket.day_end != bucket.day:
        date_s = (
            f"{bucket.day.strftime('%Y%m%d')}_{bucket.day_end.strftime('%Y%m%d')}"
        )
    else:
        date_s = bucket.day.strftime("%Y%m%d")
    if bucket.start_time is not None:
        time_s = bucket.start_time.strftime("%H%M")
    else:
        time_s = "1200"
    geo = coords_geo_slug(bucket.lat, bucket.lon, region_code=region_code)
    exp = (export_moment or datetime.now()).strftime("%H%M%S")
    slug = f"{date_s}_{time_s}_{geo}_exp{exp}"
    if seq > 0:
        slug = f"{slug}_b{seq:02d}"
    return slug


def unique_checklist_slug(
    bucket: ChecklistBucket,
    used: Set[str],
    *,
    region_code: str = "",
    export_moment: Optional[datetime] = None,
) -> str:
    """在 ``used`` 中保证唯一的 checklist slug。"""
    seq = 0
    while True:
        slug = checklist_export_slug(
            bucket,
            region_code=region_code,
            export_moment=export_moment,
            seq=seq,
        )
        if slug not in used:
            used.add(slug)
            return slug
        seq += 1
