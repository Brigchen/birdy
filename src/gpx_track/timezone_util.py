# -*- coding: utf-8 -*-
"""EXIF / GPX 时间时区：IANA 名称与统一 UTC 时间轴换算。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo, available_timezones

LOCAL_TZ = "local"
DEFAULT_EXIF_TZ = "Asia/Shanghai"
DEFAULT_GPX_TZ = "UTC"

_LEGACY_MAP = {
    "beijing": "Asia/Shanghai",
    "cst": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "utc": "UTC",
    "gmt": "UTC",
    "local": LOCAL_TZ,
    "none": LOCAL_TZ,
}

_COMMON_TZ = [
    "Asia/Shanghai",
    "UTC",
    "Asia/Hong_Kong",
    "Asia/Taipei",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Asia/Seoul",
    "Asia/Kolkata",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "Australia/Sydney",
    "Pacific/Auckland",
]


def normalize_tz_name(name: Optional[str]) -> str:
    if not name or not str(name).strip():
        return DEFAULT_EXIF_TZ
    raw = str(name).strip()
    low = raw.lower()
    if low in _LEGACY_MAP:
        return _LEGACY_MAP[low]
    if low == LOCAL_TZ:
        return LOCAL_TZ
    try:
        ZoneInfo(raw)
        return raw
    except Exception:
        pass
    for z in available_timezones():
        if z.lower() == low:
            return z
    return raw


def wall_clock_to_utc_naive(when: datetime, tz_name: str) -> datetime:
    """将无时区时间视为 tz_name 当地墙钟，转为 UTC 的 naive datetime。"""
    tz = normalize_tz_name(tz_name)
    if tz == LOCAL_TZ:
        return when
    if when.tzinfo is not None:
        return when.astimezone(timezone.utc).replace(tzinfo=None)
    z = ZoneInfo(tz)
    return when.replace(tzinfo=z).astimezone(timezone.utc).replace(tzinfo=None)


def tz_label(tz_name: str, ref: Optional[datetime] = None) -> str:
    tz = normalize_tz_name(tz_name)
    if tz == LOCAL_TZ:
        return "不转换（同一时钟）"
    ref = ref or datetime.now()
    try:
        z = ZoneInfo(tz)
        aware = ref.replace(tzinfo=z)
        off = aware.utcoffset()
        if off is None:
            return tz
        secs = int(off.total_seconds())
        sign = "+" if secs >= 0 else "-"
        secs = abs(secs)
        h, rem = divmod(secs, 3600)
        m = rem // 60
        if m:
            return f"{tz} (UTC{sign}{h}:{m:02d})"
        return f"{tz} (UTC{sign}{h})"
    except Exception:
        return tz


def timezone_combo_entries() -> List[Tuple[str, str]]:
    """(显示文本, 时区 id) 供 QComboBox 使用。"""
    out: List[Tuple[str, str]] = [(tz_label(LOCAL_TZ), LOCAL_TZ)]
    seen = {LOCAL_TZ}
    for z in _COMMON_TZ:
        z = normalize_tz_name(z)
        if z not in seen:
            out.append((tz_label(z), z))
            seen.add(z)
    for z in sorted(available_timezones()):
        if z in seen:
            continue
        out.append((tz_label(z), z))
        seen.add(z)
    return out


def set_combo_timezone(combo, tz_name: str) -> None:
    """将 QComboBox 设为指定时区（按 itemData 或文本匹配）。"""
    tz = normalize_tz_name(tz_name)
    idx = combo.findData(tz)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return
    for i in range(combo.count()):
        if combo.itemData(i) == tz or combo.itemText(i) == tz:
            combo.setCurrentIndex(i)
            return
    combo.setEditText(tz_label(tz) if tz != LOCAL_TZ else tz_label(LOCAL_TZ))


def read_combo_timezone(combo) -> str:
    data = combo.currentData()
    if data:
        return normalize_tz_name(str(data))
    text = (combo.currentText() or "").strip()
    if not text:
        return DEFAULT_EXIF_TZ
    for i in range(combo.count()):
        if combo.itemText(i) == text and combo.itemData(i):
            return normalize_tz_name(str(combo.itemData(i)))
    if text.startswith("不转换"):
        return LOCAL_TZ
    return normalize_tz_name(text)
