# -*- coding: utf-8 -*-
"""观鸟记录导出：由 GPX 轨迹按 checklist 时间窗计算 Dist Traveled (Miles)。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from gpx_track.gpx_io import GpxPoint, load_gpx_many, resolve_gpx_path_list
from gpx_track.gpx_match import (
    TrackTimeAlignment,
    _haversine_km,
    _timeline_index,
    alignment_from_tz,
)
from gpx_track.timezone_util import DEFAULT_EXIF_TZ, DEFAULT_GPX_TZ, normalize_tz_name

from .scan import ChecklistBucket

_KM_TO_MI = 0.621371


def track_distance_km_in_time_window(
    track: Sequence[GpxPoint],
    start: datetime,
    end: datetime,
    *,
    alignment: Optional[TrackTimeAlignment] = None,
) -> float:
    """
    沿 GPX 轨迹累计 [start, end] 时间窗内路程（公里）。
    与观鸟轨迹图相同的时间轴对齐（EXIF/GPX 时区）。
    """
    if not track or start is None or end is None:
        return 0.0
    if end < start:
        start, end = end, start
    align = alignment or TrackTimeAlignment()
    ts, pts = _timeline_index(track, align)
    if len(ts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(ts)):
        t0, t1 = ts[i - 1], ts[i]
        if t1 < start:
            continue
        if t0 > end:
            break
        if t1 >= start and t0 <= end:
            p0, p1 = pts[i - 1], pts[i]
            total += _haversine_km(p0.lat, p0.lon, p1.lat, p1.lon)
    return total


def format_distance_miles(km: float) -> str:
    if km <= 0:
        return ""
    mi = km * _KM_TO_MI
    if mi < 0.01:
        return "0.01"
    s = f"{mi:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0.01"


class GpxTravelDistanceResolver:
    """按 GPX 与照片时间窗计算每个 checklist 的行驶里程（英里）。"""

    def __init__(
        self,
        gpx_path: str = "",
        *,
        gpx_paths: Optional[Sequence[str]] = None,
        exif_tz: str = DEFAULT_EXIF_TZ,
        gpx_tz: str = DEFAULT_GPX_TZ,
    ) -> None:
        paths = resolve_gpx_path_list(gpx_path, gpx_paths)
        if not paths:
            raise FileNotFoundError("未指定有效的 GPX 文件")
        self._track = load_gpx_many(paths)
        self._align = alignment_from_tz(
            normalize_tz_name(exif_tz),
            normalize_tz_name(gpx_tz),
        )

    def miles_for_bucket(self, bucket: ChecklistBucket) -> Optional[str]:
        if not self._track or bucket.start_time is None:
            return None
        end = bucket.end_time or bucket.start_time
        km = track_distance_km_in_time_window(
            self._track,
            bucket.start_time,
            end,
            alignment=self._align,
        )
        if km <= 0:
            return None
        return format_distance_miles(km)


def try_create_gpx_resolver(
    gpx_file_path: Optional[str] = None,
    *,
    gpx_file_paths: Optional[Sequence[str]] = None,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> Optional[GpxTravelDistanceResolver]:
    paths = resolve_gpx_path_list(gpx_file_path, gpx_file_paths)
    if not paths:
        return None
    return GpxTravelDistanceResolver(
        gpx_paths=paths,
        exif_tz=exif_tz,
        gpx_tz=gpx_tz,
    )
