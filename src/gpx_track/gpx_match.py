# -*- coding: utf-8 -*-
"""
照片 EXIF 时间 ↔ GPX 时间 匹配（按 IANA 时区换算到 UTC 时间轴）

默认：EXIF 视为 Asia/Shanghai，GPX 视为 UTC。
在统一 UTC 时间轴上插值 GPX 得到 (lat, lon, ele)。
"""

from __future__ import annotations

import math
import os
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .gpx_io import GpxPoint, load_gpx, load_gpx_many, resolve_gpx_path_list
from .timezone_util import (
    DEFAULT_EXIF_TZ,
    DEFAULT_GPX_TZ,
    LOCAL_TZ,
    normalize_tz_name,
    tz_label,
    wall_clock_to_utc_naive,
)

# 兼容旧 GUI 导入
TZ_LABELS = {}  # deprecated

_PREFER_EXIF_GPS_KM = 2.0


@dataclass(frozen=True)
class TrackTimeAlignment:
    exif_tz: str = DEFAULT_EXIF_TZ
    gpx_tz: str = DEFAULT_GPX_TZ
    photo_offset_hours: Optional[int] = None
    gpx_offset_hours: Optional[int] = None

    def photo_to_timeline(self, when_exif: datetime) -> datetime:
        if self.photo_offset_hours is not None:
            return when_exif - timedelta(hours=int(self.photo_offset_hours))
        return wall_clock_to_utc_naive(when_exif, self.exif_tz)

    def gpx_to_timeline(self, when_gpx: datetime) -> datetime:
        if self.gpx_offset_hours is not None:
            return when_gpx - timedelta(hours=int(self.gpx_offset_hours))
        return wall_clock_to_utc_naive(when_gpx, self.gpx_tz)


def alignment_from_tz(
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> TrackTimeAlignment:
    return TrackTimeAlignment(
        exif_tz=normalize_tz_name(exif_tz),
        gpx_tz=normalize_tz_name(gpx_tz),
    )


def describe_time_alignment(align: TrackTimeAlignment) -> str:
    exif_l = tz_label(align.exif_tz)
    gpx_l = tz_label(align.gpx_tz)
    return f"EXIF → {exif_l}，GPX → {gpx_l}（对齐到 UTC）"


def photo_exif_to_utc_naive(when: datetime, tz_name: str) -> datetime:
    return wall_clock_to_utc_naive(when, tz_name)


def _photo_time(path: str) -> Optional[datetime]:
    try:
        from record_submit.exif_read import read_datetime_original
    except ImportError:
        read_datetime_original = None  # type: ignore
    if read_datetime_original is not None:
        t = read_datetime_original(path)
        if t is not None:
            return t
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def _read_photo_gps(path: str) -> Optional[Tuple[float, float]]:
    try:
        from record_submit.exif_read import read_gps_from_image

        return read_gps_from_image(path)
    except Exception:
        return None


def _times_index(track: Sequence[GpxPoint]) -> Tuple[List[datetime], List[GpxPoint]]:
    ts: List[datetime] = []
    pts: List[GpxPoint] = []
    for p in track:
        if p.time is not None:
            ts.append(p.time)
            pts.append(p)
    return ts, pts


def _timeline_index(
    track: Sequence[GpxPoint], align: TrackTimeAlignment
) -> Tuple[List[datetime], List[GpxPoint]]:
    ts: List[datetime] = []
    pts: List[GpxPoint] = []
    for p in track:
        if p.time is not None:
            ts.append(align.gpx_to_timeline(p.time))
            pts.append(p)
    return ts, pts


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


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


def _interp_on_timeline(
    ts: Sequence[datetime],
    pts: Sequence[GpxPoint],
    when_tl: datetime,
) -> Optional[Tuple[float, float, Optional[float], float]]:
    """在 GPX 时间轴上插值；超出范围时用最近端点（delta_s 记录实际时间差）。"""
    if not ts:
        return None
    # GPX 开晚了：照片早于首个轨迹点 → 用首点坐标，delta_s = 时间差
    if when_tl <= ts[0]:
        p = pts[0]
        return (p.lat, p.lon, p.ele, abs((ts[0] - when_tl).total_seconds()))
    # GPX 关早了：照片晚于末尾轨迹点 → 用末点坐标，delta_s = 时间差
    if when_tl >= ts[-1]:
        p = pts[-1]
        return (p.lat, p.lon, p.ele, abs((ts[-1] - when_tl).total_seconds()))
    i = bisect_left(ts, when_tl)
    if i == 0:
        p = pts[0]
        return (p.lat, p.lon, p.ele, abs((ts[0] - when_tl).total_seconds()))
    t0, t1 = ts[i - 1], ts[i]
    p0, p1 = pts[i - 1], pts[i]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        p = pts[i]
        return (p.lat, p.lon, p.ele, abs((t1 - when_tl).total_seconds()))
    r = (when_tl - t0).total_seconds() / span
    lat = p0.lat + (p1.lat - p0.lat) * r
    lon = p0.lon + (p1.lon - p0.lon) * r
    ele = None
    if p0.ele is not None and p1.ele is not None:
        ele = p0.ele + (p1.ele - p0.ele) * r
    elif p0.ele is not None:
        ele = p0.ele
    elif p1.ele is not None:
        ele = p1.ele
    delta = min(
        abs((t0 - when_tl).total_seconds()), abs((t1 - when_tl).total_seconds())
    )
    return (lat, lon, ele, delta)


def _resolve_lat_lon(
    path: str,
    lat_gpx: float,
    lon_gpx: float,
) -> Tuple[float, float, str]:
    gps = _read_photo_gps(path)
    if gps:
        glat, glon = gps
        if _haversine_km(glat, glon, lat_gpx, lon_gpx) <= _PREFER_EXIF_GPS_KM:
            return glat, glon, "exif_gps"
    return lat_gpx, lon_gpx, "gpx_interp"


def interpolate_track_at(
    track: Sequence[GpxPoint], when: datetime
) -> Optional[Tuple[float, float, Optional[float]]]:
    ts, pts = _times_index(track)
    if not ts:
        return None
    got = _interp_on_timeline(ts, pts, when)
    if got is None:
        return None
    lat, lon, ele, _ = got
    return (lat, lon, ele)


def interpolate_track_at_aligned(
    track: Sequence[GpxPoint],
    when_exif: datetime,
    align: TrackTimeAlignment,
) -> Optional[Tuple[float, float, Optional[float], float]]:
    when_tl = align.photo_to_timeline(when_exif)
    ts, pts = _timeline_index(track, align)
    return _interp_on_timeline(ts, pts, when_tl)


def detect_track_time_alignment(
    track: Sequence[GpxPoint],
    image_paths: Sequence[str],
    *,
    gpx_path: Optional[str] = None,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
    **kwargs,
) -> TrackTimeAlignment:
    return alignment_from_tz(exif_tz, gpx_tz)


def detect_photo_gpx_offset_hours(
    track: Sequence[GpxPoint],
    image_paths: Sequence[str],
    **kwargs,
) -> int:
    exif_tz = kwargs.get("exif_tz", DEFAULT_EXIF_TZ)
    gpx_tz = kwargs.get("gpx_tz", DEFAULT_GPX_TZ)
    align = alignment_from_tz(exif_tz, gpx_tz)
    ref = datetime(2024, 1, 15, 12, 0, 0)
    return int(
        (ref - align.photo_to_timeline(ref)).total_seconds() // 3600
    )


def match_photos_to_track(
    image_paths: Sequence[str],
    track: Sequence[GpxPoint],
    max_delta_seconds: float = 24 * 3600,
    *,
    tz_offset_hours: Optional[int] = None,
    alignment: Optional[TrackTimeAlignment] = None,
    gpx_path: Optional[str] = None,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> List[Dict]:
    if alignment is None:
        if tz_offset_hours is not None:
            alignment = TrackTimeAlignment(
                photo_offset_hours=int(tz_offset_hours), gpx_offset_hours=0
            )
        else:
            alignment = alignment_from_tz(exif_tz, gpx_tz)

    ts_tl, pts = _timeline_index(track, alignment)
    out: List[Dict] = []
    if not ts_tl:
        return out

    for path in image_paths:
        when = _photo_time(path)
        if when is None:
            continue
        when_tl = alignment.photo_to_timeline(when)
        got = _interp_on_timeline(ts_tl, pts, when_tl)
        if got is None:
            continue
        lat, lon, ele, delta_s = got
        if delta_s > max_delta_seconds:
            continue
        lat, lon, pos_src = _resolve_lat_lon(path, lat, lon)
        out.append(
            {
                "path": path,
                "time": when,
                "when_track": when_tl,
                "exif_tz": alignment.exif_tz,
                "gpx_tz": alignment.gpx_tz,
                "lat": lat,
                "lon": lon,
                "ele": ele,
                "delta_s": delta_s,
                "pos_source": pos_src,
            }
        )
    return out


try:
    from geo_encoder import write_gps_exif
except ImportError:
    write_gps_exif = None  # type: ignore


def batch_write_gps_from_gpx(
    image_folder: str,
    gpx_path: str = "",
    *,
    gpx_paths: Optional[Sequence[str]] = None,
    max_delta_hours: float = 48.0,
    recursive: bool = True,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> Dict[str, int]:
    if write_gps_exif is None:
        raise RuntimeError("geo_encoder.write_gps_exif 不可用")

    paths = resolve_gpx_path_list(gpx_path, gpx_paths)
    if not paths:
        raise FileNotFoundError("未指定有效的 GPX 文件")
    track = load_gpx_many(paths)
    ts, _ = _times_index(track)
    if not ts:
        raise ValueError("GPX 轨迹点缺少时间戳，无法与照片匹配")

    folder = Path(image_folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    exts = {".jpg", ".jpeg"}
    files: List[str] = []
    it = folder.rglob("*") if recursive else folder.iterdir()
    for f in it:
        if f.is_file() and f.suffix.lower() in exts:
            files.append(str(f))

    align = alignment_from_tz(exif_tz, gpx_tz)
    matched = 0
    written = 0
    skipped_no_time = 0
    skipped_delta = 0
    endpoint_matched = 0
    max_delta = max_delta_hours * 3600.0

    ts_tl, _ = _timeline_index(track, align)

    for path in files:
        when = _photo_time(path)
        if when is None:
            skipped_no_time += 1
            continue
        got = interpolate_track_at_aligned(track, when, align)
        if got is None:
            skipped_delta += 1
            continue
        lat, lon, ele, delta_s = got
        if delta_s > max_delta:
            skipped_delta += 1
            continue
        # 判断是否为端点匹配（照片时间在 GPX 范围之外）
        when_tl = align.photo_to_timeline(when)
        if ts_tl and (when_tl < ts_tl[0] or when_tl > ts_tl[-1]):
            endpoint_matched += 1
        lat, lon, _ = _resolve_lat_lon(path, lat, lon)
        matched += 1
        if write_gps_exif(path, lat, lon, ele, verbose=False):
            written += 1

    if endpoint_matched > 0:
        print(
            f"  GPX 匹配：{endpoint_matched} 张照片在 GPX 时间范围外，"
            f"已按最近端点坐标匹配（时间差 ≤ {max_delta_hours:.0f} 小时）"
        )

    return {
        "total": len(files),
        "matched": matched,
        "written": written,
        "skipped": len(files) - matched,
        "skipped_no_time": skipped_no_time,
        "skipped_delta": skipped_delta,
        "endpoint_matched": endpoint_matched,
    }
