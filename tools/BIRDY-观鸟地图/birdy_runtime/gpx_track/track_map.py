# -*- coding: utf-8 -*-
"""生成观鸟行迹图、物种分布图与海拔剖面（PNG）。"""

from __future__ import annotations

import math
import os
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AnnotationBbox,
    HPacker,
    OffsetImage,
    TextArea,
)
from PIL import Image, ImageDraw, ImageFilter

from .amap_basemap import (
    fetch_amap_basemap_rgba,
    fit_lon_to_aspect,
    geocode_address_wgs84,
    is_dark_basemap_style,
    is_satellite_basemap_style,
    normalize_basemap_style,
    gcj_bounds_from_lonlats,
    regeo_address_components,
    regeo_place_label,
    region_city_tokens,
    reserve_south_lat,
    wgs84_to_map_lonlat,
)
from .gpx_io import GpxPoint, load_gpx_many, resolve_gpx_path_list
from .gpx_match import (
    DEFAULT_EXIF_TZ,
    DEFAULT_GPX_TZ,
    TrackTimeAlignment,
    _times_index,
    alignment_from_tz,
    describe_time_alignment,
    match_photos_to_track,
)
from .photo_collect import BirdPhoto, collect_bird_photos
from .timezone_util import wall_clock_to_utc_naive

# 图内标题日期统一显示为北京时间
MAP_DISPLAY_TZ = ZoneInfo("Asia/Shanghai")

# 手机 2K 竖屏正式导出（宽 × 高，像素）
EXPORT_WIDTH_PX = 1440
EXPORT_HEIGHT_PX = 2560
EXPORT_DPI = 120
PREVIEW_WIDTH_PX = EXPORT_WIDTH_PX
PREVIEW_HEIGHT_PX = EXPORT_HEIGHT_PX
PREVIEW_DPI = EXPORT_DPI

# 照片 EXIF 时刻与 GPX 时间轴相差超过此值（秒）则不绘制记录点
DEFAULT_GPX_MATCH_MAX_DELTA_S = 60 * 60  # 1 小时：GPX 开晚了/关早了仍可按最近端点匹配
# 正式导出 PNG 时鸟图标注上限（避免画布过大 + 大量缩略图导致内存崩溃）
EXPORT_MAX_MARKERS = 150
# 预览模式鸟图标注上限
PREVIEW_MAX_MARKERS = 40

# 地图鸟图、物种标注与海拔文字相对原尺寸放大约 1/3
_MARKER_SIZE_SCALE = 4 / 3
# 裁圆前取画面中心正方形边长 = min(宽,高) × 此比例（1.0=短边铺满，避免只剩局部鸟体）
_THUMB_CENTER_CROP_RATIO = 1.0
# 物种名距地图左右边界至少为地图宽度的此比例
_MAP_LABEL_X_MARGIN_FRAC = 1 / 50.0
# 物种名字号 = 地图轴高度（像素）/ 此除数
_MAP_SPECIES_LABEL_HEIGHT_DIV = 100.0
# 鸟图直径 = 整图高度 × 此比例（预览/导出一致）
_MAP_THUMB_HEIGHT_FRAC = 1 / 32.0
# 海拔剖面鸟图相对地图略小（原 48/64 比例）
_ELEV_THUMB_HEIGHT_FRAC = _MAP_THUMB_HEIGHT_FRAC * (48 / 64.0)

# 海拔剖面（绿色主题）
ELEV_LINE_COLOR = "#1B5E20"
ELEV_FILL_COLOR = "#D5F5E3"
ELEV_HIGHLIGHT_COLOR = "#27AE60"
ELEV_SPECIES_COLOR = "#E74C3C"
ELEV_PANEL_BORDER = "#BDBDBD"
ELEV_SPECIES_NAME_COLOR = "#1B5E20"
ELEV_LEADER_COLOR = "#7F8C8D"
ELEV_GRID_COLOR = "#D5D8DC"
ELEV_AXIS_COLOR = "#222222"
ELEV_LABEL_COLOR = "#333333"


def _up_marker_size(n: int | float) -> int:
    return max(1, round(float(n) * _MARKER_SIZE_SCALE))


def _track_map_thumb_diameters(height_px: int) -> Tuple[int, int]:
    """地图/海拔鸟图直径（px），相对整图高度固定比例，预览与导出一致。"""
    h = max(1, int(height_px))
    map_d = max(16, int(round(h * _MAP_THUMB_HEIGHT_FRAC)))
    elev_d = max(12, int(round(h * _ELEV_THUMB_HEIGHT_FRAC)))
    return map_d, elev_d


_CJK_FONT_CONFIGURED = False
_COUNT_NUM_FONT_RESOLVED: Optional[str] = None

_COUNT_NUM_FONT_CANDIDATES = (
    "Haettenschweiler",
    "Bahnschrift SemiBold Condensed",
    "Franklin Gothic Medium Condensed",
    "Arial Narrow",
)


def _configure_matplotlib_cjk() -> None:
    """配置 matplotlib 中文显示（Windows 优先微软雅黑/黑体）。"""
    global _CJK_FONT_CONFIGURED
    if _CJK_FONT_CONFIGURED:
        return
    chosen: Optional[str] = None
    win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    font_files = [
        win / "msyh.ttc",
        win / "msyhbd.ttc",
        win / "simhei.ttf",
        win / "simsun.ttc",
    ]
    for fp in font_files:
        if fp.is_file():
            try:
                fm.fontManager.addfont(str(fp))
                chosen = fm.FontProperties(fname=str(fp)).get_name()
                break
            except Exception:
                continue
    if not chosen:
        for f in fm.fontManager.ttflist:
            n = f.name or ""
            if any(
                k in n
                for k in ("YaHei", "SimHei", "PingFang", "Noto Sans CJK", "Source Han")
            ):
                chosen = n
                break
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "Arial"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_CONFIGURED = True


def _resolve_count_num_fontfamily() -> str:
    """种数数字：优先瘦高展示字体（Windows 常见 Haettenschweiler）。"""
    global _COUNT_NUM_FONT_RESOLVED
    if _COUNT_NUM_FONT_RESOLVED is not None:
        return _COUNT_NUM_FONT_RESOLVED
    known = {f.name.lower() for f in fm.fontManager.ttflist}
    for name in _COUNT_NUM_FONT_CANDIDATES:
        if name.lower() in known:
            _COUNT_NUM_FONT_RESOLVED = name
            return name
        for fam in known:
            if name.lower() in fam:
                _COUNT_NUM_FONT_RESOLVED = fam
                return fam
    _COUNT_NUM_FONT_RESOLVED = "sans-serif"
    return _COUNT_NUM_FONT_RESOLVED


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _photo_path_key(path: str) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path)


def _dedupe_photos_by_species_radius(
    photos: Sequence[BirdPhoto],
    radius_km: float,
    *,
    dropped: Optional[List[Tuple[str, str]]] = None,
) -> List[BirdPhoto]:
    radius_km = max(0.01, float(radius_km))
    picked: List[BirdPhoto] = []
    placed: Dict[str, List[Tuple[float, float]]] = {}
    for ph in photos:
        if ph.lat is None or ph.lon is None:
            continue
        sp = ph.species_cn
        ok = True
        for la, lo in placed.get(sp, []):
            if haversine_km(ph.lat, ph.lon, la, lo) < radius_km:
                ok = False
                break
        if ok:
            picked.append(ph)
            placed.setdefault(sp, []).append((ph.lat, ph.lon))
        elif dropped is not None:
            dropped.append(
                (
                    ph.path,
                    f"同种「{sp}」在 {radius_km:g} km 内已有标注，去重省略",
                )
            )
    return picked


def _track_cumulative_km(track: Sequence[GpxPoint]) -> Tuple[List[float], List[float]]:
    dist = [0.0]
    ele: List[float] = []
    prev: Optional[GpxPoint] = None
    for p in track:
        if p.ele is not None:
            ele.append(float(p.ele))
        else:
            ele.append(ele[-1] if ele else 0.0)
        if prev is not None:
            dist.append(dist[-1] + haversine_km(prev.lat, prev.lon, p.lat, p.lon))
        prev = p
    if len(ele) < len(dist):
        ele.extend([ele[-1]] * (len(dist) - len(ele)))
    return dist, ele[: len(dist)]


def _thumb_square_box(
    w: int, h: int, ratio: float = _THUMB_CENTER_CROP_RATIO
) -> Tuple[int, int, int, int]:
    """中心正方形：默认铺满短边，再裁成圆。"""
    side = min(max(1, int(w)), max(1, int(h)))
    crop = max(1, int(round(side * float(ratio))))
    crop = min(crop, int(w), int(h))
    left = (int(w) - crop) // 2
    top = (int(h) - crop) // 2
    return left, top, left + crop, top + crop


def _circular_thumb_rgba(path: str, diameter: int = 44) -> np.ndarray:
    """正圆裁切鸟图 + 柔和阴影，返回 RGBA [0,1]。"""
    d = max(16, int(diameter))
    pad = max(6, d // 5)
    canvas = Image.new("RGBA", (d + 2 * pad, d + 2 * pad), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((0, 0, d - 1, d - 1), fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, d // 12)))
    canvas.paste(shadow, (pad + 2, pad + 4), shadow)

    im = Image.open(path).convert("RGB")
    w, h = im.size
    left, top, right, bottom = _thumb_square_box(w, h)
    im = im.crop((left, top, right, bottom))
    im = im.resize((d, d), Image.Resampling.LANCZOS)
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, d - 1, d - 1), fill=255)
    circle = Image.new("RGBA", (d, d))
    circle.paste(im, (0, 0))
    circle.putalpha(mask)
    canvas.paste(circle, (pad, pad), circle)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _thumb_display_px(diameter: int, dpi: float) -> float:
    """OffsetImage zoom=d/边长 时，画布上的实际像素直径。"""
    return float(diameter) * max(float(dpi), 1.0) / 72.0


def _thumb_offset_zoom(arr: np.ndarray, diameter: int, dpi: float) -> float:
    """与 2.0.87 前一致：按点缩放，鸟图约 diameter×dpi/72 像素（比锁像素更大）。"""
    _ = dpi
    side = max(int(arr.shape[0]), int(arr.shape[1]), 1)
    return float(diameter) / float(side)


def _map_xy(lon: float, lat: float, *, use_gcj: bool) -> Tuple[float, float]:
    if use_gcj:
        return wgs84_to_map_lonlat(lon, lat)
    return lon, lat


def _to_beijing_wall(when: datetime, source_tz: str) -> datetime:
    """将 naive 时间按 source_tz 解释，转为北京时间 naive。"""
    utc = wall_clock_to_utc_naive(when, source_tz)
    return (
        utc.replace(tzinfo=timezone.utc)
        .astimezone(MAP_DISPLAY_TZ)
        .replace(tzinfo=None)
    )


def _observation_date_label(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    *,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> str:
    beijing_times: List[datetime] = []
    for p in track:
        if p.time is not None:
            beijing_times.append(_to_beijing_wall(p.time, gpx_tz))
    for ph in photos:
        if ph.when is not None:
            beijing_times.append(_to_beijing_wall(ph.when, exif_tz))
    if not beijing_times:
        t = datetime.now(MAP_DISPLAY_TZ).replace(tzinfo=None)
        return f"{t.year}年{t.month}月{t.day}日"
    days = {(t.year, t.month, t.day) for t in beijing_times}
    if len(days) <= 1:
        t = min(beijing_times)
        return f"{t.year}年{t.month}月{t.day}日"
    t0, t1 = min(beijing_times), max(beijing_times)
    if t0.year == t1.year and t0.month == t1.month:
        return f"{t0.year}年{t0.month}月{t0.day}日-{t1.day}日"
    if t0.year == t1.year:
        return f"{t0.year}年{t0.month}月{t0.day}日-{t1.month}月{t1.day}日"
    return f"{t0.year}年{t0.month}月{t0.day}日-{t1.year}年{t1.month}月{t1.day}日"


def _observation_time_range_label(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    *,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> str:
    beijing_times: List[datetime] = []
    for p in track:
        if p.time is not None:
            beijing_times.append(_to_beijing_wall(p.time, gpx_tz))
    for ph in photos:
        if ph.when is not None:
            beijing_times.append(_to_beijing_wall(ph.when, exif_tz))
    if not beijing_times:
        return ""
    t0, t1 = min(beijing_times), max(beijing_times)
    return f"{t0.strftime('%H:%M')}-{t1.strftime('%H:%M')}"


def _distinct_species_count(photos: Sequence[BirdPhoto]) -> int:
    names = {ph.species_cn.strip() for ph in photos if (ph.species_cn or "").strip()}
    return len(names)


def resolve_track_map_place(
    *,
    location_name: str = "",
    province: str = "",
    city: str = "",
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
) -> str:
    loc = (location_name or "").strip()
    if not loc:
        loc = f"{(province or '').strip()}{(city or '').strip()}".strip()
    if not loc:
        lons, lats = _collect_lonlats(track, photos)
        if lons:
            lon_c = (min(lons) + max(lons)) / 2
            lat_c = (min(lats) + max(lats)) / 2
            loc = regeo_place_label(lon_c, lat_c) or ""
    return loc


def resolve_track_map_titles(
    *,
    location_name: str = "",
    province: str = "",
    city: str = "",
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> Tuple[str, str, str]:
    """返回 (地点, 「观鸟记录」, 观测日期)。"""
    loc = resolve_track_map_place(
        location_name=location_name,
        province=province,
        city=city,
        track=track,
        photos=photos,
    )
    date_s = _observation_date_label(
        track, photos, exif_tz=exif_tz, gpx_tz=gpx_tz
    )
    return loc, "观鸟记录", date_s


def resolve_track_map_title(
    *,
    preview: bool,
    location_name: str = "",
    province: str = "",
    city: str = "",
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
) -> str:
    """兼容旧调用：地点 + 观鸟记录 + 日期。"""
    _ = preview
    place, map_title, date_s = resolve_track_map_titles(
        location_name=location_name,
        province=province,
        city=city,
        track=track,
        photos=photos,
        exif_tz=exif_tz,
        gpx_tz=gpx_tz,
    )
    parts = [p for p in (place, map_title, date_s) if p]
    return "\n".join(parts)


def _map_logo_target_width_frac(logo_width_ratio: float) -> float:
    """角标 Logo 宽度占地图宽比例（基于水印 logo_width_ratio，较全图水印更克制）。"""
    r = float(logo_width_ratio or 0.30)
    return min(0.16, max(0.06, r * 0.40))


def _load_signature_logo_rgba(
    logo_path: str,
    target_w_px: int,
    target_h_max_px: int,
    *,
    fg_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Optional[np.ndarray]:
    """加载签名 Logo（RGBA float 0–1，透明底剪影，默认白色与水印一致）。"""
    path = (logo_path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        lg = Image.open(path).convert("RGBA")
    except OSError:
        return None
    tw = max(12, int(target_w_px))
    th_max = max(12, int(target_h_max_px))
    lw, lh = lg.size
    if lw <= 0 or lh <= 0:
        return None
    scale = min(tw / lw, th_max / lh)
    nw = max(1, int(lw * scale))
    nh = max(1, int(lh * scale))
    lg = lg.resize((nw, nh), Image.Resampling.LANCZOS)
    alpha = lg.split()[-1].point(lambda p: int(p * 0.92))
    fg = Image.new("RGBA", lg.size, fg_rgb + (0,))
    fg.putalpha(alpha)
    return np.asarray(fg, dtype=np.float32) / 255.0


# 数字地图 / 网格底图：深绿字；卫星影像：白字
MAP_INK_GREEN = "#1B5E20"
MAP_INK_GREEN_RGB = (27, 94, 32)
MAP_INK_WHITE = "#FFFFFF"
MAP_INK_WHITE_RGB = (255, 255, 255)
MAP_ACCENT_SATELLITE = "#F1C40F"
MAP_MARGIN_TITLE_X = 1.0 / 20.0
MAP_MARGIN_TITLE_Y = 1.0 / 35.0
MAP_MARGIN_SUMMARY_X = 1.0 / 15.0
# 海拔内嵌面板占用 map_ax 底部约 [0.03, 0.17]（transAxes）
ELEV_PANEL_TOP_AXES = 0.17
# 鸟图半径之外再留一点空隙，避免圆标贴上海拔框
_ELEV_MAP_THUMB_GAP_AXES = 0.025
# 海拔外框内层绘图区 [left, bottom, width, height]
_ELEV_INNER_RECT = (0.04, 0.1, 0.92, 0.80)
# 海拔 data 区相对轨迹 x/y 范围的留白比例
_ELEV_X_LEFT_FRAC = 0.00
_ELEV_X_RIGHT_FRAC = 0.00
_ELEV_Y_BOTTOM_FRAC = 0.02
_ELEV_Y_TOP_FRAC = 0.1
# 鸟种名距海拔绘图区边缘 ≥ 宽/高的 1/50
_ELEV_LABEL_MARGIN_FRAC = 1 / 50.0
SUMMARY_GAP_AXES = 0.014
TITLE_GAP_AXES = 0.018
# 「N 种鸟」相对标题顶略低，与标题左右对称
_SUMMARY_BELOW_TITLE = 0.032


@dataclass
class TitleChromeLayout:
    ha: str
    y_top: float
    box: Tuple[float, float, float, float]


@dataclass
class MapMarkerLayout:
    displays: List[Tuple[float, float]]
    label_boxes_axes: List[Tuple[float, float, float, float]]
    thumb_boxes_axes: List[Tuple[float, float, float, float]]


@dataclass
class KeyPlace:
    name: str
    lat: float
    lon: float


_KEY_PLACE_SPLIT = re.compile(r"[;；\n、]+")
_KEY_PLACE_COMMA = re.compile(r"[,，]")
# 地点标注：钴蓝，区别于鸟名深绿/白字与橙色定位点
_KEY_PLACE_COLOR = "#1A5276"
_KEY_PLACE_COLOR_DARK = "#5DADE2"
_KEY_PLACE_LABEL_HEIGHT_DIV = 86.0  # 小于鸟名除数 → 地名字略大于鸟名


def split_key_place_queries(text: str) -> List[str]:
    """逗号/分号分隔多个地点；「纬度,经度」整段保留。"""
    raw = (text or "").strip()
    if not raw:
        return []
    out: List[str] = []
    for chunk in _KEY_PLACE_SPLIT.split(raw.replace("；", ";")):
        chunk = chunk.strip()
        if not chunk:
            continue
        from .amap_basemap import parse_lonlat_query

        if parse_lonlat_query(chunk) is not None:
            out.append(chunk)
            continue
        for part in _KEY_PLACE_COMMA.split(chunk):
            name = part.strip()
            if name:
                out.append(name)
    # 去重且保序
    seen = set()
    uniq: List[str] = []
    for name in out:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(name)
    return uniq


def resolve_key_place_search_region(
    *,
    location_name: str = "",
    province: str = "",
    city: str = "",
    track: Sequence[GpxPoint] = (),
    photos: Sequence[BirdPhoto] = (),
) -> Tuple[str, str, str, Optional[Tuple[float, float]]]:
    """关键地点检索用的省/市/区与地图中心 (lat, lon)。"""
    prov = (province or "").strip()
    cit = (city or "").strip()
    dist = ""
    center: Optional[Tuple[float, float]] = None
    lons, lats = _collect_lonlats(track, photos)
    if lons:
        lon_c = 0.5 * (min(lons) + max(lons))
        lat_c = 0.5 * (min(lats) + max(lats))
        center = (lat_c, lon_c)
        if not (prov and cit):
            rp, rc, rd = regeo_address_components(lon_c, lat_c)
            prov = prov or (rp or "")
            cit = cit or (rc or "")
            dist = (rd or "").strip()
    return prov, cit, dist, center


def geocode_key_places(
    text: str,
    *,
    city: str = "",
    province: str = "",
    location_name: str = "",
    district: str = "",
    prefer_wgs84: Optional[Tuple[float, float]] = None,
    max_prefer_km: float = 80.0,
) -> Tuple[List[KeyPlace], List[str]]:
    """按地图所在城市/区域查询地名。返回 (成功列表, 失败地名)。"""
    found: List[KeyPlace] = []
    failed: List[str] = []
    hints = region_city_tokens(city, district, location_name, province)
    primary_city = (city or "").strip() or (hints[0] if hints else "")
    for name in split_key_place_queries(text):
        coords = geocode_address_wgs84(
            name,
            city=primary_city,
            region_hints=hints,
            prefer_wgs84=prefer_wgs84,
            max_prefer_km=max_prefer_km,
        )
        if coords is None:
            failed.append(name)
            continue
        lat, lon = coords
        found.append(KeyPlace(name=name, lat=float(lat), lon=float(lon)))
    return found, failed


def partition_key_places_by_view(
    bounds: Optional[Tuple[float, float, float, float]],
    places: Sequence[KeyPlace],
    *,
    pad_frac: float = 0.02,
) -> Tuple[List[KeyPlace], List[KeyPlace]]:
    """按经纬度是否落在当前地图范围内拆分；超范围常见于重名查错。"""
    if not places:
        return [], []
    if bounds is None:
        return list(places), []
    x0, x1, y0, y1 = bounds
    dx = max(x1 - x0, 1e-4) * float(pad_frac)
    dy = max(y1 - y0, 1e-4) * float(pad_frac)
    inside: List[KeyPlace] = []
    outside: List[KeyPlace] = []
    for place in places:
        ok = (
            x0 - dx <= place.lon <= x1 + dx
            and y0 - dy <= place.lat <= y1 + dy
        )
        (inside if ok else outside).append(place)
    return inside, outside


def format_key_place_alerts(written: Mapping[str, str]) -> str:
    """GUI 弹窗：查无坐标，或经纬不在当前地图范围内。"""
    parts: List[str] = []
    failed = (written.get("key_places_failed") or "").strip()
    if failed:
        parts.append(f"未能在地图所在城市/区域查到：{failed}")
    oor = (written.get("key_places_out_of_range") or "").strip()
    if oor:
        parts.append(
            "下列地点不在当前地图范围内，未标注"
            "（常见原因：地名重名查到了别处坐标）：\n"
            + oor.replace("；", "\n")
        )
    return "\n\n".join(parts)


def _is_satellite_basemap(basemap_style: str) -> bool:
    return is_satellite_basemap_style(basemap_style)


def _map_ink(
    basemap_style: str,
    *,
    on_basemap: bool = True,
) -> Tuple[str, Tuple[int, int, int], bool]:
    """返回 (文字色, Logo RGB, 是否加暗描边)。"""
    if on_basemap and is_dark_basemap_style(basemap_style):
        return MAP_INK_WHITE, MAP_INK_WHITE_RGB, True
    return MAP_INK_GREEN, MAP_INK_GREEN_RGB, False


def _map_accent_color(basemap_style: str, *, on_basemap: bool = True) -> str:
    if on_basemap and _is_satellite_basemap(basemap_style):
        return MAP_ACCENT_SATELLITE
    if on_basemap and is_dark_basemap_style(basemap_style):
        return "#F5D76E"
    return MAP_INK_GREEN


def _map_summary_style(
    basemap_style: str,
    *,
    on_basemap: bool = True,
) -> Tuple[str, str, List, str]:
    """左下种数：与标题同色——深色/卫星白字，浅色数字地图深绿。"""
    num_ff = _resolve_count_num_fontfamily()
    color, _, use_stroke = _map_ink(basemap_style, on_basemap=on_basemap)
    effects = _map_text_effects(use_stroke=use_stroke)
    return color, color, effects, num_ff


def _map_text_effects(*, use_stroke: bool) -> List:
    if not use_stroke:
        return []
    return [pe.withStroke(linewidth=2.2, foreground="#000000", alpha=0.42)]


def _ax_size_px(ax) -> Tuple[float, float]:
    fig = ax.figure
    pos = ax.get_position()
    ax_w_px = max(1.0, fig.get_figwidth() * fig.dpi * pos.width)
    ax_h_px = max(1.0, fig.get_figheight() * fig.dpi * pos.height)
    return ax_w_px, ax_h_px


def _font_pt_for_line_height_px(line_height_px: float, dpi: float) -> float:
    """将目标字高（px）换算为 matplotlib fontsize（pt）。"""
    return max(4.0, line_height_px * 72.0 / max(dpi, 1.0))


def _map_typography(ax) -> Dict[str, float]:
    """地图图内文字/Logo 尺寸：相对整图高度按比例自适应。"""
    _, ax_h_px = _ax_size_px(ax)
    dpi = ax.figure.dpi
    return {
        "title_pt": _font_pt_for_line_height_px(ax_h_px / 35.0, dpi),
        "date_pt": _font_pt_for_line_height_px(ax_h_px / 50.0, dpi),
        "count_num_pt": _font_pt_for_line_height_px(ax_h_px / 16.0, dpi),
        "count_suffix_pt": _font_pt_for_line_height_px(ax_h_px / 80.0, dpi),
        "species_pt": _font_pt_for_line_height_px(
            ax_h_px / _MAP_SPECIES_LABEL_HEIGHT_DIV, dpi
        ),
        "attribution_pt": _font_pt_for_line_height_px(ax_h_px / 100.0, dpi),
        "elev_axis_pt": _font_pt_for_line_height_px(ax_h_px / 150.0, dpi),
        "elev_species_pt": _font_pt_for_line_height_px(ax_h_px / 160.0, dpi),
        "logo_h_px": ax_h_px / 30.0,
        "ax_h_px": ax_h_px,
    }


def _data_to_axes_frac(ax, x: float, y: float) -> Tuple[float, float]:
    pt = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
    return float(pt[0]), float(pt[1])


def _data_box_to_axes_frac(
    ax, x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float, float, float]:
    ax0, ay0 = _data_to_axes_frac(ax, x0, y0)
    ax1, ay1 = _data_to_axes_frac(ax, x1, y1)
    return (
        min(ax0, ax1),
        min(ay0, ay1),
        max(ax0, ax1),
        max(ay0, ay1),
    )


def _text_height_axes_frac(ax, fontsize_pt: float) -> float:
    _, ax_h_px = _ax_size_px(ax)
    h_px = fontsize_pt * ax.figure.dpi / 72.0 * 1.14
    return h_px / max(ax_h_px, 1.0)


def _label_box_axes_frac(
    ax,
    dx: float,
    dy: float,
    name: str,
    side: int,
    off_x_pt: float,
    off_y_pt: float,
    label_fs: float,
) -> Tuple[float, float, float, float]:
    tx, ty = _offset_points_to_data(ax, dx, dy, float(side) * off_x_pt, off_y_pt)
    w = _text_width_data(ax, name, label_fs)
    h = _text_height_axes_frac(ax, label_fs) * max(
        ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9
    )
    if side > 0:
        x0, x1 = tx, tx + w
    else:
        x0, x1 = tx - w, tx
    y0, y1 = ty - h * 0.5, ty + h * 0.5
    return _data_box_to_axes_frac(ax, x0, y0, x1, y1)


def _circle_box_axes_frac(
    ax, cx: float, cy: float, r_data: float
) -> Tuple[float, float, float, float]:
    return _data_box_to_axes_frac(
        ax, cx - r_data, cy - r_data, cx + r_data, cy + r_data
    )


def _rect_overlap_axes_frac(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _track_obstacle_boxes_axes(
    ax,
    track: Sequence[GpxPoint],
    *,
    use_gcj: bool,
    x_frac_min: float = 0.0,
    x_frac_max: float = 0.58,
    y_frac_min: float = 0.0,
    y_frac_max: float = 0.45,
    pad_frac: float = 0.016,
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for p in track:
        x, y = _map_xy(p.lon, p.lat, use_gcj=use_gcj)
        fx, fy = _data_to_axes_frac(ax, x, y)
        if fx < x_frac_min or fx > x_frac_max or fy < y_frac_min or fy > y_frac_max:
            continue
        boxes.append(
            (fx - pad_frac, fy - pad_frac, fx + pad_frac, fy + pad_frac)
        )
    return boxes


def _corner_box(
    ha: str, y_top: float, block_w: float, block_h: float, *, margin_x: float
) -> Tuple[float, float, float, float]:
    y0 = y_top - block_h
    if ha == "right":
        x1 = 1.0 - margin_x
        return (x1 - block_w, y0, x1, y_top)
    if ha == "center":
        x0 = max(margin_x, 0.5 - block_w * 0.5)
        return (x0, y0, x0 + block_w, y_top)
    x0 = margin_x
    return (x0, y0, x0 + block_w, y_top)


def _rect_separation(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    if _rect_overlap_axes_frac(a, b):
        return 0.0
    dx = max(0.0, max(b[0] - a[2], a[0] - b[2]))
    dy = max(0.0, max(b[1] - a[3], a[1] - b[3]))
    if dx > 0 and dy > 0:
        return math.hypot(dx, dy)
    return dx + dy


def _box_clearance(
    box: Tuple[float, float, float, float],
    obstacles: Sequence[Tuple[float, float, float, float]],
    *,
    y_min: float,
    y_max: float,
) -> float:
    edge = min(box[0], 1.0 - box[2], box[1] - y_min, y_max - box[3])
    if obstacles:
        edge = min(edge, min(_rect_separation(box, ob) for ob in obstacles))
    return edge


def _chrome_obstacles(
    ax,
    marker_layout: Optional[MapMarkerLayout],
    track: Optional[Sequence[GpxPoint]],
    *,
    use_gcj: bool,
    elev_panel_top: float,
) -> List[Tuple[float, float, float, float]]:
    obstacles: List[Tuple[float, float, float, float]] = []
    if marker_layout is not None:
        obstacles.extend(marker_layout.label_boxes_axes)
        for tb in marker_layout.thumb_boxes_axes:
            obstacles.append(
                (tb[0] - 0.012, tb[1] - 0.012, tb[2] + 0.012, tb[3] + 0.012)
            )
    if track:
        obstacles.extend(
            _track_obstacle_boxes_axes(
                ax,
                track,
                use_gcj=use_gcj,
                x_frac_max=1.0,
                y_frac_min=0.50,
                y_frac_max=1.0,
            )
        )
    if elev_panel_top > 0.0:
        obstacles.append((0.0, 0.0, 1.0, elev_panel_top + SUMMARY_GAP_AXES))
    return obstacles


def _try_symmetric_top_chrome(
    title_w: float,
    title_h: float,
    summary_w: float,
    summary_h: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
    y_title_top: float,
) -> Optional[Tuple[TitleChromeLayout, str, Tuple[float, float, float, float]]]:
    """标题与种鸟数左右对称：左题右数或右题左数。"""
    gap = TITLE_GAP_AXES
    y_sum_top = y_title_top - _SUMMARY_BELOW_TITLE
    pairs = (("left", "right"), ("right", "left"))
    for t_ha, s_ha in pairs:
        t_box = _corner_box(
            t_ha, y_title_top, title_w, title_h, margin_x=MAP_MARGIN_TITLE_X
        )
        s_box = _corner_box(
            s_ha, y_sum_top, summary_w, summary_h, margin_x=MAP_MARGIN_SUMMARY_X
        )
        if (
            not any(_rect_overlap_axes_frac(t_box, ob, margin=gap) for ob in obstacles)
            and not any(
                _rect_overlap_axes_frac(s_box, ob, margin=SUMMARY_GAP_AXES)
                for ob in obstacles
            )
            and not _rect_overlap_axes_frac(t_box, s_box, margin=0.02)
        ):
            return TitleChromeLayout(t_ha, y_title_top, t_box), s_ha, s_box
    return None


def _pick_max_clearance_box(
    block_w: float,
    block_h: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
    *,
    y_min: float,
    y_max: float,
) -> Tuple[str, Tuple[float, float, float, float]]:
    """在全局可放位置中选周围空白最大的格子。"""
    margin = MAP_MARGIN_SUMMARY_X
    span_x = max(1e-6, 1.0 - 2.0 * margin - block_w)
    span_y = max(1e-6, y_max - y_min - block_h)
    best_box = (margin, y_min, margin + block_w, y_min + block_h)
    best_clear = -1.0
    for ix in range(21):
        x0 = margin + span_x * ix / 20.0
        for iy in range(21):
            y0 = y_min + span_y * iy / 20.0
            box = (x0, y0, x0 + block_w, y0 + block_h)
            if any(
                _rect_overlap_axes_frac(box, ob, margin=SUMMARY_GAP_AXES)
                for ob in obstacles
            ):
                continue
            clear = _box_clearance(box, obstacles, y_min=y_min, y_max=y_max)
            if clear > best_clear:
                best_clear = clear
                best_box = box
    return "left", best_box


def _pick_summary_placement(
    block_w: float,
    block_h: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
    *,
    title_ha: str,
    title_y_top: float,
    title_box: Tuple[float, float, float, float],
    elev_panel_top: float,
) -> Tuple[str, Tuple[float, float, float, float]]:
    """优先标题对侧上角（略低）；上方没空则放全局空白最大处。"""
    y_min = max(0.02, elev_panel_top + SUMMARY_GAP_AXES * 1.5)
    y_max = 0.98
    y_sum_top = min(title_y_top - _SUMMARY_BELOW_TITLE, 1.0 - MAP_MARGIN_TITLE_Y - 0.02)
    opposite = "right" if title_ha == "left" else "left"
    if title_ha == "center":
        prefer = ("left", "right")
    else:
        prefer = (opposite, title_ha if title_ha in ("left", "right") else "left")
    obs = list(obstacles) + [title_box]
    for ha in prefer:
        for dy in (0.0, 0.03, 0.06, 0.10):
            yt = y_sum_top - dy
            box = _corner_box(
                ha, yt, block_w, block_h, margin_x=MAP_MARGIN_SUMMARY_X
            )
            if box[1] < y_min:
                continue
            if any(
                _rect_overlap_axes_frac(box, ob, margin=SUMMARY_GAP_AXES)
                for ob in obs
            ):
                continue
            return ha, box
    return _pick_max_clearance_box(
        block_w, block_h, obs, y_min=y_min, y_max=y_max
    )


def _pick_summary_y(
    ax,
    *,
    x0: float,
    block_w: float,
    block_h: float,
    default_y: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
    min_y: float = 0.012,
    max_y: float = 0.86,
) -> float:
    """左下物种汇总纵坐标：物种/鸟图/轨迹布局完成后，优先上移避让。"""
    step = 0.026
    gap = SUMMARY_GAP_AXES
    candidates: List[float] = [default_y]
    for i in range(1, 22):
        candidates.append(default_y + i * step)
    for i in range(1, 8):
        candidates.append(default_y - i * step * 0.85)
    seen: set = set()
    for y in candidates:
        key = round(y, 4)
        if key in seen:
            continue
        seen.add(key)
        if y < min_y or y + block_h > max_y:
            continue
        box = (x0, y, x0 + block_w, y + block_h)
        if all(
            not _rect_overlap_axes_frac(box, ob, margin=gap) for ob in obstacles
        ):
            return y
    y = max(min_y, min(default_y, max_y - block_h))
    for _ in range(18):
        box = (x0, y, x0 + block_w, y + block_h)
        if all(
            not _rect_overlap_axes_frac(box, ob, margin=gap) for ob in obstacles
        ):
            return y
        y = min(y + step, max_y - block_h)
    return max(min_y, y)


def _measure_title_block(
    ax,
    place: str,
    map_title: str,
    date_label: str,
    *,
    logo_path: str = "",
    logo_width_ratio: float = 0.30,
) -> Tuple[float, float, bool]:
    """标题块宽高（axes 比例）及是否含 Logo。"""
    typo = _map_typography(ax)
    ax_h_px = typo["ax_h_px"]
    line_gap = 0.005
    block_w = 0.0
    block_h = 0.0

    if date_label:
        block_h += ax_h_px / 50.0 / ax_h_px + line_gap
        block_w = max(
            block_w, _text_width_axes_frac(ax, date_label, typo["date_pt"])
        )

    if place:
        block_h += ax_h_px / 35.0 / ax_h_px + line_gap
        block_w = max(
            block_w, _text_width_axes_frac(ax, place, typo["title_pt"])
        )

    if map_title:
        block_h += ax_h_px / 35.0 / ax_h_px + line_gap
        block_w = max(
            block_w, _text_width_axes_frac(ax, map_title, typo["title_pt"])
        )

    logo_path_s = (logo_path or "").strip()
    has_logo = bool(logo_path_s and os.path.isfile(logo_path_s))
    if has_logo:
        lw_frac, lh_frac = _measure_logo_axes_frac(ax, logo_path_s, logo_width_ratio)
        block_h += lh_frac
        block_w = max(block_w, lw_frac)

    return block_w, block_h, has_logo


def _measure_logo_axes_frac(
    ax, logo_path: str, logo_width_ratio: float = 0.30
) -> Tuple[float, float]:
    """签名 Logo 实际占 axes 的宽高（按绘制高度反推宽度）。"""
    typo = _map_typography(ax)
    ax_w_px, ax_h_px = _ax_size_px(ax)
    h_px = typo["logo_h_px"]
    h_frac = h_px / max(ax_h_px, 1.0) + 0.010
    w_frac = _map_logo_target_width_frac(logo_width_ratio) + 0.016
    try:
        with Image.open(logo_path) as im:
            lw, lh = im.size
        if lh > 0 and lw > 0:
            w_frac = max(w_frac, (h_px * (lw / lh)) / max(ax_w_px, 1.0) + 0.016)
    except OSError:
        pass
    return w_frac, h_frac


def _axes_box_overlap_area(
    box: Tuple[float, float, float, float],
    obstacles: Sequence[Tuple[float, float, float, float]],
    *,
    margin: float,
) -> float:
    inflated = (
        box[0] - margin,
        box[1] - margin,
        box[2] + margin,
        box[3] + margin,
    )
    return sum(_rect_intersection_area(inflated, ob) for ob in obstacles)


def _pick_title_anchor(
    *,
    block_w: float,
    block_h: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
    y_top: float,
) -> Tuple[float, str, float]:
    """左上 → 右上 → 中上，必要时下移整块，避让鸟图/鸟名。"""
    gap = TITLE_GAP_AXES
    step = 0.022
    margin_x = MAP_MARGIN_TITLE_X
    max_x_right = 1.0 - margin_x

    def fits(box: Tuple[float, float, float, float]) -> bool:
        return all(
            not _rect_overlap_axes_frac(box, ob, margin=gap) for ob in obstacles
        )

    candidates: List[Tuple[float, str, float, Tuple[float, float, float, float]]] = []
    for y_shift in (0.0, 0.03, 0.06, 0.09, 0.12, 0.16):
        yt = y_top - y_shift
        if yt - block_h < 0.40:
            break
        x0 = margin_x
        candidates.append((x0, "left", yt, (x0, yt - block_h, x0 + block_w, yt)))
        for i in range(1, 16):
            x0 = margin_x + i * step
            if x0 + block_w > 0.50:
                break
            candidates.append(
                (x0, "left", yt, (x0, yt - block_h, x0 + block_w, yt))
            )
        x_right = max_x_right
        candidates.append(
            (x_right, "right", yt, (x_right - block_w, yt - block_h, x_right, yt))
        )
        for i in range(1, 16):
            x_right = max_x_right - i * step
            if x_right - block_w < 0.50:
                break
            candidates.append(
                (
                    x_right,
                    "right",
                    yt,
                    (x_right - block_w, yt - block_h, x_right, yt),
                )
            )
        x_center = max(margin_x, min(0.5 - block_w * 0.5, max_x_right - block_w))
        candidates.append(
            (
                0.5,
                "center",
                yt,
                (x_center, yt - block_h, x_center + block_w, yt),
            )
        )

    best: Optional[Tuple[float, str, float]] = None
    best_ov = 1e18
    for x_a, ha, yt, box in candidates:
        if fits(box):
            return x_a, ha, yt
        ov = _axes_box_overlap_area(box, obstacles, margin=gap)
        score = ov + y_top - yt
        if score < best_ov:
            best_ov = score
            best = (x_a, ha, yt)
    if best is not None:
        return best
    return margin_x, "left", y_top


def _draw_map_inset_title(
    ax,
    place: str,
    map_title: str,
    date_label: str,
    *,
    logo_path: str = "",
    logo_width_ratio: float = 0.30,
    basemap_style: str = "normal",
    on_basemap: bool = True,
    marker_layout: Optional[MapMarkerLayout] = None,
    track: Optional[Sequence[GpxPoint]] = None,
    use_gcj: bool = True,
    elev_panel_top: float = 0.0,
    preset: Optional[TitleChromeLayout] = None,
) -> TitleChromeLayout:
    """图内标题：日期（H/50）→ 地点 / 观鸟记录（各 H/35）→ Logo（H/30）。"""
    typo = _map_typography(ax)
    fs_title = typo["title_pt"]
    fs_date = typo["date_pt"]
    ax_h_px = typo["ax_h_px"]
    text_color, logo_rgb, use_stroke = _map_ink(basemap_style, on_basemap=on_basemap)
    effects = _map_text_effects(use_stroke=use_stroke)
    line_gap = 0.005
    y_top = 1.0 - MAP_MARGIN_TITLE_Y

    block_w, block_h, _ = _measure_title_block(
        ax,
        place,
        map_title,
        date_label,
        logo_path=logo_path,
        logo_width_ratio=logo_width_ratio,
    )
    if preset is not None:
        ha, y_top = preset.ha, preset.y_top
        if ha == "right":
            x_anchor = preset.box[2]
        elif ha == "center":
            x_anchor = 0.5
        else:
            x_anchor = preset.box[0]
        title_box = preset.box
    else:
        obstacles = _chrome_obstacles(
            ax,
            marker_layout,
            track,
            use_gcj=use_gcj,
            elev_panel_top=elev_panel_top,
        )
        x_anchor, ha, y_top = _pick_title_anchor(
            block_w=block_w,
            block_h=block_h,
            obstacles=obstacles,
            y_top=y_top,
        )
        if ha == "right":
            title_box = (x_anchor - block_w, y_top - block_h, x_anchor, y_top)
        elif ha == "center":
            title_box = (
                0.5 - block_w * 0.5,
                y_top - block_h,
                0.5 + block_w * 0.5,
                y_top,
            )
        else:
            title_box = (x_anchor, y_top - block_h, x_anchor + block_w, y_top)
    logo_align = (0.5, 1) if ha == "center" else ((1, 1) if ha == "right" else (0, 1))
    y = y_top

    if date_label:
        ax.text(
            x_anchor,
            y,
            date_label,
            transform=ax.transAxes,
            fontsize=fs_date,
            color=text_color,
            va="top",
            ha=ha,
            zorder=40,
            path_effects=effects,
        )
        y -= ax_h_px / 50.0 / ax_h_px + line_gap

    if place:
        ax.text(
            x_anchor,
            y,
            place,
            fontsize=fs_title,
            fontweight="bold",
            color=text_color,
            va="top",
            ha=ha,
            transform=ax.transAxes,
            zorder=40,
            path_effects=effects,
        )
        y -= ax_h_px / 35.0 / ax_h_px + line_gap

    if map_title:
        ax.text(
            x_anchor,
            y,
            map_title,
            transform=ax.transAxes,
            fontsize=fs_title,
            fontweight="bold",
            color=text_color,
            va="top",
            ha=ha,
            zorder=40,
            path_effects=effects,
        )
        y -= ax_h_px / 35.0 / ax_h_px + line_gap

    target_h_px = typo["logo_h_px"]
    fig = ax.figure
    logo_arr = _load_signature_logo_rgba(
        logo_path,
        int(max(target_h_px * 4.0, target_h_px)),
        int(target_h_px),
        fg_rgb=logo_rgb,
    )
    if logo_arr is not None:
        lh, lw = logo_arr.shape[:2]
        zoom = target_h_px * 72.0 / (max(lh, 1) * fig.dpi)
        y -= target_h_px / ax_h_px + 0.004
        im = OffsetImage(logo_arr, zoom=zoom, interpolation="bilinear")
        ab = AnnotationBbox(
            im,
            (x_anchor, y),
            xycoords=ax.transAxes,
            box_alignment=logo_align,
            frameon=False,
            pad=0,
            zorder=41,
        )
        ax.add_artist(ab)
    return TitleChromeLayout(ha, y_top, title_box)


def _draw_map_attribution(
    ax,
    *,
    y_frac: float = 0.006,
    basemap_style: str = "normal",
) -> None:
    """左下角地图来源（海拔图下方）。"""
    typo = _map_typography(ax)
    text_color, _, use_stroke = _map_ink(basemap_style, on_basemap=True)
    ax.text(
        MAP_MARGIN_TITLE_X,
        y_frac,
        "高德地图api",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=typo["attribution_pt"],
        color=text_color,
        alpha=0.88,
        zorder=43,
        path_effects=_map_text_effects(use_stroke=use_stroke),
    )


def _text_width_factor(text: str) -> float:
    """中文近似方块字，宽度按字号估算，避免避让框偏窄。"""
    if not text:
        return 0.90
    if all(ch.isdigit() or ch in "m.-+ " for ch in text):
        return 0.62
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return 1.08
    return 0.90


def _text_width_axes_frac(ax, text: str, fontsize_pt: float) -> float:
    """文本在 axes 坐标系下的宽度占比（用于排版）。"""
    _, ax_w_px = _ax_size_px(ax)
    dpi = ax.figure.dpi
    factor = _text_width_factor(text)
    w_px = max(1.0, len(text) * fontsize_pt * dpi / 72.0 * factor)
    return w_px / ax_w_px


def _text_width_data(ax, text: str, fontsize_pt: float) -> float:
    x0, x1 = ax.get_xlim()
    return _text_width_axes_frac(ax, text, fontsize_pt) * max(x1 - x0, 1e-9)


def _data_span_to_points(ax, x: float, y: float, delta_data: float) -> float:
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x + delta_data, y))
    return abs(float(p1[0] - p0[0]))


def _accent_text_effects() -> List:
    return [pe.withStroke(linewidth=1.8, foreground="#000000", alpha=0.38)]


def _summary_block_size(ax, n_sp: int) -> Tuple[float, float, str]:
    typo = _map_typography(ax)
    num_s = str(n_sp)
    block_h = 1.0 / 16.0 + 1.0 / 80.0
    block_w = (
        _text_width_axes_frac(ax, num_s, typo["count_num_pt"])
        + _text_width_axes_frac(ax, "种鸟", typo["count_suffix_pt"])
        + 0.016
    )
    return block_w, block_h, num_s


def _draw_map_summary(
    ax,
    photos: Sequence[BirdPhoto],
    marker_layout: Optional[MapMarkerLayout],
    track: Sequence[GpxPoint],
    *,
    default_y: float = 0.055,
    basemap_style: str = "normal",
    on_basemap: bool = True,
    use_gcj: bool = True,
    elev_panel_top: float = 0.0,
    attribution_y: float = 0.006,
    title_layout: Optional[TitleChromeLayout] = None,
    preset_ha: Optional[str] = None,
    preset_box: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """种鸟数：优先与标题左右对称且略低；上方没空则放全局空白最大处。"""
    _ = default_y
    n_sp = _distinct_species_count(photos)
    if n_sp <= 0:
        return
    typo = _map_typography(ax)
    num_color, suf_color, effects, num_ff = _map_summary_style(
        basemap_style, on_basemap=on_basemap
    )
    block_w, block_h, num_s = _summary_block_size(ax, n_sp)
    if preset_ha and preset_box:
        ha, box = preset_ha, preset_box
    else:
        obstacles = _chrome_obstacles(
            ax,
            marker_layout,
            track,
            use_gcj=use_gcj,
            elev_panel_top=elev_panel_top,
        )
        attr_h = _text_height_axes_frac(ax, typo["attribution_pt"]) + 0.008
        obstacles.append((0.0, 0.0, 0.42, attribution_y + attr_h))
        t_ha = title_layout.ha if title_layout else "left"
        t_top = (
            title_layout.y_top if title_layout else (1.0 - MAP_MARGIN_TITLE_Y)
        )
        t_box = title_layout.box if title_layout else (0.0, 0.86, 0.28, 0.98)
        ha, box = _pick_summary_placement(
            block_w,
            block_h,
            obstacles,
            title_ha=t_ha,
            title_y_top=t_top,
            title_box=t_box,
            elev_panel_top=elev_panel_top,
        )
    y_mid = box[1] + block_h * 0.48
    x_anchor = box[2] if ha == "right" else box[0]
    loc = "center right" if ha == "right" else "center left"
    num_props = {
        "fontsize": typo["count_num_pt"],
        "fontweight": "bold",
        "fontfamily": num_ff,
        "color": num_color,
        "path_effects": effects,
    }
    suf_props = {
        "fontsize": typo["count_suffix_pt"],
        "fontweight": "bold",
        "color": suf_color,
        "path_effects": effects,
    }
    sep_pt = max(4.0, typo["count_num_pt"] * 0.2)
    pack = HPacker(
        children=[
            TextArea(num_s, textprops=num_props),
            TextArea("种鸟", textprops=suf_props),
        ],
        align="baseline",
        pad=0,
        sep=sep_pt,
    )
    ab = AnchoredOffsetbox(
        loc=loc,
        child=pack,
        bbox_to_anchor=(x_anchor, y_mid),
        bbox_transform=ax.transAxes,
        frameon=False,
        pad=0.0,
        borderpad=0.0,
    )
    ab.set_zorder(42)
    ax.add_artist(ab)


def _draw_map_chrome(
    ax,
    place: str,
    map_title: str,
    date_label: str,
    photos: Sequence[BirdPhoto],
    marker_layout: Optional[MapMarkerLayout],
    track: Sequence[GpxPoint],
    *,
    logo_path: str = "",
    logo_width_ratio: float = 0.30,
    basemap_style: str = "normal",
    on_basemap: bool = True,
    use_gcj: bool = True,
    elev_south_frac: float = 0.0,
) -> None:
    """标题 / Logo / 种鸟数最后放置：先对称试顶角，再各自避让。"""
    elev_panel_top = ELEV_PANEL_TOP_AXES if elev_south_frac > 0 else 0.0
    obstacles = _chrome_obstacles(
        ax,
        marker_layout,
        track,
        use_gcj=use_gcj,
        elev_panel_top=elev_panel_top,
    )
    title_w, title_h, _ = _measure_title_block(
        ax,
        place,
        map_title,
        date_label,
        logo_path=logo_path,
        logo_width_ratio=logo_width_ratio,
    )
    n_sp = _distinct_species_count(photos)
    y_title_top = 1.0 - MAP_MARGIN_TITLE_Y
    preset_title: Optional[TitleChromeLayout] = None
    preset_sum_ha: Optional[str] = None
    preset_sum_box: Optional[Tuple[float, float, float, float]] = None
    if n_sp > 0:
        sum_w, sum_h, _ = _summary_block_size(ax, n_sp)
        paired = _try_symmetric_top_chrome(
            title_w, title_h, sum_w, sum_h, obstacles, y_title_top
        )
        if paired is not None:
            preset_title, preset_sum_ha, preset_sum_box = paired
    title_layout = _draw_map_inset_title(
        ax,
        place,
        map_title,
        date_label,
        logo_path=logo_path,
        logo_width_ratio=logo_width_ratio,
        basemap_style=basemap_style,
        on_basemap=on_basemap,
        marker_layout=marker_layout,
        track=track,
        use_gcj=use_gcj,
        elev_panel_top=elev_panel_top,
        preset=preset_title,
    )
    _draw_map_summary(
        ax,
        photos,
        marker_layout or MapMarkerLayout([], [], []),
        track,
        basemap_style=basemap_style,
        on_basemap=on_basemap,
        use_gcj=use_gcj,
        elev_panel_top=elev_panel_top,
        title_layout=title_layout,
        preset_ha=preset_sum_ha,
        preset_box=preset_sum_box,
    )


def _thumb_radius_data(ax, thumb_diameter: int) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pos = ax.get_position()
    fig = ax.figure
    fig_w_px = max(1.0, fig.get_figwidth() * fig.dpi * pos.width)
    fig_h_px = max(1.0, fig.get_figheight() * fig.dpi * pos.height)
    rx = abs(x1 - x0) * (thumb_diameter / fig_w_px) * 0.52
    ry = abs(y1 - y0) * (thumb_diameter / fig_h_px) * 0.52
    return max(rx, ry, 1e-9)


def _circle_overlap_fraction(d: float, r: float) -> float:
    if r <= 0:
        return 0.0
    if d >= 2.0 * r:
        return 0.0
    if d <= 0:
        return 1.0
    a = r * r * math.acos(max(-1.0, min(1.0, d / (2.0 * r))))
    part = 2.0 * a - 0.5 * d * math.sqrt(max(0.0, 4.0 * r * r - d * d))
    return min(1.0, max(0.0, part / (math.pi * r * r)))


def _min_center_distance(r: float, max_overlap: float) -> float:
    """两圆半径均为 r 时，使面积重叠率 ≤ max_overlap 的最小圆心距。"""
    if r <= 0:
        return 0.0
    lo, hi = 0.0, 2.01 * r
    for _ in range(40):
        mid = (lo + hi) * 0.5
        if _circle_overlap_fraction(mid, r) > max_overlap:
            lo = mid
        else:
            hi = mid
    return hi * 1.02


def _layout_marker_displays(
    anchors: List[Tuple[float, float]],
    ax,
    thumb_diameter: int,
    *,
    max_overlap: float = 0.04,
    max_iters: int | None = None,
) -> List[Tuple[float, float]]:
    """
    鸟图重叠时：序号靠后的圆外移，锚点保留 GPS，绘制引线指向原位置。
    """
    n = len(anchors)
    if n <= 1:
        return list(anchors)
    if max_iters is None:
        max_iters = max(80, n * 4)
    r = _thumb_radius_data(ax, thumb_diameter)
    min_sep = _min_center_distance(r, max_overlap)
    max_leader = r * 3.2
    displays = [list(a) for a in anchors]
    for _ in range(max_iters):
        moved = False
        for j in range(1, n):
            ax0, ay0 = anchors[j]
            for i in range(j):
                dx = displays[j][0] - displays[i][0]
                dy = displays[j][1] - displays[i][1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    ang = (i * 2.1 + j * 0.9) % (2 * math.pi)
                    dx, dy = math.cos(ang), math.sin(ang)
                    d = 1.0
                if _circle_overlap_fraction(d, r) > max_overlap:
                    need = max(min_sep - d, min_sep * 0.12)
                    displays[j][0] += dx / d * need
                    displays[j][1] += dy / d * need
                    moved = True
            d_anchor = math.hypot(
                displays[j][0] - ax0, displays[j][1] - ay0
            )
            if d_anchor > max_leader:
                displays[j][0] = ax0 + (displays[j][0] - ax0) / d_anchor * max_leader
                displays[j][1] = ay0 + (displays[j][1] - ay0) / d_anchor * max_leader
                moved = True
        if not moved:
            break
    return [(d[0], d[1]) for d in displays]


def _data_span_y_to_points(ax, x: float, y: float, delta_data: float) -> float:
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x, y + delta_data))
    return abs(float(p1[1] - p0[1]))


def _text_size_display(text: str, fontsize_pt: float, dpi: float) -> Tuple[float, float]:
    scale = dpi / 72.0
    w = max(fontsize_pt * scale * 0.72, len(text) * fontsize_pt * scale * 0.92)
    h = fontsize_pt * scale * 1.14
    return w, h


def _rect_overlap_display(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _circle_box_display(
    ax, cx: float, cy: float, r_data: float
) -> Tuple[float, float, float, float]:
    rx = _data_span_to_points(ax, cx, cy, r_data)
    ry = _data_span_y_to_points(ax, cx, cy, r_data)
    xd, yd = ax.transData.transform((cx, cy))
    return (xd - rx, yd - ry, xd + rx, yd + ry)


def _label_box_display(
    ax,
    dx: float,
    dy: float,
    name: str,
    side: int,
    off_x_pt: float,
    off_y_pt: float,
    label_fs: float,
) -> Tuple[float, float, float, float]:
    tx, ty = _offset_points_to_data(ax, dx, dy, float(side) * off_x_pt, off_y_pt)
    xd, yd = ax.transData.transform((tx, ty))
    w, h = _text_size_display(name, label_fs, ax.figure.dpi)
    if side > 0:
        return (xd, yd - h * 0.5, xd + w, yd + h * 0.5)
    return (xd - w, yd - h * 0.5, xd, yd + h * 0.5)


def _map_label_x_pad_data(ax) -> float:
    """地图 x 方向内边距（数据坐标），约为可视宽度的 1/50。"""
    x0, x1 = ax.get_xlim()
    return max((x1 - x0) * _MAP_LABEL_X_MARGIN_FRAC, 1e-9)


def _label_text_span_data(
    ax,
    dx: float,
    dy: float,
    name: str,
    side: int,
    off_x: float,
    dy_off: float,
    label_fs: float,
) -> Tuple[float, float]:
    """物种名在数据坐标下的左右边界（含 ha left/right）。"""
    anchor_x, _ = _offset_points_to_data(ax, dx, dy, side * off_x, float(dy_off))
    lw = _text_width_data(ax, name, label_fs)
    if side > 0:
        return anchor_x, anchor_x + lw
    return anchor_x - lw, anchor_x


def _label_x_margin_overflow(ax, left: float, right: float) -> float:
    """超出左右 1/50 边距的数据坐标量，0 表示完全在内。"""
    x0, x1 = ax.get_xlim()
    pad = _map_label_x_pad_data(ax)
    over = 0.0
    if left < x0 + pad:
        over += x0 + pad - left
    if right > x1 - pad:
        over += right - (x1 - pad)
    return over


# 鸟名相对圆形鸟图边缘的基准间距（相对历史默认值减半）
_SPECIES_LABEL_GAP_SCALE = 0.5


def _species_label_base_off(
    ax, dx: float, dy: float, r_thumb: float, label_fs: float, thumb_diameter: int
) -> float:
    edge_pt = _data_span_to_points(ax, dx, dy, r_thumb * 1.08)
    s = _SPECIES_LABEL_GAP_SCALE
    return max(
        thumb_diameter * 0.30 * s,
        edge_pt * 0.78 * s + max(2.0, label_fs * 0.12) * s,
    )


def _layout_species_labels(
    ax,
    items: Sequence[BirdPhoto],
    displays: Sequence[Tuple[float, float]],
    r_thumb: float,
    label_fs: float,
    thumb_diameter: int,
) -> List[Tuple[int, float, float]]:
    """为每个物种名选择左右侧与偏移，避让鸟图圆与其它鸟名框。"""
    n = len(items)
    if n == 0:
        return []
    circle_boxes = [
        _circle_box_display(ax, displays[i][0], displays[i][1], r_thumb)
        for i in range(n)
    ]
    order = sorted(
        range(n),
        key=lambda i: -sum(
            1
            for j in range(n)
            if j != i
            and math.hypot(
                displays[i][0] - displays[j][0],
                displays[i][1] - displays[j][1],
            )
            < r_thumb * 2.6
        ),
    )
    out: List[Tuple[int, float, float]] = [(1, 6.0, 0.0)] * n
    mults = (1.0, 1.12, 1.28, 1.45, 1.65, 1.9, 2.2, 2.55, 3.0, 3.5, 4.0, 4.6, 5.2)
    dy_offs = (0, 6, -6, 12, -12, 18, -18, 24, -24, 30, -30, 36, -36, 42, -42)
    passes = _label_layout_refine_passes(n)

    def _label_box_for(
        idx: int, side: int, off_x: float, dy_off: float
    ) -> Tuple[float, float, float, float]:
        name = items[idx].species_cn
        dx, dy = displays[idx]
        return _label_box_display(
            ax, dx, dy, name, side, off_x, float(dy_off), label_fs
        )

    def _overlap_score(idx: int, layouts: Sequence[Tuple[int, float, float]]) -> float:
        side, off_x, dy_off = layouts[idx]
        box_i = _label_box_for(idx, side, off_x, dy_off)
        worst = 0.0
        if _rect_overlap_display(box_i, circle_boxes[idx], margin=3.0):
            worst = max(worst, 0.35)
        for j in range(n):
            if j == idx:
                continue
            sj, ox, dy = layouts[j]
            box_j = _label_box_for(j, sj, ox, dy)
            if _rect_overlap_display(box_i, circle_boxes[j], margin=2.0):
                worst = max(worst, 0.25)
            if _rect_overlap_display(box_i, box_j, margin=2.0):
                worst = max(
                    worst,
                    _rect_overlap_fraction(
                        _inflate_display_box(box_i),
                        _inflate_display_box(box_j),
                    ),
                )
        return worst

    def _pick_one(
        idx: int,
        placed: Sequence[Tuple[float, float, float, float]],
    ) -> Tuple[int, float, float, Tuple[float, float, float, float]]:
        name = items[idx].species_cn
        dx, dy = displays[idx]
        edge_pt = _data_span_to_points(ax, dx, dy, r_thumb * 1.08)
        base_off = _species_label_base_off(
            ax, dx, dy, r_thumb, label_fs, thumb_diameter
        )
        best_side, best_off_x, best_dy = 1, base_off, 0.0
        best_score = 1e18
        best_box = _label_box_for(idx, 1, base_off, 0.0)
        for side in (-1, 1):
            for mult in mults:
                off_x = base_off * mult
                for dy_off in dy_offs:
                    box = _label_box_display(
                        ax, dx, dy, name, side, off_x, float(dy_off), label_fs
                    )
                    score = 0.0
                    if _rect_overlap_display(box, circle_boxes[idx], margin=3.0):
                        score += 120.0
                    for j, cbox in enumerate(circle_boxes):
                        if j == idx:
                            continue
                        if _rect_overlap_display(box, cbox, margin=2.0):
                            score += 55.0
                    for lbox in placed:
                        if _rect_overlap_display(box, lbox, margin=2.0):
                            score += 90.0
                    xl, xr = _label_text_span_data(
                        ax, dx, dy, name, side, off_x, float(dy_off), label_fs
                    )
                    margin_over = _label_x_margin_overflow(ax, xl, xr)
                    if margin_over > 0:
                        score += 800.0 + margin_over * 120.0
                    score += mult * 0.8 + abs(dy_off) * 0.04
                    if score < best_score:
                        best_score = score
                        best_side, best_off_x, best_dy = side, off_x, float(dy_off)
                        best_box = box
        return best_side, best_off_x, best_dy, best_box

    for pass_idx in range(passes):
        placed_labels: List[Tuple[float, float, float, float]] = []
        if pass_idx == 0:
            process_order = order
        else:
            process_order = sorted(
                range(n),
                key=lambda i: (-_overlap_score(i, out), displays[i][0]),
            )
        for idx in process_order:
            side, off_x, dy_off, box = _pick_one(idx, placed_labels)
            out[idx] = (side, off_x, dy_off)
            placed_labels.append(box)
        if max(_overlap_score(i, out) for i in range(n)) <= 0.05:
            break

    for idx in range(n):
        side, off_x, dy_off = out[idx]
        name = items[idx].species_cn
        dx, dy = displays[idx]
        base_off = _species_label_base_off(
            ax, dx, dy, r_thumb, label_fs, thumb_diameter
        )
        xl, xr = _label_text_span_data(
            ax, dx, dy, name, side, off_x, dy_off, label_fs
        )
        if _label_x_margin_overflow(ax, xl, xr) <= 0:
            continue
        best = out[idx]
        best_over = _label_x_margin_overflow(
            ax,
            *_label_text_span_data(
                ax, dx, dy, name, best[0], best[1], best[2], label_fs
            ),
        )
        for side in (-1, 1):
            for mult in (1.0, 1.1, 1.22, 1.35):
                off_x = base_off * mult
                for dy_off in (0, 6, -6, 12, -12, 18, -18):
                    xl, xr = _label_text_span_data(
                        ax, dx, dy, name, side, off_x, dy_off, label_fs
                    )
                    over = _label_x_margin_overflow(ax, xl, xr)
                    if over >= best_over:
                        continue
                    box = _label_box_for(idx, side, off_x, dy_off)
                    if _rect_overlap_display(box, circle_boxes[idx], margin=2.0):
                        continue
                    best_over = over
                    best = (side, off_x, dy_off)
        out[idx] = best

    return out


def _elevation_south_reserve_frac(
    thumb_diameter: int, map_height_px: int, *, dpi: float = EXPORT_DPI
) -> float:
    """叠加海拔时，地图南侧应预留的 axes 高度比例（含鸟图半径）。"""
    _ = dpi
    thumb = float(thumb_diameter) / max(float(map_height_px), 1.0)
    return min(0.40, ELEV_PANEL_TOP_AXES + 0.5 * thumb + _ELEV_MAP_THUMB_GAP_AXES)


def _subplot_aspect_wh(ax) -> float:
    pos = ax.get_position()
    fig = ax.figure
    return (pos.width / max(pos.height, 1e-9)) * (
        fig.get_figwidth() / max(fig.get_figheight(), 1e-9)
    )


def _expand_lonlat_bounds(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    aspect_wh: float,
    *,
    pad_ratio: float = 0.05,
) -> Tuple[float, float, float, float]:
    """扩展经纬度范围，使底图区域宽高比与子图一致（铺满宽度）。"""
    w = lon_max - lon_min
    h = lat_max - lat_min
    if w < 1e-9:
        w = 0.01
    if h < 1e-9:
        h = w / max(aspect_wh, 0.2)
    cx = (lon_min + lon_max) / 2.0
    cy = (lat_min + lat_max) / 2.0
    data_ar = w / h
    if data_ar < aspect_wh:
        w = h * aspect_wh
    else:
        h = w / aspect_wh
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    return (
        cx - w / 2 - pad_w,
        cx + w / 2 + pad_w,
        cy - h / 2 - pad_h,
        cy + h / 2 + pad_h,
    )


def _build_timed_profile(
    track: Sequence[GpxPoint],
    align: Optional[TrackTimeAlignment] = None,
) -> Tuple[List[datetime], List[float], List[float]]:
    ts_raw, pts = _times_index(track)
    if not ts_raw:
        return [], [], []
    if align is None:
        align = TrackTimeAlignment()
    ts = [align.gpx_to_timeline(t) for t in ts_raw]
    dist: List[float] = [0.0]
    ele: List[float] = []
    for i, p in enumerate(pts):
        if p.ele is not None:
            ele.append(float(p.ele))
        else:
            ele.append(ele[-1] if ele else 0.0)
        if i > 0:
            p0 = pts[i - 1]
            dist.append(
                dist[-1]
                + haversine_km(p0.lat, p0.lon, p.lat, p.lon)
            )
    return ts, dist, ele


def _profile_at_time(
    ts: Sequence[datetime],
    dist_km: Sequence[float],
    ele: Sequence[float],
    when: datetime,
) -> Optional[Tuple[float, float]]:
    if not ts or not dist_km:
        return None
    if when <= ts[0]:
        return (dist_km[0], ele[0] if ele else 0.0)
    if when >= ts[-1]:
        return (dist_km[-1], ele[-1] if ele else 0.0)
    i = bisect_left(ts, when)
    if i == 0:
        return (dist_km[0], ele[0] if ele else 0.0)
    t0, t1 = ts[i - 1], ts[i]
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return (dist_km[min(i, len(dist_km) - 1)], ele[min(i, len(ele) - 1)] if ele else 0.0)
    r = (when - t0).total_seconds() / span
    d = dist_km[i - 1] + (dist_km[i] - dist_km[i - 1]) * r
    e0 = ele[i - 1] if ele else 0.0
    e1 = ele[i] if ele else e0
    e = e0 + (e1 - e0) * r
    return (d, e)


def _lonlat_bounds(
    lons: Sequence[float], lats: Sequence[float]
) -> Optional[Tuple[float, float, float, float]]:
    if not lons or not lats:
        return None
    return min(lons), max(lons), min(lats), max(lats)


def _track_lonlat_bounds(
    track: Sequence[GpxPoint],
) -> Optional[Tuple[float, float, float, float]]:
    if not track:
        return None
    return _lonlat_bounds([p.lon for p in track], [p.lat for p in track])


def _photo_lonlat_bounds(
    photos: Sequence[BirdPhoto],
) -> Optional[Tuple[float, float, float, float]]:
    lons: List[float] = []
    lats: List[float] = []
    for ph in photos:
        if ph.lat is None or ph.lon is None:
            continue
        lons.append(ph.lon)
        lats.append(ph.lat)
    return _lonlat_bounds(lons, lats)


def _intersect_lonlat_bounds(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    lon_min = max(a[0], b[0])
    lon_max = min(a[1], b[1])
    lat_min = max(a[2], b[2])
    lat_max = min(a[3], b[3])
    if lon_min >= lon_max or lat_min >= lat_max:
        return None
    return lon_min, lon_max, lat_min, lat_max


def _include_nearby_key_places(
    bounds: Optional[Tuple[float, float, float, float]],
    places: Sequence[KeyPlace],
    *,
    expand: float = 0.55,
) -> Optional[Tuple[float, float, float, float]]:
    """把视野附近的关键地点并入范围；过远的不拉远整图。"""
    extras = [(p.lon, p.lat) for p in places]
    if not extras:
        return bounds
    if bounds is None:
        lons = [x for x, _ in extras]
        lats = [y for _, y in extras]
        return min(lons), max(lons), min(lats), max(lats)
    x0, x1, y0, y1 = bounds
    dx = max(x1 - x0, 1e-4) * float(expand)
    dy = max(y1 - y0, 1e-4) * float(expand)
    keep = [
        (lon, lat)
        for lon, lat in extras
        if x0 - dx <= lon <= x1 + dx and y0 - dy <= lat <= y1 + dy
    ]
    if not keep:
        return bounds
    lons = [x0, x1] + [p[0] for p in keep]
    lats = [y0, y1] + [p[1] for p in keep]
    return min(lons), max(lons), min(lats), max(lats)


def _resolve_map_view_bounds(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    key_places: Sequence[KeyPlace] = (),
) -> Optional[Tuple[float, float, float, float]]:
    """
    地图可视范围：轨迹 bbox 与匹配鸟图 GPS bbox 的交集。
    轨迹常含非观鸟行程，仅用全轨迹或仅鸟图都会偏大/偏偏；取交集限定视野。
    """
    track_b = _track_lonlat_bounds(track)
    photo_b = _photo_lonlat_bounds(photos)
    if track_b and photo_b:
        inter = _intersect_lonlat_bounds(track_b, photo_b)
        base = inter if inter is not None else photo_b
    elif photo_b:
        base = photo_b
    else:
        base = track_b
    # 关键地点不再拉远整图；超范围地点由 partition 后弹窗提示
    _ = key_places
    return base


def _estimate_map_lonlat_extent(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    *,
    width_px: int,
    height_px: int,
    elev_south_frac: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    """与出图接近的经纬范围（含宽高比扩展），用于判断地点是否落在图内。"""
    view = _resolve_map_view_bounds(track, photos)
    if view is None:
        return None
    aspect = float(width_px) / max(float(height_px), 1.0)
    x0, x1, y0, y1 = _expand_lonlat_bounds(*view, aspect)
    if elev_south_frac > 0:
        x0, x1, y0, y1 = reserve_south_lat(x0, x1, y0, y1, elev_south_frac)
        x0, x1, y0, y1 = fit_lon_to_aspect(x0, x1, y0, y1, aspect)
    return x0, x1, y0, y1


def _collect_lonlats(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
) -> Tuple[List[float], List[float]]:
    bounds = _resolve_map_view_bounds(track, photos)
    if bounds is not None:
        lon_min, lon_max, lat_min, lat_max = bounds
        return [lon_min, lon_max], [lat_min, lat_max]
    lons: List[float] = []
    lats: List[float] = []
    for p in track:
        lons.append(p.lon)
        lats.append(p.lat)
    for ph in photos:
        if ph.lat is None or ph.lon is None:
            continue
        lons.append(ph.lon)
        lats.append(ph.lat)
    return lons, lats


_GPS_CLUSTER_MAX_COLS = 5
_GPS_DOT_COLOR = "#E67E22"
# 同行鸟图圆心距 = 直径 + gap；gap 为直径的 2/5（相对原先 1/5 再加大 1/5）
_THUMB_GAP_RATIO = 0.4
# 定位圆点相对历史默认直径放大 100%（面积 ×4）
_GPS_DOT_AREA_SCALE = 4.0


@dataclass
class _GpsClusterLayout:
    anchor: Tuple[float, float]
    side: int
    group: List[Tuple[BirdPhoto, float, float]]
    positions: List[Tuple[float, float]]
    row_first_indices: List[int]


def _px_as_km_at_view_center(ax, px: float) -> float:
    """将显示像素距离换算为当前视野中心附近的大致公里数。"""
    px = abs(float(px))
    if px < 1e-6:
        return 0.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    p0 = ax.transData.transform((cx, cy))
    x2, y2 = ax.transData.inverted().transform((p0[0] + px, p0[1]))
    x3, y3 = ax.transData.inverted().transform((p0[0], p0[1] + px))
    # x 为经度、y 为纬度（GCJ/WGS 同形）
    d_x = haversine_km(cy, cx, cy, x2)
    d_y = haversine_km(cy, cx, y3, cx)
    return max(d_x, d_y, 1e-6)


def _layout_cluster_radius_km(ax, thumb_diameter: int, radius_km: float) -> float:
    """
    布局合并半径：至少为物种去重半径；大范围地图上再按鸟图像素尺寸放大，
    避免缩略图彼此重叠却未合并成横向网格。
    """
    visual_km = _px_as_km_at_view_center(ax, float(thumb_diameter) * 2.8)
    return max(0.01, float(radius_km), float(visual_km))


def _cluster_photos_by_radius(
    entries: Sequence[Tuple[BirdPhoto, float, float]],
    radius_km: float,
) -> List[List[Tuple[BirdPhoto, float, float]]]:
    """
    按半径合并邻近 GPS 为同一布局簇（可含多种鸟）。
    锚点取成员经纬度均值对应的地图坐标。
    """
    radius_km = max(0.01, float(radius_km))
    clusters: List[List[Tuple[BirdPhoto, float, float]]] = []
    centers: List[Tuple[float, float]] = []  # (lat, lon)
    for ph, ax_x, ay in entries:
        if ph.lat is None or ph.lon is None:
            continue
        lat, lon = float(ph.lat), float(ph.lon)
        best_i: Optional[int] = None
        best_d = radius_km
        for i, (cla, clo) in enumerate(centers):
            d = haversine_km(lat, lon, cla, clo)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            clusters.append([(ph, ax_x, ay)])
            centers.append((lat, lon))
            continue
        clusters[best_i].append((ph, ax_x, ay))
        members = clusters[best_i]
        centers[best_i] = (
            sum(float(m[0].lat) for m in members) / len(members),  # type: ignore[arg-type]
            sum(float(m[0].lon) for m in members) / len(members),  # type: ignore[arg-type]
        )
    # 簇内按拍摄时间排序，便于横向阅读
    for g in clusters:
        g.sort(key=lambda t: t[0].when or datetime.min)
    return clusters


def _cluster_anchor_xy(
    group: Sequence[Tuple[BirdPhoto, float, float]],
    *,
    use_gcj: bool,
) -> Tuple[float, float]:
    """簇锚点：成员经纬度均值映射到地图坐标。"""
    if not group:
        return 0.0, 0.0
    lat = sum(float(ph.lat) for ph, _, _ in group) / len(group)  # type: ignore[arg-type]
    lon = sum(float(ph.lon) for ph, _, _ in group) / len(group)  # type: ignore[arg-type]
    return _map_xy(lon, lat, use_gcj=use_gcj)


def _cluster_grid_metrics(
    thumb_diameter: int, label_fs: float, *, dpi: float = EXPORT_DPI
) -> Tuple[float, float, float, float, float]:
    """返回 col_step, row_step, lead_px, gap_px, label_h（布局像素，与 2.0.87 前一致）。"""
    d = float(thumb_diameter)
    gap = d * _THUMB_GAP_RATIO
    col_step = d + gap
    # 与 _cluster_label_xytext 一致：圆上缘以上再留鸟名高度
    _, oy_pt, _, _ = _cluster_label_xytext(0, label_fs, thumb_diameter, dpi=dpi)
    label_clear_px = float(oy_pt) * float(dpi) / 72.0
    label_h = max(float(label_fs) * dpi / 72.0 * 1.2, d * 0.2)
    row_step = d + gap + label_clear_px + label_h * 0.5
    lead_px = gap + d * 0.5
    return col_step, row_step, lead_px, gap, label_h


def _cluster_side_from_anchor(ax, anchor_x: float) -> int:
    """+1 右侧，-1 左侧（选可视范围较大一侧）。"""
    x0, x1 = ax.get_xlim()
    if x1 <= x0:
        return 1
    space_left = anchor_x - x0
    space_right = x1 - anchor_x
    return 1 if space_right >= space_left else -1


def _layout_cluster_thumb_grid(
    ax,
    anchor: Tuple[float, float],
    count: int,
    side: int,
    thumb_diameter: int,
    label_fs: float,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """GPS 圆点左/右侧网格；中心距 = 直径 + 直径×2/5，行距含上方鸟名高度。"""
    if count <= 0:
        return [], []
    ax_x, ax_y = anchor
    max_cols = _GPS_CLUSTER_MAX_COLS
    n_rows = (count + max_cols - 1) // max_cols
    dpi = float(ax.figure.dpi)
    col_step, row_step, lead_px, _, _ = _cluster_grid_metrics(
        thumb_diameter, label_fs, dpi=dpi
    )
    positions: List[Tuple[float, float]] = []
    row_first: List[int] = []
    for i in range(count):
        row = i // max_cols
        col = i % max_cols
        if col == 0:
            row_first.append(i)
        dx_px = side * (lead_px + col * col_step)
        dy_px = (row - (n_rows - 1) * 0.5) * row_step
        ox, oy = _offset_points_to_data(ax, ax_x, ax_y, dx_px, dy_px)
        positions.append((ox, oy))
    return positions, row_first


def _pick_side_keeping_grid_on_map(
    ax,
    anchor: Tuple[float, float],
    count: int,
    thumb_diameter: int,
    label_fs: float,
) -> int:
    """优先选能把整块网格留在图内的一侧；否则退回空间较大侧。"""
    preferred = _cluster_side_from_anchor(ax, anchor[0])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad_x = (x1 - x0) * 0.02
    pad_y = (y1 - y0) * 0.02

    def _fits(side: int) -> bool:
        positions, _ = _layout_cluster_thumb_grid(
            ax, anchor, count, side, thumb_diameter, label_fs
        )
        if not positions:
            return True
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        return (
            min(xs) >= x0 + pad_x
            and max(xs) <= x1 - pad_x
            and min(ys) >= y0 + pad_y
            and max(ys) <= y1 - pad_y
        )

    if _fits(preferred):
        return preferred
    other = -preferred
    if _fits(other):
        return other
    return preferred


def _clamp_positions_into_axes(
    ax,
    positions: Sequence[Tuple[float, float]],
    thumb_diameter: int,
) -> List[Tuple[float, float]]:
    """将鸟图中心钳制在图内，避免被裁切消失。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # 约半个鸟图直径的边距（数据坐标，按直径参数而非放大后显示尺寸）
    p0 = ax.transData.transform((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    q = ax.transData.inverted().transform(
        (p0[0] + float(thumb_diameter) * 0.55, p0[1])
    )
    margin_x = abs(q[0] - 0.5 * (x0 + x1))
    q2 = ax.transData.inverted().transform(
        (p0[0], p0[1] + float(thumb_diameter) * 0.55)
    )
    margin_y = abs(q2[1] - 0.5 * (y0 + y1))
    lo_x, hi_x = x0 + margin_x, x1 - margin_x
    lo_y, hi_y = y0 + margin_y, y1 - margin_y
    if hi_x <= lo_x or hi_y <= lo_y:
        return list(positions)
    return [
        (min(max(px, lo_x), hi_x), min(max(py, lo_y), hi_y))
        for px, py in positions
    ]


def _min_thumb_sep_px(thumb_diameter: int, dpi: float = EXPORT_DPI) -> float:
    _ = dpi
    return float(thumb_diameter) * (1.0 + _THUMB_GAP_RATIO * 0.85)


def _count_px_conflicts(
    ax,
    positions: Sequence[Tuple[float, float]],
    occupied: Sequence[Tuple[float, float]],
    min_sep_px: float,
) -> int:
    if not positions or not occupied:
        return 0
    pts = [ax.transData.transform((x, y)) for x, y in positions]
    obs = [ax.transData.transform((x, y)) for x, y in occupied]
    hits = 0
    for p in pts:
        for o in obs:
            if math.hypot(float(p[0] - o[0]), float(p[1] - o[1])) < min_sep_px:
                hits += 1
    return hits


def _separate_positions_px(
    ax,
    positions: Sequence[Tuple[float, float]],
    min_sep_px: float,
    *,
    max_iters: int = 90,
) -> List[Tuple[float, float]]:
    """在显示像素空间推开圆心，避免不同簇鸟图重叠。"""
    n = len(positions)
    if n <= 1:
        return list(positions)
    pts = [
        [float(p[0]), float(p[1])]
        for p in (ax.transData.transform((x, y)) for x, y in positions)
    ]
    for _ in range(max_iters):
        moved = False
        for j in range(n):
            for i in range(j):
                dx = pts[j][0] - pts[i][0]
                dy = pts[j][1] - pts[i][1]
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    ang = (i * 2.1 + j * 0.9) % (2 * math.pi)
                    dx, dy, d = math.cos(ang), math.sin(ang), 1.0
                if d >= min_sep_px:
                    continue
                push = (min_sep_px - d) * 0.52
                nx, ny = dx / d, dy / d
                pts[j][0] += nx * push
                pts[j][1] += ny * push
                pts[i][0] -= nx * push
                pts[i][1] -= ny * push
                moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    out: List[Tuple[float, float]] = []
    for p in pts:
        x, y = inv.transform((p[0], p[1]))
        out.append((float(x), float(y)))
    return out


def _cluster_label_xytext(
    idx: int,
    label_fs: float,
    thumb_diameter: int,
    *,
    dpi: float = EXPORT_DPI,
) -> Tuple[float, float, str, str]:
    """
    鸟名统一放在鸟图上方，且不与圆形重叠。
    返回 annotate 用的 offset points（pt）：圆半径(px→pt) + 字号余量。
    """
    _ = idx
    dpi = max(float(dpi), 1.0)
    # 布局按直径参数计像素，与 2.0.87 前网格一致
    radius_pt = (float(thumb_diameter) * 0.52) * (72.0 / dpi)
    pad_pt = max(float(label_fs) * 0.35, 4.0)
    gap = radius_pt + pad_pt
    return 0.0, gap, "center", "bottom"


def _label_box_axes_frac_at(
    ax,
    dx: float,
    dy: float,
    name: str,
    ox_pt: float,
    oy_pt: float,
    label_fs: float,
    ha: str,
    va: str,
) -> Tuple[float, float, float, float]:
    # annotate 偏移为 pt；_offset_points_to_data 按显示像素换算
    dpi = float(ax.figure.dpi)
    ox_px = float(ox_pt) * dpi / 72.0
    oy_px = float(oy_pt) * dpi / 72.0
    tx, ty = _offset_points_to_data(ax, dx, dy, ox_px, oy_px)
    w = _text_width_data(ax, name, label_fs)
    h = _text_height_axes_frac(ax, label_fs) * max(
        ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9
    )
    if ha == "center":
        x0, x1 = tx - w * 0.5, tx + w * 0.5
    elif ha == "right":
        x0, x1 = tx - w, tx
    else:
        x0, x1 = tx, tx + w
    if va == "bottom":
        y0, y1 = ty, ty + h
    elif va == "top":
        y0, y1 = ty - h, ty
    else:
        y0, y1 = ty - h * 0.5, ty + h * 0.5
    return _data_box_to_axes_frac(ax, x0, y0, x1, y1)


def _add_photo_markers(
    ax,
    photos: Sequence[BirdPhoto],
    *,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 44,
    use_gcj: bool = False,
    compact_labels: bool = False,
    resolve_overlaps: bool = True,
    basemap_style: str = "normal",
    on_basemap: bool = True,
    radius_km: float = 1.0,
) -> MapMarkerLayout:
    shown = list(photos)
    if max_markers is not None:
        shown = shown[: max(0, max_markers)]

    entries: List[Tuple[BirdPhoto, float, float]] = []
    for ph in shown:
        if ph.lat is None or ph.lon is None:
            continue
        ax_x, ax_y = _map_xy(ph.lon, ph.lat, use_gcj=use_gcj)
        entries.append((ph, ax_x, ax_y))

    if not entries:
        return MapMarkerLayout([], [], [])

    # 合并半径 = max(去重半径, 约 2.8 个鸟图直径对应的公里)
    merge_km = _layout_cluster_radius_km(ax, thumb_diameter, radius_km)
    groups = _cluster_photos_by_radius(entries, merge_km)

    label_fs = _map_typography(ax)["species_pt"]
    r_thumb = _thumb_radius_data(ax, thumb_diameter)
    label_color, _, use_stroke = _map_ink(basemap_style, on_basemap=on_basemap)
    label_effects = _map_text_effects(use_stroke=use_stroke)
    leader_color = _GPS_DOT_COLOR
    _ = compact_labels
    _ = resolve_overlaps

    cluster_layouts: List[_GpsClusterLayout] = []
    occupied: List[Tuple[float, float]] = []
    min_sep_px = _min_thumb_sep_px(thumb_diameter, float(ax.figure.dpi))
    for group in groups:
        anchor = _cluster_anchor_xy(group, use_gcj=use_gcj)
        preferred = _pick_side_keeping_grid_on_map(
            ax, anchor, len(group), thumb_diameter, label_fs
        )
        side = preferred
        best_hits = 10**9
        for cand in (preferred, -preferred):
            trial, _ = _layout_cluster_thumb_grid(
                ax, anchor, len(group), cand, thumb_diameter, label_fs
            )
            hits = _count_px_conflicts(ax, trial, occupied, min_sep_px)
            if hits < best_hits:
                best_hits = hits
                side = cand
                if hits == 0:
                    break
        positions, row_first = _layout_cluster_thumb_grid(
            ax, anchor, len(group), side, thumb_diameter, label_fs
        )
        positions = _clamp_positions_into_axes(ax, positions, thumb_diameter)
        occupied.extend(positions)
        cluster_layouts.append(
            _GpsClusterLayout(
                anchor=anchor,
                side=side,
                group=list(group),
                positions=positions,
                row_first_indices=row_first,
            )
        )

    if resolve_overlaps and cluster_layouts:
        flat_pos: List[Tuple[float, float]] = []
        spans: List[int] = []
        for layout in cluster_layouts:
            spans.append(len(layout.positions))
            flat_pos.extend(layout.positions)
        flat_pos = _separate_positions_px(ax, flat_pos, min_sep_px)
        flat_pos = _clamp_positions_into_axes(ax, flat_pos, thumb_diameter)
        k = 0
        for layout, npos in zip(cluster_layouts, spans):
            layout.positions = list(flat_pos[k : k + npos])
            k += npos
        r_thumb = _thumb_radius_data(ax, thumb_diameter)

    flat_displays: List[Tuple[float, float]] = []
    label_boxes_axes: List[Tuple[float, float, float, float]] = []
    thumb_boxes_axes: List[Tuple[float, float, float, float]] = []

    for layout in cluster_layouts:
        ax_x, ax_y = layout.anchor
        ax.scatter(
            [ax_x],
            [ax_y],
            s=max(18, thumb_diameter // 2) * _GPS_DOT_AREA_SCALE,
            c=_GPS_DOT_COLOR,
            edgecolors="white",
            linewidths=0.7,
            zorder=7,
        )
        for row_idx in layout.row_first_indices:
            if row_idx >= len(layout.positions):
                continue
            fx, fy = layout.positions[row_idx]
            ax.plot(
                [ax_x, fx],
                [ax_y, fy],
                color=leader_color,
                linewidth=1.35,
                alpha=0.92,
                zorder=6,
                solid_capstyle="round",
                clip_on=False,
            )

        for idx, ((ph, _, _), (dx, dy)) in enumerate(
            zip(layout.group, layout.positions)
        ):
            flat_displays.append((dx, dy))
            thumb_boxes_axes.append(_circle_box_axes_frac(ax, dx, dy, r_thumb))
            ox_pt, oy_pt, ha, va = _cluster_label_xytext(
                idx, label_fs, thumb_diameter, dpi=float(ax.figure.dpi)
            )
            label_boxes_axes.append(
                _label_box_axes_frac_at(
                    ax,
                    dx,
                    dy,
                    ph.species_cn,
                    ox_pt,
                    oy_pt,
                    label_fs,
                    ha,
                    va,
                )
            )
            try:
                arr = _circular_thumb_rgba(ph.path, thumb_diameter)
                zoom = _thumb_offset_zoom(
                    arr, thumb_diameter, float(ax.figure.dpi)
                )
                imagebox = OffsetImage(arr, zoom=zoom)
                ab = AnnotationBbox(
                    imagebox,
                    (dx, dy),
                    frameon=False,
                    pad=0,
                    zorder=8,
                    annotation_clip=False,
                )
                ab.set_clip_on(False)
                ax.add_artist(ab)
            except Exception:
                ax.scatter(
                    [dx],
                    [dy],
                    c=_GPS_DOT_COLOR,
                    s=max(28, thumb_diameter),
                    zorder=8,
                    clip_on=False,
                )
            ax.annotate(
                ph.species_cn,
                (dx, dy),
                textcoords="offset points",
                xytext=(ox_pt, oy_pt),
                ha=ha,
                va=va,
                fontsize=label_fs,
                color=label_color,
                path_effects=label_effects,
                zorder=9,
                annotation_clip=False,
                clip_on=False,
            )

    return MapMarkerLayout(
        displays=flat_displays,
        label_boxes_axes=label_boxes_axes,
        thumb_boxes_axes=thumb_boxes_axes,
    )


def _draw_track_on_ax(
    ax,
    track: Sequence[GpxPoint],
    *,
    use_gcj: bool,
    on_basemap: bool,
) -> None:
    if not track:
        return
    xs = [_map_xy(p.lon, p.lat, use_gcj=use_gcj)[0] for p in track]
    ys = [_map_xy(p.lon, p.lat, use_gcj=use_gcj)[1] for p in track]
    if on_basemap:
        line_color, line_w, pt_c, pt_ec, z = "#E67E22", 2.8, "#FFFFFF", "#E67E22", 7
        start_s, end_s = 72, 72
        edge = "white"
    else:
        line_color, line_w, pt_c, pt_ec, z = "#2980B9", 2.2, "#3498DB", None, 2
        start_s, end_s = 64, 64
        edge = None
    ax.plot(
        xs,
        ys,
        color=line_color,
        linewidth=line_w,
        linestyle="solid",
        zorder=z,
        solid_capstyle="round",
    )
    step = max(1, len(xs) // 80)
    kw = dict(
        c=pt_c,
        s=10 if not on_basemap else 12,
        alpha=0.45 if not on_basemap else 0.7,
        zorder=z,
    )
    if pt_ec:
        kw["edgecolors"] = pt_ec
        kw["linewidths"] = 0.6
    ax.scatter(xs[::step], ys[::step], **kw)
    ax.scatter(
        [xs[0]], [ys[0]], c="#27AE60", s=start_s, zorder=z, edgecolors=edge
    )
    ax.scatter(
        [xs[-1]], [ys[-1]], c="#E74C3C", s=end_s, zorder=z, edgecolors=edge
    )


def _key_place_ink(basemap_style: str, *, on_basemap: bool) -> Tuple[str, bool]:
    if on_basemap and is_dark_basemap_style(basemap_style):
        return _KEY_PLACE_COLOR_DARK, True
    return _KEY_PLACE_COLOR, on_basemap and is_satellite_basemap_style(basemap_style)


def _add_key_place_markers(
    ax,
    places: Sequence[KeyPlace],
    marker_layout: MapMarkerLayout,
    *,
    use_gcj: bool,
    thumb_diameter: int,
    basemap_style: str,
    on_basemap: bool,
    elev_south_frac: float = 0.0,
) -> MapMarkerLayout:
    """绘制关键地点：蓝色菱形 + 略大于鸟名的地名；地名可 x 重叠、只靠 y 错开。"""
    if not places:
        return marker_layout
    typo = _map_typography(ax)
    label_fs = _font_pt_for_line_height_px(
        typo["ax_h_px"] / _KEY_PLACE_LABEL_HEIGHT_DIV, ax.figure.dpi
    )
    color, use_stroke = _key_place_ink(basemap_style, on_basemap=on_basemap)
    effects = _map_text_effects(use_stroke=use_stroke) or [
        pe.withStroke(linewidth=1.6, foreground="#FFFFFF", alpha=0.72)
    ]
    pin_s = max(36, int(thumb_diameter * 0.38))
    r_pin = _thumb_radius_data(ax, max(12, int(thumb_diameter * 0.28)))
    y_min = 0.04 + (ELEV_PANEL_TOP_AXES if elev_south_frac > 0 else 0.0)
    y_max = 0.92
    x_min, x_max = 0.03, 0.97
    hard: List[Tuple[float, float, float, float]] = []
    hard.extend(marker_layout.thumb_boxes_axes)
    hard.append((0.0, 0.78, 1.0, 1.0))
    if elev_south_frac > 0:
        hard.append((0.0, 0.0, 1.0, ELEV_PANEL_TOP_AXES + 0.01))
    place_labels: List[Tuple[float, float, float, float]] = []

    pin_nudge = (
        (0.0, 0.0),
        (8.0, 8.0),
        (-8.0, 8.0),
        (8.0, -8.0),
        (-8.0, -8.0),
        (12.0, 0.0),
        (-12.0, 0.0),
        (0.0, 12.0),
        (0.0, -12.0),
    )
    step_pt = max(label_fs * 1.22, 11.0)
    y_offs = [step_pt * 0.75]
    for k in range(1, 18):
        y_offs.append(step_pt * 0.75 + k * step_pt)
        y_offs.append(-(step_pt * 0.75 + (k - 1) * step_pt))

    def _inside(box: Tuple[float, float, float, float]) -> bool:
        return (
            box[0] >= x_min
            and box[2] <= x_max
            and box[1] >= y_min
            and box[3] <= y_max
        )

    def _hits_hard(box: Tuple[float, float, float, float]) -> bool:
        return any(
            _rect_overlap_axes_frac(box, ob, margin=0.006) for ob in hard
        )

    def _hits_place_name(box: Tuple[float, float, float, float]) -> bool:
        return any(
            _rect_overlap_axes_frac(box, ob, margin=0.002) for ob in place_labels
        )

    def _pin_free(box: Tuple[float, float, float, float]) -> bool:
        return all(
            not _rect_overlap_axes_frac(box, ob, margin=0.008) for ob in hard
        )

    for pl in places:
        bx, by = _map_xy(pl.lon, pl.lat, use_gcj=use_gcj)
        px, py = bx, by
        pin_box = _circle_box_axes_frac(ax, px, py, r_pin)
        if not _pin_free(pin_box):
            for ox, oy in pin_nudge[1:]:
                nx, ny = _offset_points_to_data(ax, bx, by, ox, oy)
                cand = _circle_box_axes_frac(ax, nx, ny, r_pin)
                if _pin_free(cand) and _inside(cand):
                    px, py = nx, ny
                    pin_box = cand
                    break
        ax.scatter(
            [px],
            [py],
            s=pin_s,
            marker="D",
            c=color,
            edgecolors="white",
            linewidths=1.15,
            zorder=10,
            clip_on=False,
        )
        chosen = (0.0, y_offs[0], "center", "bottom")
        chosen_box = _label_box_axes_frac_at(
            ax, px, py, pl.name, 0.0, y_offs[0], label_fs, "center", "bottom"
        )
        fallback: Optional[Tuple[float, float, str, str, Tuple[float, float, float, float]]] = None
        for oy in y_offs:
            va = "bottom" if oy >= 0 else "top"
            box = _label_box_axes_frac_at(
                ax, px, py, pl.name, 0.0, oy, label_fs, "center", va
            )
            if not _inside(box) or _hits_place_name(box):
                continue
            if fallback is None:
                fallback = (0.0, oy, "center", va, box)
            if _hits_hard(box) or _rect_overlap_axes_frac(box, pin_box, margin=0.003):
                continue
            chosen = (0.0, oy, "center", va)
            chosen_box = box
            fallback = None
            break
        if fallback is not None:
            chosen = fallback[:4]
            chosen_box = fallback[4]
        ox, oy, ha, va = chosen
        ax.annotate(
            pl.name,
            (px, py),
            textcoords="offset points",
            xytext=(ox, oy),
            ha=ha,
            va=va,
            fontsize=label_fs,
            color=color,
            fontstyle="italic",
            path_effects=effects,
            zorder=11,
            annotation_clip=False,
            clip_on=False,
        )
        marker_layout.displays.append((px, py))
        marker_layout.label_boxes_axes.append(chosen_box)
        marker_layout.thumb_boxes_axes.append(pin_box)
        place_labels.append(chosen_box)
        hard.append(pin_box)
    return marker_layout


def _plot_map_ax(
    ax,
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    place: str,
    *,
    map_title: str = "观鸟记录",
    date_label: str = "",
    logo_path: str = "",
    logo_width_ratio: float = 0.30,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 44,
    basemap_style: str = "normal",
    map_width_px: int = 1080,
    map_height_px: int = 1200,
    compact_labels: bool = False,
    resolve_overlaps: bool = True,
    summary_default_y: float = 0.19,
    radius_km: float = 1.0,
    elev_south_frac: float = 0.0,
    key_places: Sequence[KeyPlace] = (),
) -> str:
    """
    绘制地图子图（高德底图 + GCJ-02 叠加）。
    返回 basemap 状态：ok / fallback / none / no_key。
    """
    raw = (basemap_style or "normal").strip().lower()
    if raw in ("none", "off", "grid", "无", "网格"):
        style = "none"
        _draw_track_on_ax(ax, track, use_gcj=False, on_basemap=False)
        ax.set_xlabel("经度", fontsize=11)
        ax.set_ylabel("纬度", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.set_aspect("equal", adjustable="datalim")
        view_bounds = _resolve_map_view_bounds(track, photos)
        if view_bounds is not None:
            x0, x1, y0, y1 = view_bounds
        else:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
        x0, x1, y0, y1 = _expand_lonlat_bounds(
            x0, x1, y0, y1, _subplot_aspect_wh(ax)
        )
        if elev_south_frac > 0:
            x0, x1, y0, y1 = reserve_south_lat(x0, x1, y0, y1, elev_south_frac)
            x0, x1, y0, y1 = fit_lon_to_aspect(
                x0, x1, y0, y1, _subplot_aspect_wh(ax)
            )
        # 先定视野再排布鸟图，保证像素间距按最终坐标系换算
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        marker_layout = _add_photo_markers(
            ax,
            photos,
            max_markers=max_markers,
            thumb_diameter=thumb_diameter,
            use_gcj=False,
            compact_labels=compact_labels,
            resolve_overlaps=resolve_overlaps,
            basemap_style="none",
            on_basemap=False,
            radius_km=radius_km,
        )
        marker_layout = _add_key_place_markers(
            ax,
            key_places,
            marker_layout,
            use_gcj=False,
            thumb_diameter=thumb_diameter,
            basemap_style="none",
            on_basemap=False,
            elev_south_frac=elev_south_frac,
        )
        _draw_map_chrome(
            ax,
            place,
            map_title,
            date_label,
            photos,
            marker_layout,
            track,
            logo_path=logo_path,
            logo_width_ratio=logo_width_ratio,
            basemap_style="none",
            on_basemap=False,
            use_gcj=False,
            elev_south_frac=elev_south_frac,
        )
        return "none"

    style = normalize_basemap_style(raw)
    view_bounds = _resolve_map_view_bounds(track, photos)
    if view_bounds is None:
        lons, lats = [], []
    else:
        lon_min, lon_max, lat_min, lat_max = view_bounds
        lons, lats = [lon_min, lon_max], [lat_min, lat_max]
    if not lons:
        ax.axis("off")
        _draw_map_chrome(
            ax,
            place,
            map_title,
            date_label,
            photos,
            None,
            track,
            logo_path=logo_path,
            logo_width_ratio=logo_width_ratio,
            basemap_style=style,
            on_basemap=True,
            use_gcj=True,
            elev_south_frac=elev_south_frac,
        )
        return "fallback"

    basemap_ok = False
    try:
        lon_min, lon_max, lat_min, lat_max = gcj_bounds_from_lonlats(lons, lats)
        lon_min, lon_max, lat_min, lat_max = _expand_lonlat_bounds(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            _subplot_aspect_wh(ax),
        )
        img, extent = fetch_amap_basemap_rgba(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            width_px=map_width_px,
            height_px=map_height_px,
            style=style,
            south_reserve_frac=elev_south_frac,
        )
        ax.imshow(
            img,
            extent=extent,
            origin="upper",
            aspect="auto",
            zorder=0,
            interpolation="bilinear",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        basemap_ok = True
        # 底图已按画布拉伸铺满，勿再用 equal+box 收缩坐标系（会左右留白）
        ax.set_aspect("auto")
    except ValueError as e:
        if "API Key" in str(e) or "api_key" in str(e):
            ax.clear()
            _plot_map_ax(
                ax,
                track,
                photos,
                place,
                map_title=map_title,
                date_label=date_label,
                logo_path=logo_path,
                logo_width_ratio=logo_width_ratio,
                max_markers=max_markers,
                thumb_diameter=thumb_diameter,
                basemap_style="none",
                map_width_px=map_width_px,
                map_height_px=map_height_px,
                compact_labels=compact_labels,
                resolve_overlaps=resolve_overlaps,
                summary_default_y=summary_default_y,
                radius_km=radius_km,
                elev_south_frac=elev_south_frac,
                key_places=key_places,
            )
            return "no_key"
    except Exception:
        pass

    if not basemap_ok:
        ax.clear()
        _plot_map_ax(
            ax,
            track,
            photos,
            place,
            map_title=map_title,
            date_label=date_label,
            logo_path=logo_path,
            logo_width_ratio=logo_width_ratio,
            max_markers=max_markers,
            thumb_diameter=thumb_diameter,
            basemap_style="none",
            map_width_px=map_width_px,
            map_height_px=map_height_px,
            compact_labels=compact_labels,
            resolve_overlaps=resolve_overlaps,
            summary_default_y=summary_default_y,
            radius_km=radius_km,
            elev_south_frac=elev_south_frac,
            key_places=key_places,
        )
        return "fallback"

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_track_on_ax(ax, track, use_gcj=True, on_basemap=True)
    marker_layout = _add_photo_markers(
        ax,
        photos,
        max_markers=max_markers,
        thumb_diameter=thumb_diameter,
        use_gcj=True,
        compact_labels=compact_labels,
        resolve_overlaps=resolve_overlaps,
        basemap_style=style,
        on_basemap=True,
        radius_km=radius_km,
    )
    marker_layout = _add_key_place_markers(
        ax,
        key_places,
        marker_layout,
        use_gcj=True,
        thumb_diameter=thumb_diameter,
        basemap_style=style,
        on_basemap=True,
        elev_south_frac=elev_south_frac,
    )
    _draw_map_chrome(
        ax,
        place,
        map_title,
        date_label,
        photos,
        marker_layout,
        track,
        logo_path=logo_path,
        logo_width_ratio=logo_width_ratio,
        basemap_style=style,
        on_basemap=True,
        use_gcj=True,
        elev_south_frac=elev_south_frac,
    )
    return "ok"


def _offset_points_to_data(
    ax,
    x: float,
    y: float,
    ox_pt: float,
    oy_pt: float,
) -> Tuple[float, float]:
    disp = ax.transData.transform((x, y))
    off = ax.transData.inverted().transform((disp[0] + ox_pt, disp[1] + oy_pt))
    return float(off[0]), float(off[1])


def _elev_plot_limits(
    y_min: float, y_max: float, x_max: float
) -> Tuple[float, float, float, float]:
    """海拔内层 axes 的 x/y 范围：上宽下窄、右宽左紧。"""
    x1 = max(float(x_max), 0.01)
    y_rng = max(float(y_max) - float(y_min), 1e-6)
    return (
        -x1 * _ELEV_X_LEFT_FRAC,
        x1 * (1.0 + _ELEV_X_RIGHT_FRAC),
        y_min - y_rng * _ELEV_Y_BOTTOM_FRAC,
        y_max + y_rng * _ELEV_Y_TOP_FRAC,
    )


def _elev_label_pads(
    plot_x0: float, plot_x1: float, plot_y0: float, plot_y1: float
) -> Tuple[float, float]:
    x_rng = max(plot_x1 - plot_x0, 1e-6)
    y_rng = max(plot_y1 - plot_y0, 1e-6)
    return x_rng * _ELEV_LABEL_MARGIN_FRAC, y_rng * _ELEV_LABEL_MARGIN_FRAC


def _ensure_ax_display(ax) -> None:
    """inset 在未 draw 时 bbox 会严重偏小，占空会被放大到整张图。"""
    try:
        bbox = ax.get_window_extent()
    except Exception:
        bbox = None
    if bbox is not None and bbox.width >= 8 and bbox.height >= 8:
        return
    try:
        ax.figure.canvas.draw()
    except Exception:
        return


def _display_px_to_data(ax, dx_px: float, dy_px: float) -> Tuple[float, float]:
    """把屏幕像素换算成当前 axes 的数据宽高。"""
    _ensure_ax_display(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    p0 = ax.transData.transform((cx, cy))
    x1d, y1d = ax.transData.inverted().transform(
        (p0[0] + dx_px, p0[1] + dy_px)
    )
    return abs(float(x1d) - cx), abs(float(y1d) - cy)


def _elev_text_height_data(ax, fontsize_pt: float) -> float:
    dpi = max(float(ax.figure.dpi), 1.0)
    h_px = max(1.0, float(fontsize_pt) * dpi / 72.0)
    _w, h = _display_px_to_data(ax, 0.0, h_px)
    return max(h, 1e-9)


def _elev_label_ha_for_marker(ad: float, data_x_max: float) -> str:
    """左半图左对齐、右半图右对齐，避免鸟名超出绘图区。"""
    mid = max(float(data_x_max), 1e-6) * 0.5
    return "left" if ad <= mid else "right"


def _elev_name_size_data(ax, name: str, fontsize_pt: float) -> Tuple[float, float]:
    """按字号像素换算占空，不再用 inset 的错误 axes 高度去放大。"""
    dpi = max(float(ax.figure.dpi), 1.0)
    em_px = max(1.0, float(fontsize_pt) * dpi / 72.0)
    n = max(len(name or ""), 1)
    w, h = _display_px_to_data(ax, n * em_px, em_px)
    return max(w, 1e-9), max(h, 1e-9)


def _elev_label_box_data(
    ax,
    lx: float,
    ly: float,
    name: str,
    fontsize_pt: float,
    *,
    ha: str = "center",
    va: str = "center",
) -> Tuple[float, float, float, float]:
    w, h = _elev_name_size_data(ax, name, fontsize_pt)
    if ha == "left":
        x0, x1 = lx, lx + w
    elif ha == "right":
        x0, x1 = lx - w, lx
    else:
        x0, x1 = lx - w * 0.5, lx + w * 0.5
    if va == "bottom":
        y0, y1 = ly, ly + h
    elif va == "top":
        y0, y1 = ly - h, ly
    else:
        y0, y1 = ly - h * 0.5, ly + h * 0.5
    return (x0, y0, x1, y1)


def _rect_overlap_data(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _box_inside_elev_plot(
    box: Tuple[float, float, float, float],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    pad_x: float,
    pad_y: float,
) -> bool:
    return (
        box[0] >= x_min + pad_x
        and box[2] <= x_max - pad_x
        and box[1] >= y_min + pad_y
        and box[3] <= y_max - pad_y
    )


ELEV_LABEL_MAX_OVERLAP = 0.0
ELEV_LABEL_REFINE_PASSES_MIN = 6
ELEV_LABEL_REFINE_PASSES_MAX = 36


def _label_layout_refine_passes(n: int) -> int:
    """鸟种名/标注重叠消解迭代次数（随标注数增加）。"""
    if n <= 1:
        return 1
    return max(
        ELEV_LABEL_REFINE_PASSES_MIN,
        min(ELEV_LABEL_REFINE_PASSES_MAX, 4 + n // 5),
    )


def _rect_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _rect_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _rect_overlap_fraction(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    inter = _rect_intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    denom = min(_rect_area(a), _rect_area(b))
    if denom <= 0.0:
        return 1.0
    return inter / denom


def _inflate_display_box(
    box: Tuple[float, float, float, float],
    px: float = 4.0,
) -> Tuple[float, float, float, float]:
    return (box[0] - px, box[1] - px, box[2] + px, box[3] + px)


def _fix_display_box(
    box: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (
        min(box[0], box[2]),
        min(box[1], box[3]),
        max(box[0], box[2]),
        max(box[1], box[3]),
    )


def _elev_label_box_display(
    ax,
    lx: float,
    ly: float,
    name: str,
    fontsize_pt: float,
    *,
    ha: str = "center",
) -> Tuple[float, float, float, float]:
    box = _elev_label_box_data(ax, lx, ly, name, fontsize_pt, ha=ha)
    p0 = ax.transData.transform((box[0], box[1]))
    p1 = ax.transData.transform((box[2], box[3]))
    return _fix_display_box(
        (float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
    )


def _max_label_overlap_fraction(
    box: Tuple[float, float, float, float],
    others: Sequence[Tuple[float, float, float, float]],
    *,
    inflate_px: float = 4.0,
) -> float:
    inflated = _inflate_display_box(box, px=inflate_px)
    worst = 0.0
    for ob in others:
        worst = max(
            worst,
            _rect_overlap_fraction(inflated, _inflate_display_box(ob, px=inflate_px)),
        )
    return worst


def _elev_text_fits_in_plot(
    ax,
    lx: float,
    ly: float,
    name: str,
    fontsize_pt: float,
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
    ha: str = "center",
) -> bool:
    box = _elev_label_box_data(ax, lx, ly, name, fontsize_pt, ha=ha)
    return _box_inside_elev_plot(
        box, x_min, y_min, x_max, y_max, pad_x=pad_x, pad_y=pad_y
    )


def _iter_elev_label_candidates(
    ax,
    ad: float,
    ae: float,
    name: str,
    label_fs: float,
    *,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
) -> List[Tuple[float, float, int, float]]:
    """返回 (label_x, label_y, 优先级 tier, 同 tier 内 tie-break)。tier=0 为垂直列（ld=ad）。"""
    x_rng = max(x_max - x_min, 1e-6)
    y_rng = max(y_max - y_min, 1e-6)
    ha = _elev_label_ha_for_marker(ad, data_x_max)
    out: List[Tuple[float, float, int, float]] = []
    seen: set = set()

    def add(ld: float, le: float, tier: int, tie: float) -> None:
        if le < y_min + pad_y or le > y_max - pad_y:
            return
        if not _elev_text_fits_in_plot(
            ax,
            ld,
            le,
            name,
            label_fs,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            pad_x=pad_x,
            pad_y=pad_y,
            ha=ha,
        ):
            return
        key = (round(ld, 5), round(le, 5))
        if key in seen:
            return
        seen.add(key)
        out.append((ld, le, tier, tie))

    if x_min <= ad <= x_max:
        h = _elev_text_height_data(ax, label_fs)
        y_lo = y_min + pad_y + h * 0.5
        y_hi = y_max - pad_y - h * 0.5
        if y_lo <= y_hi:
            step = max(h * 0.34, y_rng * 0.02)
            n = max(12, min(200, int((y_hi - y_lo) / step) + 1))
            for i in range(n):
                le = y_lo + (y_hi - y_lo) * i / max(n - 1, 1)
                tie = abs(le - ae) / y_rng
                add(ad, le, 0, tie)

    for oy in (8, -8, 12, -12, 18, -18, 26, -26, 36, -36, 48, -48):
        ld, le = _offset_points_to_data(ax, ad, ae, 0.0, float(oy))
        if abs(ld - ad) > x_rng * 0.001:
            continue
        tie = abs(le - ae) / y_rng
        add(ad, le, 0, tie + 0.001)

    for ox in (6, -6, 10, -10, 14, -14, 18, -18, 24, -24):
        for oy in (8, -8, 14, -14, 22, -22, 32, -32):
            ld, le = _offset_points_to_data(ax, ad, ae, float(ox), float(oy))
            tie = abs(le - ae) / y_rng + abs(ld - ad) / x_rng * 1.5
            add(ld, le, 2, tie)

    for dy in (-0.12, -0.08, -0.04, 0.04, 0.08, 0.12):
        for dx in (-0.05, -0.025, 0.025, 0.05):
            ld = ad + dx * x_rng
            le = ae + dy * y_rng
            tie = abs(le - ae) / y_rng + abs(ld - ad) / x_rng * 2.0
            add(ld, le, 3, tie)
    return out


def _pick_elev_label_position(
    ax,
    ad: float,
    ae: float,
    name: str,
    label_fs: float,
    *,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
    blocked_disp: Sequence[Tuple[float, float, float, float]],
    other_disp: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float, float]:
    """返回 (lx, ly, max_overlap_fraction)。"""
    all_obstacles = list(blocked_disp) + list(other_disp)
    ha = _elev_label_ha_for_marker(ad, data_x_max)
    candidates = _iter_elev_label_candidates(
        ax,
        ad,
        ae,
        name,
        label_fs,
        data_x_max=data_x_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    best_ld, best_le = ad, ae
    best_overlap = 1.0

    x_tol = max(x_max - x_min, 1e-6) * 0.002

    def _pick_in_tiers(tiers: Sequence[int]) -> Optional[Tuple[float, float, float]]:
        tier_best: Optional[Tuple[float, float, float]] = None
        tier_score = 1e18
        for ld, le, cand_tier, tie in candidates:
            if cand_tier not in tiers:
                continue
            box = _elev_label_box_display(ax, ld, le, name, label_fs, ha=ha)
            ov = _max_label_overlap_fraction(box, all_obstacles)
            if ov > ELEV_LABEL_MAX_OVERLAP:
                continue
            col_penalty = 0.0 if abs(ld - ad) <= x_tol else 0.04
            score = ov * 800.0 + tie + col_penalty + cand_tier * 0.01
            if score < tier_score:
                tier_score = score
                tier_best = (ld, le, ov)
        return tier_best

    vertical = _pick_in_tiers((0,))
    if vertical is not None:
        return vertical
    for tiers in ((2,), (3,)):
        picked = _pick_in_tiers(tiers)
        if picked is not None:
            return picked

    fb_score = 1e18
    for ld, le, cand_tier, tie in candidates:
        box = _elev_label_box_display(ax, ld, le, name, label_fs, ha=ha)
        ov = _max_label_overlap_fraction(box, all_obstacles)
        score = ov * 1200.0 + cand_tier * 5.0 + tie
        if score < fb_score:
            fb_score = score
            best_ld, best_le, best_overlap = ld, le, ov
    return best_ld, best_le, best_overlap


def _elev_worst_pairwise_overlap(
    boxes: Sequence[Tuple[float, float, float, float]],
) -> float:
    worst = 0.0
    for i in range(len(boxes)):
        bi = _inflate_display_box(boxes[i])
        for j in range(i + 1, len(boxes)):
            worst = max(worst, _rect_overlap_fraction(bi, _inflate_display_box(boxes[j])))
    return worst


def _try_vertical_elev_label(
    ax,
    ad: float,
    ae: float,
    name: str,
    label_fs: float,
    *,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
    obstacles: Sequence[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float]]:
    """在记录点 x=ad 的垂直列上找不重叠的 y（左/右对齐由 ad 所在半区决定）。"""
    if not (x_min <= ad <= x_max):
        return None
    ha = _elev_label_ha_for_marker(ad, data_x_max)
    y_rng = max(y_max - y_min, 1e-6)
    h = _elev_text_height_data(ax, label_fs)
    y_lo = y_min + pad_y + h * 0.5
    y_hi = y_max - pad_y - h * 0.5
    if y_lo > y_hi:
        return None
    crowd = max(1, len(obstacles))
    step = max(
        h * 0.26,
        y_rng * 0.015 / min(3.0, math.sqrt(float(crowd))),
    )
    n_slots = max(12, min(200, int((y_hi - y_lo) / step) + 1))
    le_cands = [y_lo + (y_hi - y_lo) * i / max(n_slots - 1, 1) for i in range(n_slots)]
    le_cands.sort(key=lambda le: (0 if le >= ae else 1, abs(le - ae), le))
    for le in le_cands:
        if not _elev_text_fits_in_plot(
            ax,
            ad,
            le,
            name,
            label_fs,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            pad_x=pad_x,
            pad_y=pad_y,
            ha=ha,
        ):
            continue
        box = _elev_label_box_display(ax, ad, le, name, label_fs, ha=ha)
        if _max_label_overlap_fraction(box, obstacles, inflate_px=1.5) <= ELEV_LABEL_MAX_OVERLAP:
            return ad, le
    return None


def _elev_layout_overlap_score(
    idx: int,
    layouts: Sequence[Tuple[float, float, float, float, str]],
    *,
    ax,
    label_fs: float,
    data_x_max: float,
    x_min: float,
    x_max: float,
    blocked_disp: Sequence[Tuple[float, float, float, float]],
) -> float:
    ad, _, ld, le, name = layouts[idx]
    ha = _elev_label_ha_for_marker(ad, data_x_max)
    box_i = _elev_label_box_display(ax, ld, le, name, label_fs, ha=ha)
    others = list(blocked_disp)
    for j, item in enumerate(layouts):
        if j == idx:
            continue
        ad_j, _, ld_j, le_j, name_j = item
        ha_j = _elev_label_ha_for_marker(ad_j, data_x_max)
        others.append(
            _elev_label_box_display(ax, ld_j, le_j, name_j, label_fs, ha=ha_j)
        )
    return _max_label_overlap_fraction(box_i, others)


def _elev_refine_one_marker(
    idx: int,
    markers: Sequence[Tuple[float, float, str]],
    layouts: List[Tuple[float, float, float, float, str]],
    *,
    ax,
    label_fs: float,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
    blocked_disp: Sequence[Tuple[float, float, float, float]],
    layout_boxes_fn,
) -> None:
    ad, ae, name = markers[idx]
    other_disp = layout_boxes_fn(layouts, skip_idx=idx)[len(blocked_disp) :]
    obstacles = list(blocked_disp) + other_disp
    picked = _try_vertical_elev_label(
        ax,
        ad,
        ae,
        name,
        label_fs,
        data_x_max=data_x_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pad_x=pad_x,
        pad_y=pad_y,
        obstacles=obstacles,
    )
    if picked is not None:
        layouts[idx] = (ad, ae, picked[0], picked[1], name)
        return
    ld, le, _ = _pick_elev_label_position(
        ax,
        ad,
        ae,
        name,
        label_fs,
        data_x_max=data_x_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        pad_x=pad_x,
        pad_y=pad_y,
        blocked_disp=blocked_disp,
        other_disp=other_disp,
    )
    layouts[idx] = (ad, ae, ld, le, name)


def _cluster_elev_marker_groups(
    ax,
    markers: Sequence[Tuple[float, float, str]],
    label_fs: float,
) -> List[List[int]]:
    """把横向过近、鸟名框会互相挡住的点收成同一列。"""
    n = len(markers)
    if n == 0:
        return []
    widths = [_text_width_data(ax, m[2], label_fs) for m in markers]
    x0, x1 = ax.get_xlim()
    ax_w_px, _ = _ax_size_px(ax)
    px_to_data = max(x1 - x0, 1e-9) / max(ax_w_px, 1.0)
    min_merge = 32.0 * px_to_data
    order = sorted(range(n), key=lambda i: (markers[i][0], i))
    groups: List[List[int]] = [[order[0]]]
    for i in order[1:]:
        prev = groups[-1][-1]
        dx = markers[i][0] - markers[prev][0]
        # 鸟名框允许 x 重叠，近到会叠字的点收成一列、只靠 y 错开
        thresh = max(min_merge, widths[prev], widths[i])
        if dx <= thresh:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _pack_elev_cluster_stack(
    ax,
    cluster: Sequence[Tuple[float, float, str]],
    label_fs: float,
    *,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
    blocked_disp: Sequence[Tuple[float, float, float, float]],
) -> Optional[List[Tuple[float, float]]]:
    """在簇的共用 x 上、优先点正上方紧密叠放鸟名。返回与 cluster 同序的 (ld, le)。"""
    n = len(cluster)
    if n <= 0:
        return []
    ads = [float(m[0]) for m in cluster]
    names = [m[2] for m in cluster]
    col_x = float(sorted(ads)[n // 2])
    max_w = max(_text_width_data(ax, name, label_fs) for name in names)
    ha = _elev_label_ha_for_marker(col_x, data_x_max)
    if ha == "left":
        col_x = min(max(col_x, x_min + pad_x), max(x_min + pad_x, x_max - pad_x - max_w))
    else:
        col_x = max(min(col_x, x_max - pad_x), min(x_max - pad_x, x_min + pad_x + max_w))
    ha = _elev_label_ha_for_marker(col_x, data_x_max)

    h = _elev_text_height_data(ax, label_fs)
    gap = h * 1.08
    y_lo = y_min + pad_y + h * 0.5
    y_hi = y_max - pad_y - h * 0.5
    if y_lo > y_hi:
        return None
    if n > 1 and (n - 1) * gap > y_hi - y_lo:
        gap = max(h * 1.0, (y_hi - y_lo) / (n - 1))
    need = (n - 1) * gap
    ae_max = max(float(m[1]) for m in cluster)
    ae_min = min(float(m[1]) for m in cluster)

    def centers_from_top(top: float) -> List[float]:
        return [top - i * gap for i in range(n)]

    def stack_ok(centers: Sequence[float]) -> bool:
        if any(c < y_lo - 1e-6 or c > y_hi + 1e-6 for c in centers):
            return False
        for i, c in enumerate(centers):
            if not _elev_text_fits_in_plot(
                ax,
                col_x,
                c,
                names[i],
                label_fs,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                pad_x=pad_x,
                pad_y=pad_y,
                ha=ha,
            ):
                return False
            box = _elev_label_box_display(ax, col_x, c, names[i], label_fs, ha=ha)
            if _max_label_overlap_fraction(box, blocked_disp, inflate_px=1.5) > 0.02:
                return False
        return True

    step = max(h * 0.18, (y_hi - y_lo) * 0.01)
    tops: List[float] = []
    t = y_hi
    while t + 1e-9 >= y_lo + need:
        tops.append(t)
        t -= step

    def top_score(top: float) -> Tuple[int, float]:
        cents = centers_from_top(top)
        bottom = cents[-1]
        if bottom >= ae_max + h * 0.35:
            return (0, abs(bottom - (ae_max + h * 0.65)))
        if cents[0] <= ae_min - h * 0.35:
            return (2, abs(cents[0] - (ae_min - h * 0.65)))
        return (1, abs(bottom - ae_max))

    for top in sorted(tops, key=top_score):
        cents = centers_from_top(top)
        if stack_ok(cents):
            return [(col_x, c) for c in cents]
    return None


def _layout_elevation_species_labels(
    ax,
    markers: Sequence[Tuple[float, float, str]],
    label_fs: float,
    *,
    data_x_max: float,
    blocked: Sequence[Tuple[float, float, float, float]] = (),
) -> List[Tuple[float, float, float, float, str]]:
    """每个鸟名先放最高处；与已放的重叠则只改后一个的 y，直到不重叠。"""
    if not markers:
        return []
    _ensure_ax_display(ax)
    _plot_x0, _plot_x1 = ax.get_xlim()
    plot_y0, plot_y1 = ax.get_ylim()
    _pad_x, pad_y = _elev_label_pads(_plot_x0, _plot_x1, plot_y0, plot_y1)
    placed_boxes: List[Tuple[float, float, float, float]] = list(blocked)
    layouts: List[Optional[Tuple[float, float, float, float, str]]] = [None] * len(
        markers
    )
    order = sorted(range(len(markers)), key=lambda i: (markers[i][0], i))

    for idx in order:
        ad, ae, name = markers[idx]
        ha = _elev_label_ha_for_marker(ad, data_x_max)
        _w, h = _elev_name_size_data(ax, name, label_fs)
        gap = h * 0.12
        y_hi = plot_y1 - pad_y - h * 0.5

        def _box_at(y: float) -> Tuple[float, float, float, float]:
            return _elev_label_box_data(ax, ad, y, name, label_fs, ha=ha)

        y = y_hi
        for _ in range(max(16, len(placed_boxes) * 3 + 8)):
            hits = [
                prev
                for prev in placed_boxes
                if _rect_overlap_data(_box_at(y), prev)
            ]
            if not hits:
                break
            y = min(prev[1] for prev in hits) - gap - h * 0.5

        placed_boxes.append(_box_at(y))
        layouts[idx] = (ad, ae, ad, y, name)

    return [item for item in layouts if item is not None]


def _elev_unstick_overlaps(
    ax,
    layouts: List[Tuple[float, float, float, float, str]],
    *,
    label_fs: float,
    data_x_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pad_x: float,
    pad_y: float,
) -> None:
    """最后再沿垂直方向推开仍相交的鸟名。"""
    n = len(layouts)
    if n <= 1:
        return
    h = _elev_text_height_data(ax, label_fs)
    y_lo = y_min + pad_y + h * 0.5
    y_hi = y_max - pad_y - h * 0.5
    if y_lo > y_hi:
        return
    step = max(h * 0.55, (y_hi - y_lo) * 0.03)
    for _ in range(max(16, n * 4)):
        worst_i = worst_j = -1
        worst = 0.0
        for i in range(n):
            ad_i, _, ld_i, le_i, name_i = layouts[i]
            ha_i = _elev_label_ha_for_marker(ld_i, data_x_max)
            bi = _elev_label_box_display(ax, ld_i, le_i, name_i, label_fs, ha=ha_i)
            for j in range(i + 1, n):
                ad_j, _, ld_j, le_j, name_j = layouts[j]
                ha_j = _elev_label_ha_for_marker(ld_j, data_x_max)
                bj = _elev_label_box_display(
                    ax, ld_j, le_j, name_j, label_fs, ha=ha_j
                )
                ov = _rect_overlap_fraction(
                    _inflate_display_box(bi), _inflate_display_box(bj)
                )
                if ov > worst:
                    worst = ov
                    worst_i, worst_j = i, j
        if worst <= 1e-9 or worst_i < 0:
            break
        ad, ae, ld, le, name = layouts[worst_j]
        other_le = layouts[worst_i][3]
        direction = 1.0 if le >= other_le else -1.0
        le_new = min(max(le + direction * step, y_lo), y_hi)
        ha = _elev_label_ha_for_marker(ld, data_x_max)
        if not _elev_text_fits_in_plot(
            ax,
            ld,
            le_new,
            name,
            label_fs,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            pad_x=pad_x,
            pad_y=pad_y,
            ha=ha,
        ):
            le_new = min(max(le - direction * step, y_lo), y_hi)
        layouts[worst_j] = (ad, ae, ld, le_new, name)


def _elev_tick_step(y_range: float) -> int:
    if y_range <= 120:
        return 20
    if y_range <= 300:
        return 20
    if y_range <= 600:
        return 50
    return 100


def _elev_axis_limits(ele: Sequence[float]) -> Tuple[float, float]:
    if not ele:
        return 0.0, 100.0
    lo, hi = float(min(ele)), float(max(ele))
    pad = max(8.0, (hi - lo) * 0.08)
    step = _elev_tick_step(hi - lo + 2 * pad)
    y0 = math.floor((lo - pad) / step) * step
    y1 = math.ceil((hi + pad) / step) * step
    if y1 - y0 < step * 2:
        y1 = y0 + step * 2
    return y0, y1


def _elevation_highlight_labels(
    dist: Sequence[float],
    ele: Sequence[float],
) -> List[Tuple[float, float, str]]:
    """最高、最低与终点海拔标注（参考图圆点 + 数值）。"""
    if len(dist) < 2:
        return []
    pts: List[Tuple[float, float, str]] = []
    imax = int(np.argmax(ele))
    imin = int(np.argmin(ele))
    for idx in (imax, imin, len(dist) - 1):
        d, e = float(dist[idx]), float(ele[idx])
        label = f"{int(round(e))}m"
        if any(abs(d - pd) < 0.05 and abs(e - pe) < 0.5 for pd, pe, _ in pts):
            continue
        pts.append((d, e, label))
    return pts


def _style_elevation_panel_frame(ax) -> None:
    """海拔图外层面板：白底 + 灰色外框（含坐标刻度区域）。"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_facecolor("#FFFFFF")
    ax.patch.set_edgecolor(ELEV_PANEL_BORDER)
    ax.patch.set_linewidth(1.2)
    ax.patch.set_visible(True)


def _create_elevation_panel_axes(map_ax):
    """创建外层白底灰框 + 内层绘图区（坐标与标签均在外框内）。"""
    outer = map_ax.inset_axes(
        [0.04, 0.03, 0.92, 0.14],
        transform=map_ax.transAxes,
        zorder=30,
    )
    _style_elevation_panel_frame(outer)
    inner = outer.inset_axes(list(_ELEV_INNER_RECT))
    inner.set_facecolor("#FFFFFF")
    inner.patch.set_edgecolor("none")
    inner.patch.set_linewidth(0.0)
    return outer, inner


def _style_elevation_inset_ax(
    ax,
    *,
    y_min: float,
    y_max: float,
    x_max: float,
    tick_fs: float,
    label_fs: float,
) -> None:
    """海拔内层绘图区：浅绿填充、水平网格、轴端标签。"""
    ax.set_facecolor("#FFFFFF")
    ax.patch.set_edgecolor("none")
    x_lo, x_hi, y_lo, y_hi = _elev_plot_limits(y_min, y_max, x_max)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    y_step = _elev_tick_step(y_max - y_min)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=min(12, max(4, int(x_max) + 1)), integer=True, steps=[1, 2, 5, 10])
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.55,
        color=ELEV_GRID_COLOR,
        alpha=0.95,
        zorder=0,
    )
    ax.grid(axis="x", visible=False)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        length=3.0,
        width=0.65,
        colors=ELEV_AXIS_COLOR,
        labelsize=tick_fs,
        pad=1,
    )
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color(ELEV_AXIS_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x1 = max(x_max, 0.01)
    ax.annotate(
        "",
        xy=(x1, y_min),
        xytext=(x1 * 0.992, y_min),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.75,
            color=ELEV_AXIS_COLOR,
            mutation_scale=8,
        ),
        zorder=4,
    )
    ax.annotate(
        "",
        xy=(0.0, y_min),
        xytext=(0.0, y_max),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=0.75,
            color=ELEV_AXIS_COLOR,
            mutation_scale=8,
        ),
        zorder=4,
    )

    ax.text(
        0.0,
        1.06,
        "海拔(m)",
        transform=ax.transAxes,
        fontsize=label_fs,
        color=ELEV_LABEL_COLOR,
        ha="left",
        va="top",
        clip_on=False,
        zorder=5,
    )
    ax.text(
        1.00,
        0.02,
        "里程(km)",
        transform=ax.transAxes,
        fontsize=label_fs,
        color=ELEV_LABEL_COLOR,
        ha="right",
        va="bottom",
        clip_on=False,
        zorder=5,
    )


def _plot_elevation_ax(
    ax,
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    *,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 36,
    align: Optional[TrackTimeAlignment] = None,
    inset: bool = True,
    map_ax=None,
) -> None:
    if align is None:
        align = TrackTimeAlignment()
    ts_prof, dist_prof, ele_prof = _build_timed_profile(track, align)
    if not dist_prof:
        return
    y_min, y_max = _elev_axis_limits(ele_prof)
    x_max = float(dist_prof[-1])
    x_min = 0.0
    if map_ax is not None:
        typo = _map_typography(map_ax)
        tick_fs = typo["elev_axis_pt"]
        axis_label_fs = typo["elev_axis_pt"]
        species_fs = typo["elev_species_pt"]
    else:
        tick_fs = float(_up_marker_size(6))
        axis_label_fs = float(_up_marker_size(6))
        species_fs = float(_up_marker_size(6))

    if inset:
        _style_elevation_inset_ax(
            ax,
            y_min=y_min,
            y_max=y_max,
            x_max=x_max,
            tick_fs=tick_fs,
            label_fs=axis_label_fs,
        )
    else:
        _style_elevation_panel_frame(ax)
        ax.set_xlim(0.0, max(x_max, 0.01))
        ax.set_ylim(y_min, y_max)
        y_step = _elev_tick_step(y_max - y_min)
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="-", linewidth=0.55, color=ELEV_GRID_COLOR)
        ax.set_xlabel("里程(km)", fontsize=axis_label_fs, color=ELEV_LABEL_COLOR)
        ax.set_ylabel("海拔(m)", fontsize=axis_label_fs, color=ELEV_LABEL_COLOR)

    ax.fill_between(
        dist_prof,
        ele_prof,
        y_min,
        color=ELEV_FILL_COLOR,
        zorder=1,
        interpolate=True,
    )
    ax.plot(
        dist_prof,
        ele_prof,
        color=ELEV_LINE_COLOR,
        linewidth=1.8,
        solid_capstyle="round",
        zorder=3,
    )

    highlight_blocked: List[Tuple[float, float, float, float]] = []
    hl_fs = max(4.0, tick_fs * 0.95)
    for d, e, label in _elevation_highlight_labels(dist_prof, ele_prof):
        ax.scatter(
            [d],
            [e],
            c=ELEV_HIGHLIGHT_COLOR,
            s=24,
            zorder=4,
            edgecolors="white",
            linewidths=0.8,
        )
        ld, le = _offset_points_to_data(ax, d, e, 0.0, 7.0)
        y_top = ax.get_ylim()[1]
        text_h = _elev_text_height_data(ax, hl_fs)
        le = min(le, y_top - text_h * 0.25)
        ax.text(
            ld,
            le,
            label,
            ha="center",
            va="bottom",
            fontsize=hl_fs,
            color=ELEV_LABEL_COLOR,
            clip_on=False,
            zorder=5,
        )
        highlight_blocked.append(
            _elev_label_box_data(ax, ld, le, label, hl_fs, va="bottom")
        )
    shown = list(photos)
    if max_markers is not None:
        shown = shown[: max(0, max_markers)]
    markers: List[Tuple[float, float, str]] = []
    for ph in shown:
        when_tl = ph.when_track
        if when_tl is None and ph.when is not None:
            when_tl = align.photo_to_timeline(ph.when)
        if when_tl is None:
            continue
        pt = _profile_at_time(ts_prof, dist_prof, ele_prof, when_tl)
        if pt is None:
            continue
        markers.append((pt[0], pt[1], ph.species_cn))
    markers.sort(key=lambda m: m[0])
    species_layouts = _layout_elevation_species_labels(
        ax,
        markers,
        species_fs,
        data_x_max=x_max,
        blocked=highlight_blocked,
    )
    leader_thresh = max(x_max, y_max - y_min) * 0.008
    for ad, ae, ld, le, name in species_layouts:
        label_ha = _elev_label_ha_for_marker(ld, x_max)
        ax.scatter(
            [ad],
            [ae],
            c=ELEV_SPECIES_COLOR,
            s=max(16, thumb_diameter // 2),
            zorder=6,
            edgecolors="white",
            linewidths=0.7,
        )
        if math.hypot(ld - ad, le - ae) > leader_thresh:
            ax.annotate(
                "",
                xy=(ld, le),
                xytext=(ad, ae),
                arrowprops=dict(
                    arrowstyle="->",
                    color=ELEV_LEADER_COLOR,
                    lw=0.85,
                    shrinkA=2.5,
                    shrinkB=2.5,
                ),
                zorder=5,
            )
        ax.text(
            ld,
            le,
            name,
            ha=label_ha,
            va="center",
            fontsize=species_fs,
            color=ELEV_SPECIES_NAME_COLOR,
            fontweight="medium",
            clip_on=False,
            zorder=7,
        )


def _figure_size_inches(width_px: int, height_px: int, dpi: int) -> Tuple[float, float]:
    return width_px / dpi, height_px / dpi


def _encode_path_reason_lines(pairs: Sequence[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for path, reason in pairs:
        lines.append(f"{_photo_path_key(path)}|{reason}")
    return "\n".join(lines)


def _decode_path_reason_lines(raw: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ln in raw.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if "|" in ln:
            path, reason = ln.split("|", 1)
            out.append((path.strip(), reason.strip()))
        else:
            out.append((ln, ""))
    return out


def _append_photo_report_section(
    lines: List[str],
    *,
    title: str,
    raw_paths: str = "",
    raw_pairs: str = "",
    default_reason: str = "",
) -> None:
    paths_only = [ln for ln in raw_paths.split("\n") if ln.strip()]
    pairs = _decode_path_reason_lines(raw_pairs) if raw_pairs.strip() else []
    if not paths_only and not pairs:
        return
    if pairs:
        lines.append(f"{title} {len(pairs)} 张")
        for path, reason in pairs:
            lines.append(f"  · {os.path.basename(path)}")
            detail = reason or default_reason
            if detail:
                lines.append(f"    {detail}")
            lines.append(f"    {path}")
        return
    reason = default_reason or "无法在地图上标注"
    lines.append(f"{title} {len(paths_only)} 张：{reason}")
    for path in paths_only:
        lines.append(f"  · {os.path.basename(path)}")
        lines.append(f"    {path}")


def iter_skipped_photo_log_lines(written: Mapping[str, str]) -> List[str]:
    """从 generate_track_maps 返回值生成未绘制鸟图的日志行（含文件名与路径）。"""
    lines: List[str] = []
    stats = (written.get("map_photo_stats") or "").strip()
    if stats:
        lines.append(f"鸟图统计：{stats}")

    _append_photo_report_section(
        lines,
        title="未扫描纳入",
        raw_pairs=written.get("excluded_at_collect") or "",
    )
    _append_photo_report_section(
        lines,
        title="未绘制",
        raw_paths=written.get("skipped_photo_paths") or "",
        default_reason=(
            written.get("skipped_photo_reason")
            or "与 GPS/GPX 无法对应，无法在地图上标注"
        ),
    )
    _append_photo_report_section(
        lines,
        title="去重省略",
        raw_pairs=written.get("deduped_photos") or "",
    )
    trunc_pairs = (written.get("truncated_photos") or "").strip()
    trunc_n = int(written.get("markers_truncated") or 0)
    if trunc_pairs:
        _append_photo_report_section(
            lines,
            title="未绘制（数量上限）",
            raw_pairs=trunc_pairs,
        )
    elif trunc_n > 0:
        if written.get("preview_only") == "1":
            limit = written.get("preview_max_photos") or str(PREVIEW_MAX_MARKERS)
            lines.append(
                f"未绘制 {trunc_n} 张：预览模式最多显示 {limit} 张（按时间优先）"
            )
        else:
            limit = written.get("export_max_markers") or str(EXPORT_MAX_MARKERS)
            lines.append(
                f"未绘制 {trunc_n} 张：导出标注上限 {limit} 张（按时间优先）"
            )

    if lines:
        return lines

    n = int(written.get("skipped_time_mismatch") or 0)
    if n > 0:
        return [
            f"未绘制 {n} 张：与 GPX 时间差超过允许范围或无拍摄时间"
        ]
    return []


def generate_track_maps(
    *,
    reports_dir: str,
    gpx_path: Optional[str] = None,
    gpx_paths: Optional[Sequence[str]] = None,
    photo_folder: str,
    use_gpx_track: bool = True,
    use_exif_gps: bool = True,
    radius_km: float = 1.0,
    include_elevation: bool = True,
    basemap_style: str = "normal",
    preview_only: bool = False,
    preview_max_photos: int = PREVIEW_MAX_MARKERS,
    max_gpx_match_delta_s: float = DEFAULT_GPX_MATCH_MAX_DELTA_S,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
    location_name: str = "",
    province: str = "",
    city: str = "",
    logo_path: str = "",
    logo_width_ratio: float = 0.30,
    show_key_places: bool = False,
    key_places_text: str = "",
) -> Dict[str, str]:
    """
    生成 PNG。preview_only 时最多标注 preview_max_photos 张鸟图；预览与正式导出均为 1440×2560（2K 竖屏）像素，鸟图比例一致。
    """
    _configure_matplotlib_cjk()

    # 未使用 GPX 时强制走照片 EXIF GPS，并忽略传入的 GPX 路径
    if not use_gpx_track:
        use_exif_gps = True
        gpx_path = None
        gpx_paths = None

    reports = Path(reports_dir).expanduser().resolve()
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "轨迹图预览" if preview_only else "轨迹图"
    out_path = reports / f"{prefix}_{ts}.png"

    track: List[GpxPoint] = []
    gpx_files = resolve_gpx_path_list(gpx_path, gpx_paths)
    if use_gpx_track and gpx_files:
        track = load_gpx_many(gpx_files)

    collect_stats: Dict[str, object] = {}
    photos = collect_bird_photos(photo_folder, require_gps=False, stats=collect_stats)
    excluded_at_collect: List[Tuple[str, str]] = list(
        collect_stats.get("excluded") or []
    )
    n_scanned = int(collect_stats.get("scanned") or 0)
    n_collected = len(photos)

    time_align: Optional[TrackTimeAlignment] = None
    skipped_time_mismatch = 0
    skipped_unmapped_paths: List[str] = []
    deduped_pairs: List[Tuple[str, str]] = []
    truncated_pairs: List[Tuple[str, str]] = []
    skipped_reason = ""
    n_matched = 0
    matched_records: List[Dict] = []
    if track and use_gpx_track:
        paths = [p.path for p in photos]
        time_align = alignment_from_tz(exif_tz, gpx_tz)
        max_delta = max(60.0, float(max_gpx_match_delta_s))
        matched_records = match_photos_to_track(
            paths,
            track,
            alignment=time_align,
            max_delta_seconds=max_delta,
            exif_tz=exif_tz,
            gpx_tz=gpx_tz,
        )
        skipped_reason = (
            "与 GPX 无法对应（与轨迹时刻相差超过 1 小时、无拍摄时间或无法插值位置）"
        )
        by_path = {_photo_path_key(m["path"]): m for m in matched_records}
        enriched: List[BirdPhoto] = []
        for ph in photos:
            m = by_path.get(_photo_path_key(ph.path))
            if not m:
                skipped_unmapped_paths.append(_photo_path_key(ph.path))
                continue
            enriched.append(
                BirdPhoto(
                    path=ph.path,
                    species_cn=ph.species_cn,
                    when=ph.when or m.get("time"),
                    when_track=m.get("when_track"),
                    lat=m["lat"],
                    lon=m["lon"],
                    ele=m.get("ele"),
                )
            )
        photos = enriched
        n_matched = len(photos)
        skipped_time_mismatch = len(skipped_unmapped_paths)
    elif track:
        time_align = TrackTimeAlignment()
        n_matched = len(photos)
    elif use_exif_gps:
        skipped_reason = "照片中无 EXIF GPS，无法在地图上标注"
        for ph in photos:
            if ph.lat is None or ph.lon is None:
                skipped_unmapped_paths.append(_photo_path_key(ph.path))
        photos = [p for p in photos if p.lat is not None and p.lon is not None]
        n_matched = len(photos)
        skipped_time_mismatch = len(skipped_unmapped_paths)
    else:
        photos = []

    photos = _dedupe_photos_by_species_radius(
        photos, radius_km, dropped=deduped_pairs
    )
    n_after_dedupe = len(photos)
    n_with_gps = sum(
        1 for p in photos if p.lat is not None and p.lon is not None
    )
    if preview_only:
        marker_limit = preview_max_photos
    else:
        marker_limit = EXPORT_MAX_MARKERS if n_with_gps > EXPORT_MAX_MARKERS else None

    if marker_limit is not None and n_with_gps > marker_limit:
        for ph in photos[marker_limit:]:
            if ph.lat is None or ph.lon is None:
                continue
            if preview_only:
                reason = f"预览模式最多显示 {marker_limit} 张（按时间优先）"
            else:
                reason = f"导出标注上限 {marker_limit} 张（按时间优先）"
            truncated_pairs.append((ph.path, reason))

    if not track and photos:
        track = [
            GpxPoint(time=ph.when, lat=ph.lat, lon=ph.lon, ele=ph.ele)  # type: ignore
            for ph in photos
            if ph.lat is not None and ph.lon is not None
        ]
        track.sort(key=lambda p: p.time or datetime.min)

    has_elev = bool(
        include_elevation
        and track
        and any(p.ele is not None for p in track)
    )

    if preview_only:
        width_px, height_px, dpi = PREVIEW_WIDTH_PX, PREVIEW_HEIGHT_PX, PREVIEW_DPI
    else:
        width_px, height_px, dpi = EXPORT_WIDTH_PX, EXPORT_HEIGHT_PX, EXPORT_DPI
    thumb_map, thumb_elev = _track_map_thumb_diameters(height_px)

    fig_w, fig_h = _figure_size_inches(width_px, height_px, dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    ax_map = fig.add_axes([0, 0, 1, 1])

    title_place, title_map, title_date = resolve_track_map_titles(
        location_name=location_name,
        province=province,
        city=city,
        track=track,
        photos=photos,
        exif_tz=exif_tz,
        gpx_tz=gpx_tz,
    )
    key_places: List[KeyPlace] = []
    key_place_failed: List[str] = []
    key_places_oor: List[KeyPlace] = []
    if show_key_places and (key_places_text or "").strip():
        elev_frac = (
            _elevation_south_reserve_frac(thumb_map, height_px, dpi=dpi)
            if has_elev
            else 0.0
        )
        map_extent = _estimate_map_lonlat_extent(
            track,
            photos,
            width_px=width_px,
            height_px=height_px,
            elev_south_frac=elev_frac,
        )
        rp, rc, rd, center = resolve_key_place_search_region(
            location_name=location_name or title_place,
            province=province,
            city=city,
            track=track,
            photos=photos,
        )
        max_km = 80.0
        if map_extent is not None:
            x0, x1, y0, y1 = map_extent
            max_km = min(200.0, max(80.0, haversine_km(y0, x0, y1, x1) * 3.0))
        key_places, key_place_failed = geocode_key_places(
            key_places_text,
            city=rc,
            province=rp,
            location_name=location_name or title_place,
            district=rd,
            prefer_wgs84=center,
            max_prefer_km=max_km,
        )
        key_places, key_places_oor = partition_key_places_by_view(
            map_extent, key_places
        )
    marker_kw = dict(
        compact_labels=False,
        resolve_overlaps=True,
        radius_km=float(radius_km),
    )
    basemap_status = _plot_map_ax(
        ax_map,
        track,
        photos,
        title_place,
        map_title=title_map,
        date_label=title_date,
        max_markers=marker_limit,
        thumb_diameter=thumb_map,
        basemap_style=basemap_style,
        map_width_px=width_px,
        map_height_px=height_px,
        logo_path=logo_path,
        logo_width_ratio=logo_width_ratio,
        summary_default_y=0.19 if has_elev else 0.055,
        elev_south_frac=_elevation_south_reserve_frac(thumb_map, height_px, dpi=dpi)
        if has_elev
        else 0.0,
        key_places=key_places,
        **marker_kw,
    )

    if has_elev:
        _ax_elev_outer, ax_elev = _create_elevation_panel_axes(ax_map)
        _ = _ax_elev_outer
        _plot_elevation_ax(
            ax_elev,
            track,
            photos,
            max_markers=marker_limit,
            thumb_diameter=thumb_elev,
            align=time_align,
            inset=True,
            map_ax=ax_map,
        )

    if basemap_status == "ok":
        _draw_map_attribution(ax_map, y_frac=0.006, basemap_style=basemap_style)

    fig.savefig(
        str(out_path),
        dpi=dpi,
        facecolor="white",
        pad_inches=0,
        bbox_inches=None,
    )
    plt.close(fig)

    title_parts = [p for p in (title_place, title_map, title_date) if p]
    title = "\n".join(title_parts)
    written: Dict[str, str] = {
        "track_png": str(out_path),
        "map_basemap": basemap_status,
        "map_title": title,
        "map_coord_source": "gpx" if (track and use_gpx_track) else "exif",
    }
    if time_align is not None:
        written["time_align_desc"] = describe_time_alignment(time_align)
        written["gpx_match_exif_tz"] = exif_tz
        written["gpx_match_gpx_tz"] = gpx_tz
    if len(gpx_files) > 1:
        written["gpx_file_count"] = str(len(gpx_files))
    exif_cnt = sum(1 for m in matched_records if m.get("pos_source") == "exif_gps")
    if exif_cnt:
        written["map_pos_exif_gps"] = str(exif_cnt)
    pipeline = (
        f"扫描 {n_scanned} → 纳入 {n_collected} → 可标注 {n_matched}"
        f" → 去重后 {n_after_dedupe}"
    )
    if marker_limit is not None:
        pipeline += f" → 绘制 {min(n_with_gps, marker_limit)}"
    else:
        pipeline += f" → 绘制 {n_with_gps}"
    written["map_photo_stats"] = pipeline
    if preview_only:
        written["preview_only"] = "1"
        written["preview_max_photos"] = str(preview_max_photos)
    else:
        written["export_max_markers"] = str(EXPORT_MAX_MARKERS)
    if excluded_at_collect:
        written["excluded_at_collect"] = _encode_path_reason_lines(
            excluded_at_collect
        )
    if show_key_places:
        written["key_places_drawn"] = str(len(key_places))
        if key_place_failed:
            written["key_places_failed"] = "；".join(key_place_failed)
        if key_places_oor:
            written["key_places_out_of_range"] = "；".join(
                f"{p.name} ({p.lat:.5f}, {p.lon:.5f})" for p in key_places_oor
            )
    if skipped_time_mismatch > 0:
        written["skipped_time_mismatch"] = str(skipped_time_mismatch)
    if skipped_unmapped_paths:
        written["skipped_photo_paths"] = "\n".join(skipped_unmapped_paths)
        if skipped_reason:
            written["skipped_photo_reason"] = skipped_reason
    if deduped_pairs:
        written["deduped_photos"] = _encode_path_reason_lines(deduped_pairs)
    if truncated_pairs:
        written["truncated_photos"] = _encode_path_reason_lines(truncated_pairs)
    if marker_limit is not None and n_with_gps > marker_limit:
        written["markers_truncated"] = str(n_with_gps - marker_limit)
    if has_elev and not preview_only:
        written["elevation_png"] = str(out_path)
    return written
