# -*- coding: utf-8 -*-
"""生成观鸟行迹图、物种分布图与海拔剖面（PNG）。"""

from __future__ import annotations

import math
import os
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageFilter

from .amap_basemap import (
    fetch_amap_basemap_rgba,
    gcj_bounds_from_lonlats,
    regeo_place_label,
    wgs84_to_map_lonlat,
)
from .gpx_io import GpxPoint, load_gpx
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

# 手机 2K 竖屏正式导出（宽 × 高，像素）
EXPORT_WIDTH_PX = 1440
EXPORT_HEIGHT_PX = 2560
EXPORT_DPI = 120
PREVIEW_WIDTH_PX = 2160
PREVIEW_HEIGHT_PX = 3840
PREVIEW_DPI = 120

# 照片 EXIF 时刻与 GPX 时间轴相差超过此值（秒）则不绘制记录点
DEFAULT_GPX_MATCH_MAX_DELTA_S = 30 * 60
# 正式导出 PNG 时鸟图标注上限（避免画布过大 + 大量缩略图导致内存崩溃）
EXPORT_MAX_MARKERS = 80

# 地图鸟图、物种标注与海拔文字相对原尺寸放大约 1/3
_MARKER_SIZE_SCALE = 4 / 3


def _up_marker_size(n: int | float) -> int:
    return max(1, round(float(n) * _MARKER_SIZE_SCALE))


_CJK_FONT_CONFIGURED = False


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


def _dedupe_photos_by_species_radius(
    photos: Sequence[BirdPhoto],
    radius_km: float,
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
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((d, d), Image.Resampling.LANCZOS)
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, d - 1, d - 1), fill=255)
    circle = Image.new("RGBA", (d, d))
    circle.paste(im, (0, 0))
    circle.putalpha(mask)
    canvas.paste(circle, (pad, pad), circle)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _map_xy(lon: float, lat: float, *, use_gcj: bool) -> Tuple[float, float]:
    if use_gcj:
        return wgs84_to_map_lonlat(lon, lat)
    return lon, lat


def _observation_date_label(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
) -> str:
    times: List[datetime] = []
    for p in track:
        if p.time is not None:
            times.append(p.time)
    for ph in photos:
        if ph.when is not None:
            times.append(ph.when)
    if not times:
        t = datetime.now()
    else:
        t = min(times)
    return f"{t.year}年{t.month}月{t.day}日"


def resolve_track_map_title(
    *,
    preview: bool,
    location_name: str = "",
    province: str = "",
    city: str = "",
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
) -> str:
    """标题：地点 + 观鸟纪录（日期）。"""
    _ = preview
    loc = (location_name or "").strip()
    if not loc:
        loc = f"{(province or '').strip()}{(city or '').strip()}".strip()
    if not loc:
        lons, lats = _collect_lonlats(track, photos)
        if lons:
            lon_c = (min(lons) + max(lons)) / 2
            lat_c = (min(lats) + max(lats)) / 2
            loc = regeo_place_label(lon_c, lat_c) or ""
    date_s = _observation_date_label(track, photos)
    if loc:
        return f"{loc}观鸟纪录（{date_s}）"
    return f"观鸟纪录（{date_s}）"


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
    max_overlap: float = 0.2,
    max_iters: int = 32,
) -> List[Tuple[float, float]]:
    """仅对鸟图圆做碰撞分离；重叠>max_overlap 才外移，保持沿轨迹聚集。"""
    n = len(anchors)
    if n <= 1:
        return list(anchors)
    r = _thumb_radius_data(ax, thumb_diameter)
    displays = [list(a) for a in anchors]
    any_overlap = False
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(
                displays[j][0] - displays[i][0],
                displays[j][1] - displays[i][1],
            )
            if _circle_overlap_fraction(d, r) > max_overlap:
                any_overlap = True
                break
        if any_overlap:
            break
    if not any_overlap:
        return [(d[0], d[1]) for d in displays]

    min_sep = _min_center_distance(r, max_overlap)
    max_leader = r * 2.4
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = displays[j][0] - displays[i][0]
                dy = displays[j][1] - displays[i][1]
                d = math.hypot(dx, dy)
                if d < 1e-12:
                    ang = (i * 2.1 + j * 1.3) % (2 * math.pi)
                    dx, dy = math.cos(ang), math.sin(ang)
                    d = 1.0
                if _circle_overlap_fraction(d, r) > max_overlap:
                    need = max(min_sep - d, min_sep * 0.15)
                    push = need / 2.0
                    displays[i][0] -= dx / d * push
                    displays[i][1] -= dy / d * push
                    displays[j][0] += dx / d * push
                    displays[j][1] += dy / d * push
                    moved = True
        for i in range(n):
            ax0, ay0 = anchors[i]
            dx = displays[i][0] - ax0
            dy = displays[i][1] - ay0
            d = math.hypot(dx, dy)
            if d > max_leader:
                displays[i][0] = ax0 + dx / d * max_leader
                displays[i][1] = ay0 + dy / d * max_leader
                moved = True
        if not moved:
            break
    return [(d[0], d[1]) for d in displays]


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


def _collect_lonlats(
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
) -> Tuple[List[float], List[float]]:
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


def _add_photo_markers(
    ax,
    photos: Sequence[BirdPhoto],
    *,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 44,
    use_gcj: bool = False,
    compact_labels: bool = False,
    resolve_overlaps: bool = True,
) -> None:
    shown = list(photos)
    if max_markers is not None:
        shown = shown[: max(0, max_markers)]
    anchors: List[Tuple[float, float]] = []
    items: List[BirdPhoto] = []
    for ph in shown:
        if ph.lat is None or ph.lon is None:
            continue
        ax_x, ax_y = _map_xy(ph.lon, ph.lat, use_gcj=use_gcj)
        anchors.append((ax_x, ax_y))
        items.append(ph)

    displays = (
        _layout_marker_displays(anchors, ax, thumb_diameter, max_overlap=0.2)
        if resolve_overlaps and len(anchors) > 1
        else list(anchors)
    )

    label_off = int(thumb_diameter * (0.38 if compact_labels else 0.48))
    label_fs = _up_marker_size(8 if compact_labels else 9)
    r_thumb = _thumb_radius_data(ax, thumb_diameter)
    for ph, (ax_x, ax_y), (dx, dy) in zip(items, anchors, displays):
        shifted = math.hypot(dx - ax_x, dy - ax_y) > r_thumb * 0.12
        if shifted:
            ax.plot(
                [ax_x, dx],
                [ax_y, dy],
                color="#555555",
                linewidth=0.9,
                alpha=0.75,
                zorder=6,
                solid_capstyle="round",
            )
            ax.scatter(
                [ax_x],
                [ax_y],
                s=max(12, thumb_diameter // 3),
                c="#E67E22",
                edgecolors="white",
                linewidths=0.5,
                zorder=7,
            )
        try:
            arr = _circular_thumb_rgba(ph.path, thumb_diameter)
            zoom = thumb_diameter / max(arr.shape[0], arr.shape[1])
            imagebox = OffsetImage(arr, zoom=zoom)
            ab = AnnotationBbox(
                imagebox,
                (dx, dy),
                frameon=False,
                pad=0,
                zorder=8,
            )
            ax.add_artist(ab)
        except Exception:
            ax.scatter([dx], [dy], c="#E67E22", s=28, zorder=8)
        ax.annotate(
            ph.species_cn,
            (dx, dy),
            textcoords="offset points",
            xytext=(0, label_off),
            ha="center",
            va="bottom",
            fontsize=label_fs,
            color="#1a1a1a",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="#999",
                alpha=0.92,
            ),
            zorder=9,
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
        line_color, line_w, pt_c, pt_ec, z = "#F1C40F", 3.0, "#FFFFFF", "#2980B9", 7
        start_s, end_s = 72, 72
        edge = "white"
    else:
        line_color, line_w, pt_c, pt_ec, z = "#2980B9", 2.2, "#3498DB", None, 2
        start_s, end_s = 64, 64
        edge = None
    ax.plot(xs, ys, color=line_color, linewidth=line_w, label="轨迹", zorder=z)
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
        [xs[0]], [ys[0]], c="#27AE60", s=start_s, label="起点", zorder=z, edgecolors=edge
    )
    ax.scatter(
        [xs[-1]], [ys[-1]], c="#E74C3C", s=end_s, label="终点", zorder=z, edgecolors=edge
    )


def _plot_map_ax(
    ax,
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    title: str,
    *,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 44,
    basemap_style: str = "digital",
    map_width_px: int = 1080,
    map_height_px: int = 1200,
    compact_labels: bool = False,
    resolve_overlaps: bool = True,
) -> str:
    """
    绘制地图子图（高德底图 + GCJ-02 叠加）。
    返回 basemap 状态：ok / fallback / none / no_key。
    """
    style = (basemap_style or "digital").lower()
    if style in ("none", "off", "grid"):
        _draw_track_on_ax(ax, track, use_gcj=False, on_basemap=False)
        _add_photo_markers(
            ax,
            photos,
            max_markers=max_markers,
            thumb_diameter=thumb_diameter,
            use_gcj=False,
            compact_labels=compact_labels,
            resolve_overlaps=resolve_overlaps,
        )
        ax.set_xlabel("经度", fontsize=11)
        ax.set_ylabel("纬度", fontsize=11)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.set_aspect("equal", adjustable="datalim")
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x0, x1, y0, y1 = _expand_lonlat_bounds(
            x0, x1, y0, y1, _subplot_aspect_wh(ax)
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        return "none"

    lons, lats = _collect_lonlats(track, photos)
    if not lons:
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis("off")
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
    except ValueError as e:
        if "API Key" in str(e) or "api_key" in str(e):
            ax.clear()
            _plot_map_ax(
                ax,
                track,
                photos,
                title,
                max_markers=max_markers,
                thumb_diameter=thumb_diameter,
                basemap_style="none",
                map_width_px=map_width_px,
                map_height_px=map_height_px,
                compact_labels=compact_labels,
                resolve_overlaps=resolve_overlaps,
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
            title,
            max_markers=max_markers,
            thumb_diameter=thumb_diameter,
            basemap_style="none",
            map_width_px=map_width_px,
            map_height_px=map_height_px,
            compact_labels=compact_labels,
            resolve_overlaps=resolve_overlaps,
        )
        return "fallback"

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")

    _draw_track_on_ax(ax, track, use_gcj=True, on_basemap=True)
    _add_photo_markers(
        ax,
        photos,
        max_markers=max_markers,
        thumb_diameter=thumb_diameter,
        use_gcj=True,
        compact_labels=compact_labels,
        resolve_overlaps=resolve_overlaps,
    )
    ax.set_title(title, fontsize=14, pad=10)
    leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.88)
    if leg:
        leg.get_frame().set_facecolor("white")
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


def _layout_elevation_label_offsets(
    ax,
    markers: Sequence[Tuple[float, float, str]],
) -> List[Tuple[int, int]]:
    """为海拔图物种名选择偏移，尽量降低标签框重叠。"""
    if not markers:
        return []
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_rng = max(x1 - x0, 1e-6)
    y_rng = max(y1 - y0, 1e-6)
    candidates = [
        (0, _up_marker_size(10)),
        (0, -_up_marker_size(12)),
        (_up_marker_size(14), _up_marker_size(5)),
        (-_up_marker_size(14), _up_marker_size(5)),
        (0, _up_marker_size(18)),
        (0, -_up_marker_size(20)),
        (_up_marker_size(18), -_up_marker_size(8)),
        (-_up_marker_size(18), -_up_marker_size(8)),
        (0, _up_marker_size(26)),
        (0, -_up_marker_size(28)),
        (_up_marker_size(24), _up_marker_size(10)),
        (-_up_marker_size(24), _up_marker_size(10)),
        (_up_marker_size(10), _up_marker_size(22)),
        (-_up_marker_size(10), _up_marker_size(22)),
    ]
    placed: List[Tuple[float, float, float, float]] = []
    out: List[Tuple[int, int]] = []
    for d, e, name in markers:
        char_w = x_rng * max(0.018, 0.009 * len(name)) * _MARKER_SIZE_SCALE
        char_h = y_rng * 0.085 * _MARKER_SIZE_SCALE
        chosen = candidates[0]
        for ox, oy in candidates:
            lx, ly = _offset_points_to_data(ax, d, e, float(ox), float(oy))
            overlap = False
            for px, py, pw, ph in placed:
                if abs(lx - px) < (char_w + pw) * 0.72 and abs(ly - py) < (char_h + ph) * 0.72:
                    overlap = True
                    break
            if not overlap:
                chosen = (ox, oy)
                lx, ly = _offset_points_to_data(ax, d, e, float(ox), float(oy))
                placed.append((lx, ly, char_w, char_h))
                break
        else:
            i = len(placed)
            chosen = candidates[i % len(candidates)]
            ox, oy = chosen
            lx, ly = _offset_points_to_data(ax, d, e, float(ox), float(oy))
            placed.append((lx, ly, char_w, char_h))
        out.append(chosen)
    return out


def _plot_elevation_ax(
    ax,
    track: Sequence[GpxPoint],
    photos: Sequence[BirdPhoto],
    *,
    max_markers: Optional[int] = None,
    thumb_diameter: int = 36,
    align: Optional[TrackTimeAlignment] = None,
) -> None:
    if align is None:
        align = TrackTimeAlignment()
    ts_prof, dist_prof, ele_prof = _build_timed_profile(track, align)
    if not dist_prof:
        return
    ax.plot(dist_prof, ele_prof, color="#27AE60", linewidth=1.8, label="海拔", zorder=2)
    ax.fill_between(dist_prof, ele_prof, alpha=0.12, color="#27AE60")
    ax.set_xlabel("累计距离 (km)", fontsize=_up_marker_size(10))
    ax.set_ylabel("海拔 (m)", fontsize=_up_marker_size(10))
    ax.grid(True, linestyle=":", alpha=0.55)
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
    ax.relim()
    ax.autoscale_view()
    offsets = _layout_elevation_label_offsets(ax, markers)
    label_fs = _up_marker_size(6)
    for (best_d, e, name), (ox, oy) in zip(markers, offsets):
        ax.scatter(
            [best_d],
            [e],
            c="#E67E22",
            s=max(_up_marker_size(22), thumb_diameter - _up_marker_size(8)),
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.annotate(
            name,
            (best_d, e),
            textcoords="offset points",
            xytext=(ox, oy),
            ha="center",
            va="center",
            fontsize=label_fs,
            color="#111111",
            fontweight="medium",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="#666666",
                alpha=0.95,
                linewidth=0.5,
            ),
            zorder=6,
        )


def _figure_size_inches(width_px: int, height_px: int, dpi: int) -> Tuple[float, float]:
    return width_px / dpi, height_px / dpi


def generate_track_maps(
    *,
    reports_dir: str,
    gpx_path: Optional[str] = None,
    photo_folder: str,
    use_gpx_track: bool = True,
    use_exif_gps: bool = True,
    radius_km: float = 1.0,
    include_elevation: bool = True,
    basemap_style: str = "digital",
    preview_only: bool = False,
    preview_max_photos: int = 20,
    max_gpx_match_delta_s: float = DEFAULT_GPX_MATCH_MAX_DELTA_S,
    exif_tz: str = DEFAULT_EXIF_TZ,
    gpx_tz: str = DEFAULT_GPX_TZ,
    location_name: str = "",
    province: str = "",
    city: str = "",
) -> Dict[str, str]:
    """
    生成 PNG。preview_only 时最多标注 preview_max_photos 张鸟图；正式保存为 1440×2560（2K 竖屏）像素。
    """
    _configure_matplotlib_cjk()

    reports = Path(reports_dir).expanduser().resolve()
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "轨迹图预览" if preview_only else "轨迹图"
    out_path = reports / f"{prefix}_{ts}.png"

    track: List[GpxPoint] = []
    if use_gpx_track and gpx_path and os.path.isfile(gpx_path):
        track = load_gpx(gpx_path)

    photos = collect_bird_photos(photo_folder, require_gps=False)

    time_align: Optional[TrackTimeAlignment] = None
    skipped_time_mismatch = 0
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
        by_path = {m["path"]: m for m in matched_records}
        enriched: List[BirdPhoto] = []
        for ph in photos:
            m = by_path.get(ph.path)
            if not m:
                skipped_time_mismatch += 1
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
    elif track:
        time_align = TrackTimeAlignment()
    elif use_exif_gps:
        photos = [p for p in photos if p.lat is not None and p.lon is not None]
    else:
        photos = []

    photos = _dedupe_photos_by_species_radius(photos, radius_km)
    n_with_gps = sum(
        1 for p in photos if p.lat is not None and p.lon is not None
    )
    if preview_only:
        marker_limit = preview_max_photos
    else:
        marker_limit = EXPORT_MAX_MARKERS if n_with_gps > EXPORT_MAX_MARKERS else None

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
        thumb_map, thumb_elev = _up_marker_size(52), _up_marker_size(40)
    else:
        width_px, height_px, dpi = EXPORT_WIDTH_PX, EXPORT_HEIGHT_PX, EXPORT_DPI
        thumb_map, thumb_elev = _up_marker_size(64), _up_marker_size(48)

    fig_w, fig_h = _figure_size_inches(width_px, height_px, dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")

    if has_elev:
        # 地图加高；海拔区高度为原先的 1/2（原 4:2 → 现 8:1）
        gs = GridSpec(2, 1, figure=fig, height_ratios=[8, 1], hspace=0.02)
        ax_map = fig.add_subplot(gs[0])
        ax_elev = fig.add_subplot(gs[1], sharex=None)
    else:
        ax_map = fig.add_subplot(111)
        ax_elev = None

    map_height_px = int(height_px * (8 / 9) if has_elev else height_px)
    title = resolve_track_map_title(
        preview=preview_only,
        location_name=location_name,
        province=province,
        city=city,
        track=track,
        photos=photos,
    )
    marker_kw = dict(
        compact_labels=preview_only,
        resolve_overlaps=True,
    )
    basemap_status = _plot_map_ax(
        ax_map,
        track,
        photos,
        title,
        max_markers=marker_limit,
        thumb_diameter=thumb_map,
        basemap_style=basemap_style,
        map_width_px=width_px,
        map_height_px=map_height_px,
        **marker_kw,
    )

    if ax_elev is not None:
        _plot_elevation_ax(
            ax_elev,
            track,
            photos,
            max_markers=marker_limit,
            thumb_diameter=thumb_elev,
            align=time_align,
        )

    pad = 0.04 if has_elev else 0.08
    adjust_kw: Dict[str, float] = {
        "left": 0.06,
        "right": 0.98,
        "top": 0.96,
        "bottom": 0.05,
    }
    if has_elev:
        adjust_kw["hspace"] = 0.02
    fig.subplots_adjust(**adjust_kw)
    fig.savefig(
        str(out_path),
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=pad,
    )
    plt.close(fig)

    written: Dict[str, str] = {
        "track_png": str(out_path),
        "map_basemap": basemap_status,
        "map_title": title,
    }
    if time_align is not None:
        written["time_align_desc"] = describe_time_alignment(time_align)
        written["gpx_match_exif_tz"] = exif_tz
        written["gpx_match_gpx_tz"] = gpx_tz
    exif_cnt = sum(1 for m in matched_records if m.get("pos_source") == "exif_gps")
    if exif_cnt:
        written["map_pos_exif_gps"] = str(exif_cnt)
    if skipped_time_mismatch > 0:
        written["skipped_time_mismatch"] = str(skipped_time_mismatch)
    if (
        not preview_only
        and marker_limit is not None
        and n_with_gps > marker_limit
    ):
        written["markers_truncated"] = str(n_with_gps - marker_limit)
    if has_elev and not preview_only:
        written["elevation_png"] = str(out_path)
    return written
