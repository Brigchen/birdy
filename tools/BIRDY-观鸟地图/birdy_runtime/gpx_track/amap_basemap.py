# -*- coding: utf-8 -*-
"""高德地图底图：使用项目 Web 服务 Key（amap_api_config.json）加载瓦片并拼接。

矢量主题（幻影黑/月光银等）对应 JS API 的 amap://styles/*；官方未提供同款栅格瓦片，
故在标准矢量底图上做风格化调色以接近官方主题观感。卫星有/无路网使用真实瓦片图层。
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

TILE_SIZE = 256
MAX_TILES_PER_AXIS = 40
TILE_TIMEOUT = 12

# (内部 id, 下拉显示名) — 与高德 JS mapStyle 命名对齐
BASEMAP_STYLE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("normal", "标准（默认）"),
    ("dark", "幻影黑"),
    ("light", "月光银"),
    ("whitesmoke", "远山黛"),
    ("fresh", "草色青"),
    ("grey", "雅士灰"),
    ("graffiti", "涂鸦"),
    ("macaron", "马卡龙"),
    ("blue", "靛青蓝"),
    ("darkblue", "极夜蓝"),
    ("wine", "酱籽"),
    ("satellite", "无路网卫星"),
    ("satellite_roads", "有路网卫星"),
)

_VECTOR_THEME_IDS = frozenset(
    {
        "normal",
        "digital",  # 旧配置兼容
        "dark",
        "light",
        "whitesmoke",
        "fresh",
        "grey",
        "graffiti",
        "macaron",
        "blue",
        "darkblue",
        "wine",
    }
)
_DARK_THEME_IDS = frozenset({"dark", "darkblue", "wine", "grey"})
_SATELLITE_IDS = frozenset(
    {
        "satellite",
        "satellite_roads",
        "sat",
        "影像",
        "卫星",
        "无路网卫星",
        "有路网卫星",
    }
)


def normalize_basemap_style(style: Optional[str]) -> str:
    """归一化底图风格 id。"""
    s = (style or "normal").strip().lower()
    aliases = {
        "digital": "normal",
        "amap://styles/normal": "normal",
        "amap://styles/dark": "dark",
        "amap://styles/light": "light",
        "amap://styles/whitesmoke": "whitesmoke",
        "amap://styles/fresh": "fresh",
        "amap://styles/grey": "grey",
        "amap://styles/graffiti": "graffiti",
        "amap://styles/macaron": "macaron",
        "amap://styles/blue": "blue",
        "amap://styles/darkblue": "darkblue",
        "amap://styles/wine": "wine",
        "sat": "satellite",
        "影像": "satellite",
        "卫星": "satellite",
        "无路网卫星": "satellite",
        "有路网卫星": "satellite_roads",
        "satellite_road": "satellite_roads",
        "sat_roads": "satellite_roads",
    }
    s = aliases.get(s, s)
    if s.startswith("amap://styles/"):
        s = s.split("/")[-1]
    known = {cid for cid, _ in BASEMAP_STYLE_CHOICES}
    return s if s in known else "normal"


def is_satellite_basemap_style(style: Optional[str]) -> bool:
    s = normalize_basemap_style(style)
    return s in ("satellite", "satellite_roads")


def is_dark_basemap_style(style: Optional[str]) -> bool:
    """深色底图：标题/标注宜用浅色字。"""
    s = normalize_basemap_style(style)
    return s in _DARK_THEME_IDS or is_satellite_basemap_style(s)


def get_effective_amap_key() -> str:
    """与 geo_encoder / GUI「高德API」一致。"""
    from geo_encoder import _effective_amap_key

    return _effective_amap_key()


def wgs84_to_map_lonlat(lon: float, lat: float) -> Tuple[float, float]:
    """WGS84 → GCJ-02，供高德底图与轨迹叠加对齐。"""
    from geo_encoder import wgs84_to_gcj02

    glat, glon = wgs84_to_gcj02(lat, lon)
    return glon, glat


def _vector_tile_url(x: int, y: int, z: int, api_key: str) -> str:
    sub = (x + y) % 4 + 1
    key_q = f"&key={api_key}" if api_key else ""
    return (
        f"https://webrd0{sub}.is.autonavi.com/appmaptile"
        f"?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}{key_q}"
    )


def _satellite_tile_url(x: int, y: int, z: int, api_key: str) -> str:
    sub = (x + y) % 4 + 1
    key_q = f"&key={api_key}" if api_key else ""
    return (
        f"https://webst0{sub}.is.autonavi.com/appmaptile"
        f"?style=6&x={x}&y={y}&z={z}{key_q}"
    )


def _road_overlay_tile_url(x: int, y: int, z: int, api_key: str) -> str:
    """卫星路网注记层（含透明通道，可叠在影像上）。"""
    sub = (x + y) % 4 + 1
    key_q = f"&key={api_key}" if api_key else ""
    return (
        f"https://webst0{sub}.is.autonavi.com/appmaptile"
        f"?style=8&x={x}&y={y}&z={z}{key_q}"
    )


def _tile_urls_for_style(
    x: int, y: int, z: int, style: str, api_key: str
) -> List[str]:
    s = normalize_basemap_style(style)
    if s == "satellite":
        return [_satellite_tile_url(x, y, z, api_key)]
    if s == "satellite_roads":
        return [
            _satellite_tile_url(x, y, z, api_key),
            _road_overlay_tile_url(x, y, z, api_key),
        ]
    return [_vector_tile_url(x, y, z, api_key)]


def _download_tile(session, url: str) -> Image.Image:
    r = session.get(url, timeout=TILE_TIMEOUT)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGBA")


def _compose_tile_layers(layers: Sequence[Image.Image]) -> Image.Image:
    if not layers:
        return Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (240, 240, 240, 255))
    out = layers[0]
    for layer in layers[1:]:
        if layer.size != out.size:
            layer = layer.resize(out.size, Image.Resampling.BILINEAR)
        out = Image.alpha_composite(out, layer)
    return out


def _apply_vector_theme_look(img: Image.Image, style: str) -> Image.Image:
    """在标准矢量底图上近似高德官方 mapStyle 主题。"""
    s = normalize_basemap_style(style)
    if s in ("normal", "satellite", "satellite_roads"):
        return img

    rgba = img.convert("RGBA")
    rgb = rgba.convert("RGB")
    alpha = rgba.getchannel("A")

    if s == "dark":
        g = ImageOps.grayscale(rgb)
        inv = ImageOps.invert(g)
        tinted = ImageOps.colorize(inv, black="#0A1628", white="#A8C4D8")
        rgb = ImageEnhance.Contrast(tinted).enhance(1.15)
        rgb = ImageEnhance.Brightness(rgb).enhance(0.92)
    elif s == "darkblue":
        g = ImageOps.grayscale(rgb)
        inv = ImageOps.invert(g)
        tinted = ImageOps.colorize(inv, black="#020814", white="#6FA8D8")
        rgb = ImageEnhance.Contrast(tinted).enhance(1.2)
        rgb = ImageEnhance.Brightness(rgb).enhance(0.88)
    elif s == "wine":
        g = ImageOps.grayscale(rgb)
        inv = ImageOps.invert(g)
        tinted = ImageOps.colorize(inv, black="#1A080C", white="#D4A090")
        rgb = ImageEnhance.Contrast(tinted).enhance(1.1)
        rgb = ImageEnhance.Brightness(rgb).enhance(0.9)
    elif s == "light":
        rgb = ImageEnhance.Color(rgb).enhance(0.55)
        rgb = ImageEnhance.Brightness(rgb).enhance(1.12)
        rgb = ImageEnhance.Contrast(rgb).enhance(0.92)
    elif s == "whitesmoke":
        arr = np.asarray(rgb, dtype=np.float32)
        grey = arr.mean(axis=2, keepdims=True)
        arr = grey * 0.72 + arr * 0.28
        arr[..., 2] = np.clip(arr[..., 2] * 1.04 + 6, 0, 255)
        arr = np.clip(arr * 1.06 + 8, 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    elif s == "fresh":
        arr = np.asarray(rgb, dtype=np.float32)
        arr[..., 1] = np.clip(arr[..., 1] * 1.18 + 8, 0, 255)
        arr[..., 0] = np.clip(arr[..., 0] * 0.92, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.95, 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        rgb = ImageEnhance.Color(rgb).enhance(1.08)
    elif s == "grey":
        g = ImageOps.grayscale(rgb)
        rgb = ImageOps.colorize(g, black="#2A2A2E", white="#E8E8EA")
        rgb = ImageEnhance.Contrast(rgb).enhance(0.95)
    elif s == "graffiti":
        rgb = ImageEnhance.Color(rgb).enhance(1.55)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.2)
        arr = np.asarray(rgb, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.06 + 4, 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    elif s == "macaron":
        arr = np.asarray(rgb, dtype=np.float32)
        pastel = np.array([255.0, 230.0, 235.0], dtype=np.float32)
        arr = arr * 0.62 + pastel * 0.38
        arr[..., 1] = np.clip(arr[..., 1] * 1.04, 0, 255)
        rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        rgb = ImageEnhance.Color(rgb).enhance(0.85)
        rgb = ImageEnhance.Brightness(rgb).enhance(1.06)
    elif s == "blue":
        arr = np.asarray(rgb, dtype=np.float32)
        arr[..., 2] = np.clip(arr[..., 2] * 1.22 + 12, 0, 255)
        arr[..., 0] = np.clip(arr[..., 0] * 0.82, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 0.95 + 4, 0, 255)
        rgb = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        rgb = ImageEnhance.Color(rgb).enhance(1.05)

    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def _lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    n = 2.0**zoom
    px = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    py = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
        * TILE_SIZE
    )
    return px, py


def _global_pixel_to_lonlat(px: float, py: float, zoom: int) -> Tuple[float, float]:
    n = 2.0**zoom
    lon = px / (n * TILE_SIZE) * 360.0 - 180.0
    lat_rad = math.atan(
        math.sinh(math.pi * (1.0 - 2.0 * py / (n * TILE_SIZE)))
    )
    lat = math.degrees(lat_rad)
    return lon, lat


def _pick_zoom(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    width_px: int,
    height_px: int,
) -> int:
    width_px = max(320, int(width_px))
    height_px = max(240, int(height_px))
    for z in range(18, 3, -1):
        x0, y0 = _lonlat_to_global_pixel(lon_min, lat_max, z)
        x1, y1 = _lonlat_to_global_pixel(lon_max, lat_min, z)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w <= width_px * 2.2 and h <= height_px * 2.2 and w >= width_px * 0.35:
            return z
    return 12


def _pad_bounds(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    pad_ratio: float = 0.14,
    min_span_deg: float = 0.004,
) -> Tuple[float, float, float, float]:
    dlon = lon_max - lon_min
    dlat = lat_max - lat_min
    if dlon < min_span_deg:
        c = (lon_min + lon_max) / 2
        lon_min, lon_max = c - min_span_deg / 2, c + min_span_deg / 2
        dlon = min_span_deg
    if dlat < min_span_deg:
        c = (lat_min + lat_max) / 2
        lat_min, lat_max = c - min_span_deg / 2, c + min_span_deg / 2
        dlat = min_span_deg
    pad_lon = dlon * pad_ratio
    pad_lat = dlat * pad_ratio
    return (
        lon_min - pad_lon,
        lon_max + pad_lon,
        lat_min - pad_lat,
        lat_max + pad_lat,
    )


def fetch_amap_basemap_rgba(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    *,
    width_px: int,
    height_px: int,
    style: str = "normal",
    api_key: Optional[str] = None,
    zoom: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    下载并裁剪高德瓦片，返回 RGBA [0,1] 与 extent (lon_min, lon_max, lat_min, lat_max)。
    """
    import requests

    key = (api_key or get_effective_amap_key()).strip()
    if not key:
        raise ValueError(
            "高德 API Key 未配置，请编辑 src/amap_api_config.json 填写 api_key"
        )

    style = normalize_basemap_style(style)

    lon_min, lon_max, lat_min, lat_max = _pad_bounds(
        lon_min, lon_max, lat_min, lat_max
    )
    if zoom is None:
        zoom = _pick_zoom(lon_min, lon_max, lat_min, lat_max, width_px, height_px)

    px_west, px_north = _lonlat_to_global_pixel(lon_min, lat_max, zoom)
    px_east, px_south = _lonlat_to_global_pixel(lon_max, lat_min, zoom)

    tx0 = int(math.floor(px_west / TILE_SIZE))
    tx1 = int(math.floor(px_east / TILE_SIZE))
    ty0 = int(math.floor(px_north / TILE_SIZE))
    ty1 = int(math.floor(px_south / TILE_SIZE))

    if tx1 - tx0 + 1 > MAX_TILES_PER_AXIS or ty1 - ty0 + 1 > MAX_TILES_PER_AXIS:
        if zoom <= 4:
            raise ValueError("轨迹范围过大，无法生成底图，请缩小区域或缩短 GPX")
        return fetch_amap_basemap_rgba(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            width_px=width_px,
            height_px=height_px,
            style=style,
            api_key=key,
            zoom=zoom - 1,
        )

    cols = tx1 - tx0 + 1
    rows = ty1 - ty0 + 1
    mosaic = Image.new("RGBA", (cols * TILE_SIZE, rows * TILE_SIZE), (240, 240, 240, 255))
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Birdy/1.0 (track-map; amap-tiles)",
            "Referer": "https://lbs.amap.com/",
        }
    )

    failed = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            urls = _tile_urls_for_style(tx, ty, zoom, style, key)
            try:
                layers = [_download_tile(session, u) for u in urls]
                tile = _compose_tile_layers(layers)
                mosaic.paste(
                    tile,
                    ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE),
                )
            except Exception:
                failed += 1

    if failed > cols * rows // 2:
        raise RuntimeError("高德底图瓦片下载失败过多，请检查网络与 API Key 权限")

    if style in _VECTOR_THEME_IDS and style not in ("normal", "digital"):
        mosaic = _apply_vector_theme_look(mosaic, style)

    crop_x0 = int(round(px_west - tx0 * TILE_SIZE))
    crop_x1 = int(round(px_east - tx0 * TILE_SIZE))
    crop_y0 = int(round(px_north - ty0 * TILE_SIZE))
    crop_y1 = int(round(px_south - ty0 * TILE_SIZE))
    crop_x0 = max(0, min(crop_x0, mosaic.width - 1))
    crop_x1 = max(crop_x0 + 1, min(crop_x1, mosaic.width))
    crop_y0 = max(0, min(crop_y0, mosaic.height - 1))
    crop_y1 = max(crop_y0 + 1, min(crop_y1, mosaic.height))

    cropped = mosaic.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    if width_px > 0 and height_px > 0:
        cropped = cropped.resize((width_px, height_px), Image.Resampling.LANCZOS)

    out_lon_min, out_lat_max = _global_pixel_to_lonlat(
        tx0 * TILE_SIZE + crop_x0, ty0 * TILE_SIZE + crop_y0, zoom
    )
    out_lon_max, out_lat_min = _global_pixel_to_lonlat(
        tx0 * TILE_SIZE + crop_x1, ty0 * TILE_SIZE + crop_y1, zoom
    )

    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    extent = (out_lon_min, out_lon_max, out_lat_min, out_lat_max)
    return arr, extent


def regeo_place_label(lon: float, lat: float) -> Optional[str]:
    """逆地理编码，返回简短地名（省市区县），失败返回 None。"""
    prov, city, _district = regeo_address_components(lon, lat)
    parts = [p for p in (prov, city) if p]
    if parts:
        return "".join(parts[:2])
    return None


def regeo_address_components(
    lon: float, lat: float,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """逆地理编码，返回 (省, 市, 区县)；失败时均为 None。"""
    import requests

    key = get_effective_amap_key()
    if not key:
        return None, None, None
    glon, glat = wgs84_to_map_lonlat(lon, lat)
    try:
        r = requests.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params={
                "key": key,
                "location": f"{glon:.6f},{glat:.6f}",
                "extensions": "base",
                "output": "json",
            },
            timeout=TILE_TIMEOUT,
        )
        data = r.json()
        if data.get("status") != "1":
            return None, None, None
        comp = (data.get("regeocode") or {}).get("addressComponent") or {}

        def _one(key: str) -> Optional[str]:
            v = comp.get(key)
            if isinstance(v, list):
                v = v[0] if v else ""
            v = (v or "").strip()
            return v or None

        province = _one("province")
        city = _one("city") or _one("district")
        district = _one("district")
        return province, city, district
    except Exception:
        return None, None, None


def gcj_bounds_from_lonlats(
    lons: List[float], lats: List[float]
) -> Tuple[float, float, float, float]:
    if not lons:
        raise ValueError("无有效坐标")
    mlons: List[float] = []
    mlats: List[float] = []
    for lon, lat in zip(lons, lats):
        glon, glat = wgs84_to_map_lonlat(lon, lat)
        mlons.append(glon)
        mlats.append(glat)
    return min(mlons), max(mlons), min(mlats), max(mlats)
