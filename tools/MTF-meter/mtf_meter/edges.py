# -*- coding: utf-8 -*-
"""定位 ISO 12233 斜边 ROI：只测 L/R 黑斜块、上方 T、中心偏下斜方块。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 检测在此长边尺度上进行，避免高分辨率下霍夫最小线长过大、整图一条都找不到。
_DETECT_LONG_SIDE = 1280

# Super Image Test / ISO 12233 综合卡：几何中心是圆环，四角是星形图，不宜当斜边。
# 只保留真正的黑色斜方块：左右、上方长条、画面中心偏下。
_SLOTS = (
    # slot, zone, x0, y0, x1, y1  （相对画幅）
    ("left", "corner", 0.00, 0.22, 0.26, 0.78),
    ("right", "corner", 0.74, 0.22, 1.00, 0.78),
    ("top", "corner", 0.30, 0.03, 0.70, 0.36),
    ("center", "center", 0.30, 0.50, 0.70, 0.84),
)
_MAX_SLOTS = len(_SLOTS)
_MIN_STEP = 22.0


def _slant_ok(angle_deg: float) -> bool:
    """ISO 12233 斜边约 5°；排除水平/垂直（含轻微相机倾斜）和 45° 类图案。"""
    a = abs(float(angle_deg)) % 180.0
    dev_h = min(a, 180.0 - a)
    dev_v = abs(a - 90.0)
    slant = min(dev_h, dev_v)
    return 3.0 <= slant <= 15.0


def _roi_side(h: int, w: int) -> int:
    return int(np.clip(round(min(h, w) * 0.12), 72, 360))


def _work_image(g8: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = g8.shape[:2]
    long = max(h, w)
    if long <= _DETECT_LONG_SIDE:
        return g8, 1.0
    scale = float(_DETECT_LONG_SIDE) / float(long)
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    work = cv2.resize(g8, (nw, nh), interpolation=cv2.INTER_AREA)
    return work, scale


def _cells(h: int, w: int) -> List[Tuple[str, str, int, int, int, int]]:
    """L / R / T / 中下四区，避开中央圆环与四角星形图。"""
    out = []
    for slot, zone, fx0, fy0, fx1, fy1 in _SLOTS:
        x1 = int(round(w * fx0))
        y1 = int(round(h * fy0))
        x2 = int(round(w * fx1))
        y2 = int(round(h * fy1))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 40 or y2 - y1 < 40:
            continue
        out.append((slot, zone, x1, y1, x2, y2))
    return out


def _line_step_contrast(
    crop: np.ndarray, x1: int, y1: int, x2: int, y2: int, dist: int = 6
) -> float:
    """沿线两侧亮度差：黑白斜方块很高，圆环/星形图往往较低或很碎。"""
    h, w = crop.shape[:2]
    ang = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(ang), math.cos(ang)
    n = max(12, int(math.hypot(x2 - x1, y2 - y1)))
    ts = np.linspace(0.18, 0.82, n)
    xs = x1 + ts * (x2 - x1)
    ys = y1 + ts * (y2 - y1)
    a: List[float] = []
    b: List[float] = []
    for x, y in zip(xs, ys):
        xa = int(round(x + nx * dist))
        ya = int(round(y + ny * dist))
        xb = int(round(x - nx * dist))
        yb = int(round(y - ny * dist))
        if 0 <= xa < w and 0 <= ya < h:
            a.append(float(crop[ya, xa]))
        if 0 <= xb < w and 0 <= yb < h:
            b.append(float(crop[yb, xb]))
    if len(a) < 6 or len(b) < 6:
        return 0.0
    return abs(float(np.median(a)) - float(np.median(b)))


def _best_slanted_line(
    crop: np.ndarray,
    ox: int,
    oy: int,
    min_len: int,
) -> Optional[Tuple[int, int, int, int, float, float]]:
    ch, cw = crop.shape[:2]
    if ch < 32 or cw < 32:
        return None
    blur = cv2.GaussianBlur(crop, (5, 5), 1.0)
    edges = cv2.Canny(blur, 50, 150)
    ml = max(22, int(min_len))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(18, ml // 4),
        minLineLength=ml,
        maxLineGap=8,
    )
    if lines is None:
        return None
    best = None
    best_score = -1.0
    for row in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not _slant_ok(ang):
            continue
        contrast = _line_step_contrast(blur, x1, y1, x2, y2)
        if contrast < _MIN_STEP:
            continue
        score = length * (1.0 + contrast / 40.0)
        if score > best_score:
            best_score = score
            best = (x1 + ox, y1 + oy, x2 + ox, y2 + oy, length, ang)
    return best


def _square_on_midpoint(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    side: int,
    w: int,
    h: int,
) -> Optional[Tuple[int, int, int, int]]:
    cx = int(round(0.5 * (x1 + x2)))
    cy = int(round(0.5 * (y1 + y2)))
    half = side // 2
    xa = cx - half
    ya = cy - half
    xb = xa + side
    yb = ya + side
    if xa < 0:
        xb -= xa
        xa = 0
    if ya < 0:
        yb -= ya
        ya = 0
    if xb > w:
        xa -= xb - w
        xb = w
    if yb > h:
        ya -= yb - h
        yb = h
    xa, ya = max(0, xa), max(0, ya)
    if xb - xa < 48 or yb - ya < 48:
        return None
    return xa, ya, xb, yb


def find_edge_rois(
    gray: np.ndarray,
    max_rois: int = _MAX_SLOTS,
) -> List[Dict]:
    """
    长边缩到约 1280 后，只在 L/R/T/中下四区各找一条 3–15° 高对比斜边，框映射回原图。
    不搜四角星形图、正中圆环、底部其它图案。
    """
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    h0, w0 = g8.shape
    work, scale = _work_image(g8)
    hh, ww = work.shape
    min_len = max(24, int(round(min(hh, ww) * 0.05)))
    cap = max(1, min(_MAX_SLOTS, int(max_rois)))
    found: List[Dict] = []
    for slot, zone, x1, y1, x2, y2 in _cells(hh, ww):
        if len(found) >= cap:
            break
        crop = work[y1:y2, x1:x2]
        line = _best_slanted_line(crop, x1, y1, min_len)
        if line is None:
            continue
        lx1, ly1, lx2, ly2, length, ang = line
        if scale != 1.0:
            inv = 1.0 / scale
            lx1 = int(round(lx1 * inv))
            ly1 = int(round(ly1 * inv))
            lx2 = int(round(lx2 * inv))
            ly2 = int(round(ly2 * inv))
            length = float(length * inv)
        side = _roi_side(h0, w0)
        box = _square_on_midpoint(lx1, ly1, lx2, ly2, side, w0, h0)
        if box is None:
            continue
        xa, ya, xb, yb = box
        found.append(
            {
                "x1": xa,
                "y1": ya,
                "x2": xb,
                "y2": yb,
                "length": length,
                "angle_deg": ang,
                "zone": zone,
                "slot": slot,
            }
        )
    return found
