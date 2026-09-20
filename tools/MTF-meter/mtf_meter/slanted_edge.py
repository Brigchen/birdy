# -*- coding: utf-8 -*-
"""ISO 12233 斜边 SFR：ESF → LSF → MTF，求 MTF50/30/10。"""

from __future__ import annotations

from math import erf
from typing import Dict, Optional, Tuple

import numpy as np


def theoretical_gaussian_mtf50(sigma_px: float) -> float:
    """高斯 LSF（σ 像素）的 MTF50（cycles/pixel）。"""
    if sigma_px <= 0:
        return 0.0
    # MTF(f)=exp(-2 π² σ² f²)=0.5 → f = sqrt(ln2)/(σ π √2)
    return float(np.sqrt(np.log(2.0)) / (sigma_px * np.pi * np.sqrt(2.0)))


def make_slanted_edge(
    h: int = 160,
    w: int = 160,
    angle_deg: float = 5.0,
    sigma: float = 1.2,
    low: float = 40.0,
    high: float = 210.0,
) -> np.ndarray:
    """合成斜边图（亮度 float64），用于单元测试。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = np.deg2rad(angle_deg)
    # 过中心、法向 (cos, sin) 的有符号距离
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    dist = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    scale = sigma * np.sqrt(2.0)
    edge = np.vectorize(erf)(dist / max(1e-6, scale))
    img = low + (high - low) * (0.5 * (edge + 1.0))
    return img


def _bin_esf(
    gray: np.ndarray,
    nx: float,
    ny: float,
    oversample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    t = (xs - cx) * nx + (ys - cy) * ny
    t_min = float(t.min())
    t_max = float(t.max())
    n_bins = max(32, int(round((t_max - t_min) * oversample)) + 1)
    edges = np.linspace(t_min, t_max, n_bins + 1)
    idx = np.clip(np.digitize(t.ravel(), edges) - 1, 0, n_bins - 1)
    weights = np.bincount(idx, minlength=n_bins).astype(np.float64)
    acc = np.bincount(idx, weights=gray.ravel(), minlength=n_bins)
    ok = weights > 0
    esf = np.zeros(n_bins, dtype=np.float64)
    esf[ok] = acc[ok] / weights[ok]
    # 空仓线性插值
    if ok.any() and not ok.all():
        xp = np.flatnonzero(ok)
        esf[~ok] = np.interp(np.flatnonzero(~ok), xp, esf[ok])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, esf


def _mtf_from_esf(
    esf: np.ndarray,
    dx_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lsf = np.gradient(esf, dx_px)
    # 去均值，避免直流泄漏
    lsf = lsf - np.mean(lsf)
    win = np.hamming(len(lsf))
    spec = np.abs(np.fft.rfft(lsf * win))
    if spec[0] <= 1e-12:
        spec[0] = 1e-12
    mtf = spec / spec[0]
    freq = np.fft.rfftfreq(len(lsf), d=dx_px)
    # 有限差分 sinc 校正
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sinc(freq * dx_px)
        corr = np.ones_like(mtf)
        nz = np.abs(sinc) > 1e-6
        corr[nz] = 1.0 / sinc[nz]
    mtf = np.clip(mtf * corr, 0.0, 4.0)
    mtf[0] = 1.0
    return freq, mtf


def _crossing(freq: np.ndarray, mtf: np.ndarray, level: float) -> Optional[float]:
    """MTF 降到 level 的频率（cycles/pixel）。"""
    if len(freq) < 3:
        return None
    # 从峰值之后找第一次下穿
    peak_i = int(np.argmax(mtf[: max(2, len(mtf) // 3)]))
    y = mtf[peak_i:]
    x = freq[peak_i:]
    below = np.where(y <= level)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(x[0])
    y0, y1 = float(y[i - 1]), float(y[i])
    x0, x1 = float(x[i - 1]), float(x[i])
    if abs(y1 - y0) < 1e-12:
        return x1
    t = (level - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def slanted_edge_sfr(
    gray: np.ndarray,
    oversample: int = 4,
) -> Optional[Dict[str, float]]:
    """
    对含一条主导斜边的 ROI 计算 SFR。
    返回 cycles/pixel 的 MTF50/30/10、过锐峰值、边对比度。
    """
    if gray is None or gray.size < 64:
        return None
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        return None
    h, w = g.shape
    if h < 24 or w < 24:
        return None
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    mag = np.hypot(gx, gy)
    thr = float(np.percentile(mag, 80))
    mask = mag >= max(thr, 1e-6)
    if int(mask.sum()) < 20:
        return None
    mx = float(np.mean(gx[mask]))
    my = float(np.mean(gy[mask]))
    nlen = float(np.hypot(mx, my))
    if nlen < 1e-8:
        return None
    nx, ny = mx / nlen, my / nlen
    # 保证沿法向由暗到亮
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    probe = 8.0
    y0 = int(np.clip(round(cy - ny * probe), 0, h - 1))
    x0 = int(np.clip(round(cx - nx * probe), 0, w - 1))
    y1 = int(np.clip(round(cy + ny * probe), 0, h - 1))
    x1 = int(np.clip(round(cx + nx * probe), 0, w - 1))
    if g[y1, x1] < g[y0, x0]:
        nx, ny = -nx, -ny
    centers, esf = _bin_esf(g, nx, ny, oversample=max(2, int(oversample)))
    if len(esf) < 24:
        return None
    dx = float(np.mean(np.diff(centers))) if len(centers) > 1 else 1.0 / oversample
    if dx <= 1e-8:
        return None
    contrast = float(np.percentile(esf, 90) - np.percentile(esf, 10))
    if contrast < 8.0:
        return None
    freq, mtf = _mtf_from_esf(esf, dx)
    mtf50 = _crossing(freq, mtf, 0.50)
    mtf30 = _crossing(freq, mtf, 0.30)
    mtf10 = _crossing(freq, mtf, 0.10)
    if mtf50 is None:
        return None
    # 超过奈奎斯特（0.5 cy/px）视为 ROI 内不是单条斜边（图案/文字干扰）
    if not np.isfinite(mtf50) or float(mtf50) >= 0.48:
        return None
    ang = float(np.degrees(np.arctan2(ny, nx)))
    return {
        "mtf50_cy_px": float(mtf50),
        "mtf30_cy_px": float(mtf30) if mtf30 is not None else float("nan"),
        "mtf10_cy_px": float(mtf10) if mtf10 is not None else float("nan"),
        "mtf_peak": float(np.max(mtf[1 : max(2, len(mtf) // 4)])),
        "contrast": contrast,
        "edge_angle_deg": ang,
        "nyquist_cy_px": 0.5,
    }


def cy_px_to_lp_mm(cy_px: Optional[float], pixels_per_mm: Optional[float]) -> Optional[float]:
    if cy_px is None or pixels_per_mm is None:
        return None
    if not np.isfinite(cy_px) or pixels_per_mm <= 0:
        return None
    return float(cy_px * pixels_per_mm)
