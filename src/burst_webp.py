# -*- coding: utf-8 -*-
"""
连拍序列 → 动画 WebP 或 MP4：白平衡、按裁剪框自动曝光、按每帧标定点+裁剪区裁剪（越界补边）、
可选叠加水印（连拍时元数据按首张）、按「每秒几张」设置帧时长；MP4 便于不支持动图 WebP 的客户端播放。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from burst_anchor import (
    FrameLayout,
    crop_bgr_with_pad,
    geom_from_first,
    in_bounds_crop_xyxy,
    layout_valid,
    meter_box_in_padded_crop,
)
from image_io import imread_bgr
from watermark_generator import WatermarkOptions, render_watermark_on_pil_image

ProgressCb = Optional[Callable[[int, int, str], None]]


def _emit_export_progress(
    progress: ProgressCb,
    log_terminal: bool,
    cur: int,
    tot: int,
    msg: str,
    log_tag: str = "[Birdy WebP导出]",
) -> None:
    """GUI progress + 终端日志（连拍导出：WebP 或视频）。"""
    if progress:
        try:
            progress(cur, tot, msg)
        except Exception:
            pass
    if log_terminal:
        print(f"{log_tag} {cur}/{tot} {msg}", flush=True)


def _burst_align_log(msg: str) -> None:
    """连拍对齐每帧位移与中间量，打印到运行 Birdy 的终端。"""
    print(f"[Birdy 连拍对齐] {msg}", flush=True)


def _subsec_ascii_to_microseconds(subsec: str) -> int:
    """
    EXIF SubSecTime*：ASCII 数字表示「小数点后的有效数字」。
    例如 '027' → 0.027s → 27000µs；'12' → 0.12s。
    微秒 clamp 到 [0, 999999] 以写入 datetime。
    """
    s = (subsec or "").strip()
    if not s or not s.isdigit():
        return 0
    val = int(s)
    denom = 10 ** len(s)
    usec = int(round(val / float(denom) * 1_000_000.0))
    return int(np.clip(usec, 0, 999_999))


def _merge_datetime_with_subsec(dt: datetime, subsec: str) -> datetime:
    us = _subsec_ascii_to_microseconds(subsec)
    return dt.replace(microsecond=us)


def _parse_exif_dt(s: str) -> Optional[datetime]:
    s = str(s).strip()
    if len(s) < 19:
        return None
    try:
        return datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _try_exif_datetime(path: str) -> Optional[datetime]:
    """
    读取拍摄时间，尽量带上 SubSecTimeOriginal 等子秒信息；
    否则连拍同秒多帧时间差为 0，推断间隔会错误地变成约 1s。
    """
    try:
        from PIL.ExifTags import IFD
    except ImportError:
        IFD = None  # type: ignore[misc, assignment]

    try:
        with Image.open(path) as im:
            if hasattr(im, "getexif"):
                ex = im.getexif()
                if ex is not None:
                    exif_ifd: dict = {}
                    if IFD is not None:
                        try:
                            exif_ifd = ex.get_ifd(IFD.Exif)
                        except Exception:
                            exif_ifd = {}
                    if exif_ifd:
                        # (日期标签, 与之匹配的子秒标签) — 与常见相机 EXIF 一致
                        pairs = (
                            (36867, 37521),  # DateTimeOriginal + SubSecTimeOriginal
                            (36868, 37522),  # DateTimeDigitized + SubSecTimeDigitized
                            (36867, 37520),  # 部分机型子秒只在 SubSecTime
                        )
                        for dk, sk in pairs:
                            if dk not in exif_ifd:
                                continue
                            raw_dt = exif_ifd.get(dk)
                            if not raw_dt:
                                continue
                            dt = _parse_exif_dt(str(raw_dt))
                            if not dt:
                                continue
                            sub = ""
                            if sk in exif_ifd and exif_ifd.get(sk) is not None:
                                sub = str(exif_ifd.get(sk)).strip()
                            return _merge_datetime_with_subsec(dt, sub)
                    # 主时间写在顶层 IFD 时，子秒仍在 Exif 子 IFD
                    for k in (36867, 306, 36868):
                        if k not in ex:
                            continue
                        s = ex.get(k)
                        if not s:
                            continue
                        dt = _parse_exif_dt(str(s))
                        if not dt:
                            continue
                        sub = ""
                        for sk in (37521, 37522, 37520):
                            if exif_ifd and sk in exif_ifd and exif_ifd.get(sk):
                                sub = str(exif_ifd.get(sk)).strip()
                                break
                        return _merge_datetime_with_subsec(dt, sub)
            if hasattr(im, "_getexif"):
                raw = im._getexif()
                if raw:
                    for k in (36867, 306, 36868):
                        if k in raw:
                            s = raw.get(k)
                            if s:
                                dt = _parse_exif_dt(str(s))
                                if dt:
                                    return dt
    except Exception:
        pass
    return None


def sort_paths_by_capture_time(paths: List[str]) -> List[str]:
    """按 EXIF 拍摄时间排序；无 EXIF 时用修改时间。"""

    def sort_key(p: str) -> Tuple[float, str]:
        dt = _try_exif_datetime(p)
        if dt is not None:
            return (dt.timestamp(), p.lower())
        try:
            st = os.stat(p)
            if getattr(st, "st_mtime_ns", None) is not None:
                return (st.st_mtime_ns / 1e9, p.lower())
            return (float(st.st_mtime), p.lower())
        except OSError:
            return (0.0, p.lower())

    return sorted(paths, key=sort_key)


def infer_shot_interval_ms(paths_ordered: List[str]) -> Tuple[float, str]:
    """推断相邻帧间隔（毫秒）及说明文案。"""
    if len(paths_ordered) < 2:
        return 200.0, "单张默认间隔 200 ms"

    n = len(paths_ordered)
    dts: List[Optional[datetime]] = [_try_exif_datetime(p) for p in paths_ordered]
    if all(x is not None for x in dts):
        dlist: List[datetime] = [x for x in dts if x is not None]  # type: ignore[assignment]
        lacks_subsec = all(d.microsecond == 0 for d in dlist)
        span_ms = (dlist[-1] - dlist[0]).total_seconds() * 1000.0
        avg_span = span_ms / float(n - 1) if n >= 2 and span_ms > 0 else None

        deltas: List[float] = []
        for i in range(n - 1):
            ms = (dlist[i + 1] - dlist[i]).total_seconds() * 1000.0
            if 0 < ms < 120_000:
                deltas.append(ms)

        # 有 SubSec（至少一帧带微秒）：相邻帧中位数最可靠
        if not lacks_subsec and deltas:
            med = float(np.median(deltas))
            return _clamp_ms(med), "EXIF 相邻帧间隔（中位数，含 SubSec）"

        # 无 SubSec（全部为整秒）：相邻差多为 0 被丢弃，用首张—末张总时长均摊
        if lacks_subsec and avg_span is not None and 0 < avg_span < 120_000:
            return _clamp_ms(float(avg_span)), "EXIF 首张—末张平均间隔（无 SubSec）"

        if deltas:
            med = float(np.median(deltas))
            return _clamp_ms(med), "EXIF 相邻帧间隔（中位数）"

        if avg_span is not None and 0 < avg_span < 120_000:
            return _clamp_ms(float(avg_span)), "EXIF 首张—末张平均间隔"

    mt: List[float] = []
    for p in paths_ordered:
        try:
            st = os.stat(p)
            if getattr(st, "st_mtime_ns", None) is not None:
                mt.append(st.st_mtime_ns / 1e9)
            else:
                mt.append(float(st.st_mtime))
        except OSError:
            mt.append(0.0)
    span2_ms = (mt[-1] - mt[0]) * 1000.0 if n >= 2 else 0.0
    avg_mtime = span2_ms / float(n - 1) if span2_ms > 0 and n >= 2 else None

    deltas2: List[float] = []
    for i in range(len(mt) - 1):
        ms = (mt[i + 1] - mt[i]) * 1000.0
        if 0 < ms < 120_000:
            deltas2.append(ms)
    if deltas2:
        med = float(np.median(deltas2))
        return _clamp_ms(med), "文件修改时间差（中位数）"
    if avg_mtime is not None and 0 < avg_mtime < 120_000:
        return _clamp_ms(float(avg_mtime)), "文件修改时间首张—末张平均间隔"
    return 200.0, "默认 200 ms（无法可靠推断间隔）"


def _clamp_ms(x: float) -> float:
    return float(np.clip(x, 10.0, 60_000.0))


def gray_world_balance_factors(bgr: np.ndarray) -> Tuple[float, float, float]:
    """首张灰世界：返回 (B 乘子, G 乘子, R 乘子)，供连拍后续帧复用。"""
    if bgr is None or bgr.size == 0:
        return 1.0, 1.0, 1.0
    img = bgr.astype(np.float32)
    b_mean = float(np.mean(img[:, :, 0])) + 1e-6
    g_mean = float(np.mean(img[:, :, 1])) + 1e-6
    r_mean = float(np.mean(img[:, :, 2])) + 1e-6
    k = (b_mean + g_mean + r_mean) / 3.0
    return (k / b_mean, k / g_mean, k / r_mean)


def gray_world_white_balance_with_factors(
    bgr: np.ndarray, fb: float, fg: float, fr: float
) -> np.ndarray:
    """用给定乘子做灰世界白平衡（BGR）。"""
    if bgr is None or bgr.size == 0:
        return bgr
    img = bgr.astype(np.float32)
    out = np.empty_like(img)
    out[:, :, 0] = np.clip(img[:, :, 0] * fb, 0, 255)
    out[:, :, 1] = np.clip(img[:, :, 1] * fg, 0, 255)
    out[:, :, 2] = np.clip(img[:, :, 2] * fr, 0, 255)
    return out.astype(np.uint8)


def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """简单灰世界白平衡（BGR）。"""
    fb, fg, fr = gray_world_balance_factors(bgr)
    return gray_world_white_balance_with_factors(bgr, fb, fg, fr)


def _wb_burst_pipeline(
    raw_bgr: List[np.ndarray], opts: BurstWebpBuildOptions
) -> List[np.ndarray]:
    """连拍：首张统计白平衡系数，后续帧套用。"""
    n = len(raw_bgr)
    if n == 0:
        return []
    if opts.enable_white_balance and n >= 2:
        fb, fg, fr = gray_world_balance_factors(raw_bgr[0])
        return [gray_world_white_balance_with_factors(x, fb, fg, fr) for x in raw_bgr]
    if opts.enable_white_balance:
        return [gray_world_white_balance(x) for x in raw_bgr]
    return raw_bgr


def _crop_frame_maybe_ae(
    fr: np.ndarray,
    lay: FrameLayout,
    geom,
    opts: BurstWebpBuildOptions,
) -> np.ndarray:
    """先按布局裁剪，再按裁剪框内有效区域测光做自动曝光（比全图 gamma 快得多）。"""
    crop = crop_bgr_with_pad(fr, lay, geom)
    if not bool(getattr(opts, "enable_auto_exposure", False)):
        return crop
    strength = float(
        np.clip(float(getattr(opts, "auto_exposure_strength", 1.0)), 0.0, 3.0)
    )
    if strength <= 0.0:
        return crop
    from auto_exposure import auto_expose_bgr

    h, w = fr.shape[:2]
    box = meter_box_in_padded_crop(lay, geom, w, h)
    return auto_expose_bgr(crop, strength=strength, detect=False, meter_box=box)


def _auto_expose_by_crop_layouts(
    frames: List[np.ndarray],
    lays: List[FrameLayout],
    geom,
    opts: BurstWebpBuildOptions,
) -> List[np.ndarray]:
    """兼容旧路径：对整图按裁剪框测光做自动曝光。导出主路径已改为裁剪后再曝光。"""
    if not bool(getattr(opts, "enable_auto_exposure", False)):
        return frames
    strength = float(
        np.clip(float(getattr(opts, "auto_exposure_strength", 1.0)), 0.0, 3.0)
    )
    if strength <= 0.0:
        return frames
    from auto_exposure import auto_expose_bgr

    out: List[np.ndarray] = []
    for fr, lay in zip(frames, lays):
        h, w = fr.shape[:2]
        box = in_bounds_crop_xyxy(lay, geom, w, h)
        out.append(
            auto_expose_bgr(fr, strength=strength, detect=False, meter_box=box)
        )
    return out


def _resize_long_edge(bgr: np.ndarray, max_long: int) -> np.ndarray:
    if max_long <= 0:
        return bgr
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_long:
        return bgr
    sc = max_long / float(m)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def infer_crop_center_norm_from_birds(
    bgr: np.ndarray, birds: List[dict]
) -> Tuple[float, float]:
    """
    根据鸟框并集中心得到归一化裁剪中心 (nx,ny)∈[0,1]²；无检测框则 (0.5,0.5)。
    """
    if bgr is None or bgr.size == 0:
        return (0.5, 0.5)
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0 or not birds:
        return (0.5, 0.5)
    x1 = min(int(b["bbox"][0]) for b in birds)
    y1 = min(int(b["bbox"][1]) for b in birds)
    x2 = max(int(b["bbox"][2]) for b in birds)
    y2 = max(int(b["bbox"][3]) for b in birds)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (
        float(np.clip(cx / float(w), 0.0, 1.0)),
        float(np.clip(cy / float(h), 0.0, 1.0)),
    )


def crop_window_rect_pixels(
    w: int,
    h: int,
    retention: float,
    cx_norm: float,
    cy_norm: float,
) -> Tuple[int, int, int, int]:
    """
    与 _center_crop_around_point 相同的裁剪窗口：返回 (x0, y0, crop_w, crop_h)。
    """
    r = float(np.clip(retention, 0.25, 1.0))
    if r >= 0.999 or w <= 0 or h <= 0:
        return 0, 0, w, h
    nw = max(1, int(round(w * r)))
    nh = max(1, int(round(h * r)))
    nw = min(nw, w)
    nh = min(nh, h)
    cx = float(np.clip(cx_norm, 0.0, 1.0)) * float(w)
    cy = float(np.clip(cy_norm, 0.0, 1.0)) * float(h)
    x0 = int(round(cx - 0.5 * float(nw)))
    y0 = int(round(cy - 0.5 * float(nh)))
    x0 = max(0, min(w - nw, x0))
    y0 = max(0, min(h - nh, y0))
    return x0, y0, nw, nh


def _center_crop_around_point(
    bgr: np.ndarray,
    retention: float,
    cx_norm: float,
    cy_norm: float,
) -> np.ndarray:
    """
    以 (cx_norm, cy_norm) 为裁剪窗口中心（先尽量对准，再夹紧到图像内），
    保留约 retention 比例的宽与高（稳定裁边）。
    """
    r = float(np.clip(retention, 0.25, 1.0))
    if r >= 0.999:
        return bgr
    h, w = bgr.shape[:2]
    x0, y0, nw, nh = crop_window_rect_pixels(w, h, r, cx_norm, cy_norm)
    return bgr[y0 : y0 + nh, x0 : x0 + nw].copy()


def _ecc_gray_scene_align(gray_u8: np.ndarray) -> np.ndarray:
    """
    强低频灰度，用于连拍对齐：让相位/ECC 主要响应场景整体位移，
    减弱鸟等小目标在画面中部对全局平移/旋转估计的拉动。
    先盒式核再大 σ 高斯，尽量抹掉鸟体中频纹理，只留大尺度背景趋势。
    """
    if gray_u8.size == 0:
        return gray_u8.astype(np.float32)
    h, w = gray_u8.shape[:2]
    m = min(h, w)
    k = 9 if m >= 32 else 5
    g = cv2.GaussianBlur(gray_u8, (k, k), 0).astype(np.float32)
    if m >= 48:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=3.0, sigmaY=3.0)
    return g


def _ecc_disk_mask(
    h: int, w: int, cx: float, cy: float, radius_frac: float
) -> np.ndarray:
    """
    人工指定的追踪圆：圆内 255、圆外 0（与首张归一化坐标一致）。
    radius_frac 为半径占 min(h,w) 的比例。
    """
    m = np.zeros((h, w), dtype=np.uint8)
    rf = float(np.clip(radius_frac, 0.02, 0.45))
    R = max(4.0, min(rf * float(min(h, w)), 0.48 * float(min(h, w))))
    fx = float(np.clip(cx, 0.0, 1.0)) * float(max(0, w - 1))
    fy = float(np.clip(cy, 0.0, 1.0)) * float(max(0, h - 1))
    yy, xx = np.indices((h, w), dtype=np.float64)
    dist2 = (xx - fx) ** 2 + (yy - fy) ** 2
    m[dist2 <= R * R] = 255
    return m


def _ecc_border_ring_mask(h: int, w: int) -> np.ndarray:
    """
    仅四周边带参与对齐（uint8 0/255）：中心矩形权重为 0，
    鸟从停飞起时中部强运动不会进入相关与 ECC，整帧位移跟边带背景走。
    """
    m = np.zeros((h, w), dtype=np.uint8)
    band = max(12, int(round(0.24 * float(min(h, w)))))
    if 2 * band + 8 >= min(h, w):
        band = max(8, min(h, w) // 5)
    m[:band, :] = 255
    m[h - band :, :] = 255
    m[:, :band] = 255
    m[:, w - band :] = 255
    return m


def _clamp_align_translation(M: np.ndarray, w: int, h: int, max_frac: float) -> None:
    lim = float(max_frac * min(h, w))
    M[0, 2] = float(np.clip(M[0, 2], -lim, lim))
    M[1, 2] = float(np.clip(M[1, 2], -lim, lim))


def _clamp_align_euclidean(
    M: np.ndarray, w: int, h: int, max_frac: float, max_rot_rad: float
) -> None:
    """MOTION_EUCLIDEAN 的 2x3：旋转 + 平移，夹紧避免离谱解导致抖动。"""
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    theta = float(np.clip(theta, -max_rot_rad, max_rot_rad))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    _clamp_align_translation(M, w, h, max_frac)


def _euclidean_params_from_M(M: np.ndarray) -> Tuple[float, float, float]:
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    return theta, float(M[0, 2]), float(M[1, 2])


def _M_from_euclidean_params(theta: float, tx: float, ty: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)


def _smooth_align_param_track(x: np.ndarray, radius: int) -> np.ndarray:
    """对称滑动平均（边界 replicate），压低帧间参数抖动。"""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 1 or radius <= 0:
        return x.astype(np.float64, copy=False)
    k = 2 * radius + 1
    pad = np.pad(x, (radius, radius), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / float(k)
    return np.convolve(pad, kernel, mode="valid")


def _median_smooth_1d(x: np.ndarray, radius: int) -> np.ndarray:
    """滑动中值，去掉鸟起飞等造成的单帧位移尖峰。"""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 1 or radius <= 0:
        return x.astype(np.float64, copy=False)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def _velocity_limit_track(x: np.ndarray, max_step: float) -> np.ndarray:
    """相邻帧参数变化不超过 max_step，抑制残留的大幅错估。"""
    x = np.asarray(x, dtype=np.float64).copy()
    n = int(x.size)
    if n <= 1 or max_step <= 0:
        return x
    for i in range(1, n):
        x[i] = float(np.clip(x[i], x[i - 1] - max_step, x[i - 1] + max_step))
    return x


def _roi_norm_rect_to_pixel_box(
    w: int, h: int, roi_norm: Tuple[float, float, float, float], min_side: int = 32
) -> Tuple[int, int, int, int]:
    """归一化矩形 (x0,y0,x1,y1) → 像素半开盒 [x0,x1)×[y0,y1)，排序并保证最小边长。"""
    a, b, c, d = (float(x) for x in roi_norm)
    x0n, x1n = sorted((max(0.0, min(1.0, a)), max(0.0, min(1.0, c))))
    y0n, y1n = sorted((max(0.0, min(1.0, b)), max(0.0, min(1.0, d))))
    x0 = int(round(x0n * float(max(0, w - 1))))
    x1 = int(round(x1n * float(max(0, w - 1))))
    y0 = int(round(y0n * float(max(0, h - 1))))
    y1 = int(round(y1n * float(max(0, h - 1))))
    if x1 <= x0:
        x1 = min(w, x0 + min_side)
    if y1 <= y0:
        y1 = min(h, y0 + min_side)
    if x1 - x0 < min_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_side // 2)
        x1 = min(w, x0 + min_side)
    if y1 - y0 < min_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_side // 2)
        y1 = min(h, y0 + min_side)
    x1 = max(min(w, x1), x0 + 1)
    y1 = max(min(h, y1), y0 + 1)
    return x0, y0, x1, y1


def _inflate_pixel_rect_xyxy(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    w: int,
    h: int,
    pad_x: int,
    pad_y: int,
) -> Tuple[int, int, int, int]:
    """轴对齐矩形各边外扩 pad_* 后裁入 [0,w)×[0,h)。"""
    xa = max(0, int(x0) - int(pad_x))
    ya = max(0, int(y0) - int(pad_y))
    xb = min(int(w), int(x1) + int(pad_x))
    yb = min(int(h), int(y1) + int(pad_y))
    if xb <= xa:
        xb = min(w, xa + 1)
    if yb <= ya:
        yb = min(h, ya + 1)
    return xa, ya, xb, yb


def _make_feature_detector(name: str) -> Tuple[str, Any]:
    u = (name or "ORB").strip().upper()
    if u == "BRISK":
        return "BRISK", cv2.BRISK_create(thresh=20)
    if u == "SIFT":
        try:
            return "SIFT", cv2.SIFT_create(nfeatures=800)
        except Exception:
            return _make_feature_detector("ORB")
    return "ORB", cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8)


def _match_feature_descriptors(kind: str, d0: Any, d1: Any) -> List[Any]:
    if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
        return []
    norm = cv2.NORM_HAMMING if kind in ("ORB", "BRISK") else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    pairs = bf.knnMatch(d0, d1, k=2)
    good: List[Any] = []
    for pr in pairs:
        if len(pr) < 2:
            continue
        m, n = pr[0], pr[1]
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def _roi_align_reproj_thresh(w: int, h: int, det_name: str) -> float:
    """
    全幅连拍上 ORB/SIFT 匹配位移的内点阈值（像素）：4px 在 5k 级图上过严，
    易在 n_match 下降时出现「中位数被离群点拉飞 → n_in=0」而整帧放弃对齐。
    """
    m = float(min(int(h), int(w)))
    base = max(6.5, 0.0020 * m)
    if (det_name or "").strip().upper() == "SIFT":
        base = max(base, 7.5)
    return float(min(22.0, base))


def _ransac_refine_translation_displacements(
    d: np.ndarray,
    thr: float,
    max_refine: int = 5,
    max_hypotheses: int = 800,
) -> Tuple[np.ndarray, int]:
    """
    位移样本 d_i ≈ 常向量 t（warp 约定下 pts_cur - pts_ref）。
    随机假设 t=d_j，数 inlier，取最优后再中位数迭代收紧。
    """
    n = int(d.shape[0])
    if n < 4:
        t = np.median(d, axis=0) if n > 0 else np.zeros(2, dtype=np.float64)
        err = np.linalg.norm(d - t, axis=1) if n > 0 else np.array([])
        return t, int(np.count_nonzero(err < thr))
    rng = np.random.default_rng(12345 + n)
    n_hyp = int(min(max_hypotheses, max(60, 25 * n)))
    best_cnt = -1
    best_t = np.median(d, axis=0).copy()
    for _ in range(n_hyp):
        j = int(rng.integers(0, n))
        t0 = d[j].astype(np.float64)
        err = np.linalg.norm(d - t0, axis=1)
        cnt = int(np.count_nonzero(err < thr))
        if cnt > best_cnt:
            best_cnt = cnt
            best_t = t0.copy()
    t = best_t.copy()
    for _ in range(int(max_refine)):
        err = np.linalg.norm(d - t, axis=1)
        inl = err < thr
        n_inl = int(np.count_nonzero(inl))
        if n_inl < 3:
            break
        t_new = np.median(d[inl], axis=0)
        if float(np.max(np.abs(t_new - t))) < 1e-6:
            t = t_new
            break
        t = t_new
    err = np.linalg.norm(d - t, axis=1)
    return t.astype(np.float64), int(np.count_nonzero(err < thr))


def _estimate_translation_from_roi_matches(
    pts_cur: np.ndarray,
    pts_ref: np.ndarray,
    reproj_thresh: float,
    max_refine: int = 5,
) -> Tuple[np.ndarray, int]:
    """
    ROI 内匹配点 → 纯平移 2×3（无旋转）。先迭代中位数；内点不足时用 RANSAC 式假设精化。

    平移分量与 ``cv2.warpAffine(..., WARP_INVERSE_MAP)``、``findTransformECC`` 一致：
    ``dst(x,y)=src(x+t_x,y+t_y)``，故对同一物理点应取 ``t ≈ median(pts_cur - pts_ref)``。
    """
    if pts_cur.shape[0] < 4 or pts_ref.shape[0] < 4:
        return np.eye(2, 3, dtype=np.float32), 0
    d = (pts_cur.astype(np.float64) - pts_ref.astype(np.float64)).reshape(-1, 2)
    thr = float(reproj_thresh)
    t = np.median(d, axis=0)
    for _ in range(int(max_refine)):
        err = np.linalg.norm(d - t, axis=1)
        inl = err < thr
        n_inl = int(np.count_nonzero(inl))
        if n_inl < 4:
            break
        t_new = np.median(d[inl], axis=0)
        if float(np.max(np.abs(t_new - t))) < 1e-6:
            t = t_new
            break
        t = t_new
    err = np.linalg.norm(d - t, axis=1)
    n_in = int(np.count_nonzero(err < thr))
    if n_in < 4 and d.shape[0] >= 4:
        t2, n2 = _ransac_refine_translation_displacements(
            d, thr, max_refine=max_refine
        )
        if n2 > n_in:
            t, n_in = t2, n2
    if n_in < 4 and d.shape[0] >= 4 and thr < 20.0:
        thr_loose = min(20.0, thr * 1.85)
        t2, n2 = _ransac_refine_translation_displacements(
            d, thr_loose, max_refine=max_refine
        )
        if n2 > n_in:
            t, n_in = t2, n2
            thr = thr_loose
    M = np.array(
        [[1.0, 0.0, float(t[0])], [0.0, 1.0, float(t[1])]], dtype=np.float32
    )
    err = np.linalg.norm(d - t, axis=1)
    n_in = int(np.count_nonzero(err < thr))
    return M, n_in


def _estimate_euclidean_from_roi_matches(
    pts_cur: np.ndarray,
    pts_ref: np.ndarray,
    reproj_thresh: float,
) -> Tuple[np.ndarray, int]:
    """
    ROI 内匹配点 → 欧氏变换（旋转 + 平移）2×3。
    使用 cv2.estimateAffinePartial2D 估计相似变换（平移+旋转+尺度），
    再经 _similarity_to_euclidean_2x3 去尺度得到纯旋转+平移。

    estimateAffinePartial2D(src, dst) 返回 M 使得 dst ≈ M * src，
    与 warpAffine(..., WARP_INVERSE_MAP) 方向一致：传入 M 即可将 cur 逆映射到 ref 坐标系。
    """
    if pts_cur.shape[0] < 4 or pts_ref.shape[0] < 4:
        return np.eye(2, 3, dtype=np.float32), 0
    thr = float(reproj_thresh)
    try:
        M_raw, inliers = cv2.estimateAffinePartial2D(
            pts_ref,
            pts_cur,
            method=cv2.RANSAC,
            ransacReprojThreshold=thr,
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), 0
    if M_raw is None:
        return np.eye(2, 3, dtype=np.float32), 0
    n_in = int(np.count_nonzero(inliers)) if inliers is not None else 0
    if n_in < 4:
        return np.eye(2, 3, dtype=np.float32), 0
    A_eucl = _similarity_to_euclidean_2x3(M_raw)
    return A_eucl, n_in


def _roi_pair_estimate_euclidean(
    det_name: str,
    kp_ref: Any,
    d_ref: Any,
    kp_cur: Any,
    d_cur: Any,
    max_pair_disp: float,
    reproj_thr: float,
) -> Tuple[np.ndarray, int, int, int]:
    """
    一对「参考图 / 当前图」在已有 mask 上提好的关键点与描述子，估计整图欧氏变换（旋转+平移）2×3。
    返回 (A_try, n_in, n_match, n_kept_disp_gate)。
    """
    A_id = np.eye(2, 3, dtype=np.float32)
    if (
        kp_ref is None
        or kp_cur is None
        or d_ref is None
        or d_cur is None
        or len(kp_ref) < 4
        or len(kp_cur) < 4
    ):
        return A_id, 0, 0, 0
    matches = _match_feature_descriptors(det_name, d_ref, d_cur)
    n_match = len(matches)
    if n_match < 4:
        return A_id, 0, n_match, 0
    pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in matches])
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
    disp = np.linalg.norm(pts_ref - pts_cur, axis=1)
    keep = disp <= max_pair_disp
    n_kept = int(np.count_nonzero(keep))
    if n_kept >= 4:
        pts_cur = pts_cur[keep]
        pts_ref = pts_ref[keep]
    A_try, n_in = _estimate_euclidean_from_roi_matches(
        pts_cur, pts_ref, reproj_thr
    )
    return A_try, n_in, n_match, n_kept


def _similarity_to_euclidean_2x3(A: np.ndarray) -> np.ndarray:
    """将相似/仿射子矩阵压成真旋转（去掉尺度），保留平移。"""
    R = A[:2, :2].astype(np.float64)
    t = A[:2, 2].astype(np.float64).copy()
    U, _, Vt = np.linalg.svd(R)
    Re = U @ Vt
    if np.linalg.det(Re) < 0:
        U2 = U.copy()
        U2[:, 1] *= -1.0
        Re = U2 @ Vt
    out = np.zeros((2, 3), dtype=np.float32)
    out[:2, :2] = Re.astype(np.float32)
    out[:2, 2] = t.astype(np.float32)
    return out


def _clamp_affine_euclidean(
    A: np.ndarray, w: int, h: int, max_deg: float, max_t_frac: float
) -> None:
    md = math.radians(max_deg)
    theta = math.atan2(float(A[1, 0]), float(A[0, 0]))
    theta = float(np.clip(theta, -md, md))
    c, s = math.cos(theta), math.sin(theta)
    A[0, 0], A[0, 1] = float(c), float(-s)
    A[1, 0], A[1, 1] = float(s), float(c)
    lim = float(max_t_frac * min(h, w))
    A[0, 2] = float(np.clip(float(A[0, 2]), -lim, lim))
    A[1, 2] = float(np.clip(float(A[1, 2]), -lim, lim))


def _align_burst_roi_euclidean_features(
    proc: List[np.ndarray],
    roi_norm: Tuple[float, float, float, float],
    feature_name: str,
    on_frame: Optional[Callable[[int, int], None]] = None,
    align_debug: Optional[List[Dict[str, Any]]] = None,
) -> List[np.ndarray]:
    """
    首张 ROI 内提点作全局锚；后续帧在 ROI 外扩区提点。
    j≥2 时链式对齐上一已对齐帧并与首张结果择优；若链式单步变换过大则改首张（抑错配顶夹）。
    每 6 帧强制与首张重锚一次以抑制链式累积漂移。输出为整幅欧氏变换 warp + BORDER_CONSTANT 留白。
    支持旋转（±5°）+ 平移估计。
    """
    if len(proc) < 2:
        return proc
    h0, w0 = proc[0].shape[:2]
    bx0, by0, bx1, by1 = _roi_norm_rect_to_pixel_box(w0, h0, roi_norm)
    mask0 = np.zeros((h0, w0), dtype=np.uint8)
    mask0[by0:by1, bx0:bx1] = 255
    _roi_align_max_t_frac = 0.14
    _roi_align_max_deg = 5.0
    lim_pix = int(math.ceil(_roi_align_max_t_frac * float(min(h0, w0))))
    pad_search = max(40, lim_pix + 28)

    det_name, detector = _make_feature_detector(feature_name)
    reproj_thr = _roi_align_reproj_thresh(w0, h0, det_name)
    ref_gray = cv2.cvtColor(proc[0], cv2.COLOR_BGR2GRAY)
    kp0, d0 = detector.detectAndCompute(ref_gray, mask0)
    if align_debug is not None:
        align_debug.append(
            {
                "i": 0,
                "mode": "roi_euclid",
                "det": det_name,
                "roi_px": [bx0, by0, bx1, by1],
                "nkp0": len(kp0) if kp0 else 0,
                "search_pad": int(pad_search),
            }
        )

    aligned: List[np.ndarray] = [proc[0]]
    border_val = (245, 245, 245)

    roi_w = max(1, bx1 - bx0)
    roi_h = max(1, by1 - by0)
    max_pair_disp = float(2 * pad_search + max(roi_w, roi_h))
    lim_t_abs = float(_roi_align_max_t_frac * float(min(h0, w0)))
    roi_reanchor_period = 6
    chain_spike_lim = max(130.0, 0.036 * float(min(h0, w0)))
    chain_rot_spike_deg = 3.0
    _burst_align_log(
        f"ROI 欧氏模式 图幅={w0}x{h0} det={det_name} roi_px=[{bx0},{by0},{bx1},{by1}] "
        f"search_pad={pad_search} max_pair_disp={max_pair_disp:.1f} "
        f"reproj_thr={reproj_thr:.1f}px clamp|t|<={lim_t_abs:.1f}px "
        f"clamp|θ|<={_roi_align_max_deg:.1f}° "
        f"n_kp_ref={len(kp0) if kp0 else 0} 总帧={len(proc)} "
        f"j≥2链式+首张择优; 抑链跳>{chain_spike_lim:.0f}px或>{chain_rot_spike_deg:.1f}°→首张; "
        f"每{roi_reanchor_period}帧重锚首张"
    )

    for j in range(1, len(proc)):
        gray = cv2.cvtColor(proc[j], cv2.COLOR_BGR2GRAY)
        sx0, sy0, sx1, sy1 = _inflate_pixel_rect_xyxy(
            bx0, by0, bx1, by1, w0, h0, pad_search, pad_search
        )
        mask_cur = np.zeros((h0, w0), dtype=np.uint8)
        mask_cur[sy0:sy1, sx0:sx1] = 255
        kp1, d1 = detector.detectAndCompute(gray, mask_cur)
        A_eucl = np.eye(2, 3, dtype=np.float32)
        n_in = 0
        n_match = 0
        n_kept = 0
        tx_est = ty_est = 0.0
        theta_est_deg = 0.0
        log_note = ""
        ref_src = ""
        if (
            kp0 is None
            or kp1 is None
            or d0 is None
            or d1 is None
            or len(kp0) < 4
            or len(kp1) < 4
        ):
            log_note = (
                f"跳过对齐: 首张或当前 kp 不足 "
                f"(k0={0 if kp0 is None else len(kp0)} k1={0 if kp1 is None else len(kp1)})"
            )
        elif j == 1:
            A_try, n_in, n_match, n_kept = _roi_pair_estimate_euclidean(
                det_name, kp0, d0, kp1, d1, max_pair_disp, reproj_thr
            )
            ref_src = "首张"
            tx_est = float(A_try[0, 2])
            ty_est = float(A_try[1, 2])
            theta_est_deg = math.degrees(math.atan2(float(A_try[1, 0]), float(A_try[0, 0])))
            if n_in >= 4:
                A_eucl = A_try
                _clamp_affine_euclidean(
                    A_eucl, w0, h0, max_deg=_roi_align_max_deg, max_t_frac=_roi_align_max_t_frac
                )
                theta_app_deg = math.degrees(math.atan2(float(A_eucl[1, 0]), float(A_eucl[0, 0])))
                log_note = (
                    f"[{ref_src}] 估计 θ={theta_est_deg:.2f}° t=({tx_est:.3f},{ty_est:.3f}) n_in={n_in} "
                    f"→ 夹紧后 θ={theta_app_deg:.2f}° t=({float(A_eucl[0, 2]):.3f},{float(A_eucl[1, 2]):.3f})"
                )
            else:
                A_eucl = np.eye(2, 3, dtype=np.float32)
                n_in = 0
                log_note = (
                    f"[{ref_src}] 跳过对齐: 内点不足 n_in=0 "
                    f"(估计曾 θ={theta_est_deg:.2f}° t=({tx_est:.3f},{ty_est:.3f}))"
                )
        else:
            A_f, n_f, nm_f, nk_f = _roi_pair_estimate_euclidean(
                det_name, kp0, d0, kp1, d1, max_pair_disp, reproj_thr
            )
            if j % roi_reanchor_period == 0:
                A_try, n_in, n_match, n_kept = A_f, n_f, nm_f, nk_f
                ref_src = f"首张(每{roi_reanchor_period}帧重锚)"
                if n_in < 4:
                    ref_gray_p = cv2.cvtColor(aligned[j - 1], cv2.COLOR_BGR2GRAY)
                    kp_p, d_p = detector.detectAndCompute(ref_gray_p, mask0)
                    A_ch, n_ch, nm_ch, nk_ch = _roi_pair_estimate_euclidean(
                        det_name, kp_p, d_p, kp1, d1, max_pair_disp, reproj_thr
                    )
                    if n_ch > n_in:
                        A_try, n_in, n_match, n_kept = A_ch, n_ch, nm_ch, nk_ch
                        ref_src = "链式(重锚首张不足回退)"
            else:
                ref_gray_p = cv2.cvtColor(aligned[j - 1], cv2.COLOR_BGR2GRAY)
                kp_p, d_p = detector.detectAndCompute(ref_gray_p, mask0)
                A_ch, n_ch, nm_ch, nk_ch = _roi_pair_estimate_euclidean(
                    det_name, kp_p, d_p, kp1, d1, max_pair_disp, reproj_thr
                )
                mag_ch = math.hypot(float(A_ch[0, 2]), float(A_ch[1, 2]))
                mag_f = math.hypot(float(A_f[0, 2]), float(A_f[1, 2]))
                theta_ch_deg = math.degrees(math.atan2(float(A_ch[1, 0]), float(A_ch[0, 0])))
                theta_f_deg = math.degrees(math.atan2(float(A_f[1, 0]), float(A_f[0, 0])))
                lim_warn = 0.082 * float(min(h0, w0))
                rot_spike = abs(theta_ch_deg) > chain_rot_spike_deg and abs(theta_ch_deg - theta_f_deg) > 1.5
                heavy_spike = mag_ch > chain_spike_lim and (
                    mag_ch > mag_f * 2.0 + 45.0
                    or mag_ch - mag_f > 95.0
                    or mag_ch > lim_warn
                )
                chain_spike = (
                    n_ch >= 4
                    and n_f >= 4
                    and (heavy_spike or rot_spike)
                    and not (n_ch > n_f + 28)
                )
                if chain_spike:
                    A_try, n_in, n_match, n_kept = A_f, n_f, nm_f, nk_f
                    spike_reason = "平移" if heavy_spike else "旋转"
                    ref_src = (
                        f"首张(抑链{spike_reason}跳‖链‖={mag_ch:.0f}‖首‖={mag_f:.0f}"
                        f" θ链={theta_ch_deg:.1f}° θ首={theta_f_deg:.1f}° lim~{lim_warn:.0f})"
                    )
                else:
                    cand: List[Tuple[str, np.ndarray, int, int, int]] = [
                        ("链式j-1", A_ch, n_ch, nm_ch, nk_ch),
                        ("首张", A_f, n_f, nm_f, nk_f),
                    ]
                    ref_src, A_try, n_in, n_match, n_kept = max(
                        cand,
                        key=lambda c: (
                            c[2],
                            c[3],
                            1 if c[0].startswith("链") else 0,
                        ),
                    )
            tx_est = float(A_try[0, 2])
            ty_est = float(A_try[1, 2])
            theta_est_deg = math.degrees(math.atan2(float(A_try[1, 0]), float(A_try[0, 0])))
            if n_in >= 4:
                A_eucl = A_try
                _clamp_affine_euclidean(
                    A_eucl, w0, h0, max_deg=_roi_align_max_deg, max_t_frac=_roi_align_max_t_frac
                )
                theta_app_deg = math.degrees(math.atan2(float(A_eucl[1, 0]), float(A_eucl[0, 0])))
                log_note = (
                    f"[{ref_src}] 估计 θ={theta_est_deg:.2f}° t=({tx_est:.3f},{ty_est:.3f}) n_in={n_in} "
                    f"→ 夹紧后 θ={theta_app_deg:.2f}° t=({float(A_eucl[0, 2]):.3f},{float(A_eucl[1, 2]):.3f})"
                )
            else:
                A_eucl = np.eye(2, 3, dtype=np.float32)
                n_in = 0
                log_note = (
                    f"[{ref_src}] 跳过对齐: 内点不足 n_in=0 "
                    f"(估计曾 θ={theta_est_deg:.2f}° t=({tx_est:.3f},{ty_est:.3f}))"
                )
        _burst_align_log(
            f"ROI 欧氏帧 j={j}/{len(proc) - 1} search=[{sx0},{sy0},{sx1},{sy1}] "
            f"n_kp1={0 if kp1 is None else len(kp1)} n_match={n_match} n_kept_disp_gate={n_kept} "
            f"reproj_thr={reproj_thr:.1f}px {log_note}"
        )
        warped = cv2.warpAffine(
            proc[j],
            A_eucl,
            (w0, h0),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        aligned.append(warped)
        if align_debug is not None:
            theta_app_deg = math.degrees(math.atan2(float(A_eucl[1, 0]), float(A_eucl[0, 0])))
            align_debug.append(
                {
                    "i": j,
                    "mode": "roi_euclid",
                    "det": det_name,
                    "nin": n_in,
                    "nkp1": len(kp1) if kp1 else 0,
                    "tx": float(A_eucl[0, 2]),
                    "ty": float(A_eucl[1, 2]),
                    "theta_deg": theta_app_deg,
                    "search_px": [sx0, sy0, sx1, sy1],
                    "align_ref": ref_src or "-",
                }
            )
        if on_frame is not None:
            try:
                on_frame(j, len(proc) - 1)
            except Exception:
                pass

    return aligned


def _draw_dashed_rect_bgr(
    bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
    gap: int = 5,
) -> None:
    """在 BGR 图上画轴对齐虚线矩形（像素坐标，含边）。"""
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    h, w = bgr.shape[:2]
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    step = max(2, int(dash) + int(gap))

    def edge_h(y: int, xa: int, xb: int) -> None:
        if xa >= xb:
            return
        x = xa
        while x < xb:
            x2 = min(xb, x + dash)
            if x2 > x:
                cv2.line(bgr, (x, y), (x2 - 1, y), color, thickness, cv2.LINE_AA)
            x += step

    def edge_v(x: int, ya: int, yb: int) -> None:
        if ya >= yb:
            return
        y = ya
        while y < yb:
            y2 = min(yb, y + dash)
            if y2 > y:
                cv2.line(bgr, (x, y), (x, y2 - 1), color, thickness, cv2.LINE_AA)
            y += step

    edge_h(y0, x0, x1 + 1)
    edge_h(y1, x0, x1 + 1)
    edge_v(x0, y0, y1 + 1)
    edge_v(x1, y0, y1 + 1)


def _draw_burst_align_debug_overlay(
    bgr: np.ndarray,
    frame_i: int,
    opts: "BurstWebpBuildOptions",
    dbg: Optional[Dict[str, Any]],
) -> None:
    """
    测试用：在 BGR 帧上绘制对齐 ROI 虚线框与判据（仅 ASCII，避免字体问题）。
    """
    if bgr is None or bgr.size == 0:
        return
    h, w = bgr.shape[:2]
    if w < 8 or h < 8:
        return
    ovl = bgr
    if dbg is None:
        dbg = {}
    mode = str(dbg.get("mode", ""))
    yellow = (0, 220, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    mns = float(min(h, w))
    rect_th = max(4, int(round(0.007 * mns)))
    rect_th2 = max(3, rect_th - 1)
    dash_w = max(14, int(round(0.02 * mns)))
    gap_w = max(7, int(round(0.012 * mns)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(max(1.1, min(2.0, mns / 380.0)))
    font_th = max(2, int(round(0.55 * font_scale)))
    if opts.align_track_roi_norm is not None and mode != "ecc_ring":
        rx0, ry0, rx1, ry1 = _roi_norm_rect_to_pixel_box(
            w, h, opts.align_track_roi_norm
        )
        _draw_dashed_rect_bgr(
            ovl, rx0, ry0, rx1, ry1, yellow, rect_th, dash_w, gap_w
        )
        spx = dbg.get("search_px")
        if (
            frame_i > 0
            and isinstance(spx, (list, tuple))
            and len(spx) >= 4
        ):
            ex0, ey0, ex1, ey1 = (int(spx[0]), int(spx[1]), int(spx[2]), int(spx[3]))
            if ex1 > ex0 and ey1 > ey0:
                # search_px 记在「对齐前当前帧」像素系；叠画在「对齐后」全图上，
                # 纯平移 warp 下 cur 点 (cx,cy) 落在输出 (cx-tx, cy-ty)（与 warpAffine+INVERSE 一致）。
                txa = float(dbg.get("tx", 0) or 0.0)
                tya = float(dbg.get("ty", 0) or 0.0)
                ox0 = int(round(float(ex0) - txa))
                ox1 = int(round(float(ex1) - txa))
                oy0 = int(round(float(ey0) - tya))
                oy1 = int(round(float(ey1) - tya))
                ox0, ox1 = sorted((max(0, ox0), min(w - 1, ox1)))
                oy0, oy1 = sorted((max(0, oy0), min(h - 1, oy1)))
                if ox1 > ox0 and oy1 > oy0:
                    orange = (0, 140, 255)
                    _draw_dashed_rect_bgr(
                        ovl, ox0, oy0, ox1, oy1, orange, rect_th2, dash_w, gap_w
                    )
    lines: List[str] = []
    if dbg.get("err"):
        lines.append(f"F{frame_i} ERR={dbg.get('err')}")
    elif frame_i == 0 and mode == "ecc_ring":
        lines.append(
            f"F0 ECC work={dbg.get('rw', '?')}x{dbg.get('rh', '?')} "
            f"sc={dbg.get('scale', 0):.3f}"
        )
    elif mode == "ecc_ring":
        lines.append(
            f"F{frame_i} ECC tx={dbg.get('txf', 0):.1f} ty={dbg.get('tyf', 0):.1f}"
        )
    elif mode in ("roi_trans", "roi_rigid"):
        if frame_i == 0:
            spad = dbg.get("search_pad", "")
            pad_note = f" pad={spad}" if spad != "" else ""
            lines.append(
                f"F0 ROI {dbg.get('det', '?')} k0={dbg.get('nkp0', 0)} "
                f"px={dbg.get('roi_px', [])}{pad_note}"
            )
        else:
            lines.append(
                f"F{frame_i} {dbg.get('det', '?')} in={dbg.get('nin', 0)} "
                f"k1={dbg.get('nkp1', 0)}"
            )
            ar = str(dbg.get("align_ref", "") or "").strip()
            txty = f"tx={dbg.get('tx', 0):.1f} ty={dbg.get('ty', 0):.1f}"
            if ar and ar != "-":
                txty += f" ref={ar}"
            if dbg.get("search_px"):
                txty += " 橙=提特征外扩区(已换算)"
            lines.append(txty)
    elif frame_i == 0:
        lines.append("F0 align (no roi debug row)")
    y = max(28, int(round(0.032 * float(h))))
    line_gap = max(10, int(round(0.018 * float(h))))
    pad_x = max(6, int(round(0.01 * float(w))))
    for ln in lines[:5]:
        (tw, th), bl = cv2.getTextSize(ln, font, font_scale, font_th)
        cv2.rectangle(
            ovl,
            (pad_x, y - th - bl - 2),
            (pad_x + tw + 8, y + bl + 4),
            black,
            -1,
        )
        cv2.putText(
            ovl,
            ln,
            (pad_x + 4, y),
            font,
            font_scale,
            white,
            font_th,
            cv2.LINE_AA,
        )
        y += th + line_gap


def _align_burst_stack_ecc(
    proc: List[np.ndarray],
    max_work_edge: float = 1280.0,
    on_frame: Optional[Callable[[int, int], None]] = None,
    align_track_roi_norm: Optional[Tuple[float, float, float, float]] = None,
    align_feature_detector: str = "ORB",
    align_debug: Optional[List[Dict[str, Any]]] = None,
) -> List[np.ndarray]:
    """
    以 proc[0] 为参考对齐后续帧。
    - 人工 ROI：框内特征 + 欧氏变换（旋转+平移），整图逆 warp、常量边界留白。
    - 未设 ROI：边带掩膜 + 相位/ECC + 时序平滑。

    on_frame：每对齐完一帧调用 on_frame(当前索引 j, 总对齐全帧数 n-1)，便于导出/预览进度。
    """
    if len(proc) < 2:
        return proc
    if align_track_roi_norm is not None:
        return _align_burst_roi_euclidean_features(
            proc,
            align_track_roi_norm,
            align_feature_detector,
            on_frame=on_frame,
            align_debug=align_debug,
        )

    h0, w0 = proc[0].shape[:2]
    scale = min(1.0, float(max_work_edge) / float(max(w0, h0)))
    rw = max(1, int(round(w0 * scale)))
    rh = max(1, int(round(h0 * scale)))
    _burst_align_log(
        f"ECC 边带模式 全尺寸={w0}x{h0} work={rw}x{rh} scale={scale:.4f} 总帧={len(proc)}"
    )

    ref_s = cv2.resize(proc[0], (rw, rh), interpolation=cv2.INTER_AREA)
    ref_u8 = cv2.cvtColor(ref_s, cv2.COLOR_BGR2GRAY)
    ref_g = _ecc_gray_scene_align(ref_u8)
    mask = _ecc_border_ring_mask(rh, rw)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 160, 1e-7)
    raw_ms: List[np.ndarray] = []

    for j in range(1, len(proc)):
        cur_s = cv2.resize(proc[j], (rw, rh), interpolation=cv2.INTER_AREA)
        cur_u8 = cv2.cvtColor(cur_s, cv2.COLOR_BGR2GRAY)
        cg = _ecc_gray_scene_align(cur_u8)

        M_small = np.eye(2, 3, dtype=np.float32)
        ph_dx = ph_dy = 0.0
        try:
            try:
                shift, _resp = cv2.phaseCorrelate(ref_g, cg, mask)
            except TypeError:
                shift, _resp = cv2.phaseCorrelate(ref_g, cg)
            ph_dx, ph_dy = float(shift[0]), float(shift[1])
        except cv2.error:
            pass

        best = np.eye(2, 3, dtype=np.float32)
        Mt = M_small.copy()
        ecc_ok = False
        try:
            _, Mt = cv2.findTransformECC(
                ref_g,
                cg,
                Mt,
                cv2.MOTION_TRANSLATION,
                criteria,
                mask,
                7,
            )
            _clamp_align_translation(Mt, rw, rh, max_frac=0.15)
            best = Mt
            ecc_ok = True
        except cv2.error:
            pass

        M = best.copy()
        M[0, 2] /= scale
        M[1, 2] /= scale
        _clamp_align_translation(M, w0, h0, max_frac=0.15)
        raw_ms.append(M)
        _burst_align_log(
            f"ECC 帧 j={j}/{len(proc) - 1} 粗相关 phase_xy=({ph_dx:.3f},{ph_dy:.3f}) "
            f"work 上 ecc_tx_ty=({float(best[0, 2]):.3f},{float(best[1, 2]):.3f}) "
            f"ecc_ok={ecc_ok} → 全幅 raw_tx_ty=({float(M[0, 2]):.3f},{float(M[1, 2]):.3f})"
        )

    n1 = len(raw_ms)
    txs = np.empty(n1, dtype=np.float64)
    tys = np.empty(n1, dtype=np.float64)
    for i, M in enumerate(raw_ms):
        _, tx, ty = _euclidean_params_from_M(M)
        txs[i] = tx
        tys[i] = ty

    txs_raw = txs.astype(np.float64, copy=True)
    tys_raw = tys.astype(np.float64, copy=True)

    med_r = 1 if n1 >= 3 else 0
    txs = _median_smooth_1d(txs, med_r)
    tys = _median_smooth_1d(tys, med_r)
    txs_med = txs.astype(np.float64, copy=True)
    tys_med = tys.astype(np.float64, copy=True)
    lim = max(5.0, 0.014 * float(min(h0, w0)))
    txs = _velocity_limit_track(txs, lim)
    tys = _velocity_limit_track(tys, lim)
    txs_vel = txs.astype(np.float64, copy=True)
    tys_vel = tys.astype(np.float64, copy=True)
    smooth_r = 2 if n1 >= 4 else (1 if n1 >= 2 else 0)
    txs = _smooth_align_param_track(txs, smooth_r)
    tys = _smooth_align_param_track(tys, smooth_r)
    for ii in range(n1):
        jj = ii + 1
        _burst_align_log(
            f"ECC 帧 j={jj} 位移链 raw=({txs_raw[ii]:.3f},{tys_raw[ii]:.3f}) "
            f"median_r{med_r}=({txs_med[ii]:.3f},{tys_med[ii]:.3f}) "
            f"vel_lim={lim:.1f} → ({txs_vel[ii]:.3f},{tys_vel[ii]:.3f}) "
            f"smooth_r{smooth_r}=({txs[ii]:.3f},{tys[ii]:.3f})"
        )

    if align_debug is not None:
        align_debug.append(
            {
                "i": 0,
                "mode": "ecc_ring",
                "scale": float(scale),
                "rw": rw,
                "rh": rh,
            }
        )
        for jj in range(1, len(proc)):
            ii = jj - 1
            align_debug.append(
                {
                    "i": jj,
                    "mode": "ecc_ring",
                    "txf": float(txs[ii]),
                    "tyf": float(tys[ii]),
                }
            )

    aligned: List[np.ndarray] = [proc[0]]
    for j in range(1, len(proc)):
        i = j - 1
        M = _M_from_euclidean_params(0.0, float(txs[i]), float(tys[i]))
        tx_b = float(M[0, 2])
        ty_b = float(M[1, 2])
        _clamp_align_translation(M, w0, h0, max_frac=0.15)
        _burst_align_log(
            f"ECC 帧 j={j} 最终 warp: smooth 输入=({tx_b:.3f},{ty_b:.3f}) "
            f"夹紧后 applied=({float(M[0, 2]):.3f},{float(M[1, 2]):.3f})"
        )
        warped = cv2.warpAffine(
            proc[j],
            M,
            (w0, h0),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned.append(warped)
        if on_frame is not None:
            try:
                on_frame(j, len(proc) - 1)
            except Exception:
                pass

    return aligned


@dataclass
class BurstWebpBuildOptions:
    enable_white_balance: bool = True
    enable_auto_exposure: bool = True
    auto_exposure_strength: float = 1.0
    # 定点（三脚架）或跟踪：仅影响对话框自动找标定点；导出按 frame_layouts 裁剪。
    mode: str = "fixed"
    fps: float = 2.0
    # 每帧标定点 + 裁剪区（与输入 paths 等长）；可为 FrameLayout 或 dict。
    frame_layouts: Optional[List[Any]] = None
    max_long_edge: int = 1600
    webp_quality: int = 85
    # 与批量水印一致：在缩放后的每帧上叠加水印（不重复自动显影）。
    # 连拍 ≥2 时元数据路径固定为首张（见 _bgr_frames_to_pil_with_optional_watermark）。
    watermark_options: Optional[WatermarkOptions] = None
    watermark_source_folder: str = ""
    prefer_folder_name_as_species: bool = True
    # 叠水印时物种/左侧主题文案；非空则优先于「从水印源文件夹路径推断文件夹名」。
    watermark_species_or_theme: str = ""


def _fps_and_frame_duration_ms(opts: BurstWebpBuildOptions) -> Tuple[float, float]:
    fps = float(np.clip(float(opts.fps if opts.fps is not None else 2.0), 0.1, 60.0))
    return fps, 1000.0 / fps


def _coerce_frame_layouts(raw: Optional[List[Any]], n: int) -> List[FrameLayout]:
    if not raw:
        raise ValueError("请先在首图上设置标定点与裁剪区")
    out: List[FrameLayout] = []
    last: Optional[FrameLayout] = None
    for i in range(n):
        item = raw[i] if i < len(raw) else last
        if item is None:
            raise ValueError(f"第 {i + 1} 帧尚未设置标定点与裁剪区")
        if isinstance(item, FrameLayout):
            lay = item
        elif isinstance(item, dict):
            lay = FrameLayout.from_dict(item)
        else:
            raise ValueError("frame_layouts 格式无效")
        out.append(lay)
        last = lay
    if not layout_valid(out[0]):
        raise ValueError("请先在首图上设置标定点与裁剪区")
    return out


def _bgr_frames_to_pil_with_optional_watermark(
    frames_bgr: List[np.ndarray],
    paths_per_frame: List[str],
    opts: BurstWebpBuildOptions,
    after_each_frame: Optional[Callable[[int, int, str], None]] = None,
) -> List[Image.Image]:
    wopt = opts.watermark_options
    ws = (opts.watermark_source_folder or "").strip()
    nfr = len(frames_bgr)
    # 连拍 ≥2：水印文字/EXIF（日期、相机、GPS、物种路径等）一律按首张，减少逐张读 EXIF 与布局差异。
    wm_meta_path = (
        paths_per_frame[0]
        if len(paths_per_frame) >= 2
        else ""
    )
    out: List[Image.Image] = []
    for i, x in enumerate(frames_bgr):
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if wopt is not None and ws and i < len(paths_per_frame):
            path_for_wm = wm_meta_path or paths_per_frame[i]
            wm = render_watermark_on_pil_image(
                pil,
                path_for_wm,
                ws,
                wopt,
                opts.prefer_folder_name_as_species,
                species_or_theme_override=opts.watermark_species_or_theme,
            )
            if wm is not None:
                pil = wm
        out.append(pil)
        if after_each_frame is not None:
            try:
                after_each_frame(
                    i + 1,
                    nfr,
                    (
                        f"叠水印（首张元数据）第 {i + 1}/{nfr} 帧"
                        if wm_meta_path
                        else f"叠水印 / 合成预览图 第 {i + 1}/{nfr} 帧"
                    ),
                )
            except Exception:
                pass
    return out


def _burst_prepare_animation_pil_frames(
    paths: List[str],
    opts: BurstWebpBuildOptions,
    progress: ProgressCb,
    log_terminal: bool,
    log_tag: str,
) -> Tuple[List[Image.Image], List[str], dict]:
    """
    与写出 WebP/MP4 共用的处理链，返回 RGB PIL 帧列表、排序后的路径及元数据。
    meta 含 progress_total（最后一步留给调用方写文件）。
    """
    paths = [p for p in paths if p and os.path.isfile(p)]
    if len(paths) < 2:
        raise ValueError("至少需要 2 张图片")

    n = len(paths)
    lays = _coerce_frame_layouts(opts.frame_layouts, n)
    fps, frame_dur_ms = _fps_and_frame_duration_ms(opts)
    n_wb = 1 if opts.enable_white_balance else 0
    # 加载 n + 白平衡(可选) + 裁剪/曝光 n + 缩放 1 + 水印 n + 写出 1
    tot_pg = n + n_wb + n + 1 + n + 1
    cur_pg = 0

    def pg(msg: str) -> None:
        nonlocal cur_pg
        cur_pg += 1
        c = min(cur_pg, tot_pg)
        _emit_export_progress(progress, log_terminal, c, tot_pg, msg, log_tag)

    _emit_export_progress(
        progress,
        log_terminal,
        0,
        tot_pg,
        f"开始：{n} 帧，{fps:g} 张/秒，约 {tot_pg} 步（终端可跟踪进度）",
        log_tag,
    )

    raw_bgr: List[np.ndarray] = []
    for i, p in enumerate(paths):
        im = imread_bgr(p, raw_half_size=True)
        if im is None:
            raise RuntimeError(f"无法读取：{p}")
        raw_bgr.append(im)
        pg(f"加载 {i + 1}/{n}: {os.path.basename(p)}")

    if opts.enable_white_balance:
        pg("连拍：首张统计白平衡，后续帧沿用…")
        proc = _wb_burst_pipeline(raw_bgr, opts)
    else:
        proc = raw_bgr

    h0, w0 = proc[0].shape[:2]
    geom = geom_from_first(lays[0], w0, h0)

    cropped: List[np.ndarray] = []
    for i, (fr, lay) in enumerate(zip(proc, lays)):
        cropped.append(_crop_frame_maybe_ae(fr, lay, geom, opts))
        if opts.enable_auto_exposure:
            pg(f"裁剪并自动曝光 {i + 1}/{n}")
        else:
            pg(f"按布局裁剪 {i + 1}/{n}")
    proc = cropped

    if opts.max_long_edge > 0:
        pg(f"缩放到最长边 {opts.max_long_edge}px …")
        proc = [_resize_long_edge(x, opts.max_long_edge) for x in proc]
    else:
        pg("导出尺寸：不缩放（原分辨率）")

    h1, w1 = proc[0].shape[:2]
    for j in range(1, len(proc)):
        hj, wj = proc[j].shape[:2]
        if hj != h1 or wj != w1:
            proc[j] = cv2.resize(proc[j], (w1, h1), interpolation=cv2.INTER_AREA)

    def _wm_pg(fi: int, fn: int, msg: str) -> None:
        pg(msg)

    frames_rgb = _bgr_frames_to_pil_with_optional_watermark(
        proc, paths, opts, after_each_frame=_wm_pg
    )

    note = f"{fps:g} 张/秒"
    meta = {
        "interval_ms": float(frame_dur_ms),
        "interval_note": note,
        "frame_duration_ms": float(frame_dur_ms),
        "fps": float(fps),
        "n_frames": len(frames_rgb),
        "progress_total": tot_pg,
        "last_progress_cur": cur_pg,
        "mode": str(opts.mode or "fixed"),
    }
    return frames_rgb, paths, meta


def build_animated_webp(
    paths: List[str],
    out_path: str,
    opts: BurstWebpBuildOptions,
    progress: ProgressCb = None,
    log_terminal: bool = True,
) -> dict:
    """
    读取 paths，内部按拍摄时间排序，处理后写出动画 WebP。

    Returns:
        interval_ms, interval_note, frame_duration_ms, n_frames, out_path, format
    """
    log_tag = "[Birdy WebP导出]"
    frames_rgb, _paths_used, meta = _burst_prepare_animation_pil_frames(
        paths, opts, progress, log_terminal, log_tag
    )
    tot_pg = int(meta["progress_total"])
    dur = int(round(float(meta["frame_duration_ms"])))

    _parent = os.path.dirname(os.path.abspath(out_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)

    last_cur = int(meta.get("last_progress_cur", tot_pg - 1))
    write_cur = min(last_cur + 1, tot_pg)
    _emit_export_progress(
        progress,
        log_terminal,
        write_cur,
        tot_pg,
        f"写入 WebP：{os.path.basename(out_path)} …",
        log_tag,
    )
    try:
        frames_rgb[0].save(
            out_path,
            format="WEBP",
            save_all=True,
            append_images=frames_rgb[1:],
            duration=dur,
            loop=0,
            quality=int(np.clip(opts.webp_quality, 30, 100)),
            method=6,
        )
    except Exception as e:
        raise RuntimeError(
            f"写出动画 WebP 失败（请确认已安装带 libwebp 的 Pillow）：{e}"
        ) from e

    _emit_export_progress(
        progress,
        log_terminal,
        tot_pg,
        tot_pg,
        f"完成 → {out_path}",
        log_tag,
    )

    return {
        "interval_ms": meta["interval_ms"],
        "interval_note": meta["interval_note"],
        "frame_duration_ms": float(dur),
        "fps": float(meta.get("fps", 1000.0 / max(dur, 1))),
        "n_frames": len(frames_rgb),
        "out_path": out_path,
        "format": "webp",
        "mode": meta.get("mode", "fixed"),
    }


def build_animated_mp4(
    paths: List[str],
    out_path: str,
    opts: BurstWebpBuildOptions,
    progress: ProgressCb = None,
    log_terminal: bool = True,
) -> dict:
    """
    与 WebP 相同处理链，写出 MP4（MPEG-4 Part 2，fourcc mp4v），便于不支持动图 WebP 的客户端播放。

    帧率由「每秒几张」决定，限制在 0.25～120 fps。
    """
    log_tag = "[Birdy 视频导出]"
    frames_rgb, _paths_used, meta = _burst_prepare_animation_pil_frames(
        paths, opts, progress, log_terminal, log_tag
    )
    tot_pg = int(meta["progress_total"])
    frame_dur_ms = float(meta["frame_duration_ms"])
    fps = 1000.0 / max(frame_dur_ms, 1e-3)
    fps = float(np.clip(fps, 0.25, 120.0))

    _parent = os.path.dirname(os.path.abspath(out_path))
    if _parent:
        os.makedirs(_parent, exist_ok=True)

    def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
        rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    first = _pil_to_bgr(frames_rgb[0])
    h0, w0 = first.shape[:2]
    h0 -= h0 % 2
    w0 -= w0 % 2
    if h0 < 2 or w0 < 2:
        raise RuntimeError("视频导出：首帧有效分辨率过小。")
    first = first[:h0, :w0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w0, h0))
    if not writer.isOpened():
        raise RuntimeError(
            "无法创建 MP4 写入器（请确认 OpenCV 已编译 ffmpeg 后端，且扩展名为 .mp4、路径可写）。"
            f"文件：{out_path}"
        )

    last_cur = int(meta.get("last_progress_cur", tot_pg - 1))
    write_cur = min(last_cur + 1, tot_pg)
    _emit_export_progress(
        progress,
        log_terminal,
        write_cur,
        tot_pg,
        f"写入 MP4（{fps:.2f} fps）…",
        log_tag,
    )
    try:
        writer.write(first)
        for pil in frames_rgb[1:]:
            bgr = _pil_to_bgr(pil)
            if bgr.shape[1] != w0 or bgr.shape[0] != h0:
                bgr = cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_AREA)
            bgr = bgr[:h0, :w0]
            writer.write(bgr)
    finally:
        writer.release()

    _emit_export_progress(
        progress,
        log_terminal,
        tot_pg,
        tot_pg,
        f"完成 → {out_path}",
        log_tag,
    )

    return {
        "interval_ms": meta["interval_ms"],
        "interval_note": meta["interval_note"],
        "frame_duration_ms": frame_dur_ms,
        "n_frames": len(frames_rgb),
        "out_path": out_path,
        "format": "mp4",
        "fps": fps,
        "mode": meta.get("mode", "fixed"),
    }


def build_preview_frames_rgb(
    paths: List[str],
    opts: BurstWebpBuildOptions,
    max_long_edge: int = 720,
    max_frames: int = 24,
    progress: ProgressCb = None,
    log_terminal: bool = True,
) -> Tuple[List[Image.Image], float, str, Optional[np.ndarray]]:
    """
    用于 UI 快速预览：限制帧数与边长，不写盘。
    返回 (RGB PIL 列表, frame_duration_ms, interval_note, 首张裁剪前的 BGR)。

    progress(cur, total, msg)：阶段性进度（用于界面与 ETA）。
    log_terminal：是否在终端打印与 progress 同步的简要日志。
    """
    paths = [p for p in paths if p and os.path.isfile(p)]
    fps, frame_dur_ms = _fps_and_frame_duration_ms(opts)
    note = f"{fps:g} 张/秒"
    if not paths:
        return [], frame_dur_ms, note, None
    take = paths[: max(1, min(len(paths), max_frames))]
    lays_all = _coerce_frame_layouts(opts.frame_layouts, len(paths))
    lays = lays_all[: len(take)]

    o2 = BurstWebpBuildOptions(
        enable_white_balance=opts.enable_white_balance,
        enable_auto_exposure=opts.enable_auto_exposure,
        auto_exposure_strength=opts.auto_exposure_strength,
        mode=str(opts.mode or "fixed"),
        fps=fps,
        frame_layouts=lays,
        max_long_edge=max_long_edge,
        webp_quality=opts.webp_quality,
        watermark_options=opts.watermark_options,
        watermark_source_folder=opts.watermark_source_folder,
        prefer_folder_name_as_species=opts.prefer_folder_name_as_species,
        watermark_species_or_theme=opts.watermark_species_or_theme,
    )

    nt = len(take)
    raw_bgr: List[np.ndarray] = []
    loaded_paths: List[str] = []
    loaded_lays: List[FrameLayout] = []
    for i, p in enumerate(take):
        if log_terminal:
            print(
                f"[Birdy 动图预览] 解码 {i + 1}/{nt}: {os.path.basename(p)}",
                flush=True,
            )
        im = imread_bgr(p, raw_half_size=True)
        if im is None:
            continue
        raw_bgr.append(im)
        loaded_paths.append(p)
        loaded_lays.append(lays[i])
    if not raw_bgr:
        return [], frame_dur_ms, note, None

    n = len(loaded_paths)
    tot_steps = 1 + 1 + n + 1 + n
    cur_step = 0

    def _emit(cur: int, tot: int, msg: str) -> None:
        if progress:
            try:
                progress(cur, tot, msg)
            except Exception:
                pass
        if log_terminal:
            print(f"[Birdy 动图预览] {cur}/{tot} {msg}", flush=True)

    def tick(msg: str) -> None:
        nonlocal cur_step
        cur_step += 1
        c = min(cur_step, tot_steps)
        _emit(c, tot_steps, msg)

    _emit(0, tot_steps, f"已载入 {n} 帧；后续约 {tot_steps} 步。")

    if o2.enable_white_balance:
        tick("连拍：首张统计白平衡，其余帧沿用…")
        proc = _wb_burst_pipeline(raw_bgr, o2)
    else:
        proc = raw_bgr

    ref0 = proc[0].copy()
    h0, w0 = proc[0].shape[:2]
    geom = geom_from_first(loaded_lays[0], w0, h0)
    cropped: List[np.ndarray] = []
    for i, (fr, lay) in enumerate(zip(proc, loaded_lays)):
        cropped.append(_crop_frame_maybe_ae(fr, lay, geom, o2))
        if o2.enable_auto_exposure:
            tick(f"裁剪并自动曝光 {i + 1}/{n}…")
        else:
            tick(f"按布局裁剪 {i + 1}/{n}…")
    proc = cropped

    tick(f"缩放到预览最长边 {max_long_edge}px …")
    proc = [_resize_long_edge(x, max_long_edge) for x in proc]
    h1, w1 = proc[0].shape[:2]
    for j in range(1, len(proc)):
        if proc[j].shape[0] != h1 or proc[j].shape[1] != w1:
            proc[j] = cv2.resize(proc[j], (w1, h1), interpolation=cv2.INTER_AREA)

    def _frame_cb(fi: int, fn: int, msg: str) -> None:
        tick(msg)

    out_pil = _bgr_frames_to_pil_with_optional_watermark(
        proc, loaded_paths, o2, after_each_frame=_frame_cb
    )
    _emit(tot_steps, tot_steps, "预览处理完成")
    return out_pil, frame_dur_ms, note, ref0
