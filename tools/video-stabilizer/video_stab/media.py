# -*- coding: utf-8 -*-
"""视频枚举、抽帧与相对框换算。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v"}
BBox = Tuple[int, int, int, int]
NormBox = Tuple[float, float, float, float]


def list_videos(folder: str, recursive: bool = False) -> List[str]:
    root = Path(folder)
    if not root.is_dir():
        return []
    it = root.rglob("*") if recursive else root.iterdir()
    files = [
        str(p.resolve())
        for p in it
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    files.sort(key=str.lower)
    return files


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(n) / float(fps)


def read_frame_at(path: str, time_sec: float = 0.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    if time_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def norm_to_pixels(norm: Sequence[float], width: int, height: int) -> BBox:
    x, y, w, h = (float(v) for v in norm)
    px = int(round(x * width))
    py = int(round(y * height))
    pw = int(round(w * width))
    ph = int(round(h * height))
    px = max(0, min(px, max(0, width - 2)))
    py = max(0, min(py, max(0, height - 2)))
    pw = max(2, min(pw, width - px))
    ph = max(2, min(ph, height - py))
    return px, py, pw, ph


def fit_even_rect(norm: Sequence[float], width: int, height: int) -> BBox:
    """把相对框换成不超出画面的偶数宽高，便于写视频。"""
    x, y, w, h = norm_to_pixels(norm, width, height)
    w = max(64, w - (w % 2))
    h = max(64, h - (h % 2))
    w = min(w, width - (width % 2))
    h = min(h, height - (height % 2))
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    return x, y, w, h


def window_origin(
    focus_x: float,
    focus_y: float,
    out_w: int,
    out_h: int,
    width: int,
    height: int,
    rel_x: float = 0.5,
    rel_y: float = 0.5,
) -> Tuple[int, int]:
    """让焦点落在输出画面的相对位置，并把窗口限制在原片内部。"""
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))
    crop_x = int(round(focus_x - rel_x * out_w))
    crop_y = int(round(focus_y - rel_y * out_h))
    crop_x = max(0, min(crop_x, max(0, width - out_w)))
    crop_y = max(0, min(crop_y, max(0, height - out_h)))
    return crop_x, crop_y


def crop_video_rect(
    input_path: str,
    output_path: str,
    rect: BBox,
    *,
    log=None,
) -> Tuple[int, int]:
    """按固定矩形裁出一段无黑边视频。"""
    import cv2

    x, y, w, h = rect
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y : y + h, x : x + w]
        if crop.shape[1] != w or crop.shape[0] != h:
            crop = cv2.resize(crop, (w, h))
        writer.write(crop)
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        raise IOError("裁切后没有帧")
    if log:
        log(f"画面范围裁切: {w}x{h} @ ({x},{y})，{n} 帧")
    return w, h


def parse_norm_box(text: str) -> Optional[NormBox]:
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("框应为 x,y,w,h（相对画面 0–1）")
    vals = tuple(float(p) for p in parts)
    if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
        raise ValueError("相对框的 x,y,w,h 需在 0–1，且宽高大于 0")
    if vals[0] + vals[2] > 1.001 or vals[1] + vals[3] > 1.001:
        raise ValueError("相对框超出画面")
    return vals  # type: ignore[return-value]
