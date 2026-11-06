# -*- coding: utf-8 -*-
"""追踪稳定器：目标（鸟）保持在画面中心。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

ProgressFn = Optional[Callable[[int, int], None]]
LogFn = Optional[Callable[[str], None]]


def _log(msg: str, log: LogFn) -> None:
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def _interp_nan(values: np.ndarray) -> np.ndarray:
    idx = np.arange(len(values))
    good = np.isfinite(values)
    if not np.any(good):
        return np.zeros_like(values)
    if int(np.count_nonzero(good)) == 1:
        return np.full_like(values, float(values[good][0]))
    return np.interp(idx, idx[good], values[good]).astype(np.float64)


def _median_filter(values: np.ndarray, k: int = 5) -> np.ndarray:
    radius = max(1, k // 2)
    out = values.copy()
    n = len(values)
    for i in range(n):
        sl = values[max(0, i - radius) : min(n, i + radius + 1)]
        out[i] = float(np.median(sl))
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    csum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out[i] = (csum[b] - csum[a]) / (b - a)
    return out


def _long_absent_mask(cx: np.ndarray, min_run: int) -> np.ndarray:
    """连续漏检达到 min_run 后视为鸟已飞走，此后一直用画面中心。"""
    absent = ~np.isfinite(cx)
    lock = np.zeros(len(cx), dtype=bool)
    i = 0
    n = len(cx)
    held = False
    need = max(1, min_run)
    while i < n:
        if absent[i]:
            j = i + 1
            while j < n and absent[j]:
                j += 1
            if held or (j - i) >= need:
                held = True
                lock[i:j] = True
            i = j
            continue
        j = i + 1
        while j < n and not absent[j]:
            j += 1
        if held:
            lock[i:j] = True
        i = j
    return lock


def _smooth_centers(cx: np.ndarray, cy: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """先补漏检，再去掉单帧跳点，最后做短窗口平滑。窗口过长会让快速飞行的鸟滞后晃动。"""
    radius = max(1, min(8, int(radius)))
    sx = _median_filter(_interp_nan(cx), k=5)
    sy = _median_filter(_interp_nan(cy), k=5)
    window = radius * 2 + 1
    return _moving_average(sx, window), _moving_average(sy, window)


def _output_size(width: int, height: int, bird_h: float) -> Tuple[int, int]:
    """按鸟的高度取景，大约占画面高度的 22%，并保持原片宽高比。"""
    if bird_h > 8:
        target_h = bird_h / 0.22
    else:
        target_h = height * 0.45
    target_h = float(np.clip(target_h, height * 0.28, height * 0.62))
    out_h = int(target_h)
    out_h -= out_h % 2
    out_h = max(64, min(height, out_h))
    out_w = int(round(out_h * (width / float(height))))
    out_w -= out_w % 2
    out_w = max(64, min(width, out_w))
    return out_w, out_h


def _frame_centered(
    frame: np.ndarray, cx: float, cy: float, out_w: int, out_h: int
) -> np.ndarray:
    """把 (cx, cy) 放在输出中心。鸟飞出可裁范围时用黑边，避免把鸟挤离中心。"""
    height, width = frame.shape[:2]
    crop_x = int(round(cx - out_w / 2.0))
    crop_y = int(round(cy - out_h / 2.0))
    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    x1 = max(0, crop_x)
    y1 = max(0, crop_y)
    x2 = min(width, crop_x + out_w)
    y2 = min(height, crop_y + out_h)
    if x2 <= x1 or y2 <= y1:
        return canvas
    dx = x1 - crop_x
    dy = y1 - crop_y
    canvas[dy : dy + (y2 - y1), dx : dx + (x2 - x1)] = frame[y1:y2, x1:x2]
    return canvas


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _boxes_from_result(result) -> List[Tuple[Tuple[float, float, float, float], float]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    out: List[Tuple[Tuple[float, float, float, float], float]] = []
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        xywh = (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
        out.append((xywh, float(conf[i])))
    return out


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _flow_shift(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    box: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float]]:
    """在上一帧鸟框附近做光流，漏检时把框跟着纹理走。"""
    x, y, w, h = box
    height, width = prev_gray.shape[:2]
    pad = int(max(16, 0.5 * max(w, h)))
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(width, int(x + w) + pad)
    y1 = min(height, int(y + h) + pad)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    roi = prev_gray[y0:y1, x0:x1]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.02, minDistance=4)
    if pts is None or len(pts) < 6:
        return None
    pts = pts.copy()
    pts[:, 0, 0] += x0
    pts[:, 0, 1] += y0
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        pts,
        None,
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if nxt is None or status is None:
        return None
    good = status.reshape(-1) == 1
    if int(np.count_nonzero(good)) < 4:
        return None
    delta = (nxt - pts)[good]
    dx = float(np.median(delta[:, 0, 0]))
    dy = float(np.median(delta[:, 0, 1]))
    if abs(dx) > 60 or abs(dy) > 60:
        return None
    return dx, dy


def _track_centers_yolo(
    input_path: str,
    model_path: str,
    *,
    prefer: Optional[Tuple[float, float, float, float]] = None,
    log: LogFn = None,
    progress: ProgressFn = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """逐帧检测，并只接受离上一帧不太远的框。漏检留空，后面再插值。"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    imgsz = 640
    max_step = 0.08 * float(np.hypot(width, height))

    cx = np.full(max(total, 1), np.nan, dtype=np.float64)
    cy = np.full(max(total, 1), np.nan, dtype=np.float64)
    heights: List[float] = []
    last: Optional[Tuple[float, float]] = None
    last_box: Optional[Tuple[float, float, float, float]] = None
    prev_gray = None
    gap = 0
    hits = 0
    flow_fills = 0
    frame_idx = 0

    _log(f"YOLO 检测: {Path(model_path).name}  imgsz={imgsz}", log)
    if prefer is not None:
        _log(
            f"  优先框选区域 ({int(prefer[0])},{int(prefer[1])}) "
            f"{int(prefer[2])}x{int(prefer[3])}",
            log,
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= len(cx):
            cx = np.append(cx, np.nan)
            cy = np.append(cy, np.nan)
        results = model.predict(frame, verbose=False, imgsz=imgsz, conf=0.15)
        cands = _boxes_from_result(results[0] if results else None)
        chosen = None
        if cands and prefer is not None and last is None:
            ranked = sorted(cands, key=lambda c: _iou_xywh(c[0], prefer), reverse=True)
            if _iou_xywh(ranked[0][0], prefer) >= 0.05:
                chosen = ranked[0][0]
            else:
                pcx, pcy = _center(prefer)
                chosen = min(cands, key=lambda c: ( _center(c[0])[0] - pcx) ** 2 + (_center(c[0])[1] - pcy) ** 2)[0]
        elif cands and last is not None and gap <= 8:
            limit = max_step * (gap + 1)
            lx, ly = last
            near = [
                c[0]
                for c in cands
                if ( _center(c[0])[0] - lx) ** 2 + (_center(c[0])[1] - ly) ** 2 <= limit * limit
            ]
            if near:
                chosen = min(near, key=lambda b: (_center(b)[0] - lx) ** 2 + (_center(b)[1] - ly) ** 2)
        elif cands and (last is None or gap > 8):
            chosen = max(cands, key=lambda c: c[1])[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if chosen is not None:
            hits += 1
            gap = 0
            last_box = chosen
            last = _center(chosen)
            cx[frame_idx] = last[0]
            cy[frame_idx] = last[1]
            heights.append(float(chosen[3]))
        else:
            gap += 1
            # 只在刚丢失的短窗口内用光流。再长就当鸟已飞走，不再跟着背景挪画面。
            shift = None
            if gap <= 10 and last_box is not None and prev_gray is not None:
                shift = _flow_shift(prev_gray, gray, last_box)
            if shift is not None and last_box is not None:
                flow_fills += 1
                x, y, bw, bh = last_box
                last_box = (x + shift[0], y + shift[1], bw, bh)
                last = _center(last_box)
                cx[frame_idx] = last[0]
                cy[frame_idx] = last[1]
        prev_gray = gray
        frame_idx += 1
        if frame_idx % 60 == 0:
            _log(f"  已处理 {frame_idx}/{max(total, frame_idx)} 帧", log)
        if progress and frame_idx % 10 == 0:
            progress(frame_idx, max(total, frame_idx) * 2)

    cap.release()
    cx = cx[:frame_idx]
    cy = cy[:frame_idx]
    if hits == 0 and frame_idx:
        cx[:] = width / 2.0
        cy[:] = height / 2.0
        _log("未检出鸟，画面保持居中", log)
    missed = frame_idx - hits - flow_fills
    _log(f"检出 {hits}/{frame_idx} 帧，短时光流 {flow_fills} 帧", log)
    if missed > 0:
        _log(f"  {missed} 帧无鸟；连续离画的区段改为画面中心裁切", log)
    bird_h = float(np.median(heights)) if heights else 0.0
    if bird_h > 0:
        _log(f"  鸟高约 {bird_h:.0f}px", log)
    return cx, cy, width, height, fps, bird_h


def _create_box_tracker():
    """优先 CSRT；OpenCV 5 起主包只有 MIL 等跟踪器。"""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create(), "CSRT"
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create(), "CSRT"
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create(), "MIL"
    raise RuntimeError("当前 OpenCV 没有 CSRT/MIL 跟踪器")


def _track_centers_csrt(
    input_path: str,
    init_bbox: Tuple[int, int, int, int],
    *,
    log: LogFn = None,
    progress: ProgressFn = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise IOError("无法读取首帧")

    x0, y0, bw, bh = [int(v) for v in init_bbox]
    x0 = max(0, min(x0, max(0, width - 2)))
    y0 = max(0, min(y0, max(0, height - 2)))
    bw = max(2, min(bw, width - x0))
    bh = max(2, min(bh, height - y0))
    init_bbox = (x0, y0, bw, bh)

    tracker, tracker_name = _create_box_tracker()
    tracker.init(first, init_bbox)

    cx_list: List[float] = [x0 + bw / 2.0]
    cy_list: List[float] = [y0 + bh / 2.0]
    frame_idx = 1
    failed = 0
    _log(f"{tracker_name} 跟踪框: {init_bbox}", log)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [float(v) for v in bbox]
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            failed += 1
            if cx_list:
                cx, cy = cx_list[-1], cy_list[-1]
            else:
                cx, cy = width / 2.0, height / 2.0
        cx_list.append(cx)
        cy_list.append(cy)
        frame_idx += 1
        if progress and frame_idx % 15 == 0:
            progress(frame_idx, total * 2)

    cap.release()
    if failed:
        _log(f"⚠ 框跟踪丢失 {failed} 帧", log)
    return cx_list, cy_list, width, height, fps


def run_track_stabilizer(
    input_path: str,
    output_path: str,
    *,
    engine: str = "yolo",
    bird_model: Optional[str] = None,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    smoothing_window: int = 30,
    margin_ratio: float = 0.15,
    view_norm: Optional[Tuple[float, float, float, float]] = None,
    progress: ProgressFn = None,
    log: LogFn = None,
) -> Tuple[int, int]:
    width = height = 0
    fps = 25.0

    bird_h = 0.0
    if engine == "yolo" and bird_model and Path(bird_model).is_file():
        cx_raw, cy_raw, width, height, fps, bird_h = _track_centers_yolo(
            input_path, bird_model, prefer=init_bbox, log=log, progress=progress
        )
    else:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if init_bbox is None:
            iw = int(width * 0.4)
            ih = int(height * 0.4)
            init_bbox = ((width - iw) // 2, (height - ih) // 2, iw, ih)
        cx_list, cy_list, width, height, fps = _track_centers_csrt(
            input_path, init_bbox, log=log, progress=progress
        )
        cx_raw = np.asarray(cx_list, dtype=np.float64)
        cy_raw = np.asarray(cy_list, dtype=np.float64)
        if init_bbox is not None:
            bird_h = float(init_bbox[3])

    radius = max(1, min(8, int(smoothing_window) // 2 or 1))
    cx_raw = np.asarray(cx_raw, dtype=np.float64)
    cy_raw = np.asarray(cy_raw, dtype=np.float64)
    min_run = max(12, int(round(float(fps) * 0.4)))
    gone = _long_absent_mask(cx_raw, min_run)
    cx_track = cx_raw.copy()
    cy_track = cy_raw.copy()
    cx_track[gone] = np.nan
    cy_track[gone] = np.nan
    smooth_x, smooth_y = _smooth_centers(cx_track, cy_track, radius)
    view_rect = None
    rel_x, rel_y = 0.5, 0.5
    if view_norm is not None:
        from .media import fit_even_rect

        view_rect = fit_even_rect(view_norm, width, height)
        vx, vy, vw, vh = view_rect
        if init_bbox is not None:
            bx, by, bw, bh = init_bbox
            rel_x = (bx + bw / 2.0 - vx) / max(1, vw)
            rel_y = (by + bh / 2.0 - vy) / max(1, vh)
            rel_x = max(0.0, min(1.0, rel_x))
            rel_y = max(0.0, min(1.0, rel_y))
        out_w, out_h = vw, vh
        _log(f"自定义画面: {out_w}x{out_h} @ ({vx},{vy})", log)
    else:
        out_w, out_h = _output_size(width, height, bird_h)
    if np.any(gone):
        if view_rect is None:
            smooth_x[gone] = width / 2.0
            smooth_y[gone] = height / 2.0
            _log(f"连续无鸟 {int(np.count_nonzero(gone))} 帧，保持画面中心裁切，不补黑边", log)
        else:
            _log(f"连续无鸟 {int(np.count_nonzero(gone))} 帧，保持所画画面范围，不补黑边", log)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    total = len(smooth_x)
    if view_rect is not None:
        from .media import window_origin
    _log(
        f"追踪输出: {out_w}x{out_h}，平滑 {radius * 2 + 1} 帧，"
        f"轨迹跨度 x={smooth_x.max() - smooth_x.min():.0f} y={smooth_y.max() - smooth_y.min():.0f}",
        log,
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total:
            break
        if view_rect is not None and gone[frame_idx]:
            vx, vy, _, _ = view_rect
            cropped = frame[vy : vy + out_h, vx : vx + out_w]
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h))
            out.write(cropped)
        elif view_rect is not None:
            ox, oy = window_origin(
                smooth_x[frame_idx],
                smooth_y[frame_idx],
                out_w,
                out_h,
                width,
                height,
                rel_x,
                rel_y,
            )
            cropped = frame[oy : oy + out_h, ox : ox + out_w]
            if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                cropped = cv2.resize(cropped, (out_w, out_h))
            out.write(cropped)
        else:
            out.write(
                _frame_centered(frame, smooth_x[frame_idx], smooth_y[frame_idx], out_w, out_h)
            )
        frame_idx += 1
        if progress and frame_idx % 15 == 0:
            progress(total + frame_idx, total * 2)

    cap.release()
    out.release()
    return out_w, out_h
