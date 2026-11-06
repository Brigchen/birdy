# -*- coding: utf-8 -*-
"""三脚架固定视角：每帧对齐第一帧，再裁到全程都看得见的区域。

手持晃动常常超过画面的两成。若把校正量钳在窄裁边里，或者用指数平滑只跟一部分大位移，
背景会残留跳动，采样越出画面时就闪黑边。大位移不设上限、也不做平滑：
先对齐第一帧，再用相位相关补掉剩下的小偏差。锁定之后磨平几像素的高频残差，
并按行收掉岸线上下不一致的错位。输出窗口取所有帧的公共可视区域（面积可以不到一半）。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ProgressFn = Optional[Callable[[int, int], None]]
LogFn = Optional[Callable[[str], None]]

_EST_WIDTH = 960
_REFINE_WIDTH = 1920


def _log(msg: str, log: LogFn) -> None:
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def _resize_gray(bgr: np.ndarray, rw: int, rh: int) -> np.ndarray:
    small = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    xy = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(xy)
    out[:, 0] = M[0, 0] * xy[:, 0] + M[0, 1] * xy[:, 1] + M[0, 2]
    out[:, 1] = M[1, 0] * xy[:, 0] + M[1, 1] * xy[:, 1] + M[1, 2]
    return out


def _lock_similarity(M: np.ndarray, rw: int, rh: int) -> np.ndarray:
    """去掉缩放（呼吸感），保留旋转，并让画面中心仍落到原估计位置。"""
    a, b, tx = (float(v) for v in M[0])
    c, _d, ty = (float(v) for v in M[1])
    scale = math.hypot(a, c)
    if scale < 1e-8:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    ang = math.atan2(c, a)
    cx = (rw - 1) * 0.5
    cy = (rh - 1) * 0.5
    dx = a * cx + b * cy + tx
    dy = c * cx + M[1, 1] * cy + ty
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    tx2 = dx - cos_a * cx + sin_a * cy
    ty2 = dy - sin_a * cx - cos_a * cy
    return np.array(
        [[cos_a, -sin_a, tx2], [sin_a, cos_a, ty2]],
        dtype=np.float64,
    )


def _to_full(M: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """小图坐标上的仿射，换成原分辨率。"""
    a, b, tx = (float(v) for v in M[0])
    c, d, ty = (float(v) for v in M[1])
    return np.array(
        [
            [a, b * (sx / sy), tx * sx],
            [c * (sy / sx), d, ty * sy],
        ],
        dtype=np.float64,
    )


def _rebuild_from_center(
    angles: Sequence[float],
    dest_x: Sequence[float],
    dest_y: Sequence[float],
    rw: int,
    rh: int,
) -> List[np.ndarray]:
    cx = (rw - 1) * 0.5
    cy = (rh - 1) * 0.5
    motions: List[np.ndarray] = []
    for ang, dx, dy in zip(angles, dest_x, dest_y):
        cos_a = math.cos(float(ang))
        sin_a = math.sin(float(ang))
        tx = float(dx) - cos_a * cx + sin_a * cy
        ty = float(dy) - sin_a * cx - cos_a * cy
        motions.append(
            np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float64)
        )
    return motions


def estimate_reference_motions(
    input_path: str,
    *,
    progress: ProgressFn = None,
) -> Tuple[List[np.ndarray], int, int]:
    """返回每帧「参考坐标 → 当前帧」的相似变换（缩放已锁定为 1），以及原图宽高。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, ref_bgr = cap.read()
    if not ok or ref_bgr is None:
        cap.release()
        raise IOError("无法读取第一帧")

    rw = min(_EST_WIDTH, width)
    rh = max(64, int(round(height * (rw / float(width)))))
    sx = width / float(rw)
    sy = height / float(rh)
    ref_g = _resize_gray(ref_bgr, rw, rh)
    ref_f = ref_g.astype(np.float32)
    hann = cv2.createHanningWindow((rw, rh), cv2.CV_32F)

    p_ref = cv2.goodFeaturesToTrack(
        ref_g,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if p_ref is None or len(p_ref) < 20:
        cap.release()
        raise RuntimeError("第一帧纹理太少，无法锁定视角")
    p_ref = p_ref.reshape(-1, 1, 2).astype(np.float32)

    lk = dict(
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    angles = [0.0]
    dest_x = [(rw - 1) * 0.5]
    dest_y = [(rh - 1) * 0.5]
    prev_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    frame_idx = 1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = _resize_gray(frame, rw, rh)
        shift, resp = cv2.phaseCorrelate(ref_f, gray.astype(np.float32), hann)
        if resp < 0.02:
            seed = prev_M
        else:
            seed = _lock_similarity(
                np.array(
                    [[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]],
                    dtype=np.float64,
                ),
                rw,
                rh,
            )
        pred = _apply_affine(seed, p_ref).reshape(-1, 1, 2).astype(np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            ref_g, gray, p_ref, pred, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk
        )
        M = None
        if p1 is not None and st is not None:
            good = (st.reshape(-1) == 1) & (err.reshape(-1) < 18.0)
            if int(good.sum()) >= 12:
                src = p_ref.reshape(-1, 2)[good]
                dst = p1.reshape(-1, 2)[good]
                raw, inl = cv2.estimateAffinePartial2D(
                    src.reshape(-1, 1, 2),
                    dst.reshape(-1, 1, 2),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2000,
                    confidence=0.99,
                )
                if raw is not None and inl is not None and int(inl.sum()) >= 12:
                    M = _lock_similarity(raw, rw, rh)
        if M is None:
            M = prev_M
        else:
            cx = (rw - 1) * 0.5
            cy = (rh - 1) * 0.5
            jumped = _apply_affine(M, np.array([[cx, cy]]))[0]
            prev_c = _apply_affine(prev_M, np.array([[cx, cy]]))[0]
            step = math.hypot(float(jumped[0] - prev_c[0]), float(jumped[1] - prev_c[1]))
            if step > 0.18 * min(rw, rh):
                M = prev_M
        prev_M = M
        cx = (rw - 1) * 0.5
        cy = (rh - 1) * 0.5
        landed = _apply_affine(M, np.array([[cx, cy]]))[0]
        ang = math.atan2(float(M[1, 0]), float(M[0, 0]))
        angles.append(ang)
        dest_x.append(float(landed[0]))
        dest_y.append(float(landed[1]))
        frame_idx += 1
        if progress and frame_idx % 20 == 0:
            progress(frame_idx, max(total, frame_idx))

    cap.release()
    angles[0] = 0.0
    dest_x[0] = (rw - 1) * 0.5
    dest_y[0] = (rh - 1) * 0.5
    small = _rebuild_from_center(angles, dest_x, dest_y, rw, rh)
    full = [_to_full(M, sx, sy) for M in small]
    return full, width, height


def _full_to_small(M: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """原分辨率上的「参考 → 当前」仿射，换成缩小图坐标。"""
    a, b, tx = (float(v) for v in M[0])
    c, d, ty = (float(v) for v in M[1])
    return np.array(
        [
            [a, b * (sy / sx), tx / sx],
            [c * (sx / sy), d, ty / sy],
        ],
        dtype=np.float64,
    )


def _add_output_shift(M: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """输出画面相对参考还偏了 (dx, dy) 时，把采样点沿当前帧平移补上。"""
    out = np.array(M, dtype=np.float64, copy=True)
    out[0, 2] += float(dx)
    out[1, 2] += float(dy)
    return out


def _ground_roi(image: np.ndarray) -> np.ndarray:
    """去掉上沿天空，用地面和地平线的纹理锁细抖动。"""
    y0 = int(image.shape[0] * 0.28)
    return image[y0:]


def _refine_once(
    input_path: str,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
) -> Tuple[List[np.ndarray], int, List[float]]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    ok, ref_bgr = cap.read()
    if not ok or ref_bgr is None:
        cap.release()
        raise IOError("无法读取第一帧")

    rw = min(_REFINE_WIDTH, width)
    rh = max(64, int(round(height * (rw / float(width)))))
    sx = width / float(rw)
    sy = height / float(rh)
    ref_roi = _ground_roi(_resize_gray(ref_bgr, rw, rh).astype(np.float32))
    hann = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_32F)
    limit = 0.015 * float(min(width, height))

    dxs = [0.0]
    dys = [0.0]
    base: List[np.ndarray] = [np.array(motions[0], dtype=np.float64, copy=True)]
    for M in list(motions)[1:]:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = _resize_gray(frame, rw, rh)
        aligned = cv2.warpAffine(
            gray,
            _full_to_small(M, sx, sy),
            (rw, rh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        ali_roi = _ground_roi(aligned.astype(np.float32))
        shift, resp = cv2.phaseCorrelate(ref_roi, ali_roi, hann)
        dx = float(shift[0]) * sx
        dy = float(shift[1]) * sy
        if resp < 0.04 or abs(dx) > limit or abs(dy) > limit:
            dx, dy = 0.0, 0.0
        dxs.append(dx)
        dys.append(dy)
        base.append(np.array(M, dtype=np.float64, copy=True))
    cap.release()
    while len(base) < len(motions):
        base.append(np.array(motions[len(base)], dtype=np.float64, copy=True))
        dxs.append(0.0)
        dys.append(0.0)
    dxs[0] = 0.0
    dys[0] = 0.0
    refined = [_add_output_shift(M, dx, dy) for M, dx, dy in zip(base, dxs, dys)]
    mags = [math.hypot(dx, dy) for dx, dy in zip(dxs[1:], dys[1:]) if dx or dy]
    used = len(mags)
    return refined, used, mags


def refine_reference_motions(
    input_path: str,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
    *,
    log: LogFn = None,
) -> List[np.ndarray]:
    """粗对齐之后，用地平线以下的纹理把每帧再贴回第一帧。连做两遍，把剩下的小抖动补掉。"""
    if not motions:
        return []
    current = [np.array(M, dtype=np.float64, copy=True) for M in motions]
    used = 0
    mags: List[float] = []
    for _ in range(2):
        current, used, mags = _refine_once(input_path, current, width, height)
    if mags:
        _log(
            f"细校正: {used}/{max(1, len(current) - 1)} 帧，"
            f"平移中位 {float(np.median(mags)):.2f}px",
            log,
        )
    else:
        _log("细校正: 没有可用的亚像素修正", log)
    return current


def _gaussian_lowpass(values: Sequence[float], sigma: float) -> np.ndarray:
    """零相位高斯平滑。sigma 用帧数，越大越只留下缓慢漂移。"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 5 or sigma < 0.8:
        return arr.copy()
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _step_p95(rx: np.ndarray, ry: np.ndarray) -> float:
    if len(rx) < 3:
        return 0.0
    mag = np.hypot(np.diff(rx), np.diff(ry))
    return float(np.percentile(mag, 95))


def _preview_matrix(M: np.ndarray, origin_x: float, origin_y: float, scale: float) -> np.ndarray:
    """小预览上的像素 (i, j) 对应成片像素 (i*scale, j*scale)，再按成片矩阵采样原图。"""
    return np.array(
        [
            [
                M[0, 0] * scale,
                M[0, 1] * scale,
                M[0, 0] * origin_x + M[0, 1] * origin_y + M[0, 2],
            ],
            [
                M[1, 0] * scale,
                M[1, 1] * scale,
                M[1, 0] * origin_x + M[1, 1] * origin_y + M[1, 2],
            ],
        ],
        dtype=np.float64,
    )


def _measure_crop_residual(
    input_path: str,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """按最终成片的采样方式，量每一帧相对第一帧还偏了多少像素。

    直接从原图采样到预览，避免先缩小再变形带来的偏差。返回 (rx, ry, 被拒绝的帧数)。
    拒绝的帧沿用上一帧，不能当成已经对齐。
    """
    n = len(motions)
    rx = np.zeros(n, dtype=np.float64)
    ry = np.zeros(n, dtype=np.float64)
    rejected = 0
    if n == 0:
        return rx, ry, rejected
    try:
        left, top, out_w, out_h = stable_crop_box(motions, width, height)
    except (RuntimeError, ValueError):
        return rx, ry, n
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    ok, ref_bgr = cap.read()
    if not ok or ref_bgr is None:
        cap.release()
        raise IOError("无法读取第一帧")

    rw = min(960, out_w)
    rh = max(64, int(round(out_h * (rw / float(out_w)))))
    scale = out_w / float(rw)
    limit = 24.0

    def _roi(gray: np.ndarray, M: np.ndarray) -> np.ndarray:
        warped = cv2.warpAffine(
            gray,
            _preview_matrix(M, float(left), float(top), scale),
            (rw, rh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        y0 = int(warped.shape[0] * 0.08)
        y1 = max(y0 + 32, int(warped.shape[0] * 0.92))
        x0 = int(warped.shape[1] * 0.08)
        x1 = max(x0 + 32, int(warped.shape[1] * 0.92))
        roi = warped[y0:y1, x0:x1]
        if roi.shape[0] < 32 or roi.shape[1] < 32:
            roi = warped
        return roi.astype(np.float32)

    ref = _roi(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), motions[0])
    hann = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    filled = 1
    for index, M in enumerate(list(motions)[1:], start=1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cur = _roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), M)
        if cur.shape != ref.shape:
            cur = cv2.resize(cur, (ref.shape[1], ref.shape[0])).astype(np.float32)
        shift, resp = cv2.phaseCorrelate(ref, cur, hann)
        dx = float(shift[0]) * scale
        dy = float(shift[1]) * scale
        if resp < 0.04 or abs(dx) > limit or abs(dy) > limit:
            rx[index] = rx[index - 1]
            ry[index] = ry[index - 1]
            rejected += 1
        else:
            rx[index] = dx
            ry[index] = dy
        filled = index + 1
    cap.release()
    if filled < n:
        rx[filled:] = rx[filled - 1]
        ry[filled:] = ry[filled - 1]
        rejected += n - filled
    return rx, ry, rejected


def _apply_shift_series(
    motions: Sequence[np.ndarray],
    dx: np.ndarray,
    dy: np.ndarray,
    sign: float,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for M, x, y in zip(motions, dx, dy):
        out.append(_add_output_shift(M, sign * float(x), sign * float(y)))
    while len(out) < len(motions):
        out.append(np.array(motions[len(out)], dtype=np.float64, copy=True))
    return out


def smooth_residual_jitter(
    input_path: str,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
    *,
    log: LogFn = None,
) -> List[np.ndarray]:
    """锁定之后再磨轨迹：高频错位被抵消，只保留平滑的缓慢漂移。

    校正只有几像素。正反两个方向都量一遍帧间位移，只用能把抖动压下去的那个；
    裁切窗口会在后面按新轨迹重算，避免黑边。
    """
    copied = [np.array(M, dtype=np.float64, copy=True) for M in motions]
    if len(motions) < 8:
        return copied
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    def _score(step: float, rejected: int) -> float:
        if rejected > max(2, int(0.2 * len(motions))):
            return 1e6
        return step

    def _highpass(px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cx = np.clip(px - _gaussian_lowpass(px, sigma), -8.0, 8.0)
        cy = np.clip(py - _gaussian_lowpass(py, sigma), -8.0, 8.0)
        cx[0] = 0.0
        cy[0] = 0.0
        return cx, cy

    sigma = max(8.0, 0.30 * fps)
    rx, ry, _rejected = _measure_crop_residual(input_path, motions, width, height)
    base_step = _step_p95(rx, ry)
    if base_step < 0.12:
        _log(f"轨迹平滑: 帧间 p95 已是 {base_step:.2f}px，不再改轨迹", log)
        return copied

    corr_x, corr_y = _highpass(rx, ry)
    chosen_sign = 0.0
    current = copied
    current_step = base_step
    for sign in (1.0, -1.0):
        trial = _apply_shift_series(motions, corr_x, corr_y, sign)
        tx, ty, rejected = _measure_crop_residual(input_path, trial, width, height)
        step = _score(_step_p95(tx, ty), rejected)
        if step < current_step:
            current_step = step
            current = trial
            chosen_sign = sign
            rx, ry = tx, ty
    if chosen_sign == 0.0 or current_step > base_step * 0.85:
        _log(
            f"轨迹平滑: 帧间 p95 {base_step:.2f}px，校正没有压低抖动，保持锁定",
            log,
        )
        return copied

    for _ in range(2):
        corr_x, corr_y = _highpass(rx, ry)
        if float(np.median(np.hypot(corr_x[1:], corr_y[1:]))) < 0.05:
            break
        trial = _apply_shift_series(current, corr_x, corr_y, chosen_sign)
        tx, ty, rejected = _measure_crop_residual(input_path, trial, width, height)
        step = _score(_step_p95(tx, ty), rejected)
        if step > current_step * 0.85:
            break
        current = trial
        current_step = step
        rx, ry = tx, ty

    _log(
        f"轨迹平滑: {sigma:.0f} 帧高斯，帧间 p95 {base_step:.2f}→{current_step:.2f}px",
        log,
    )
    return current


def _center_shift(M: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
    x = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    y = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
    return float(x - cx), float(y - cy)


def _corners_inside(
    M: np.ndarray,
    corners: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    pad: float,
) -> bool:
    for x, y in corners:
        xp = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        yp = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        if xp < pad or yp < pad or xp > width - 1 - pad or yp > height - 1 - pad:
            return False
    return True


def stable_crop_box(
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
    *,
    pad: float = 2.0,
) -> Tuple[int, int, int, int]:
    """参考坐标系里、所有帧采样都不出画幅的最大轴对齐窗口。返回 (x, y, w, h)。"""
    if not motions:
        raise ValueError("没有运动估计")
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    dxs = []
    dys = []
    for M in motions:
        dx, dy = _center_shift(M, cx, cy)
        dxs.append(dx)
        dys.append(dy)
    left = int(math.ceil(max(pad - dx for dx in dxs)))
    right = int(math.floor(min(width - pad - dx for dx in dxs)))
    top = int(math.ceil(max(pad - dy for dy in dys)))
    bottom = int(math.floor(min(height - pad - dy for dy in dys)))
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    def fits(L: int, T: int, R: int, B: int) -> bool:
        if R - L < 64 or B - T < 64:
            return False
        corners = ((L, T), (R - 1, T), (R - 1, B - 1), (L, B - 1))
        return all(_corners_inside(M, corners, width, height, pad) for M in motions)

    guard = 0
    while not fits(left, top, right, bottom) and guard < 500:
        guard += 1
        push = [0.0, 0.0, 0.0, 0.0]  # 左、上内收，右、下内收
        for M in motions:
            for x, y in (
                (left, top),
                (right - 1, top),
                (right - 1, bottom - 1),
                (left, bottom - 1),
            ):
                xp = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                yp = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                if xp < pad:
                    overflow = pad - xp
                    if x <= left + 1:
                        push[0] = max(push[0], overflow)
                    else:
                        push[2] = max(push[2], overflow)
                elif xp > width - 1 - pad:
                    overflow = xp - (width - 1 - pad)
                    if x >= right - 2:
                        push[2] = max(push[2], overflow)
                    else:
                        push[0] = max(push[0], overflow)
                if yp < pad:
                    overflow = pad - yp
                    if y <= top + 1:
                        push[1] = max(push[1], overflow)
                    else:
                        push[3] = max(push[3], overflow)
                elif yp > height - 1 - pad:
                    overflow = yp - (height - 1 - pad)
                    if y >= bottom - 2:
                        push[3] = max(push[3], overflow)
                    else:
                        push[1] = max(push[1], overflow)
        step = [max(1, int(math.ceil(v))) if v > 1e-6 else 0 for v in push]
        if not any(step):
            step = [1, 1, 1, 1]
        left += step[0]
        top += step[1]
        right -= step[2]
        bottom -= step[3]
        if right - left < 64 or bottom - top < 64:
            break
    if not fits(left, top, right, bottom):
        raise RuntimeError("晃动超过可裁范围，无法得到无黑边的固定视角")

    out_w = (right - left) & ~1
    out_h = (bottom - top) & ~1
    right = left + out_w
    bottom = top + out_h
    while out_w >= 64 and out_h >= 64 and not fits(left, top, right, bottom):
        if out_w >= out_h:
            out_w -= 2
            right = left + out_w
        else:
            out_h -= 2
            bottom = top + out_h
    if out_w < 64 or out_h < 64 or not fits(left, top, right, bottom):
        raise RuntimeError("固定视角裁切过小")
    return left, top, out_w, out_h


def _output_matrix(M: np.ndarray, origin_x: float, origin_y: float) -> np.ndarray:
    """输出像素 (u, v) 对应参考坐标 (origin_x+u, origin_y+v)，再映射到当前帧。"""
    return np.array(
        [
            [
                M[0, 0],
                M[0, 1],
                M[0, 0] * origin_x + M[0, 1] * origin_y + M[0, 2],
            ],
            [
                M[1, 0],
                M[1, 1],
                M[1, 0] * origin_x + M[1, 1] * origin_y + M[1, 2],
            ],
        ],
        dtype=np.float64,
    )


def _fit_strips(
    y_norm: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    weight: np.ndarray,
    fast: bool,
) -> np.ndarray:
    """慢速帧用二次曲线。快速横移时只用中下部分的直线，避免把拖影拧成弧。"""
    coeff = np.zeros((2, 3), dtype=np.float64)
    mask = weight > 0
    if fast:
        lower = mask & (y_norm > -0.35)
        use = lower if int(np.count_nonzero(lower)) >= 3 else mask
        degree = 1
    else:
        use = mask
        degree = 2
    if int(np.count_nonzero(use)) < degree + 2:
        return coeff
    sw = np.sqrt(weight[use])
    for axis, samples in ((0, dx), (1, dy)):
        fit = np.polyfit(y_norm[use], samples[use], degree, w=sw)
        if degree == 1:
            coeff[axis, 1] = float(fit[0])
            coeff[axis, 2] = float(fit[1])
        else:
            coeff[axis] = fit
    return coeff


def _despike_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """单帧系数若和前后都差一截，改成前后的平均，去掉孤立的错行。"""
    out = np.array(coeffs, dtype=np.float64, copy=True)
    if len(out) < 3:
        return out
    mag = np.max(np.abs(out.reshape(len(out), -1)), axis=1)
    for i in range(1, len(out) - 1):
        if abs(mag[i] - mag[i - 1]) > 4.0 and abs(mag[i] - mag[i + 1]) > 4.0:
            out[i] = 0.5 * (out[i - 1] + out[i + 1])
    return out


def measure_row_field(
    input_path: str,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
    left: int,
    top: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """每一帧拟合 dx(y)、dy(y) 的二次曲线，把岸线收到和画面中部同一个第一帧。

    返回 (帧数, 2, 3)，第二维 0 是水平、1 是竖直，系数按 y²、y、常数排列，
    y 在输出高度上从 -1 到 1。
    """
    n = len(motions)
    coeffs = np.zeros((n, 2, 3), dtype=np.float64)
    if n == 0 or out_w < 64 or out_h < 64:
        return coeffs
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    ok, ref_bgr = cap.read()
    if not ok or ref_bgr is None:
        cap.release()
        raise IOError("无法读取第一帧")

    rw = min(960, out_w)
    rh = max(64, int(round(out_h * (rw / float(out_w)))))
    sx = out_w / float(rw)
    sy = out_h / float(rh)
    strips = 8

    def _preview(gray: np.ndarray, M: np.ndarray) -> np.ndarray:
        warped = cv2.warpAffine(
            gray,
            _preview_matrix(M, float(left), float(top), sx),
            (rw, rh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    ref = _preview(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), motions[0])
    x0 = int(rw * 0.08)
    x1 = max(x0 + 32, int(rw * 0.92))
    edges = [int(round(rh * i / strips)) for i in range(strips + 1)]
    hann = []
    ref_strips = []
    y_norm = []
    for i in range(strips):
        y_a, y_b = edges[i], max(edges[i] + 1, edges[i + 1])
        patch = ref[y_a:y_b, x0:x1]
        ref_strips.append(patch.astype(np.float32))
        hann.append(cv2.createHanningWindow((patch.shape[1], patch.shape[0]), cv2.CV_32F))
        y_norm.append(((y_a + y_b) * 0.5 / rh) * 2.0 - 1.0)
    y_norm_arr = np.asarray(y_norm, dtype=np.float64)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    centers = np.array([_center_shift(M, cx, cy) for M in motions], dtype=np.float64)
    motion_step = np.zeros(len(centers), dtype=np.float64)
    if len(centers) > 1:
        delta = np.diff(centers, axis=0)
        motion_step[1:] = np.hypot(delta[:, 0], delta[:, 1])
        motion_step[0] = motion_step[1]

    index = 0
    while index + 1 < n:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        index += 1
        cur = _preview(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), motions[index])
        dx = np.zeros(strips, dtype=np.float64)
        dy = np.zeros(strips, dtype=np.float64)
        weight = np.zeros(strips, dtype=np.float64)
        for i in range(strips):
            y_a, y_b = edges[i], max(edges[i] + 1, edges[i + 1])
            patch = cur[y_a:y_b, x0:x1].astype(np.float32)
            if patch.shape != ref_strips[i].shape or float(patch.std()) < 6.0:
                continue
            shift, resp = cv2.phaseCorrelate(ref_strips[i], patch, hann[i])
            sx_i = float(shift[0]) * sx
            sy_i = float(shift[1]) * sy
            shift_limit = 28.0 if motion_step[index] > 35.0 else 16.0
            if resp < 0.05 or abs(sx_i) > shift_limit or abs(sy_i) > shift_limit:
                continue
            dx[i] = sx_i
            dy[i] = sy_i
            weight[i] = float(resp) * float(patch.std())
        if int(np.count_nonzero(weight)) < 4:
            continue
        fitted = _fit_strips(
            y_norm_arr, dx, dy, weight, bool(motion_step[index] > 35.0)
        )
        probe = np.array([-1.0, 0.0, 1.0])
        peak = max(
            float(np.max(np.abs(np.polyval(fitted[0], probe)))),
            float(np.max(np.abs(np.polyval(fitted[1], probe)))),
        )
        peak_limit = 24.0 if motion_step[index] > 35.0 else 16.0
        if peak > peak_limit:
            continue
        coeffs[index] = fitted
    cap.release()
    return _despike_coeffs(coeffs)


def tame_row_field(
    coeffs: np.ndarray,
    motions: Sequence[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """把单帧行位移限制在画面还能受得住的范围里。"""
    out = np.array(coeffs, dtype=np.float64, copy=True)
    if len(out) == 0:
        return out
    _ = (motions, width, height)
    probe = np.array([-1.0, 0.0, 1.0])
    limit = 18.0
    for frame in out:
        peak = 0.0
        for axis in (0, 1):
            peak = max(peak, float(np.max(np.abs(np.polyval(frame[axis], probe)))))
        if peak > limit:
            frame *= limit / peak
    return out


def row_lock_margin(coeffs: np.ndarray) -> int:
    if coeffs.size == 0:
        return 0
    probe = np.array([-1.0, 0.0, 1.0])
    peak = 0.0
    for frame in coeffs:
        for axis in (0, 1):
            peak = max(peak, float(np.max(np.abs(np.polyval(frame[axis], probe)))))
    if peak < 0.4:
        return 0
    return int(math.ceil(peak)) + 3


def render_stabilized_frame(
    frame: np.ndarray,
    M: np.ndarray,
    left: int,
    top: int,
    out_w: int,
    out_h: int,
    coeff: Optional[np.ndarray],
    margin: int,
    measured_h: int,
) -> np.ndarray:
    """按锁定矩阵采样；coeff 给出随行变化的额外位移，用来把岸线拉回第一帧。"""
    if coeff is None or margin <= 0 or np.max(np.abs(coeff)) < 0.05:
        return cv2.warpAffine(
            frame,
            _output_matrix(M, float(left + margin), float(top + margin)),
            (out_w, out_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    ys = np.arange(out_h, dtype=np.float32)
    xs = np.arange(out_w, dtype=np.float32)
    y_old = ys + float(margin)
    x_old = xs + float(margin)
    s = ((y_old + 0.5) / float(max(measured_h, 1))) * 2.0 - 1.0
    dx = np.polyval(coeff[0], s).astype(np.float32)
    dy = np.polyval(coeff[1], s).astype(np.float32)
    x_ref = x_old[None, :] + dx[:, None]
    y_ref = y_old[:, None] + dy[:, None]
    map_x = (
        M[0, 0] * x_ref + M[0, 1] * y_ref + (M[0, 0] * left + M[0, 1] * top + M[0, 2])
    ).astype(np.float32)
    map_y = (
        M[1, 0] * x_ref + M[1, 1] * y_ref + (M[1, 0] * left + M[1, 1] * top + M[1, 2])
    ).astype(np.float32)
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def run_virtual_tripod(
    input_path: str,
    output_path: str,
    *,
    margin_ratio: float = 0.12,
    progress: ProgressFn = None,
    log: LogFn = None,
) -> Tuple[int, int]:
    """锁定第一帧视角并写出无黑边视频。margin_ratio 仅保留兼容，裁切由晃动幅度决定。"""
    _ = margin_ratio
    motions, width, height = estimate_reference_motions(input_path, progress=progress)
    motions = refine_reference_motions(
        input_path, motions, width, height, log=log
    )
    motions = smooth_residual_jitter(
        input_path, motions, width, height, log=log
    )
    crop_left, crop_top, crop_w, crop_h = stable_crop_box(motions, width, height)
    row_coeffs = tame_row_field(
        measure_row_field(
            input_path, motions, width, height, crop_left, crop_top, crop_w, crop_h
        ),
        motions,
        width,
        height,
    )
    margin = row_lock_margin(row_coeffs)
    out_w = (crop_w - 2 * margin) & ~1
    out_h = (crop_h - 2 * margin) & ~1
    if out_w < 64 or out_h < 64:
        margin = 0
        out_w, out_h = crop_w & ~1, crop_h & ~1
    left = crop_left
    top = crop_top
    if margin > 0:
        def _at(axis: int, s: float) -> np.ndarray:
            c = row_coeffs[:, axis]
            return c[:, 0] * s * s + c[:, 1] * s + c[:, 2]

        gap = np.hypot(_at(0, 1.0) - _at(0, -1.0), _at(1, 1.0) - _at(1, -1.0))
        _log(
            f"行校正: 上下相对位移 p95 {float(np.percentile(np.abs(gap), 95)):.1f}px，"
            f"四边内收 {margin}px",
            log,
        )
    area = (out_w * out_h) / float(width * height)

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    shifts = np.array([_center_shift(M, cx, cy) for M in motions], dtype=np.float64)
    p2p_x = float(shifts[:, 0].max() - shifts[:, 0].min()) if len(shifts) else 0.0
    p2p_y = float(shifts[:, 1].max() - shifts[:, 1].min()) if len(shifts) else 0.0
    _log(
        f"虚拟三脚架: 晃动 {p2p_x:.0f}×{p2p_y:.0f}px，"
        f"输出 {out_w}×{out_h}（保留面积 {area:.0%}）",
        log,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"无法写入视频: {output_path}")

    ref_small = None
    residual = []
    black_frames = 0
    total = max(1, len(motions))
    for i, M in enumerate(motions):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        stabilized = render_stabilized_frame(
            frame,
            M,
            left,
            top,
            out_w,
            out_h,
            None if margin <= 0 else row_coeffs[min(i, len(row_coeffs) - 1)],
            margin,
            crop_h,
        )
        border = np.concatenate(
            [
                stabilized[:2, :, :].reshape(-1, 3),
                stabilized[-2:, :, :].reshape(-1, 3),
                stabilized[:, :2, :].reshape(-1, 3),
                stabilized[:, -2:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        if np.all(border == 0):
            black_frames += 1
        small = cv2.resize(
            cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY),
            (320, max(64, int(round(320 * out_h / out_w)))),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        if ref_small is None:
            ref_small = small
            residual.append((0.0, 0.0))
        else:
            shift, _resp = cv2.phaseCorrelate(ref_small, small)
            residual.append((float(shift[0]), float(shift[1])))
        writer.write(stabilized)
        if progress and (i + 1) % 15 == 0:
            progress(i + 1, total)

    cap.release()
    writer.release()
    if residual:
        arr = np.array(residual, dtype=np.float64)
        rx = float(arr[:, 0].max() - arr[:, 0].min())
        ry = float(arr[:, 1].max() - arr[:, 1].min())
        _log(
            f"锚定残差 p2p {rx:.2f}×{ry:.2f}px（320px 宽预览），纯黑边帧 {black_frames}",
            log,
        )
    return out_w, out_h
