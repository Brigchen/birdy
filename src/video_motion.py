# -*- coding: utf-8 -*-
"""视频相机运动分析与裁切规划。

思路参考公开的图像防抖做法（KLT + RANSAC 相似变换、零相位高斯平滑、
按实际晃动反推固定裁切），按 Birdy 的需要独立实现，未复制任何第三方源码。

两个入口：
- load_motion()：逐帧估计相机运动，结果缓存在 ~/.birdy/video_motion_cache，
  改平滑时间/强度/裁切时直接复用，不必重新读 4K 帧。
- plan_stabilization()：把运动轨迹平滑后，算出全程不出画幅的固定裁切尺寸
  和每帧「输出坐标 → 源坐标」的采样矩阵。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

CACHE_VERSION = 1
ANALYSIS_MAX_WIDTH = 960

ProgressFn = Optional[Callable[[int, int], None]]
LogFn = Optional[Callable[[str], None]]
CancelFn = Optional[Callable[[], bool]]


@dataclass
class MotionData:
    """逐帧相机运动。steps[i] = [dx, dy, angle, log_scale, quality]。"""

    width: int
    height: int
    fps: float
    steps: np.ndarray
    cached: bool = False


@dataclass
class CropPlan:
    """裁切与采样计划。matrices 为每帧 2x3 的「输出 → 源」采样矩阵。"""

    out_w: int
    out_h: int
    matrices: List[np.ndarray]
    zoom: float
    strength: float
    motion_px: Tuple[float, float]
    smooth_seconds: float


def _log(msg: str, log: LogFn) -> None:
    if log:
        log(msg)
    else:
        print(f"[Birdy 运动分析] {msg}", flush=True)


def _cache_dir() -> Path:
    override = os.environ.get("BIRDY_MOTION_CACHE")
    path = Path(override) if override else Path.home() / ".birdy" / "video_motion_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(
    identity_path: str,
    width: int,
    height: int,
    fps: float,
    range_key: str,
) -> str:
    try:
        st = os.stat(identity_path)
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    raw = "|".join(
        [
            os.path.abspath(identity_path),
            stamp,
            f"{width}x{height}",
            f"{fps:.3f}",
            range_key,
            str(ANALYSIS_MAX_WIDTH),
            str(CACHE_VERSION),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    finally:
        cap.release()


def _load_cache(
    path: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
) -> Optional[MotionData]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return None
        if int(data["width"]) != width or int(data["height"]) != height:
            return None
        if abs(float(data["fps"]) - fps) > 0.01:
            return None
        steps = np.asarray(data["steps"], dtype=np.float64)
        if steps.ndim != 2 or steps.shape[1] != 5 or len(steps) < 1:
            return None
        if not np.isfinite(steps).all():
            return None
        if frame_count and abs(len(steps) - frame_count) > 1:
            return None
        return MotionData(width, height, fps, steps, cached=True)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cache(path: Path, motion: MotionData) -> None:
    payload = {
        "version": CACHE_VERSION,
        "width": motion.width,
        "height": motion.height,
        "fps": motion.fps,
        "frame_count": int(len(motion.steps)),
        "steps": np.round(motion.steps, 6).tolist(),
    }
    tmp = path.with_suffix(".part")
    try:
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _similarity(
    tx: float = 0.0,
    ty: float = 0.0,
    angle: float = 0.0,
    log_scale: float = 0.0,
) -> np.ndarray:
    scale = math.exp(log_scale)
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _decompose(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        math.atan2(float(matrix[1, 0]), float(matrix[0, 0])),
        math.log(max(1e-6, math.hypot(float(matrix[0, 0]), float(matrix[1, 0])))),
    )


def _analyze(
    video_path: str,
    width: int,
    height: int,
    fps: float,
    *,
    progress: ProgressFn,
    cancelled: CancelFn,
) -> MotionData:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    est_scale = min(1.0, ANALYSIS_MAX_WIDTH / float(max(1, width)))
    ew = max(32, int(round(width * est_scale)))
    eh = max(32, int(round(height * est_scale)))
    hann = cv2.createHanningWindow((ew, eh), cv2.CV_32F)

    feature_params = dict(
        maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def read_small():
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != ew or gray.shape[0] != eh:
            gray = cv2.resize(gray, (ew, eh), interpolation=cv2.INTER_AREA)
        return frame, gray

    _, prev_u8 = read_small()
    if prev_u8 is None:
        cap.release()
        raise IOError("无法读取第一帧")

    prev_f32 = prev_u8.astype(np.float32) / 255.0
    steps = [[0.0, 0.0, 0.0, 0.0, 1.0]]
    failed = 0
    index = 0

    while True:
        if cancelled and cancelled():
            cap.release()
            raise RuntimeError("已取消")
        _, cur_u8 = read_small()
        if cur_u8 is None:
            break

        rel = None
        quality = 0.0
        p0 = cv2.goodFeaturesToTrack(prev_u8, mask=None, **feature_params)
        if p0 is not None and len(p0) >= 8:
            p1, st, _err = cv2.calcOpticalFlowPyrLK(
                prev_u8, cur_u8, p0, None, **lk_params
            )
            if p1 is not None:
                back, st2, _err2 = cv2.calcOpticalFlowPyrLK(
                    cur_u8, prev_u8, p1, None, **lk_params
                )
                if back is not None:
                    forward_back = np.linalg.norm(
                        back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1
                    )
                    good = (
                        (st.ravel() != 0)
                        & (st2.ravel() != 0)
                        & (forward_back < 1.5)
                    )
                    if int(good.sum()) >= 8:
                        a = p0.reshape(-1, 2)[good]
                        b = p1.reshape(-1, 2)[good]
                        fit, inliers = cv2.estimateAffinePartial2D(
                            a,
                            b,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.0,
                            maxIters=2000,
                            confidence=0.995,
                        )
                        if fit is not None and inliers is not None:
                            quality = float(inliers.mean()) * min(
                                1.0, len(a) / 40.0
                            )
                            if quality >= 0.25:
                                rel = _similarity(
                                    float(fit[0, 2]),
                                    float(fit[1, 2]),
                                    math.atan2(float(fit[1, 0]), float(fit[0, 0])),
                                    math.log(
                                        max(
                                            1e-6,
                                            math.hypot(
                                                float(fit[0, 0]), float(fit[1, 0])
                                            ),
                                        )
                                    ),
                                )

        if rel is None:
            shift, response = cv2.phaseCorrelate(
                prev_f32, cur_u8.astype(np.float32) / 255.0, hann
            )
            if response >= 0.08:
                rel = _similarity(float(shift[0]), float(shift[1]))
                quality = min(0.6, float(response))

        if rel is None:
            failed += 1
            steps.append([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            tx, ty, angle, log_scale = _decompose(rel)
            center = np.array([ew * 0.5, eh * 0.5, 1.0])
            moved = rel @ center
            displacement = float(
                math.hypot(moved[0] - center[0], moved[1] - center[1])
            )
            if (
                not 0.94 < math.exp(log_scale) < 1.06
                or abs(angle) > 0.25
                or displacement > 0.25 * max(ew, eh)
            ):
                failed += 1
                steps.append([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                steps.append(
                    [tx / est_scale, ty / est_scale, angle, log_scale, quality]
                )

        prev_u8 = cur_u8
        prev_f32 = cur_u8.astype(np.float32) / 255.0
        index += 1
        if progress and index % 30 == 0:
            progress(index, total or index)

    cap.release()
    if failed:
        _log(f"⚠ {failed} 帧运动估计失败（按无位移处理）", None)
    progress and progress(len(steps), total or len(steps))
    return MotionData(width, height, fps, np.asarray(steps, dtype=np.float64))


def load_motion(
    video_path: str,
    width: int,
    height: int,
    fps: float,
    *,
    identity_path: Optional[str] = None,
    range_key: str = "",
    log: LogFn = None,
    progress: ProgressFn = None,
    cancelled: CancelFn = None,
    force: bool = False,
) -> MotionData:
    """读取或测量相机运动。同一个原片 + 时间范围只测一次。"""
    identity = identity_path or video_path
    key = _cache_key(identity, width, height, fps, range_key)
    cache_path = _cache_dir() / f"{key}.json"
    frame_count = _frame_count(video_path)

    if not force:
        cached = _load_cache(cache_path, width, height, fps, frame_count)
        if cached is not None:
            _log(f"命中运动分析缓存（{len(cached.steps)} 帧，跳过测量）", log)
            return cached

    _log(f"估计相机运动：{width}x{height} @ {fps:.1f}fps", log)
    motion = _analyze(
        video_path, width, height, fps, progress=progress, cancelled=cancelled
    )
    if frame_count and abs(len(motion.steps) - frame_count) > 1:
        # 某些容器帧数不准，缓存仍以实际读到的帧为准
        pass
    _save_cache(cache_path, motion)
    return motion


def camera_path(steps: np.ndarray) -> np.ndarray:
    """把逐帧相对运动累乘成「参考帧 → 当前帧」的相机路径。"""
    n = len(steps)
    paths = np.empty((n, 3, 3), dtype=np.float64)
    if n == 0:
        return paths
    paths[0] = np.eye(3)
    for i in range(1, n):
        rel = _similarity(
            float(steps[i, 0]),
            float(steps[i, 1]),
            float(steps[i, 2]),
            float(steps[i, 3]),
        )
        paths[i] = rel @ paths[i - 1]
    return paths


def smooth_series(values: np.ndarray, sigma: float) -> np.ndarray:
    """零相位高斯平滑（sigma 以帧计），边缘按常数延拓，不引入延迟。"""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 3 or sigma <= 0.01:
        return arr.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:n]


def _correction_series(motion: MotionData, smooth_seconds: float) -> np.ndarray:
    steps = motion.steps
    paths = camera_path(steps)
    coords = np.empty((len(steps), 4), dtype=np.float64)
    for i, matrix in enumerate(paths):
        coords[i] = _decompose(matrix)
    coords[:, 2] = np.unwrap(coords[:, 2])

    sigma = max(0.5, min(600.0, float(smooth_seconds) * max(1.0, motion.fps)))
    target = np.column_stack(
        [smooth_series(coords[:, j], sigma) for j in range(4)]
    )

    correction = np.empty((len(steps), 4), dtype=np.float64)
    for i in range(len(steps)):
        smooth_path = _similarity(*target[i])
        corr = paths[i] @ np.linalg.inv(smooth_path)
        correction[i] = _decompose(corr)
    correction[:, 2] = np.unwrap(correction[:, 2])
    return correction


def _crop_rect(
    scale: float, width: int, height: int, safety: float
) -> Tuple[int, int, int, int]:
    keep = 1.0 - 2.0 * max(0.0, min(0.3, float(safety)))
    cw = max(32, int(width * scale * keep)) & ~1
    ch = max(32, int(height * scale * keep)) & ~1
    cw = min(cw, width & ~1)
    ch = min(ch, height & ~1)
    return cw, ch, (width - cw) // 2, (height - ch) // 2


def _fits(
    correction: np.ndarray,
    width: int,
    height: int,
    scale: float,
    safety: float,
) -> bool:
    cw, ch, ox, oy = _crop_rect(scale, width, height, safety)
    corners = np.array(
        [
            [ox, oy, 1.0],
            [ox + cw - 1, oy, 1.0],
            [ox + cw - 1, oy + ch - 1, 1.0],
            [ox, oy + ch - 1, 1.0],
        ],
        dtype=np.float64,
    )
    for tx, ty, angle, log_scale in correction:
        matrix = _similarity(tx, ty, angle, log_scale)
        mapped = corners @ matrix.T
        if (
            mapped[:, 0].min() < -1e-6
            or mapped[:, 1].min() < -1e-6
            or mapped[:, 0].max() > width - 1 + 1e-6
            or mapped[:, 1].max() > height - 1 + 1e-6
        ):
            return False
    return True


def _best_scale(
    correction: np.ndarray, width: int, height: int, safety: float
) -> float:
    if not _fits(correction, width, height, 1.0, safety):
        lo, hi = 0.05, 1.0
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            if _fits(correction, width, height, mid, safety):
                lo = mid
            else:
                hi = mid
        return lo
    return 1.0


def plan_stabilization(
    motion: MotionData,
    *,
    smooth_seconds: float = 1.5,
    strength: float = 1.0,
    max_zoom: float = 1.8,
    safety: float = 0.0,
) -> CropPlan:
    """平滑路径并规划固定裁切；晃动过大时先收小校正，避免黑边。"""
    steps = motion.steps
    if len(steps) == 0:
        raise ValueError("没有可用的运动数据")

    width, height = motion.width, motion.height
    correction = _correction_series(motion, smooth_seconds)
    correction *= max(0.0, min(1.0, float(strength)))

    limit_shift = 0.35 * min(width, height)
    correction[:, 0] = np.clip(correction[:, 0], -limit_shift, limit_shift)
    correction[:, 1] = np.clip(correction[:, 1], -limit_shift, limit_shift)
    correction[:, 2] = np.clip(correction[:, 2], -0.25, 0.25)
    correction[:, 3] = np.clip(correction[:, 3], -0.12, 0.12)

    min_scale = 1.0 / max(1.0, float(max_zoom))
    scale = _best_scale(correction, width, height, safety)

    applied = 1.0
    if scale < min_scale:
        # 校正幅度太大：整体向「不平滑」收缩，直到裁切不超过上限
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if _best_scale(correction * mid, width, height, safety) >= min_scale:
                lo = mid
            else:
                hi = mid
        applied = lo
        correction *= applied
        scale = _best_scale(correction, width, height, safety)

    while scale > 0.05 and not _fits(
        correction, width, height, scale, safety
    ):
        scale -= 0.005

    cw, ch, ox, oy = _crop_rect(scale, width, height, safety)
    matrices: List[np.ndarray] = []
    for tx, ty, angle, log_scale in correction:
        matrix = _similarity(tx, ty, angle, log_scale)
        offset = matrix[:2, :2] @ np.array([ox, oy], dtype=np.float64)
        matrices.append(
            np.array(
                [
                    [matrix[0, 0], matrix[0, 1], matrix[0, 2] + offset[0]],
                    [matrix[1, 0], matrix[1, 1], matrix[1, 2] + offset[1]],
                ],
                dtype=np.float64,
            )
        )

    motion_px = (
        float(np.ptp(correction[:, 0])) if len(correction) else 0.0,
        float(np.ptp(correction[:, 1])) if len(correction) else 0.0,
    )
    return CropPlan(
        out_w=cw,
        out_h=ch,
        matrices=matrices,
        zoom=width / float(max(1, cw)),
        strength=applied,
        motion_px=motion_px,
        smooth_seconds=float(smooth_seconds),
    )
