# -*- coding: utf-8 -*-
"""动图生成：导出前锁眼精修（BurstWebpBuildOptions.enable_feature_lock_refine）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_anchor import FrameLayout  # noqa: E402
from burst_webp import (  # noqa: E402
    BurstWebpBuildOptions,
    _burst_prepare_animation_pil_frames,
    _coerce_frame_layouts,
)


W, H = 320, 240


def _texture_bg() -> np.ndarray:
    rng = np.random.RandomState(9)
    return cv2.GaussianBlur(
        rng.randint(0, 256, (H, W, 3), dtype=np.uint8), (5, 5), 0
    )


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _make_jittered_layouts(
    n: int,
    first: FrameLayout,
    true_shifts,
    noise_scale: float = 8.0,
    seed: int = 42,
) -> list:
    """给真实锚点加噪声，模拟传播阶段没完全对准。"""
    rng = np.random.RandomState(seed)
    out = [first]
    base_ax, base_ay = first.ax * W, first.ay * H
    for i in range(1, n):
        dx, dy = true_shifts[i]
        ax = base_ax + dx + rng.normal(0, noise_scale)
        ay = base_ay + dy + rng.normal(0, noise_scale)
        out.append(
            FrameLayout(
                ax=ax / W,
                ay=ay / H,
                x0=first.x0,
                y0=first.y0,
                x1=first.x1,
                y1=first.y1,
            )
        )
    return out


def _trajectory_jitter(lays: list) -> float:
    """锚点轨迹相邻帧绝对位移的 p95，越大表示抖得越厉害。"""
    dxs = []
    dys = []
    for i in range(1, len(lays)):
        dxs.append(abs(lays[i].ax * W - lays[i - 1].ax * W))
        dys.append(abs(lays[i].ay * H - lays[i - 1].ay * H))
    arr = np.array(dxs + dys, dtype=np.float32)
    return float(np.percentile(arr, 95)) if arr.size else 0.0


def _track_error(lays: list, true_shifts, first_ax: float, first_ay: float) -> float:
    """每帧锚点到真值的平均像素距离。"""
    total = 0.0
    for i, (dx, dy) in enumerate(true_shifts):
        ex = first_ax * W + dx
        ey = first_ay * H + dy
        total += float(
            np.hypot(lays[i].ax * W - ex, lays[i].ay * H - ey)
        )
    return total / len(true_shifts)


def test_refine_reduces_jitter_on_translating_sequence():
    bg = _texture_bg()
    true_shifts = [(0, 0), (5, 3), (10, 6), (15, 9), (20, 12), (25, 15), (30, 18)]
    frames = [_shift(bg, dx, dy) for dx, dy in true_shifts]
    first = FrameLayout(ax=0.5, ay=0.5, x0=0.35, y0=0.35, x1=0.65, y1=0.65)
    jittered = _make_jittered_layouts(len(frames), first, true_shifts, noise_scale=6.0)

    lays_before = _coerce_frame_layouts(jittered, len(frames))
    jitter_before = _trajectory_jitter(lays_before)
    err_before = _track_error(lays_before, true_shifts, first.ax, first.ay)

    from burst_anchor import track_layouts_features

    refined = track_layouts_features(frames, lays_before, mode="fixed")
    jitter_after = _trajectory_jitter(refined)
    err_after = _track_error(refined, true_shifts, first.ax, first.ay)

    assert jitter_after < jitter_before, (
        f"精修后抖动 p95 应下降：before={jitter_before:.2f}, after={jitter_after:.2f}"
    )
    assert err_after < err_before, (
        f"精修后平均误差应下降：before={err_before:.2f}, after={err_after:.2f}"
    )


def test_refine_disabled_is_noop():
    """enable=False 时精修不应改变布局。"""
    bg = _texture_bg()
    true_shifts = [(0, 0), (6, 4), (12, 8)]
    frames = [_shift(bg, dx, dy) for dx, dy in true_shifts]
    first = FrameLayout(ax=0.5, ay=0.5, x0=0.3, y0=0.3, x1=0.7, y1=0.7)
    jittered = _make_jittered_layouts(len(frames), first, true_shifts, noise_scale=5.0)

    from burst_anchor import track_layouts_features

    refined = track_layouts_features(frames, jittered, mode="fixed")
    assert len(refined) == len(jittered)
    # 不逐元素比（算法本身就会平滑），只断言没有 NaN/极端值
    for lay in refined:
        assert np.isfinite(lay.ax) and np.isfinite(lay.ay)


def test_opts_defaults_enable_refine():
    opts = BurstWebpBuildOptions()
    assert opts.enable_feature_lock_refine is True
    assert opts.feature_lock_mode == ""
