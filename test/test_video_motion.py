# -*- coding: utf-8 -*-
"""运动分析缓存与固定裁切规划。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from video_motion import (  # noqa: E402
    MotionData,
    load_motion,
    plan_stabilization,
    smooth_series,
)


def _synthetic_motion(
    width: int = 640,
    height: int = 360,
    count: int = 80,
    jitter: float = 20.0,
    angle: float = 0.005,
) -> MotionData:
    steps = np.zeros((count, 5), dtype=np.float64)
    steps[:, 4] = 1.0
    for i in range(1, count):
        steps[i, 0] = jitter * (math.sin(i * 0.9) - math.sin((i - 1) * 0.9))
        steps[i, 1] = jitter * (math.cos(i * 1.3) - math.cos((i - 1) * 1.3))
        steps[i, 2] = angle * (1 if i % 7 == 0 else -1 if i % 7 == 3 else 0)
    return MotionData(width, height, 30.0, steps)


def _inside(matrices, out_w, out_h, width, height) -> bool:
    corners = np.array(
        [[0, 0, 1], [out_w - 1, 0, 1], [out_w - 1, out_h - 1, 1], [0, out_h - 1, 1]],
        dtype=np.float64,
    )
    for matrix in matrices:
        mapped = corners @ matrix.T
        if mapped[:, 0].min() < -1e-6 or mapped[:, 1].min() < -1e-6:
            return False
        if mapped[:, 0].max() > width - 1 + 1e-6:
            return False
        if mapped[:, 1].max() > height - 1 + 1e-6:
            return False
    return True


def test_smooth_series_cuts_jitter_keeps_drift():
    index = np.arange(120, dtype=np.float64)
    raw = 0.35 * index + 4.0 * np.sin(index * math.pi / 2.0)
    smoothed = smooth_series(raw, 6.0)
    raw_step = np.diff(raw)
    smoothed_step = np.diff(smoothed)
    assert float(np.std(smoothed_step)) < float(np.std(raw_step)) * 0.2
    assert abs(float(np.mean(smoothed_step)) - 0.35) < 0.06
    assert abs(float(smoothed[-1] - raw[-1])) < 2.0
    assert abs(float(smoothed[0] - raw[0])) < 2.0


def test_plan_produces_black_free_fixed_crop():
    motion = _synthetic_motion()
    plan = plan_stabilization(motion, smooth_seconds=1.4, max_zoom=1.8)
    assert plan.out_w < motion.width and plan.out_h < motion.height
    assert plan.zoom > 1.0
    assert len(plan.matrices) == len(motion.steps)
    assert _inside(
        plan.matrices,
        plan.out_w,
        plan.out_h,
        motion.width,
        motion.height,
    )


def test_plan_limits_crop_when_shake_is_large():
    motion = _synthetic_motion(jitter=140.0, angle=0.02)
    plan = plan_stabilization(motion, smooth_seconds=1.4, max_zoom=1.2)
    assert plan.out_w >= int(motion.width / 1.2) - 4
    assert plan.out_h >= int(motion.height / 1.2) - 4
    assert plan.strength < 1.0
    assert _inside(
        plan.matrices,
        plan.out_w,
        plan.out_h,
        motion.width,
        motion.height,
    )


def test_motion_cache_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BIRDY_MOTION_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, size=(140, 220, 3), dtype=np.uint8)
    cv2.rectangle(scene, (70, 40), (140, 110), (240, 220, 60), -1)
    video = tmp_path / "shake.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 120)
    )
    assert writer.isOpened()
    for i in range(14):
        dx = int(round(6 * math.sin(i)))
        writer.write(scene[10:130, 10 + dx : 10 + dx + 200])
    writer.release()

    first = load_motion(str(video), 200, 120, 10.0)
    assert not first.cached
    assert first.steps.shape[1] == 5
    assert len(first.steps) >= 12

    second = load_motion(str(video), 200, 120, 10.0)
    assert second.cached
    assert np.allclose(first.steps, second.steps, atol=1e-5)
