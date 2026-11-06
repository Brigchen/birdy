# -*- coding: utf-8 -*-
"""三脚架模式：平移晃动应锁回同一视角，且边缘不是纯黑。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "video-stabilizer"
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))

from video_stab.tripod import (  # noqa: E402
    _fit_strips,
    _output_matrix,
    measure_row_field,
    render_stabilized_frame,
    row_lock_margin,
    run_virtual_tripod,
    smooth_residual_jitter,
    stable_crop_box,
    tame_row_field,
)


def test_tripod_locks_translation_without_black_border(tmp_path: Path):
    rng = np.random.default_rng(1)
    scene = rng.integers(0, 255, size=(300, 520, 3), dtype=np.uint8)
    cv2.rectangle(scene, (200, 120), (280, 190), (230, 230, 230), -1)
    width, height = 320, 180
    src = tmp_path / "shake.mp4"
    dst = tmp_path / "locked.mp4"
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
    )
    assert writer.isOpened()
    for i in range(24):
        dx = int(round(28 * math.sin(2 * math.pi * i / 24)))
        dy = int(round(16 * math.cos(2 * math.pi * i / 11)))
        writer.write(scene[30 + dy : 30 + dy + height, 60 + dx : 60 + dx + width])
    writer.release()

    out_w, out_h = run_virtual_tripod(str(src), str(dst))
    assert out_w >= width - 70
    assert out_h >= height - 50
    assert (out_w * out_h) / float(width * height) < 0.95

    cap = cv2.VideoCapture(str(dst))
    ok, ref = cap.read()
    assert ok
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shifts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert float(np.mean(np.all(frame[:2] == 0, axis=2))) < 0.02
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift, _resp = cv2.phaseCorrelate(ref_g, gray)
        shifts.append(shift)
    cap.release()
    arr = np.asarray(shifts)
    assert arr.shape[0] >= 20
    assert float(np.max(np.abs(arr))) < 1.0


def test_residual_smoother_cuts_high_frequency_and_keeps_drift(tmp_path: Path):
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, size=(180, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 30), (220, 130), (240, 180, 40), -1)
    src = tmp_path / "still.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    assert writer.isOpened()
    for _ in range(36):
        writer.write(frame)
    writer.release()

    motions = []
    for i in range(36):
        tx = 5.0 * math.sin(2 * math.pi * i / 3.0) + 0.08 * i
        ty = 4.0 * math.cos(2 * math.pi * i / 3.0)
        motions.append(np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64))

    smoothed = smooth_residual_jitter(str(src), motions, 320, 180, log=lambda _msg: None)
    left, top, out_w, out_h = stable_crop_box(smoothed, 320, 180)
    cap = cv2.VideoCapture(str(src))
    warped = []
    for M in smoothed:
        ok, image = cap.read()
        assert ok and image is not None
        warped.append(
            cv2.warpAffine(
                image,
                _output_matrix(M, float(left), float(top)),
                (out_w, out_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    cap.release()

    def gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ref = gray(warped[0])
    vs0 = []
    steps = []
    prev = ref
    for image in warped[1:]:
        cur = gray(image)
        shift, _resp = cv2.phaseCorrelate(ref, cur)
        vs0.append((float(shift[0]), float(shift[1])))
        step, _resp = cv2.phaseCorrelate(prev, cur)
        steps.append(math.hypot(float(step[0]), float(step[1])))
        prev = cur
    vs0_arr = np.asarray(vs0)
    step_arr = np.asarray(steps)
    assert float(np.percentile(step_arr, 95)) < 0.6
    assert float(np.ptp(vs0_arr[:, 0])) > 1.0


def test_row_lock_pulls_sheared_rows_back(tmp_path: Path):
    rng = np.random.default_rng(3)
    height, width = 160, 240
    scene = rng.integers(0, 255, size=(height, width + 48, 3), dtype=np.uint8)
    for y in range(0, height, 8):
        scene[y : y + 2, :] = (240, 220, 80)
    src = tmp_path / "shear.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    assert writer.isOpened()
    amplitudes = []
    for i in range(16):
        amp = 6.0 * math.sin(2 * math.pi * i / 5.0)
        amplitudes.append(amp)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        for y in range(height):
            shift = int(round(amp * ((y / (height - 1)) - 0.5)))
            frame[y] = scene[y, 24 + shift : 24 + shift + width]
        writer.write(frame)
    writer.release()

    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    motions = [eye.copy() for _ in range(16)]
    coeffs = measure_row_field(str(src), motions, width, height, 0, 0, width, height)
    margin = row_lock_margin(coeffs)
    assert margin > 0
    out_w = (width - 2 * margin) & ~1
    out_h = (height - 2 * margin) & ~1
    cap = cv2.VideoCapture(str(src))
    ok, first = cap.read()
    assert ok
    locked = [
        render_stabilized_frame(first, motions[0], 0, 0, out_w, out_h, coeffs[0], margin, height)
    ]
    for index, M in enumerate(motions[1:], start=1):
        ok, frame = cap.read()
        assert ok
        locked.append(
            render_stabilized_frame(frame, M, 0, 0, out_w, out_h, coeffs[index], margin, height)
        )
    cap.release()

    def band_shift(image: np.ndarray, ref: np.ndarray, y0: float, y1: float) -> float:
        h, w = ref.shape[:2]
        a = cv2.cvtColor(ref[int(h * y0) : int(h * y1)], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(image[int(h * y0) : int(h * y1)], cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift, _resp = cv2.phaseCorrelate(a, b)
        return float(shift[0])

    ref = locked[0]
    gaps = []
    for image in locked[1:]:
        top = band_shift(image, ref, 0.05, 0.25)
        bottom = band_shift(image, ref, 0.75, 0.95)
        gaps.append(abs(top - bottom))
    assert float(np.percentile(gaps, 95)) < 2.0


def test_fast_row_fit_ignores_sky_and_stays_straight():
    y = np.linspace(-1.0, 1.0, 8)
    dx = 5.0 * y
    dx[0] = 18.0
    weight = np.ones(8)
    slow = _fit_strips(y, dx, np.zeros(8), weight, fast=False)
    fast = _fit_strips(y, dx, np.zeros(8), weight, fast=True)
    assert abs(float(fast[0, 0])) < 1e-6
    assert abs(float(fast[0, 1]) - 5.0) < 1.0
    assert abs(float(slow[0, 0])) > abs(float(fast[0, 0]))
    capped = np.zeros((2, 2, 3))
    capped[:, 0, 1] = 20.0
    tamed = tame_row_field(capped, [], 100, 100)
    assert abs(float(tamed[0, 0, 1])) <= 18.0
