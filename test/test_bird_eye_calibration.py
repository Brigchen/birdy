# -*- coding: utf-8 -*-
"""鸟眼关键帧校准：眼-锚偏移、失锁找回与模型缺失降级。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_anchor import FrameLayout, track_layouts_features  # noqa: E402
from bird_eye_calibration import BirdEyeCalibrator  # noqa: E402

W, H = 320, 240
ANCHOR = (160.0, 120.0)
FIRST = FrameLayout(ax=0.5, ay=0.5, x0=0.3, y0=0.3, x1=0.7, y1=0.7)
EYE_OFF = (25.0, 12.0)


def _texture() -> np.ndarray:
    rng = np.random.RandomState(21)
    return cv2.GaussianBlur(
        rng.randint(0, 256, (H, W, 3), dtype=np.uint8), (5, 5), 0
    )


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _mark(frame: np.ndarray, idx: int) -> np.ndarray:
    """在左上角写入帧号，假眼检测器用它查真值。"""
    out = frame.copy()
    out[0, 0, 0] = 10 + idx
    return out


def _make_eye_fn(eye_positions, missing=()):
    def detect(frame, cx, cy, radius):
        idx = int(frame[0, 0, 0]) - 10
        if idx in missing or idx < 0 or idx >= len(eye_positions):
            return None
        ex, ey = eye_positions[idx]
        return ex, ey, 0.9

    return detect


def _err(lay: FrameLayout, px: float, py: float) -> float:
    return float(np.hypot(lay.ax * W - px, lay.ay * H - py))


def test_eye_calibration_keeps_anchor_semantics_and_reacquires():
    bg = _texture()
    shifts = [(0, 0), (5, 3), (10, 6), (15, 9), (20, 12), (25, 15)]
    frames = []
    eye_positions = []
    for i, (dx, dy) in enumerate(shifts):
        frames.append(_mark(_shift(bg, dx, dy), i))
        eye_positions.append((ANCHOR[0] + dx + EYE_OFF[0], ANCHOR[1] + dy + EYE_OFF[1]))
    # 第 3 帧起整帧遮挡，只有鸟眼还能"看到"目标
    for i in (3, 4, 5):
        frames[i] = _mark(np.full((H, W, 3), 110, dtype=np.uint8), i)

    eye_fn = _make_eye_fn(eye_positions)
    out = track_layouts_features(
        frames,
        [FIRST] + [None] * (len(frames) - 1),
        detect_eye_fn=eye_fn,
        calibrate_every=2,
        calibrate_on_lost=True,
    )
    # 眼在锚点右侧 25px：校准后锚点仍应对准真值，而不是跳到眼上
    for i, (dx, dy) in enumerate(shifts):
        assert not out[i].lost, f"第 {i} 帧应被眼校准找回"
        assert _err(out[i], ANCHOR[0] + dx, ANCHOR[1] + dy) < 4.0, (
            f"第 {i} 帧偏移未正确换算：{out[i].ax * W:.1f},{out[i].ay * H:.1f}"
        )

    # 同样的序列关闭眼校准：遮挡后必然失锁
    out_no_eye = track_layouts_features(
        frames, [FIRST] + [None] * (len(frames) - 1)
    )
    assert any(lay.lost for lay in out_no_eye[3:]), "无眼校准时遮挡应失锁"


def test_eye_calibration_can_start_later():
    """首帧没检出眼时，后续校准帧仍能学习偏移并工作。"""
    bg = _texture()
    shifts = [(0, 0), (5, 3), (10, 6), (15, 9), (20, 12)]
    frames = []
    eye_positions = []
    for i, (dx, dy) in enumerate(shifts):
        frames.append(_mark(_shift(bg, dx, dy), i))
        eye_positions.append((ANCHOR[0] + dx + EYE_OFF[0], ANCHOR[1] + dy + EYE_OFF[1]))

    eye_fn = _make_eye_fn(eye_positions, missing=(0,))
    out = track_layouts_features(
        frames,
        [FIRST] + [None] * (len(frames) - 1),
        detect_eye_fn=eye_fn,
        calibrate_every=1,
    )
    for i, (dx, dy) in enumerate(shifts):
        assert not out[i].lost
        assert _err(out[i], ANCHOR[0] + dx, ANCHOR[1] + dy) < 4.0


def test_merge_propagate_passes_eye_calibrator():
    """传播入口的参数能正确透传到锁眼校准。"""
    from burst_anchor import merge_propagate

    bg = _texture()
    shifts = [(0, 0), (5, 3), (10, 6), (15, 9)]
    frames = []
    eye_positions = []
    for i, (dx, dy) in enumerate(shifts):
        frames.append(_mark(_shift(bg, dx, dy), i))
        eye_positions.append(
            (ANCHOR[0] + dx + EYE_OFF[0], ANCHOR[1] + dy + EYE_OFF[1])
        )
    frames[2] = _mark(np.full((H, W, 3), 90, dtype=np.uint8), 2)

    out = merge_propagate(
        frames,
        [FIRST, None, None, None],
        mode="fixed",
        use_feature_lock=True,
        detect_eye_fn=_make_eye_fn(eye_positions),
        eye_calibrate_every=2,
    )
    assert len(out) == 4
    assert not out[2].lost
    assert _err(out[2], ANCHOR[0] + 10, ANCHOR[1] + 6) < 4.0


def test_calibrator_degrades_when_model_missing(tmp_path: Path):
    calib = BirdEyeCalibrator(model_path=str(tmp_path / "no_such_eye.pt"))
    assert calib.available() is False
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert calib.detect_anchor(img, 32, 32, 40) is None
