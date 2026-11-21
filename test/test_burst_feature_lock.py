# -*- coding: utf-8 -*-
"""连拍自动锁眼：局部特征跟踪、遮挡缺口与失锁标记。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_anchor import (  # noqa: E402
    FrameLayout,
    merge_propagate,
    track_layouts_features,
)

W, H = 320, 240
ANCHOR = (160.0, 120.0)
FIRST = FrameLayout(ax=0.5, ay=0.5, x0=0.35, y0=0.35, x1=0.65, y1=0.65)


def _texture() -> np.ndarray:
    rng = np.random.RandomState(7)
    img = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    m = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        img,
        m,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        m,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple:
    m = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    px = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    py = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return float(px), float(py)


def _err_norm(lay: FrameLayout, px: float, py: float) -> float:
    return float(np.hypot(lay.ax * W - px, lay.ay * H - py))


def test_feature_lock_follows_translation():
    base = _texture()
    shifts = [(0, 0), (6, 4), (12, 8), (18, 12), (24, 16)]
    frames = [_shift(base, dx, dy) for dx, dy in shifts]
    out = track_layouts_features(frames, [FIRST] + [None] * (len(frames) - 1))

    assert len(out) == len(frames)
    assert out[0].auto is False
    for i, (dx, dy) in enumerate(shifts):
        assert not out[i].lost, f"第 {i} 帧不应失锁"
        assert out[i].auto is True or i == 0
        assert _err_norm(out[i], ANCHOR[0] + dx, ANCHOR[1] + dy) < 3.0
    assert min(lay.conf for lay in out[1:]) > 0.4


def test_feature_lock_follows_rotation():
    base = _texture()
    angles = [0, 2, 4, 6, 8]
    frames = [_rotate(base, a) for a in angles]
    # 锚点偏离画面中心，旋转才会产生位移
    seed = FrameLayout(ax=0.60, ay=0.40, x0=0.45, y0=0.25, x1=0.75, y1=0.55)
    px0, py0 = seed.ax * W, seed.ay * H
    out = track_layouts_features(frames, [seed] + [None] * (len(frames) - 1))

    for i, a in enumerate(angles):
        ex, ey = _rotate_point(px0, py0, a)
        assert not out[i].lost, f"旋转第 {i} 帧不应失锁"
        assert _err_norm(out[i], ex, ey) < 4.0


def test_feature_lock_follows_moving_bird_on_static_background():
    """定点模式：相机不动，裁剪框贴着鸟，锁眼应跟着鸟走而不是跟着背景。"""
    rng = np.random.RandomState(3)
    w, h = 640, 480
    bg = cv2.GaussianBlur(rng.randint(0, 256, (h, w, 3), dtype=np.uint8), (5, 5), 0)
    bird = cv2.GaussianBlur(rng.randint(0, 256, (80, 110, 3), dtype=np.uint8), (3, 3), 0)
    shifts = [(0, 0), (7, 5), (14, 10), (21, 15)]
    frames = []
    for dx, dy in shifts:
        f = bg.copy()
        x0, y0 = 250 + dx, 180 + dy
        f[y0 : y0 + 80, x0 : x0 + 110] = bird
        frames.append(f)
    first = FrameLayout(
        ax=305.0 / w,
        ay=220.0 / h,
        x0=245.0 / w,
        y0=175.0 / h,
        x1=360.0 / w,
        y1=265.0 / h,
    )
    out = track_layouts_features(frames, [first] + [None] * 3)
    for i, (dx, dy) in enumerate(shifts):
        assert not out[i].lost
        assert abs(out[i].ax * w - (305 + dx)) < 3.0
        assert abs(out[i].ay * h - (220 + dy)) < 3.0


def test_feature_lock_interpolates_short_occlusion():
    base = _texture()
    shifts = [(0, 0), (6, 4), (12, 8), (18, 12)]
    frames = [_shift(base, dx, dy) for dx, dy in shifts]
    frames[2] = np.full((H, W, 3), 128, dtype=np.uint8)  # 第 3 帧整帧被遮挡
    out = track_layouts_features(frames, [FIRST] + [None] * (len(frames) - 1))

    # 中间缺口 1 帧：应线性插值并恢复，不再标记失锁
    assert out[2].lost is False
    assert abs(out[2].conf - 0.5) < 1e-6
    ex, ey = ANCHOR[0] + 12, ANCHOR[1] + 8
    assert _err_norm(out[2], ex, ey) < 4.0
    # 缺口之后的帧应重新锁上
    assert not out[3].lost
    assert _err_norm(out[3], ANCHOR[0] + 18, ANCHOR[1] + 12) < 4.0


def test_feature_lock_marks_long_occlusion_lost():
    base = _texture()
    shifts = [(0, 0), (6, 4), (12, 8), (18, 12), (24, 16), (30, 20), (36, 24)]
    frames = [_shift(base, dx, dy) for dx, dy in shifts]
    for i in range(1, 5):
        frames[i] = np.full((H, W, 3), 96, dtype=np.uint8)

    out = track_layouts_features(frames, [FIRST] + [None] * (len(frames) - 1))
    assert all(lay.lost for lay in out[1:]), "连续丢失超过 3 帧后应全部标记失锁"
    assert all(lay.auto for lay in out[1:])
    # 失锁帧仍给出可导出的固定尺寸裁剪框
    for lay in out[1:]:
        assert abs((lay.x1 - lay.x0) * W - (FIRST.x1 - FIRST.x0) * W) < 2.0
        assert abs((lay.y1 - lay.y0) * H - (FIRST.y1 - FIRST.y0) * H) < 2.0


def test_feature_lock_respects_manual_lock_and_reseeds():
    base = _texture()
    shifts = [(0, 0), (6, 4), (12, 8), (18, 12)]
    frames = [_shift(base, dx, dy) for dx, dy in shifts]
    manual = FrameLayout(
        ax=(ANCHOR[0] + 12) / W,
        ay=(ANCHOR[1] + 8) / H,
        x0=0.30,
        y0=0.30,
        x1=0.70,
        y1=0.70,
        auto=False,
        conf=1.0,
    )
    slots = [FIRST, None, manual, None]
    out = track_layouts_features(frames, slots)

    assert out[2].auto is False
    assert abs(out[2].ax - manual.ax) < 1e-9
    assert not out[2].lost
    # 手动锁定页作为新种子，后续帧继续跟随
    assert not out[3].lost
    assert _err_norm(out[3], ANCHOR[0] + 18, ANCHOR[1] + 12) < 4.0


def test_merge_propagate_feature_lock_dispatch():
    base = _texture()
    shifts = [(0, 0), (5, 3), (10, 6)]
    frames = [_shift(base, dx, dy) for dx, dy in shifts]
    out = merge_propagate(
        frames,
        [FIRST, None, None],
        mode="fixed",
        use_feature_lock=True,
    )
    assert len(out) == 3
    assert abs(out[2].ax - (ANCHOR[0] + 10) / W) < 0.02
    assert abs(out[2].ay - (ANCHOR[1] + 6) / H) < 0.02


def test_frame_layout_lost_roundtrip():
    lay = FrameLayout(ax=0.4, ay=0.6, lost=True, conf=0.2)
    back = FrameLayout.from_dict(lay.to_dict())
    assert back.lost is True
    assert abs(back.conf - 0.2) < 1e-9
