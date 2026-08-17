# -*- coding: utf-8 -*-
"""动图定点 / 跟踪：模板传播、rel 裁剪几何与越界补边。"""

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
    crop_bgr_with_pad,
    geom_from_first,
    in_bounds_crop_xyxy,
    layout_from_anchor,
    merge_propagate,
    meter_box_in_padded_crop,
    propagate_layouts,
)


def test_geom_rel_roundtrip():
    lay = FrameLayout(ax=0.50, ay=0.40, x0=0.20, y0=0.10, x1=0.80, y1=0.70)
    w, h = 200, 100
    geom = geom_from_first(lay, w, h)
    assert abs(geom.rel_x - (0.50 - 0.20) / 0.60) < 1e-6
    assert abs(geom.rel_y - (0.40 - 0.10) / 0.60) < 1e-6
    assert geom.crop_w == 120
    assert geom.crop_h == 60
    back = layout_from_anchor(lay.ax, lay.ay, geom, w, h)
    assert abs(back.x0 - lay.x0) < 1e-3
    assert abs(back.y0 - lay.y0) < 1e-3
    assert abs(back.x1 - lay.x1) < 1e-3
    assert abs(back.y1 - lay.y1) < 1e-3


def test_crop_pad_when_anchor_near_corner():
    img = np.full((100, 120, 3), 40, dtype=np.uint8)
    first = FrameLayout(ax=0.5, ay=0.5, x0=0.25, y0=0.25, x1=0.75, y1=0.75)
    geom = geom_from_first(first, 120, 100)
    corner = layout_from_anchor(0.02, 0.03, geom, 120, 100)
    assert corner.x0 < 0.0 or corner.y0 < 0.0
    crop = crop_bgr_with_pad(img, corner, geom)
    assert crop.shape[1] == geom.crop_w
    assert crop.shape[0] == geom.crop_h
    # 越界补边为黑，角落应出现黑色像素
    assert int(crop[0, 0].max()) == 0


def test_fixed_mode_follows_scene_shift():
    rng = np.random.RandomState(1)
    bg = rng.randint(40, 200, (180, 240, 3), dtype=np.uint8)
    bg[40:72, 50:82] = (0, 255, 0)
    shift_x, shift_y = 14, 9
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    f2 = cv2.warpAffine(bg, M, (240, 180), borderValue=(0, 0, 0))
    ax0, ay0 = 66 / 240.0, 56 / 180.0
    first = FrameLayout(ax=ax0, ay=ay0, x0=0.12, y0=0.12, x1=0.72, y1=0.72)
    out = propagate_layouts([bg, f2], first, mode="fixed")
    assert len(out) == 2
    assert abs(out[1].ax - (66 + shift_x) / 240.0) < 0.04
    assert abs(out[1].ay - (56 + shift_y) / 180.0) < 0.04
    geom = geom_from_first(first, 240, 180)
    assert abs(out[1].x1 - out[1].x0 - geom.crop_w / 240.0) < 0.02


def test_track_mode_follows_moving_subject():
    frames = []
    xs = [36, 72, 110]
    for x in xs:
        img = np.zeros((160, 200, 3), dtype=np.uint8)
        img[58:92, x : x + 32] = (0, 0, 255)
        frames.append(img)
    ax0 = (xs[0] + 16) / 200.0
    ay0 = 75 / 160.0
    first = FrameLayout(ax=ax0, ay=ay0, x0=0.05, y0=0.20, x1=0.45, y1=0.70)
    out = propagate_layouts(frames, first, mode="track")
    assert len(out) == 3
    assert out[2].ax > out[0].ax + 0.15
    assert abs(out[2].ax - (xs[2] + 16) / 200.0) < 0.08


def test_later_frame_click_places_crop_from_first_geom():
    """后续帧只给标定点时，裁剪框相对位置与尺寸与首图一致，并锁定。"""
    first = FrameLayout(ax=0.50, ay=0.40, x0=0.20, y0=0.10, x1=0.80, y1=0.70)
    w, h = 200, 100
    geom = geom_from_first(first, w, h)
    later = layout_from_anchor(0.60, 0.50, geom, w, h, auto=False, conf=1.0)
    assert later.auto is False
    assert abs((later.x1 - later.x0) * w - geom.crop_w) < 1.5
    assert abs((later.y1 - later.y0) * h - geom.crop_h) < 1.5
    assert abs((0.60 - later.x0) / (later.x1 - later.x0) - geom.rel_x) < 1e-3
    assert abs((0.50 - later.y0) / (later.y1 - later.y0) - geom.rel_y) < 1e-3
    seed = [first, None, None]
    seed[1] = later
    frames = [
        np.zeros((h, w, 3), dtype=np.uint8),
        np.zeros((h, w, 3), dtype=np.uint8),
        np.zeros((h, w, 3), dtype=np.uint8),
    ]
    frames[0][30:50, 80:110] = 200
    merged = merge_propagate(frames, seed, mode="fixed")
    assert merged[1].auto is False
    assert abs(merged[1].ax - 0.60) < 1e-9
    assert abs(merged[1].x0 - later.x0) < 1e-9
    assert merged[2].auto is True


def test_merge_propagate_keeps_locked_frame():
    rng = np.random.RandomState(2)
    a = rng.randint(0, 255, (80, 100, 3), dtype=np.uint8)
    a[20:40, 30:50] = 255
    b = np.roll(a, 8, axis=1)
    first = FrameLayout(ax=0.40, ay=0.375, x0=0.10, y0=0.10, x1=0.70, y1=0.70)
    locked = FrameLayout(
        ax=0.90, ay=0.90, x0=0.60, y0=0.60, x1=0.99, y1=0.99, auto=False
    )
    merged = merge_propagate([a, b], [first, locked], mode="fixed")
    assert abs(merged[1].ax - 0.90) < 1e-9
    assert merged[1].auto is False


def test_merge_track_mid_lock_does_not_drift():
    frames = []
    xs = [36, 72, 110]
    for x in xs:
        img = np.zeros((160, 200, 3), dtype=np.uint8)
        img[58:92, x : x + 32] = (0, 0, 255)
        frames.append(img)
    first = FrameLayout(
        ax=(xs[0] + 16) / 200.0,
        ay=75 / 160.0,
        x0=0.05,
        y0=0.20,
        x1=0.45,
        y1=0.70,
    )
    locked = FrameLayout(
        ax=0.20, ay=0.80, x0=0.05, y0=0.50, x1=0.40, y1=0.95, auto=False
    )
    merged = merge_propagate(frames, [first, locked, None], mode="track")
    assert merged[1].auto is False
    assert abs(merged[1].ax - 0.20) < 1e-9
    assert abs(merged[1].ay - 0.80) < 1e-9


def test_prepare_fps_and_layout_crop():
    import tempfile

    from PIL import Image

    from burst_webp import BurstWebpBuildOptions, _burst_prepare_animation_pil_frames

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            arr = np.zeros((80, 120, 3), dtype=np.uint8)
            arr[20:50, 30 + i * 8 : 60 + i * 8] = (180, 40, 20)
            fp = str(Path(td) / f"f{i}.png")
            Image.fromarray(arr).save(fp)
            paths.append(fp)
        bgrs = [cv2.imread(p) for p in paths]
        first = FrameLayout(ax=0.375, ay=0.4375, x0=0.10, y0=0.10, x1=0.70, y1=0.80)
        lays = propagate_layouts(bgrs, first, mode="fixed")
        opts = BurstWebpBuildOptions(
            enable_white_balance=False,
            enable_auto_exposure=False,
            mode="fixed",
            fps=2.0,
            frame_layouts=lays,
            max_long_edge=64,
        )
        frames, _, meta = _burst_prepare_animation_pil_frames(
            paths, opts, None, False, "[test]"
        )
        assert len(frames) == 2
        assert abs(float(meta["frame_duration_ms"]) - 500.0) < 1e-6
        assert abs(float(meta["fps"]) - 2.0) < 1e-6
        assert frames[0].size[0] > 1 and frames[0].size[1] > 1


def test_prepare_progress_skips_wb_message_when_disabled():
    import tempfile

    from PIL import Image

    from burst_webp import BurstWebpBuildOptions, _burst_prepare_animation_pil_frames

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            arr = np.zeros((40, 48, 3), dtype=np.uint8)
            arr[8:28, 8 + i * 4 : 28 + i * 4] = (40, 80, 160)
            fp = str(Path(td) / f"f{i}.png")
            Image.fromarray(arr).save(fp)
            paths.append(fp)
        first = FrameLayout(ax=0.5, ay=0.5, x0=0.15, y0=0.15, x1=0.85, y1=0.85)
        opts = BurstWebpBuildOptions(
            enable_white_balance=False,
            enable_auto_exposure=False,
            mode="fixed",
            fps=2.0,
            frame_layouts=[first, first],
            max_long_edge=32,
        )
        msgs: list = []

        def progress(_c, _t, msg: str) -> None:
            msgs.append(msg)

        _burst_prepare_animation_pil_frames(paths, opts, progress, False, "[test]")
        joined = "\n".join(msgs)
        assert "首张统计白平衡" not in joined
        assert "跳过白平衡" not in joined
        assert any("按布局裁剪" in m for m in msgs)


def test_in_bounds_crop_xyxy_matches_first_geom():
    first = FrameLayout(ax=0.5, ay=0.5, x0=0.25, y0=0.25, x1=0.75, y1=0.75)
    geom = geom_from_first(first, 120, 100)
    box = in_bounds_crop_xyxy(first, geom, 120, 100)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 >= 0 and y0 >= 0 and x1 <= 120 and y1 <= 100
    assert abs((x1 - x0) - geom.crop_w) <= 1
    assert abs((y1 - y0) - geom.crop_h) <= 1


def test_meter_box_in_padded_crop_excludes_pad():
    first = FrameLayout(ax=0.5, ay=0.5, x0=0.25, y0=0.25, x1=0.75, y1=0.75)
    w, h = 120, 100
    geom = geom_from_first(first, w, h)
    corner = layout_from_anchor(0.02, 0.03, geom, w, h)
    box = meter_box_in_padded_crop(corner, geom, w, h)
    assert box is not None
    x0, y0, x1, y1 = box
    crop = crop_bgr_with_pad(np.full((h, w, 3), 80, dtype=np.uint8), corner, geom)
    assert 0 <= x0 < x1 <= crop.shape[1]
    assert 0 <= y0 < y1 <= crop.shape[0]
    # 越界侧应有黑边，测光框不应从 (0,0) 开始
    assert x0 > 0 or y0 > 0


def test_crop_then_ae_matches_metering_on_crop():
    from burst_webp import BurstWebpBuildOptions, _crop_frame_maybe_ae

    img = np.full((80, 100, 3), 200, dtype=np.uint8)
    img[20:50, 20:50] = 25
    lay = FrameLayout(ax=0.35, ay=0.44, x0=0.20, y0=0.20, x1=0.50, y1=0.55)
    geom = geom_from_first(lay, 100, 80)
    opts = BurstWebpBuildOptions(
        enable_white_balance=False,
        enable_auto_exposure=True,
        auto_exposure_strength=1.0,
    )
    crop_ae = _crop_frame_maybe_ae(img, lay, geom, opts)
    opts_off = BurstWebpBuildOptions(
        enable_white_balance=False,
        enable_auto_exposure=False,
    )
    crop_raw = _crop_frame_maybe_ae(img, lay, geom, opts_off)
    assert crop_ae.shape == crop_raw.shape
    assert float(crop_ae.mean()) > float(crop_raw.mean()) + 5


def test_auto_expose_meters_crop_box_not_background():
    from auto_exposure import auto_expose_bgr

    img = np.full((80, 100, 3), 200, dtype=np.uint8)
    img[20:50, 20:50] = 25
    out = auto_expose_bgr(
        img, strength=1.0, detect=False, meter_box=[20, 20, 50, 50]
    )
    assert float(out[35, 35].mean()) > float(img[35, 35].mean()) + 15
    # 测光在暗框内，不应按亮背景去压暗
    assert float(out[10, 10].mean()) >= 180.0


def test_auto_expose_strength_above_one_adds_exposure():
    from auto_exposure import auto_expose_bgr

    img = np.full((64, 64, 3), 40, dtype=np.uint8)
    box = [8, 8, 56, 56]
    s1 = auto_expose_bgr(img, strength=1.0, detect=False, meter_box=box)
    s2 = auto_expose_bgr(img, strength=2.0, detect=False, meter_box=box)
    s05 = auto_expose_bgr(img, strength=0.5, detect=False, meter_box=box)
    m1 = float(s1.mean())
    m2 = float(s2.mean())
    m05 = float(s05.mean())
    m0 = float(img.mean())
    assert m1 > m0
    assert m2 > m1
    assert m0 < m05 < m1


def test_apply_exposure_strength_reuses_corrected_result():
    from auto_exposure import apply_exposure_strength, auto_expose_bgr

    img = np.full((64, 64, 3), 40, dtype=np.uint8)
    box = [8, 8, 56, 56]
    s1 = auto_expose_bgr(img, strength=1.0, detect=False, meter_box=box)
    s2_direct = auto_expose_bgr(img, strength=2.0, detect=False, meter_box=box)
    s2_apply = apply_exposure_strength(img, s1, 2.0)
    s05_direct = auto_expose_bgr(img, strength=0.5, detect=False, meter_box=box)
    s05_apply = apply_exposure_strength(img, s1, 0.5)
    assert float(s2_apply.mean()) > float(s1.mean())
    assert np.mean(np.abs(s2_direct.astype(np.float32) - s2_apply.astype(np.float32))) < 1.0
    assert np.mean(np.abs(s05_direct.astype(np.float32) - s05_apply.astype(np.float32))) < 1.0


def _seq_detector(boxes_per_frame):
    seq = iter(boxes_per_frame)

    def detect(_bgr):
        try:
            return next(seq)
        except StopIteration:
            return []

    return detect


def test_yolo_keeps_click_offset_inside_bird_box():
    frames = [np.zeros((160, 200, 3), dtype=np.uint8) for _ in range(3)]
    boxes = [
        [{"bbox": [20, 40, 60, 80], "conf": 0.91}],
        [{"bbox": [70, 40, 110, 80], "conf": 0.88}],
        [{"bbox": [120, 40, 160, 80], "conf": 0.86}],
    ]
    ax0, ay0 = 30 / 200.0, 50 / 160.0
    first = FrameLayout(ax=ax0, ay=ay0, x0=0.05, y0=0.15, x1=0.50, y1=0.70)
    out = propagate_layouts(
        frames, first, mode="track", detect_birds_fn=_seq_detector(boxes)
    )
    assert abs(out[2].ax - 130 / 200.0) < 0.03
    assert abs(out[2].ay - 50 / 160.0) < 0.03
    assert out[2].auto is True


def test_kalman_coasts_when_yolo_misses_a_frame():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
    boxes = [
        [{"bbox": [10, 40, 30, 60], "conf": 0.9}],
        [{"bbox": [30, 40, 50, 60], "conf": 0.9}],
        [],
        [{"bbox": [70, 40, 90, 60], "conf": 0.9}],
    ]
    first = FrameLayout(ax=0.20, ay=0.50, x0=0.05, y0=0.30, x1=0.45, y1=0.70)
    out = propagate_layouts(
        frames, first, mode="track", detect_birds_fn=_seq_detector(boxes)
    )
    assert abs(out[2].ax - 0.60) < 0.15
    assert abs(out[3].ax - 0.80) < 0.10


def test_yolo_propagate_skips_locked_frame():
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(3)]
    boxes = [
        [{"bbox": [10, 20, 40, 50], "conf": 0.9}],
        [{"bbox": [50, 20, 80, 50], "conf": 0.9}],
        [{"bbox": [85, 20, 115, 50], "conf": 0.9}],
    ]
    first = FrameLayout(ax=25 / 120.0, ay=35 / 80.0, x0=0.05, y0=0.15, x1=0.50, y1=0.75)
    locked = FrameLayout(
        ax=65 / 120.0, ay=35 / 80.0, x0=0.20, y0=0.15, x1=0.70, y1=0.75, auto=False
    )
    merged = merge_propagate(
        frames,
        [first, locked, None],
        mode="track",
        detect_birds_fn=_seq_detector(boxes),
    )
    assert merged[1].auto is False
    assert abs(merged[1].ax - 65 / 120.0) < 1e-9
    assert merged[2].auto is True
    assert merged[2].ax > 0.70
