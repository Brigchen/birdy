# -*- coding: utf-8 -*-
"""MP4 抽帧转 WebP：间隔与倍速。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_webp import (  # noqa: E402
    build_webp_from_mp4,
    mp4_playback_duration_ms,
    mp4_sample_indices,
)


def test_sample_keeps_every_nth_frame():
    assert mp4_sample_indices(6, 2) == [0, 2, 4]
    assert mp4_sample_indices(3, 1) == [0, 1, 2]
    assert mp4_sample_indices(0, 2) == []


def test_speed_changes_frame_duration_not_which_frames():
    """30fps、每 2 帧留 1 帧：原速约 67ms，2 倍速减半，半速加倍。"""
    base, fps1 = mp4_playback_duration_ms(30.0, 2, 1.0)
    fast, fps2 = mp4_playback_duration_ms(30.0, 2, 2.0)
    slow, fps_half = mp4_playback_duration_ms(30.0, 2, 0.5)
    assert abs(base - (2 / 30.0) * 1000.0) < 1e-6
    assert abs(fast - base / 2.0) < 1e-6
    assert abs(slow - base * 2.0) < 1e-6
    assert fps2 > fps1 > fps_half


def test_build_webp_samples_and_speeds(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(src),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    assert writer.isOpened()
    colors = (
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    )
    for bgr in colors:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:] = bgr
        writer.write(frame)
    writer.release()

    out = tmp_path / "clip_动图.webp"
    result = build_webp_from_mp4(
        str(src),
        str(out),
        frame_step=2,
        speed=2.0,
        max_long_edge=32,
        webp_quality=90,
        log_terminal=False,
    )
    assert result["n_frames"] == 3
    assert result["src_frames"] == 6
    expected_ms = max(1, int(round((2 / 10.0) / 2.0 * 1000.0)))
    assert result["frame_duration_ms"] == float(expected_ms)
    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) == 3
        assert int(im.info.get("loop", -1)) == 0
