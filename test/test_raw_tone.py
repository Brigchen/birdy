# -*- coding: utf-8 -*-
"""RAW demosaic 后的亮度与饱和度补偿。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_io import _tone_demosaic_bgr  # noqa: E402


def test_dark_frame_gets_brighter_and_more_chroma():
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    dark[:] = (40, 50, 30)
    out = _tone_demosaic_bgr(dark)
    assert float(out.mean()) > float(dark.mean())
    # 绿通道相对偏高，补偿后色差应拉开
    spread_in = float(dark.max() - dark.min())
    spread_out = float(out.max() - out.min())
    assert spread_out >= spread_in
