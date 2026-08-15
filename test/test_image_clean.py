# -*- coding: utf-8 -*-
"""图片清洗：清晰度映射与去重哈希单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_clean import (  # noqa: E402
    clarity_score_0_100,
    dhash64,
    hamming64,
    similarity_to_max_hamming,
)


def test_clarity_sharp_higher_than_blur():
    sharp = np.zeros((128, 128), dtype=np.uint8)
    sharp[:, ::8] = 255
    sharp[:, 1::8] = 255
    sharp[:, 2::8] = 255
    blur = np.full((128, 128), 128, dtype=np.uint8)
    assert clarity_score_0_100(sharp) > clarity_score_0_100(blur)


def test_dhash_identical_zero_distance():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    a = dhash64(img)
    b = dhash64(img.copy())
    assert hamming64(a, b) == 0


def test_dhash_different_positive_distance():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    b = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert hamming64(dhash64(a), dhash64(b)) > 0


def test_similarity_to_max_hamming():
    assert similarity_to_max_hamming(100) == 0
    assert similarity_to_max_hamming(0) == 64
    assert 0 < similarity_to_max_hamming(92) < 16


def test_center_bird_crop_picks_nearest_to_center():
    from image_clean import _center_bird_crop

    # 160x120 画布：左侧大鸟、中心小鸟
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 10:50, :] = 80  # left
    img[50:70, 70:90, :] = 200  # center-ish
    birds = [
        {"bbox": [10, 40, 50, 80]},
        {"bbox": [70, 50, 90, 70]},
    ]
    crop = _center_bird_crop(img, birds)
    assert crop is not None
    # 中心鸟框约 20x20
    assert crop.shape[0] == 20 and crop.shape[1] == 20
    assert int(crop.mean()) >= 190
