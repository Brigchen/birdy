# -*- coding: utf-8 -*-
"""图片清洗：清晰度映射与去重哈希单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
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
    subject_clarity_score,
)


def test_clarity_sharp_higher_than_blur():
    sharp = np.zeros((128, 128), dtype=np.uint8)
    sharp[:, ::8] = 255
    sharp[:, 1::8] = 255
    sharp[:, 2::8] = 255
    blur = np.full((128, 128), 128, dtype=np.uint8)
    assert clarity_score_0_100(sharp) > clarity_score_0_100(blur)


def test_clarity_iso_noise_not_scored_as_sharp():
    """高 ISO 颗粒不应把糊图打过默认阈值 35。"""
    rng = np.random.RandomState(7)
    blur = np.full((240, 240), 110, dtype=np.uint8)
    noisy = np.clip(
        blur.astype(np.int16) + rng.randint(-32, 33, blur.shape), 0, 255
    ).astype(np.uint8)
    sharp = np.zeros((240, 240), dtype=np.uint8)
    sharp[:, ::8] = 255
    sharp[:, 1::8] = 255
    sharp[:, 2::8] = 255
    nscore = clarity_score_0_100(noisy)
    assert nscore < 35
    assert nscore < clarity_score_0_100(sharp) - 10


def test_clarity_motion_blur_lower_than_sharp_bars():
    sharp = np.zeros((240, 240), dtype=np.uint8)
    sharp[:, ::8] = 255
    sharp[:, 1::8] = 255
    mot = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    assert clarity_score_0_100(mot) < 35
    assert clarity_score_0_100(mot) < clarity_score_0_100(sharp)


def _step_edge(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.uint8)
    g[:, n // 2 :] = 255
    return g


def test_clarity_small_image_not_inflated_vs_640():
    """小图会长边放大到 640，不再因像素少而明显高于同内容大图。"""
    s160 = clarity_score_0_100(_step_edge(160))
    s640 = clarity_score_0_100(_step_edge(640))
    s1280 = clarity_score_0_100(_step_edge(1280))
    assert s160 <= s640 + 8
    assert abs(s640 - s1280) < 1.0


def test_masked_clarity_ignores_sharp_ground():
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    img[:, ::3] = 220
    img[80:160, 80:160] = 90
    mask = np.zeros((240, 240), np.uint8)
    mask[84:156, 84:156] = 1
    full = clarity_score_0_100(img)
    masked = clarity_score_0_100(img, mask)
    assert masked < 30
    assert full > masked + 15


def test_subject_clarity_score_uses_mask_xy():
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    img[:, ::3] = 220
    img[80:160, 80:160] = 90
    birds = [
        {
            "bbox": [80, 80, 160, 160],
            "mask_xy": [[80, 80], [160, 80], [160, 160], [80, 160]],
        }
    ]
    s = subject_clarity_score(img, birds, use_full_frame=True)
    assert s < clarity_score_0_100(img)


def test_masked_clarity_full_frame_matches_crop_scale():
    """整图套掩膜与先裁鸟框计分应接近，避免小鸟剪影把峰度顶满。"""
    rng = np.random.RandomState(2)
    img = np.full((400, 560, 3), 100, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.randint(-24, 25, img.shape), 0, 255).astype(
        np.uint8
    )
    y0, x0, h, w = 150, 210, 70, 90
    blob = np.full((h, w, 3), 50, dtype=np.uint8)
    blob = cv2.GaussianBlur(blob, (0, 0), 3.5)
    img[y0 : y0 + h, x0 : x0 + w] = blob
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y0 + 4 : y0 + h - 4, x0 + 4 : x0 + w - 4] = 1
    full = clarity_score_0_100(img, mask)
    crop = clarity_score_0_100(
        img[y0 : y0 + h, x0 : x0 + w], mask[y0 : y0 + h, x0 : x0 + w]
    )
    assert abs(full - crop) < 6
    assert full < 35


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
