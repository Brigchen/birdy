# -*- coding: utf-8 -*-
"""图片清洗：目录级模糊/去重（不依赖 YOLO）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_clean import ImageCleanOptions, clean_bird_images  # noqa: E402


def _write_jpg(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def test_clean_blur_and_dedupe_without_bird_check(tmp_path: Path):
    sp = tmp_path / "species_a"
    # 清晰条纹图
    sharp = np.zeros((120, 120, 3), dtype=np.uint8)
    sharp[::3, :, :] = 255
    # 近乎纯色模糊图
    blur = np.full((120, 120, 3), 90, dtype=np.uint8)
    # 两张几乎相同的清晰图
    dup_a = sharp.copy()
    dup_b = sharp.copy()
    dup_b[0, 0, 0] = 254  # 极小差异

    p_sharp = sp / "sharp.jpg"
    p_blur = sp / "blur.jpg"
    p_dup1 = sp / "dup1.jpg"
    p_dup2 = sp / "dup2.jpg"
    _write_jpg(p_sharp, sharp)
    _write_jpg(p_blur, blur)
    _write_jpg(p_dup1, dup_a)
    _write_jpg(p_dup2, dup_b)
    assert p_sharp.exists() and p_blur.exists()

    opts = ImageCleanOptions(
        remove_no_bird=False,
        remove_blurry=True,
        dedupe=True,
        min_clarity=40,
        dup_similarity=90,
    )
    r = clean_bird_images(str(tmp_path), opts)
    assert r.total == 4
    assert r.removed_blurry >= 1
    assert not p_blur.exists()
    remaining = list(sp.glob("*.jpg"))
    assert len(remaining) <= 2
    assert r.kept == len(remaining)
