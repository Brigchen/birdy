# -*- coding: utf-8 -*-
"""clean_image_list：切割图清洗可跳过鸟检，仅做清晰度+去重。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import image_clean  # noqa: E402
from image_clean import (  # noqa: E402
    ImageCleanOptions,
    clean_image_list,
)


def _save(path: Path, w: int, h: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype("uint8")
    import cv2

    cv2.imwrite(str(path), arr)


def test_skip_bird_detection_never_runs_yolo(tmp_path):
    """清洗只做清晰度+去重：即使勾了 remove_no_bird 也不删任何图。"""
    for i in range(3):
        _save(tmp_path / f"b{i}.jpg", 120, 160, i)

    opts = ImageCleanOptions(
        remove_no_bird=True,
        remove_blurry=False,
        dedupe=False,
    )
    r = clean_image_list(
        [str(p) for p in sorted(tmp_path.glob("*.jpg"))],
        opts,
        skip_bird_detection=True,
    )
    assert r.kept == 3
    assert r.removed_no_bird == 0
    assert len(list(tmp_path.glob("*.jpg"))) == 3


def test_skip_mode_still_dedupes_and_blurs(tmp_path):
    # 两张相同图 → 去重留 1 张
    import cv2

    arr = (np.random.default_rng(1).random((160, 120, 3)) * 255).astype("uint8")
    cv2.imwrite(str(tmp_path / "a1.jpg"), arr)
    cv2.imwrite(str(tmp_path / "a2.jpg"), arr)
    # 一张纯糊图（全黑）
    cv2.imwrite(str(tmp_path / "blur.jpg"), np.zeros((160, 120, 3), "uint8"))
    opts = ImageCleanOptions(
        remove_no_bird=False,
        remove_blurry=True,
        dedupe=True,
        min_clarity=1.0,
        dup_similarity=92,
        use_full_frame_for_clarity=True,
    )
    r = clean_image_list(
        [str(p) for p in sorted(tmp_path.glob("*.jpg"))],
        opts,
        skip_bird_detection=True,
    )
    assert r.removed_duplicate == 1
    assert r.removed_blurry == 1
    assert r.kept == 1
