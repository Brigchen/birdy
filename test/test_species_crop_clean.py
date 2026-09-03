# -*- coding: utf-8 -*-
"""切割后再清洗：暂存切割、按源图分组、整图清晰度。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detect_bird_and_eye import (  # noqa: E402
    archive_identified_crop_file,
    group_crop_records_by_source,
    save_instance_crops_to_staging,
    save_union_crop_for_birds,
)
from image_clean import subject_for_clarity  # noqa: E402


def _clf():
    return {
        "order_cn": "Passeriformes",
        "family_cn": "Pycnonotidae",
        "genus_cn": "Pycnonotus",
        "species_cn": "Light-vented Bulbul",
    }


def test_save_instance_crops_one_file_per_bird(tmp_path):
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[10:30, 10:40] = 40
    img[40:70, 70:110] = 200
    birds = [
        {"bbox": [10, 10, 40, 30]},
        {"bbox": [70, 40, 110, 70]},
    ]
    src = tmp_path / "big.jpg"
    cv2.imwrite(str(src), img)
    staging = tmp_path / "staging"
    recs = save_instance_crops_to_staging(
        img, birds, str(staging), str(src), province="JS", city="NJ"
    )
    assert len(recs) == 2
    assert all(Path(r["path"]).is_file() for r in recs)
    assert recs[0]["inst"] == 1 and recs[1]["inst"] == 2
    assert recs[0]["source_path"] == str(src)
    assert recs[0]["bbox"] == [10, 10, 40, 30]


def test_group_crop_records_by_source():
    recs = [
        {"source_path": "a.jpg", "inst": 1},
        {"source_path": "b.jpg", "inst": 1},
        {"source_path": "a.jpg", "inst": 2},
    ]
    grouped = group_crop_records_by_source(recs)
    assert len(grouped["a.jpg"]) == 2
    assert len(grouped["b.jpg"]) == 1


def test_archive_identified_crop_uses_taxonomy(tmp_path):
    crop = np.full((24, 24, 3), 90, dtype=np.uint8)
    crop_path = tmp_path / "_crop_staging" / "x.jpg"
    crop_path.parent.mkdir()
    cv2.imwrite(str(crop_path), crop)
    bird = {
        "species": [
            {
                "chinese_name": "bulbul",
                "scientific_name": "Pycnonotus sinensis",
                "confidence": 0.9,
            }
        ],
        "classification": _clf(),
    }
    saved = archive_identified_crop_file(
        str(crop_path),
        bird,
        str(tmp_path / "class"),
        "src.jpg",
        province="JS",
        city="NJ",
        counter={"n": 0},
        inst_i=1,
    )
    assert saved
    assert Path(saved).is_file()
    assert "Passeriformes" in saved
    assert "Light-vented Bulbul" in saved
    assert not crop_path.exists()


def test_union_saved_only_when_two_birds_remain(tmp_path):
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    src = str(tmp_path / "big.jpg")
    cv2.imwrite(src, img)
    bird = {
        "bbox": [10, 10, 40, 30],
        "species": [{"chinese_name": "bulbul", "confidence": 0.8}],
        "classification": _clf(),
    }
    bird2 = {
        "bbox": [70, 40, 110, 70],
        "species": [{"chinese_name": "bulbul", "confidence": 0.7}],
        "classification": _clf(),
    }
    out = save_union_crop_for_birds(
        img,
        [bird, bird2],
        str(tmp_path / "class"),
        src,
        min_species_accept_confidence=None,
    )
    assert out and Path(out).is_file()
    assert out.endswith("_all.jpg")
    none = save_union_crop_for_birds(
        img,
        [bird],
        str(tmp_path / "class"),
        src,
        min_species_accept_confidence=None,
    )
    assert none is None


def test_subject_for_clarity_full_frame_keeps_whole_crop():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :] = 10
    img[15:25, 20:40] = 200
    birds = [{"bbox": [20, 15, 40, 25]}]
    full = subject_for_clarity(img, birds, use_full_frame=True)
    part = subject_for_clarity(img, birds, use_full_frame=False)
    assert full.shape == img.shape
    assert part.shape[0] == 10 and part.shape[1] == 20
