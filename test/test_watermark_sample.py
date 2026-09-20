# -*- coding: utf-8 -*-
"""水印：按物种目录随机抽样。"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image  # noqa: E402

from watermark_generator import (  # noqa: E402
    _center_bird_bbox_area,
    _compose_inline_signature_label,
    _compose_leica_style,
    _fit_logo,
    _logo_target_width,
    is_species_collection_image,
    sample_images_per_species_dir,
)


def test_sample_per_species_dir_caps_and_keeps_small_dirs(tmp_path: Path):
    sp_a = tmp_path / "白鹭"
    sp_b = tmp_path / "苍鹭"
    sp_a.mkdir()
    sp_b.mkdir()
    a_files = [str(sp_a / f"a{i}.jpg") for i in range(5)]
    b_files = [str(sp_b / f"b{i}.jpg") for i in range(2)]
    for p in a_files + b_files:
        Path(p).write_bytes(b"x")

    rng = random.Random(42)
    out = sample_images_per_species_dir(
        a_files + b_files, 3, rng=rng, area_fn=lambda _p: 1.0
    )
    assert len(out) == 5  # 3 from A + 2 from B
    assert sum(1 for p in out if Path(p).parent.name == "白鹭") == 3
    assert sum(1 for p in out if Path(p).parent.name == "苍鹭") == 2


def test_sample_per_species_dir_disabled_returns_all():
    paths = [
        r"C:\x\白鹭\1.jpg",
        r"C:\x\白鹭\2.jpg",
        r"C:\x\苍鹭\1.jpg",
    ]
    assert sample_images_per_species_dir(paths, 0) == paths
    assert sample_images_per_species_dir(paths, -1) == paths


def test_sample_is_deterministic_with_seed():
    paths = [rf"C:\x\种A\{i}.jpg" for i in range(10)]
    areas = {p: float(i) for i, p in enumerate(paths)}
    a = sample_images_per_species_dir(
        paths, 4, rng=random.Random(7), area_fn=areas.get
    )
    b = sample_images_per_species_dir(
        paths, 4, rng=random.Random(7), area_fn=areas.get
    )
    assert a == b
    assert len(a) == 4


def test_sample_skips_all_collection_images():
    base = r"C:\x\白鹭"
    paths = [
        rf"{base}\a1.jpg",
        rf"{base}\a2.jpg",
        rf"{base}\shot_00001_all.jpg",
        rf"{base}\all.jpg",
        rf"{base}\a3.jpg",
        rf"{base}\a4.jpg",
    ]
    out = sample_images_per_species_dir(paths, 10, rng=random.Random(1), area_fn=lambda _p: 1.0)
    assert len(out) == 4
    assert all(not is_species_collection_image(p) for p in out)
    assert is_species_collection_image(rf"{base}\shot_00001_all.jpg")
    assert is_species_collection_image(rf"{base}\all.jpg")
    assert not is_species_collection_image(rf"{base}\a1.jpg")


def test_sample_half_by_center_bird_size_half_random():
    base = r"C:\x\白鹭"
    paths = [rf"{base}\{i}.jpg" for i in range(6)]
    areas = {
        paths[0]: 10.0,
        paths[1]: 100.0,
        paths[2]: 80.0,
        paths[3]: 5.0,
        paths[4]: 50.0,
        paths[5]: 20.0,
    }
    out = sample_images_per_species_dir(
        paths, 4, rng=random.Random(0), area_fn=lambda p: areas[p]
    )
    assert len(out) == 4
    # per_dir=4 → 2 张最大中心鸟体 + 2 张从其余随机
    assert paths[1] in out and paths[2] in out
    rest_picked = [p for p in out if p not in (paths[1], paths[2])]
    assert len(rest_picked) == 2
    assert set(rest_picked).issubset({paths[0], paths[3], paths[4], paths[5]})


def test_center_bird_bbox_picks_nearest_to_frame_center():
    birds = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [40, 40, 80, 90]},
    ]
    area = _center_bird_bbox_area(birds, 100, 100)
    assert area == 40.0 * 50.0


def test_logo_target_width_uses_image_long_side():
    assert _logo_target_width(2000, 1000, 0.30) == 600
    assert _logo_target_width(1000, 2000, 0.30) == 600
    assert _logo_target_width(1000, 1000, 0.30) == 300
    assert _logo_target_width(400, 2000, 0.30) == 400


def test_portrait_logo_box_wider_than_width_ratio():
    logo = Image.new("RGBA", (400, 80), (0, 0, 0, 255))
    w, h = 500, 1000
    area_w = _logo_target_width(w, h, 0.30)
    area_h = max(28, int(h * 0.16))
    fitted = _fit_logo(logo, area_w, area_h)
    assert area_w == 300
    assert area_w > max(40, int(w * 0.30))
    assert fitted.size[0] == 300
    land = _compose_leica_style(
        Image.new("RGB", (w, h), (80, 80, 80)), "左", "右", logo, 0.30
    )
    assert land.size[0] > w
    inline = _compose_inline_signature_label(
        Image.new("RGB", (w, h), (80, 80, 80)),
        logo,
        "白鹭",
        "杭州",
        0.30,
        "",
    )
    assert inline.size == (w, h)
