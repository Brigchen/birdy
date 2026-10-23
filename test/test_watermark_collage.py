# -*- coding: utf-8 -*-
"""水印拼图：2/4/9 连与循环补图。"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watermark_generator import (  # noqa: E402
    _collage_orientation,
    collage_tile_count,
    collage_tile_groups,
    collage_tile_groups_sampled,
    compose_collage,
    cycle_fill_paths,
    generate_watermarks,
    normalize_collage_mode,
    WatermarkOptions,
)


def test_normalize_collage_mode():
    assert normalize_collage_mode("2h") == "2h"
    assert normalize_collage_mode("左右") == "2h"
    assert normalize_collage_mode("2v") == "2v"
    assert normalize_collage_mode("3h") == "3h"
    assert normalize_collage_mode("3v") == "3v"
    assert normalize_collage_mode("4") == "4"
    assert normalize_collage_mode("9连") == "9"
    assert normalize_collage_mode("") == "off"
    assert collage_tile_count("2h") == 2
    assert collage_tile_count("2v") == 2
    assert collage_tile_count("3h") == 3
    assert collage_tile_count("3v") == 3
    assert collage_tile_count("4") == 4
    assert collage_tile_count("9") == 9


def test_cycle_fill_and_last_group_wraps(tmp_path: Path):
    def _mk(name: str) -> str:
        p = tmp_path / name
        Image.new("RGB", (24, 48), (10, 10, 10)).save(p)
        return str(p)

    paths = [_mk(f"a{i}.jpg") for i in range(8)]
    filled = cycle_fill_paths(paths, 9)
    assert len(filled) == 9
    assert filled[:8] == paths
    assert filled[8] == paths[0]
    groups = collage_tile_groups(paths, 9)
    assert len(groups) == 1
    assert groups[0][-1] == paths[0]
    eleven = [_mk(f"b{i:02d}.jpg") for i in range(11)]
    g2 = collage_tile_groups(eleven, 9)
    assert len(g2) == 2
    assert g2[0] == eleven[:9]
    assert g2[1][0] == eleven[9]
    assert g2[1][1] == eleven[10]
    assert len(g2[1]) == 9
    assert g2[1][2] == eleven[0]


def test_compose_collage_2h_and_9(tmp_path: Path):
    a = Image.new("RGB", (40, 20), (255, 0, 0))
    b = Image.new("RGB", (40, 20), (0, 255, 0))
    out = compose_collage([a, b], 1, 2, gap=0)
    assert out.size == (80, 20)
    tiles = [Image.new("RGB", (10, 10), (i * 20, 0, 0)) for i in range(9)]
    g9 = compose_collage(tiles, 3, 3, gap=0)
    assert g9.size == (30, 30)


def test_eight_mixed_orientation_yield_two_four_collages(tmp_path: Path):
    """竖/横/方混在同一物种 8 张时，应 2 个四拼图，而不是 5 个单图×4。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    specs = [
        (30, 50),
        (50, 30),
        (40, 40),
        (35, 55),
        (55, 35),
        (38, 38),
        (32, 48),
        (48, 32),
    ]
    paths = []
    for i, (w, h) in enumerate(specs):
        p = sp / f"m{i}.jpg"
        Image.new("RGB", (w, h), (i * 25, 10, 10)).save(p)
        paths.append(str(p))
    groups = collage_tile_groups(paths, 4, per_species_budget=8)
    assert len(groups) == 2
    for g in groups:
        assert len(g) == 4
        assert len(set(g)) == 4


def test_eight_portraits_yield_two_disjoint_four_collages(tmp_path: Path):
    sp = tmp_path / "sp"
    sp.mkdir()
    paths = []
    for i in range(8):
        p = sp / f"bird_{i:02d}.jpg"
        Image.new("RGB", (30, 50), (i * 20, 10, 10)).save(p)
        paths.append(str(p))
    groups = collage_tile_groups(paths, 4)
    assert len(groups) == 2
    flat = [p for g in groups for p in g]
    assert len(set(flat)) == 8
    assert groups[0] != groups[1]


def test_varied_aspect_four_portraits_one_quad_not_four_copies(tmp_path: Path):
    """长宽比差异大时仍应 4 张不同图一组，而不是拆成 4 个「同图×4」拼图。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    paths = []
    for i, h in enumerate((40, 80, 120, 160)):
        p = sp / f"v{i}.jpg"
        Image.new("RGB", (20, h), (i * 40, 0, 0)).save(p)
        paths.append(str(p))
    groups = collage_tile_groups(paths, 4)
    assert len(groups) == 1
    assert len(set(groups[0])) == 4


def test_aspect_subpools_do_not_duplicate_same_quad(tmp_path: Path):
    """长宽比略不同时，不应出现多组都用同一前 4 张。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    paths = []
    for i, (w, h) in enumerate(
        [(20, 40), (20, 41), (20, 50), (20, 51), (20, 60), (20, 61), (20, 70), (20, 71)]
    ):
        p = sp / f"x{i}.jpg"
        Image.new("RGB", (w, h), (i, 0, 0)).save(p)
        paths.append(str(p))
    groups = collage_tile_groups(paths, 4)
    assert len(groups) == 2
    assert set(groups[0]) != set(groups[1])


def test_collage_groups_prefer_same_orientation(tmp_path: Path):
    sp = tmp_path / "sp"
    sp.mkdir()
    p1 = sp / "a.jpg"
    p2 = sp / "b.jpg"
    p3 = sp / "c.jpg"
    p4 = sp / "d.jpg"
    Image.new("RGB", (20, 40), (1, 0, 0)).save(p1)
    Image.new("RGB", (20, 38), (2, 0, 0)).save(p2)
    Image.new("RGB", (40, 20), (0, 1, 0)).save(p3)
    Image.new("RGB", (38, 20), (0, 2, 0)).save(p4)
    groups = collage_tile_groups([str(p1), str(p2), str(p3), str(p4)], 2)
    assert len(groups) == 2
    for g in groups:
        assert len(g) == 2
        sizes = []
        for p in g:
            im = Image.open(p)
            sizes.append(im.size)
            im.close()
        orient = "p" if sizes[0][1] > sizes[0][0] else "l"
        for w, h in sizes:
            o = "p" if h > w else "l"
            assert o == orient


def test_compose_collage_3h(tmp_path: Path):
    tiles = [
        Image.new("RGB", (30, 60), (i * 30, 0, 0))
        for i in range(3)
    ]
    out = compose_collage(tiles, 1, 3, gap=0)
    assert out.size == (90, 60)


def test_nine_collage_one_portrait_does_not_repeat_nine_times(tmp_path: Path):
    """竖图仅 1 张时不应单独成 9 连同图×9；应与横图合并成组。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    Image.new("RGB", (40, 80), (1, 2, 3)).save(sp / "p0.jpg")
    for i in range(11):
        Image.new("RGB", (80, 40), (i * 10, 0, 0)).save(sp / f"l{i:02d}.jpg")
    paths = [str(p) for p in sorted(sp.glob("*.jpg"))]
    groups = collage_tile_groups(paths, 9)
    assert groups[0]
    assert len(set(groups[0])) == 9
    assert len(groups) >= 1


def test_collage_sampled_eight_per_species_two_quads(tmp_path: Path):
    """8 张/种 + 四连：先横竖分池，各池抽 4，共 2 个拼图任务。"""
    sp = tmp_path / "in" / "灰头麦鸡"
    out_dir = tmp_path / "out"
    sp.mkdir(parents=True)
    for i in range(40):
        w, h = (60, 40) if i % 2 else (40, 60)
        Image.new("RGB", (w, h), (i * 5, 20, 80)).save(sp / f"{i:03d}.jpg")
    r = generate_watermarks(
        str(tmp_path / "in"),
        str(out_dir),
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=True,
            enable_camera_params=False,
            enable_auto_enhance=False,
            watermark_style="inline",
            collage_mode="4",
        ),
        prefer_folder_name_as_species=True,
        random_per_species=8,
    )
    assert r["total"] == 2
    assert r["ok"] == 2


def test_collage_batch_one_develop_pass(tmp_path: Path, monkeypatch):
    """拼图批量：4 格只合成一次，显影管线只跑一遍（非每格 YOLO/生态显影）。"""
    sp = tmp_path / "in" / "物种A"
    out_dir = tmp_path / "out"
    sp.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (120, 160), (20 * i, 40, 80)).save(sp / f"{i}.jpg", "JPEG")
    calls = {"n": 0}
    real = __import__("watermark_generator", fromlist=["apply_watermark_photo_pipeline"]).apply_watermark_photo_pipeline

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(
        "watermark_generator.apply_watermark_photo_pipeline", _count
    )
    r = generate_watermarks(
        str(tmp_path / "in"),
        str(out_dir),
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=True,
            enable_camera_params=False,
            enable_auto_enhance=False,
            watermark_style="inline",
            collage_mode="4",
        ),
        prefer_folder_name_as_species=True,
    )
    assert r["ok"] == 1
    assert calls["n"] == 1
    assert list(out_dir.glob("*.jpg"))


def test_generate_watermarks_9_cycles_when_short(tmp_path: Path):
    sp = tmp_path / "in" / "白鹭"
    out_dir = tmp_path / "out"
    sp.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (32, 24), (10 * i, 40, 80)).save(sp / f"{i}.jpg", "JPEG")
    r = generate_watermarks(
        str(tmp_path / "in"),
        str(out_dir),
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=True,
            enable_camera_params=False,
            enable_auto_enhance=False,
            watermark_style="inline",
            collage_mode="9",
        ),
        prefer_folder_name_as_species=True,
    )
    assert r["ok"] == 1
    assert r["total"] == 1
    files = list(out_dir.glob("*.jpg"))
    assert len(files) == 1
    im = Image.open(files[0])
    assert im.size[0] > 32 and im.size[1] > 24
