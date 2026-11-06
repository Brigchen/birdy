# -*- coding: utf-8 -*-
"""水印拼图：2/4/9 连与循环补图。"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watermark_generator import (  # noqa: E402
    COLLAGE_CANVAS_SIZE,
    _collage_orientation,
    collage_frame_options,
    collage_photo_boxes,
    collage_tile_count,
    collage_tile_frame_px,
    collage_tile_groups,
    collage_tile_groups_sampled,
    compose_collage,
    COLLAGE_SIDE_WHITE_SCALE,
    compose_collage_canvas,
    compose_full_species_collage,
    collect_species_representatives,
    fit_collage_tile,
    full_collage_count_parts,
    full_collage_photo_boxes,
    full_collage_title_height,
    _full_collage_count_fonts,
    _apply_post_compose_watermark_pipeline,
    _leica_side_white,
    cycle_fill_paths,
    generate_watermarks,
    is_species_collection_image,
    normalize_collage_mode,
    plan_multi_species_collages,
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
    assert normalize_collage_mode("上二下一") == "3t"
    assert normalize_collage_mode("5拼") == "5"
    assert normalize_collage_mode("6") == "6"
    assert normalize_collage_mode("24拼") == "24"
    assert normalize_collage_mode("") == "off"
    assert collage_tile_count("2h") == 2
    assert collage_tile_count("2v") == 2
    assert collage_tile_count("3h") == 3
    assert collage_tile_count("3v") == 3
    assert collage_tile_count("3t") == 3
    assert collage_tile_count("4") == 4
    assert collage_tile_count("5") == 5
    assert collage_tile_count("6") == 6
    assert collage_tile_count("9") == 9
    assert collage_tile_count("24") == 24


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


def test_collage_prefers_dissimilar_and_includes_all(tmp_path: Path):
    """同一张拼图优先画面相差大的图，文件名带 all 的合集图也入选。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    Image.new("RGB", (40, 70), (200, 10, 10)).save(sp / "a.jpg")
    Image.new("RGB", (40, 70), (210, 8, 8)).save(sp / "b.jpg")
    Image.new("RGB", (40, 70), (190, 12, 12)).save(sp / "c.jpg")
    Image.new("RGB", (40, 70), (10, 10, 220)).save(sp / "shot_00001_all.jpg")
    paths = [str(p) for p in sorted(sp.glob("*.jpg"))]
    groups = collage_tile_groups(paths, 2)
    assert any("all" in Path(p).name for g in groups for p in g)
    assert any("all" in Path(p).name for p in groups[0])
    sampled = collage_tile_groups(paths, 2, per_species_budget=2)
    assert len(sampled) == 1
    assert len(sampled[0]) == 2
    assert any("all" in Path(p).name for p in sampled[0])


def _all_count(group) -> int:
    return sum(1 for p in group if is_species_collection_image(p))


def test_collage_option_allows_one_group_photo(tmp_path: Path):
    """勾选后每张多联图最多一张 all 群体图，其余为单只图。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    colors = [
        (200, 10, 10),
        (10, 180, 10),
        (10, 10, 200),
        (200, 200, 10),
        (180, 10, 180),
        (10, 180, 180),
    ]
    paths = []
    for i, color in enumerate(colors):
        p = sp / f"bird_{i}.jpg"
        Image.new("RGB", (40, 70), color).save(p)
        paths.append(str(p))
    for i, color in enumerate(((20, 20, 20), (230, 230, 230))):
        p = sp / f"group_{i}_all.jpg"
        Image.new("RGB", (40, 70), color).save(p)
        paths.append(str(p))

    groups = collage_tile_groups(paths, 4, allow_one_all=True)
    assert len(groups) == 2
    for g in groups:
        assert len(g) == 4
        assert _all_count(g) == 1
    assert len({p for g in groups for p in g if is_species_collection_image(p)}) == 2

    one_all = paths[:5] + [paths[-1]]
    partial = collage_tile_groups(one_all, 4, allow_one_all=True)
    assert _all_count(partial[0]) == 1
    if len(partial) > 1:
        assert _all_count(partial[1]) == 0

    sampled = collage_tile_groups(
        paths, 4, per_species_budget=4, allow_one_all=True
    )
    assert len(sampled) == 1
    assert _all_count(sampled[0]) == 1
    assert sum(1 for p in sampled[0] if not is_species_collection_image(p)) == 3


def test_collage_keeps_species_apart_under_same_order(tmp_path: Path):
    """分类树只选到目时，同一属下的不同物种不能进同一张拼图。"""
    root = tmp_path / "in"
    grey = root / "鸻形目" / "鸻科" / "麦鸡属" / "灰头麦鸡"
    lapwing = root / "鸻形目" / "鸻科" / "麦鸡属" / "凤头麦鸡"
    grey.mkdir(parents=True)
    lapwing.mkdir(parents=True)
    (grey / "sub").mkdir()
    Image.new("RGB", (40, 60), (200, 0, 0)).save(grey / "1.jpg")
    Image.new("RGB", (40, 60), (180, 20, 20)).save(grey / "sub" / "2.jpg")
    Image.new("RGB", (60, 40), (0, 160, 0)).save(lapwing / "1.jpg")
    Image.new("RGB", (60, 40), (20, 180, 20)).save(lapwing / "2.jpg")
    out_dir = tmp_path / "out"
    r = generate_watermarks(
        str(root),
        str(out_dir),
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=True,
            enable_camera_params=False,
            enable_auto_enhance=False,
            watermark_style="inline",
            collage_mode="2h",
        ),
        prefer_folder_name_as_species=True,
    )
    assert r["total"] == 2
    assert r["ok"] == 2
    names = [p.name for p in out_dir.glob("*.jpg")]
    assert len(names) == 2
    assert any("灰头麦鸡" in n for n in names)
    assert any("凤头麦鸡" in n for n in names)


def test_collage_does_not_rank_by_pixel_area(tmp_path: Path):
    """大图不再优先：小的异色图应进入第一张拼图，而不是两张大红图。"""
    sp = tmp_path / "sp"
    sp.mkdir()
    Image.new("RGB", (400, 600), (200, 0, 0)).save(sp / "big.jpg")
    Image.new("RGB", (80, 120), (205, 0, 0)).save(sp / "mid.jpg")
    Image.new("RGB", (20, 40), (10, 180, 10)).save(sp / "small_green.jpg")
    paths = [str(sp / name) for name in ("big.jpg", "mid.jpg", "small_green.jpg")]
    groups = collage_tile_groups(paths, 2)
    assert "small_green.jpg" in {Path(p).name for p in groups[0]}


def test_compose_collage_3h(tmp_path: Path):
    tiles = [
        Image.new("RGB", (30, 60), (i * 30, 0, 0))
        for i in range(3)
    ]
    out = compose_collage(tiles, 1, 3, gap=0)
    assert out.size == (90, 60)


def test_collage_canvas_is_portrait_4k_with_white_frame():
    """拼图成图固定竖版 4:3 / 4K，左右裁水平居中，四周与格间为细白边。"""
    assert COLLAGE_CANVAS_SIZE == (2880, 3840)
    assert abs(COLLAGE_CANVAS_SIZE[1] / COLLAGE_CANVAS_SIZE[0] - 4 / 3) < 1e-9
    wide = Image.new("RGB", (400, 40), (255, 0, 0))
    for x in range(120, 280):
        for y in range(40):
            wide.putpixel((x, y), (0, 0, 255))
    out = compose_collage_canvas([wide, wide], "2v")
    assert out.size == COLLAGE_CANVAS_SIZE
    assert out.getpixel((0, 0)) == (255, 255, 255)
    frame = collage_tile_frame_px()
    boxes = collage_photo_boxes("2v")
    assert len(boxes) == 2
    x, y, w, h = boxes[0]
    assert out.getpixel((x - 1, y + h // 2)) == (255, 255, 255)
    assert out.getpixel((x + w // 2, y + h // 2)) == (0, 0, 255)

    top = collage_photo_boxes("5")
    assert len(top) == 5
    assert top[0][2] != top[1][2]
    assert top[2][2] > top[0][2]
    row9 = collage_photo_boxes("9")
    assert row9[0][2] != row9[1][2] != row9[2][2]
    row24 = collage_photo_boxes("24")
    assert len(row24) == 24
    assert row24[0][2] == row24[1][2] == row24[2][2] == row24[3][2]
    assert frame >= 8


def test_vertical_crop_centers_on_upper_two_fifths():
    """上下裁以画面 2/5 高度为中心，1/3 与几何中心都会被切掉。"""
    w, h = 200, 1000
    im = Image.new("RGB", (w, h), (0, 0, 0))
    for y in range(300, 350):
        for x in range(w):
            im.putpixel((x, y), (0, 180, 0))
    for y in range(360, 440):
        for x in range(w):
            im.putpixel((x, y), (255, 0, 0))
    for y in range(470, 530):
        for x in range(w):
            im.putpixel((x, y), (0, 0, 255))
    fitted = fit_collage_tile(im, (400, 100))
    assert fitted.size == (400, 100)
    assert fitted.getpixel((200, 50)) == (255, 0, 0)
    assert fitted.getpixel((200, 0)) != (0, 180, 0)
    assert fitted.getpixel((200, 0)) != (0, 0, 255)


def test_full_collage_title_band_and_uneven_rows():
    """全拼图：竖版 4K，上方 1/4 标题，各行等高且同行宽度不等。"""
    boxes = full_collage_photo_boxes(5)
    assert len(boxes) == 5
    title_h = full_collage_title_height()
    assert title_h == COLLAGE_CANVAS_SIZE[1] // 4
    assert all(b[1] >= title_h for b in boxes)
    assert boxes[0][3] == boxes[3][3]
    assert len({boxes[0][2], boxes[1][2], boxes[2][2]}) > 1
    bg = Image.new("RGB", (40, 20), (0, 180, 0))
    photos = [Image.new("RGB", (30, 40), (220, 10, 10)) for _ in range(5)]
    out = compose_full_species_collage(
        photos,
        ["翠鸟", "白鹭", "麻雀", "喜鹊", "戴胜"],
        bg,
        "杭州西湖",
        "2026.11.05",
        "5种",
        "2小时",
    )
    assert out.size == COLLAGE_CANVAS_SIZE
    assert out.getpixel((COLLAGE_CANVAS_SIZE[0] // 2, title_h // 2)) == (0, 180, 0)
    x, y, w, h = boxes[0]
    assert out.getpixel((x + w // 2, y + h // 2)) == (220, 10, 10)
    white = False
    for yy in range(title_h // 5, title_h * 4 // 5, 3):
        for xx in range(40, 700, 3):
            r, g, b = out.getpixel((xx, yy))
            if r > 220 and g > 220 and b > 220:
                white = True
                break
        if white:
            break
    assert white


def test_collect_one_photo_per_species(tmp_path: Path):
    root = tmp_path / "in"
    (root / "翠鸟").mkdir(parents=True)
    (root / "白鹭").mkdir(parents=True)
    Image.new("RGB", (12, 16), (1, 2, 3)).save(root / "翠鸟" / "a.jpg")
    Image.new("RGB", (12, 16), (4, 5, 6)).save(root / "翠鸟" / "b.jpg")
    Image.new("RGB", (16, 12), (7, 8, 9)).save(root / "白鹭" / "c.jpg")
    picks = collect_species_representatives(str(root))
    assert len(picks) == 2
    names = {name for _p, name in picks}
    assert names == {"翠鸟", "白鹭"}


def test_full_collage_picks_largest_single(tmp_path: Path):
    """全拼图每个物种取像素面积最大的单只图，更大的 all 群体图不入选。"""
    root = tmp_path / "in"
    species = root / "翠鸟"
    species.mkdir(parents=True)
    Image.new("RGB", (10, 12), (1, 1, 1)).save(species / "small.jpg")
    Image.new("RGB", (40, 30), (2, 2, 2)).save(species / "big.jpg")
    Image.new("RGB", (200, 200), (3, 3, 3)).save(species / "crowd_all.jpg")
    picks = collect_species_representatives(str(root))
    assert len(picks) == 1
    assert Path(picks[0][0]).name == "big.jpg"


def test_full_collage_follows_record_taxonomy(tmp_path: Path):
    """排列与观鸟记录导出一致：苍鹭、反嘴鹬、麻雀，而不是按中文名。"""
    root = tmp_path / "in"
    for name in ("麻雀", "反嘴鹬", "苍鹭"):
        (root / name).mkdir(parents=True)
        Image.new("RGB", (8, 8), (9, 9, 9)).save(root / name / "a.jpg")
    picks = collect_species_representatives(str(root))
    assert [name for _path, name in picks] == ["苍鹭", "反嘴鹬", "麻雀"]


def test_full_collage_count_is_number_plus_smaller_suffix():
    assert full_collage_count_parts("12种", 3) == ("12", "种鸟")
    assert full_collage_count_parts("", 7) == ("7", "种鸟")
    assert full_collage_count_parts("种鸟", 4) == ("4", "种鸟")
    draw = ImageDraw.Draw(Image.new("RGB", (32, 32)))
    num_font, suf_font, gap = _full_collage_count_fonts(draw, "12", "种鸟", 800, 960)
    num_size = int(getattr(num_font, "size", 0))
    suf_size = int(getattr(suf_font, "size", 0))
    assert num_size > 0 and suf_size > 0
    assert suf_size * 4 <= num_size
    assert gap > 0


def test_collage_outer_side_white_is_quarter(tmp_path: Path):
    """拼图外框上、左、右白边为单张的 1/4。"""
    w, h = COLLAGE_CANVAS_SIZE
    full = _leica_side_white(w, h, 1.0)
    thin = _leica_side_white(w, h, COLLAGE_SIDE_WHITE_SCALE)
    assert COLLAGE_SIDE_WHITE_SCALE == 0.25
    assert thin == max(2, int(0.01 * (w + h) * 0.25))
    assert thin * 4 <= full
    img = Image.new("RGB", (240, 320), (40, 50, 60))
    out = _apply_post_compose_watermark_pipeline(
        img,
        "x.jpg",
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=False,
            enable_camera_params=False,
            enable_auto_enhance=False,
        ),
        str(tmp_path),
        {},
        True,
        None,
    )
    expect = _leica_side_white(240, 320, COLLAGE_SIDE_WHITE_SCALE)
    y = expect + 8
    run = 0
    for x in range(out.size[0]):
        px = out.getpixel((x, y))
        if px[0] < 250 or px[1] < 250 or px[2] < 250:
            break
        run += 1
    assert run == expect
    assert run < _leica_side_white(240, 320, 1.0)


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
    # 拼图画布 2880×3840，外框水印再向外扩一圈
    assert im.size[0] > COLLAGE_CANVAS_SIZE[0]
    assert im.size[1] > COLLAGE_CANVAS_SIZE[1]
    im.close()


def test_collage_frame_drops_date_and_camera():
    single = collage_frame_options(
        WatermarkOptions(
            enable_date=True,
            enable_camera_params=True,
            enable_species=True,
            watermark_style="inline",
        )
    )
    assert single.watermark_style == "frame"
    assert single.enable_date is False
    assert single.enable_camera_params is False
    assert single.enable_species is True
    mixed = collage_frame_options(
        WatermarkOptions(
            enable_date=True,
            enable_camera_params=True,
            enable_species=True,
            collage_multi_species=True,
        )
    )
    assert mixed.enable_species is False


def test_multi_species_collage_one_photo_each(tmp_path: Path):
    """多个物种子目录合拼时每物种一张，物种名写在小图上；不按单物种拆开。"""
    root = tmp_path / "in"
    egret = root / "白鹭"
    heron = root / "苍鹭"
    egret.mkdir(parents=True)
    heron.mkdir(parents=True)
    Image.new("RGB", (40, 70), (180, 20, 20)).save(egret / "a.jpg")
    Image.new("RGB", (40, 70), (20, 20, 20)).save(egret / "shot_all.jpg")
    Image.new("RGB", (40, 70), (20, 80, 160)).save(heron / "a.jpg")
    Image.new("RGB", (70, 40), (20, 160, 80)).save(heron / "b.jpg")
    paths = [str(p) for p in root.rglob("*.jpg")]
    groups = plan_multi_species_collages(str(root), paths, 2)
    assert len(groups) == 1
    assert {name for _p, name in groups[0]} == {"白鹭", "苍鹭"}
    assert all("all" not in Path(p).name for p, _n in groups[0])

    five = tmp_path / "five"
    for i in range(5):
        sp = five / f"种{i}"
        sp.mkdir(parents=True)
        Image.new("RGB", (30, 40), (i * 40, 10, 10)).save(sp / "a.jpg")
    more = plan_multi_species_collages(
        str(five), [str(p) for p in five.rglob("*.jpg")], 4
    )
    assert len(more) == 2
    assert all(len(g) == 4 for g in more)

    canvas = compose_collage_canvas(
        [
            Image.new("RGB", (80, 120), (180, 10, 10)),
            Image.new("RGB", (80, 120), (10, 80, 160)),
        ],
        "2h",
        labels=["白鹭", "苍鹭"],
    )
    x, y, w, h = collage_photo_boxes("2h")[0]
    marked = False
    y0 = max(y, y + h - max(200, h // 12))
    for py in range(y0, y + h):
        for px in range(x + w // 2 - 120, x + w // 2 + 120):
            _r, g, b = canvas.getpixel((px, py))
            if g > 80 and b > 80:
                marked = True
                break
        if marked:
            break
    assert marked

    out_dir = tmp_path / "out"
    result = generate_watermarks(
        str(root),
        str(out_dir),
        WatermarkOptions(
            enable_location=True,
            location_text="厦门",
            enable_date=True,
            enable_species=True,
            enable_camera_params=True,
            enable_auto_enhance=False,
            watermark_style="inline",
            collage_mode="2h",
            collage_multi_species=True,
            collage_allow_one_all=True,
        ),
        prefer_folder_name_as_species=True,
        random_per_species=8,
    )
    assert result["ok"] == 1
    names = [p.name for p in out_dir.glob("*.jpg")]
    assert len(names) == 1
    assert "合拼" in names[0]
