#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpx_track.track_map import (  # noqa: E402
    _cluster_grid_metrics,
    _cluster_label_xytext,
    _cluster_photos_by_radius,
    _layout_cluster_thumb_grid,
    _observation_date_label,
)
from gpx_track.photo_collect import BirdPhoto  # noqa: E402


def test_observation_date_single_day():
    ph = BirdPhoto(
        path="a.jpg",
        species_cn="测试",
        when=datetime(2026, 8, 5, 10, 0),
        lat=24.0,
        lon=118.0,
    )
    assert _observation_date_label([], [ph]) == "2026年8月5日"


def test_observation_date_same_month_range():
    p1 = BirdPhoto(
        path="a.jpg",
        species_cn="A",
        when=datetime(2026, 8, 5, 8, 0),
        lat=24.0,
        lon=118.0,
    )
    p2 = BirdPhoto(
        path="b.jpg",
        species_cn="B",
        when=datetime(2026, 8, 8, 18, 0),
        lat=24.0,
        lon=118.0,
    )
    assert _observation_date_label([], [p1, p2]) == "2026年8月5日-8日"


def test_cluster_photos_by_radius_merges_nearby():
    """去重半径内不同种应合并为同一布局簇（避免竖条散点）。"""
    # ~0.2 km apart at lat 25
    p1 = BirdPhoto("a.jpg", "八哥", None, 25.5000, 119.7800)
    p2 = BirdPhoto("b.jpg", "黑枕王鹟", None, 25.5015, 119.7802)
    p3 = BirdPhoto("c.jpg", "暗绿绣眼鸟", None, 25.5030, 119.7801)
    # far away (~5 km north)
    p4 = BirdPhoto("d.jpg", "红隼", None, 25.5450, 119.7800)
    entries = [
        (p1, 0.1, 0.2),
        (p2, 0.11, 0.21),
        (p3, 0.12, 0.22),
        (p4, 0.5, 0.8),
    ]
    groups = _cluster_photos_by_radius(entries, radius_km=1.0)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 3]


def test_cluster_photos_by_radius_no_merge_when_far():
    p1 = BirdPhoto("a.jpg", "A", None, 25.50, 119.78)
    p2 = BirdPhoto("b.jpg", "B", None, 25.55, 119.78)
    entries = [(p1, 0.1, 0.2), (p2, 0.5, 0.8)]
    groups = _cluster_photos_by_radius(entries, radius_km=1.0)
    assert len(groups) == 2


def test_cluster_label_alternates_vertical():
    a = _cluster_label_xytext(0, 10.0, 44, dpi=120)
    b = _cluster_label_xytext(1, 10.0, 80, dpi=120)
    # 鸟名统一在上方，且偏移至少超过圆半径(pt)
    assert a[3] == "bottom" and b[3] == "bottom"
    assert a[1] > (44 * 0.5) * (72 / 120)
    assert b[1] > (80 * 0.5) * (72 / 120)


def test_cluster_grid_spacing_uses_diameter_plus_fifth():
    d = 50
    col_step, row_step, _, gap, _ = _cluster_grid_metrics(d, 10.0, dpi=120)
    assert gap == d * 0.4
    assert col_step == d + gap
    assert row_step > col_step


def test_layout_cluster_row_first_indices():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    positions, row_first = _layout_cluster_thumb_grid(
        ax, (0.2, 0.5), 7, 1, 40, 9.0
    )
    plt.close(fig)
    assert len(positions) == 7
    assert row_first == [0, 5]
    assert positions[0][0] < positions[1][0] < positions[2][0]


def test_island_scale_markers_stay_on_map():
    """大范围视野下全部鸟图应落在坐标轴内（不被推到画布外）。"""
    import matplotlib.pyplot as plt
    from datetime import datetime
    from PIL import Image
    import tempfile
    from gpx_track.track_map import _add_photo_markers

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_is_"))
    photos = []
    for i in range(10):
        p = td / f"s{i}.jpg"
        Image.new("RGB", (80, 80), (i * 20, 90, 140)).save(p)
        photos.append(
            BirdPhoto(
                str(p),
                f"种{i}",
                datetime(2026, 8, 7, 10, i),
                25.48 + 0.005 * i,
                119.82,
            )
        )
    fig, ax = plt.subplots(figsize=(9, 16), dpi=100)
    ax.set_xlim(119.70, 119.90)
    ax.set_ylim(25.40, 25.60)
    layout = _add_photo_markers(
        ax,
        photos,
        thumb_diameter=80,
        use_gcj=False,
        radius_km=1.0,
        basemap_style="none",
        on_basemap=False,
    )
    plt.close(fig)
    assert len(layout.displays) == 10
    x0, x1 = 119.70, 119.90
    y0, y1 = 25.40, 25.60
    assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in layout.displays)
