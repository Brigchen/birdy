#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpx_track.track_map import (  # noqa: E402
    _add_photo_markers,
    _choose_cluster_placement,
    _cluster_grid_metrics,
    _cluster_label_xytext,
    _cluster_photos_by_radius,
    _dedupe_photos_by_species_radius,
    _is_multi_individual_image,
    _clamp_positions_into_axes,
    _layout_cluster_thumb_grid,
    _max_cols_on_side,
    _min_thumb_sep_px,
    _observation_date_label,
    haversine_km,
)
from gpx_track.photo_collect import BirdPhoto  # noqa: E402


def test_east_up_puts_north_left_and_east_up():
    from gpx_track.track_map import _map_xy, _rotate_basemap_east_up, _track_wider_than_tall
    import numpy as np

    assert _track_wider_than_tall(114.0, 114.2, 30.50, 30.52)
    assert not _track_wider_than_tall(114.0, 114.01, 30.0, 30.3)
    south = _map_xy(114.1, 30.50, use_gcj=False, east_up=True)
    north = _map_xy(114.1, 30.52, use_gcj=False, east_up=True)
    west = _map_xy(114.0, 30.51, use_gcj=False, east_up=True)
    east = _map_xy(114.2, 30.51, use_gcj=False, east_up=True)
    assert north[0] < south[0]
    assert east[1] > west[1]
    img = np.zeros((4, 6, 3), dtype=np.float32)
    img[0, :, 0] = 1.0
    turned, extent = _rotate_basemap_east_up(img, (114.0, 114.2, 30.50, 30.52))
    assert turned.shape[0] == 6 and turned.shape[1] == 4
    assert turned[:, 0, 0].mean() == 1.0
    assert extent[0] < extent[1]
    assert extent[2] == 114.0 and extent[3] == 114.2


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


def test_layout_cluster_keeps_one_photo_per_species():
    """同一块网格里同种只出现一次（反嘴鹬不应并排两张）。"""
    p1 = BirdPhoto("a.jpg", "反嘴鹬", None, 25.5000, 119.7800)
    p2 = BirdPhoto("b.jpg", "反嘴鹬", None, 25.5004, 119.7801)
    p3 = BirdPhoto("c.jpg", "灰头麦鸡", None, 25.5002, 119.7800)
    groups = _cluster_photos_by_radius(
        [(p1, 0, 0), (p2, 0, 0), (p3, 0, 0)], radius_km=1.0
    )
    assert len(groups) == 1
    names = [t[0].species_cn for t in groups[0]]
    assert names.count("反嘴鹬") == 1
    assert names.count("灰头麦鸡") == 1


def test_layout_cluster_does_not_chain_same_species_beyond_radius():
    """中间夹其它种时，不得把半径外的第二张同种串进同一网格。"""
    # 纬度 0.009 ≈ 1.0km；0.016 ≈ 1.8km
    avocet_a = BirdPhoto("a.jpg", "反嘴鹬", None, 25.5000, 119.7800)
    other = BirdPhoto("x.jpg", "灰头麦鸡", None, 25.5065, 119.7800)
    avocet_b = BirdPhoto("b.jpg", "反嘴鹬", None, 25.5160, 119.7800)
    d_ab = haversine_km(25.5000, 119.7800, 25.5160, 119.7800)
    assert d_ab > 1.0
    groups = _cluster_photos_by_radius(
        [(avocet_a, 0, 0), (other, 0, 0), (avocet_b, 0, 0)],
        radius_km=1.0,
    )
    avocet_groups = sum(
        1
        for g in groups
        if any(t[0].species_cn == "反嘴鹬" for t in g)
    )
    assert avocet_groups == 2
    for g in groups:
        names = [t[0].species_cn for t in g]
        assert names.count("反嘴鹬") <= 1


def test_cluster_label_alternates_vertical():
    a = _cluster_label_xytext(0, 10.0, 44, dpi=120)
    b = _cluster_label_xytext(1, 10.0, 80, dpi=120)
    # 鸟名统一在上方，且偏移至少超过圆半径(pt)
    assert a[3] == "bottom" and b[3] == "bottom"
    assert a[1] > (44 * 0.5) * (72 / 120)
    assert b[1] > (80 * 0.5) * (72 / 120)


def test_cluster_grid_spacing_uses_diameter_plus_fifth():
    d = 50
    dpi = 120.0
    col_step, row_step, _, gap, _ = _cluster_grid_metrics(d, 10.0, dpi=dpi)
    assert gap == float(d) * 0.4
    assert col_step == float(d) + gap
    assert row_step > col_step


def test_cluster_block_bbox_covers_rendered_thumbs():
    """图块包围盒应包住每张 OffsetImage 窗口（网格步长仍用 diameter）。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        _cluster_block_bbox_px,
        _iter_map_thumb_artists,
        _plan_cluster_layouts,
        _thumb_cluster_index_from_gid,
    )

    fig, ax = plt.subplots(figsize=(8, 12), dpi=120)
    ax.set_xlim(115.10, 115.18)
    ax.set_ylim(30.18, 30.26)
    photos = [
        BirdPhoto(f"{i}.jpg", f"测试鸟种{i}", None, 30.20, 115.12 + 0.0001 * i)
        for i in range(6)
    ]
    layouts = _plan_cluster_layouts(
        ax,
        photos,
        max_markers=None,
        thumb_diameter=80,
        use_gcj=False,
        radius_km=0.3,
        east_up=False,
        top_axes=None,
        bottom_axes=None,
        track_xy=[],
    )
    assert layouts and any(lay.positions for lay in layouts)
    from gpx_track.track_map import _add_photo_markers

    _add_photo_markers(
        ax, photos, thumb_diameter=80, use_gcj=False,
        basemap_style="none", on_basemap=False, radius_km=0.3,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for cluster_idx, layout in enumerate(layouts):
        if not layout.positions:
            continue
        pts_px = [ax.transData.transform(p) for p in layout.positions]
        names = [ph.species_cn for ph, _, _ in layout.group]
        bb = _cluster_block_bbox_px(
            pts_px, thumb_diameter=80, label_fs=9.0, dpi=120.0, species_names=names,
        )
        for ab in _iter_map_thumb_artists(ax):
            gid = ab.get_gid() if hasattr(ab, "get_gid") else ""
            if _thumb_cluster_index_from_gid(gid or "") != cluster_idx:
                continue
            ext = ab.get_window_extent(renderer)
            assert bb[0] <= float(ext.x0) + 1.0
            assert bb[1] >= float(ext.x1) - 1.0
            assert bb[2] <= float(ext.y0) + 1.0
            assert bb[3] >= float(ext.y1) - 1.0
    plt.close(fig)


def test_layout_cluster_row_first_indices():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    positions, row_first = _layout_cluster_thumb_grid(
        ax, (0.2, 0.5), 7, 1, 40, 9.0, max_cols=5
    )
    plt.close(fig)
    assert len(positions) == 7
    assert row_first == [0, 5]
    assert positions[0][0] < positions[1][0] < positions[2][0]


def test_bird_labels_stay_below_title_band():
    """多行鸟名的上沿应落在标题块之下，下沿离开底边。"""
    import matplotlib.pyplot as plt

    from gpx_track.track_map import (
        _label_box_axes_frac_at,
        _marker_center_y_bounds,
        _title_band_bottom_axes,
    )

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    ax.set_xlim(115.0, 115.2)
    ax.set_ylim(30.0, 30.3)
    fig.canvas.draw()
    bottom, top = _title_band_bottom_axes(
        ax, "黄石西塞山", "月亮山", "2026-10-02", elev_south_frac=0.18
    )
    assert top < 0.95
    assert bottom >= 0.17
    bounds = _marker_center_y_bounds(
        ax, 80, 11.0, top_axes=top, bottom_axes=bottom
    )
    assert bounds is not None
    # 锚点贴在上沿，夹紧后鸟名仍低于标题
    pos, _ = _layout_cluster_thumb_grid(ax, (115.05, 30.28), 6, 1, 80, 11.0, max_cols=2)
    clamped = _clamp_positions_into_axes(ax, pos, 80, y_bounds=bounds)
    oy = _cluster_label_xytext(0, 11.0, 80, dpi=100)
    for x, y in clamped:
        box = _label_box_axes_frac_at(
            ax, x, y, "白胸翡翠", oy[0], oy[1], 11.0, oy[2], oy[3]
        )
        assert box[3] <= top + 0.02
    plt.close(fig)


def test_compass_aligns_with_title_and_summary_below():
    from gpx_track.track_map import _compass_axes_box, _try_symmetric_top_chrome

    y_top = 0.97
    box = _compass_axes_box("left", y_top)
    from gpx_track.track_map import _compass_center

    cx, cy = _compass_center("left", None, elev_panel_top=0.0, y_title_top=y_top)
    assert cx > 0.8
    assert abs(cy - (y_top - 0.02)) < 1e-6
    paired = _try_symmetric_top_chrome(0.22, 0.12, 0.18, 0.08, [], y_top)
    assert paired is not None
    _title, _ha, sum_box = paired
    assert sum_box[3] <= box[1] + 1e-6


def test_distant_photos_stay_near_their_gps():
    """相距较远的鸟图应留在各自记录点附近，而不是收成一个中心。"""
    import tempfile
    import matplotlib.pyplot as plt
    from PIL import Image

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_near_"))
    photos = []
    coords = [(30.20, 115.10), (30.24, 115.16)]
    for i, (lat, lon) in enumerate(coords):
        p = td / f"{i}.jpg"
        Image.new("RGB", (40, 40), (80, 80, 80)).save(p)
        photos.append(BirdPhoto(str(p), f"鸟{i}", datetime(2026, 10, 4, 9, i), lat, lon))
    fig, ax = plt.subplots(figsize=(6, 8), dpi=80)
    ax.set_xlim(115.05, 115.22)
    ax.set_ylim(30.16, 30.28)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=40, use_gcj=False, radius_km=1.0,
        basemap_style="none", on_basemap=False,
    )
    plt.close(fig)
    assert len(layout.displays) == 2
    for (lat, lon), (x, y) in zip(coords, layout.displays):
        assert abs(x - lon) < 0.03
        assert abs(y - lat) < 0.03


def test_matrix_shape_prefers_filled_grid():
    from gpx_track.track_map import _matrix_shape

    assert _matrix_shape(10) == (5, 2)
    assert _matrix_shape(12) == (4, 3)
    assert _matrix_shape(6) == (3, 2)
    # 14 张应排成 5×3 扁矩阵，而不是 2×7 超长竖条
    assert _matrix_shape(14) == (5, 3)
    assert _matrix_shape(16) == (4, 4)
    # 大簇不再封顶 5 列变成超高竖条
    c48, r48 = _matrix_shape(48)
    assert c48 >= 6 and r48 <= 8
    c51, r51 = _matrix_shape(51)
    assert c51 >= 6 and r51 <= 8
    from gpx_track.track_map import _cluster_col_cap
    assert _cluster_col_cap(2, 48) == 8
    assert _cluster_col_cap(7, 47) == 5
    assert _matrix_shape(47, max_cols=5)[0] == 5


def test_top_left_cluster_uses_wide_matrix_not_two_cols():
    """左上角记录点两侧有空时，应排宽矩阵，不能挤成两列超长竖条。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 32), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    side, cols, pos, _rf = _choose_cluster_placement(
        ax, (0.18, 0.82), 14, 80, 8.0, [], 80 * 1.34
    )
    plt.close(fig)
    rows = (14 + cols - 1) // cols
    assert len(pos) == 14
    assert cols >= 4, (side, cols, rows)
    assert rows <= 4, (side, cols, rows)


def test_large_cluster_does_not_straddle():
    """几十张的大簇不得水平居中骑在记录点上盖住地图。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 32), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    side, cols, pos, _rf = _choose_cluster_placement(
        ax, (0.42, 0.52), 51, 80, 8.0, [], 80 * 1.34
    )
    plt.close(fig)
    rows = (51 + cols - 1) // cols
    assert len(pos) == 51
    assert side != 0
    assert cols >= 6, (side, cols, rows)
    assert rows <= 8, (side, cols, rows)


def test_large_cluster_keeps_track_side_after_window_expand():
    """扩窗后图中线会变；大簇仍按轨迹范围中线留在原侧，避免被吸到湖面。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18, 32), dpi=120)
    # 模拟向左扩窗后：簇在图中线左侧，但在轨迹中线右侧
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    track = [(0.30 + 0.16 * i / 20.0, 0.2 + 0.03 * i) for i in range(21)]
    side, cols, pos, _rf = _choose_cluster_placement(
        ax, (0.44, 0.50), 48, 80, 8.0, [], 80 * 1.34,
        ignore_fit=True, track_xy=track,
    )
    plt.close(fig)
    assert side == 1, (side, cols)
    assert cols >= 6
    assert len(pos) == 48


def test_crowded_edge_cluster_widens_map():
    """贴边且鸟图很多时，向该点外侧扩大经纬度范围。"""
    from gpx_track.track_map import _expand_bounds_for_photo_packs

    photos = [
        BirdPhoto(f"{i}.jpg", f"鸟{i}", None, 30.201, 115.101)
        for i in range(12)
    ]
    lon0, lon1, lat0, lat1 = 115.10, 115.20, 30.18, 30.26
    out0, out1, out_lat0, out_lat1 = _expand_bounds_for_photo_packs(
        lon0, lon1, lat0, lat1, photos,
        width_px=2160, height_px=3840, thumb_diameter=120,
    )
    assert out0 < lon0 - 0.01
    assert out1 >= lon1 - 1e-9
    assert out_lat1 - out_lat0 >= (lat1 - lat0) - 1e-9


def test_cluster_regions_do_not_overlap_and_clear_track():
    """两点各有多张鸟图时，摆放区不交错，且圆心不压在轨迹上。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _cluster_region_bbox, _point_seg_dist, _species_label_up_px

    photos = []
    for i in range(4):
        photos.append(BirdPhoto(f"a{i}.jpg", f"甲{i}", None, 30.20, 115.10))
    for i in range(4):
        photos.append(BirdPhoto(f"b{i}.jpg", f"乙{i}", None, 30.201, 115.102))
    fig, ax = plt.subplots(figsize=(6, 10), dpi=100)
    ax.set_xlim(115.08, 115.14)
    ax.set_ylim(30.18, 30.24)
    fig.canvas.draw()
    track = [(115.08 + 0.001 * k, 30.20) for k in range(60)]
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=70, use_gcj=False,
        basemap_style="none", on_basemap=False, track_xy=track,
    )
    assert len(layout.displays) == 8
    dpi = float(ax.figure.dpi)
    pad = 70 * 0.62
    label_up = _species_label_up_px(70, 9.0, dpi)
    px = [ax.transData.transform(p) for p in layout.displays]
    groups = [px[:4], px[4:]]
    bb0 = _cluster_region_bbox(groups[0], pad, label_up)
    bb1 = _cluster_region_bbox(groups[1], pad, label_up)
    assert bb0[1] < bb1[0] or bb1[1] < bb0[0] or bb0[3] < bb1[2] or bb1[3] < bb0[2]
    tpx = [ax.transData.transform(p) for p in track]
    for p in px:
        nearest = min(
            _point_seg_dist(p[0], p[1], tpx[i][0], tpx[i][1], tpx[i + 1][0], tpx[i + 1][1])
            for i in range(len(tpx) - 1)
        )
        assert nearest > 70 * 0.45
    plt.close(fig)


def test_two_left_edge_clusters_do_not_stack():
    """左缘两个记录点的鸟图不得被夹到同一位置互相重叠。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        _cluster_occupancy_pad_px,
        _cluster_region_bbox,
        _rects_overlap,
        _species_label_up_px,
    )

    photos = []
    for i in range(4):
        photos.append(BirdPhoto(f"a{i}.jpg", f"甲{i}", None, 30.200, 115.100))
    for i in range(4):
        photos.append(BirdPhoto(f"b{i}.jpg", f"乙{i}", None, 30.218, 115.100))
    fig, ax = plt.subplots(figsize=(8, 14), dpi=100)
    ax.set_xlim(115.095, 115.135)
    ax.set_ylim(30.185, 30.245)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=70, use_gcj=False, radius_km=0.3,
        basemap_style="none", on_basemap=False,
    )
    assert len(layout.displays) == 8
    dpi = float(ax.figure.dpi)
    pad = _cluster_occupancy_pad_px(70, dpi)
    label_up = _species_label_up_px(70, 9.0, dpi)
    px = [ax.transData.transform(p) for p in layout.displays]
    bb0 = _cluster_region_bbox(px[:4], pad, label_up)
    bb1 = _cluster_region_bbox(px[4:], pad, label_up)
    plt.close(fig)
    assert not _rects_overlap(bb0, bb1, 12.0)


def test_cluster_block_clears_vertical_track():
    """鸟图圆心不得落在轨迹附近（整块避让后圆心仍须让开轨迹）。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _point_seg_dist, _thumb_display_px

    photos = [
        BirdPhoto(f"t{i}.jpg", f"鸟{i}", None, 30.20 + 0.0001 * i, 115.105)
        for i in range(8)
    ]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.set_xlim(115.08, 115.16)
    ax.set_ylim(30.18, 30.24)
    fig.canvas.draw()
    track = [(115.128, 30.18 + 0.001 * k) for k in range(60)]
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=70, use_gcj=False, radius_km=1.0,
        basemap_style="none", on_basemap=False, track_xy=track,
    )
    dpi = float(ax.figure.dpi)
    min_dist = _thumb_display_px(70, dpi) * 0.35
    px = [ax.transData.transform(p) for p in layout.displays]
    tpx = [ax.transData.transform(p) for p in track]
    plt.close(fig)
    assert layout.displays
    for p in px:
        nearest = min(
            _point_seg_dist(
                float(p[0]),
                float(p[1]),
                float(tpx[i][0]),
                float(tpx[i][1]),
                float(tpx[i + 1][0]),
                float(tpx[i + 1][1]),
            )
            for i in range(len(tpx) - 1)
        )
        assert nearest > min_dist


def test_max_cols_left_edge_forces_right_or_wrap():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    left_cols = _max_cols_on_side(ax, (0.04, 0.5), -1, 3, 90, 9.0)
    right_cols = _max_cols_on_side(ax, (0.04, 0.5), 1, 3, 90, 9.0)
    assert left_cols <= 1
    assert right_cols >= 2
    side, cols, pos, _rf = _choose_cluster_placement(
        ax, (0.04, 0.5), 3, 90, 9.0, [], 90 * 1.34
    )
    plt.close(fig)
    assert side == 1 or cols == 1
    assert len(pos) == 3
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    # 不得全挤在同一点
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    assert spread > 0.02


def test_left_edge_markers_do_not_overlap():
    """左缘多张鸟图应向右或分行，不能夹回后叠在一起。"""
    import tempfile
    import matplotlib.pyplot as plt
    from PIL import Image

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_le_"))
    photos = []
    for i, name in enumerate(("白鹡鸰", "白胸翡翠", "凸胸鹦鹉")):
        p = td / f"{name}.jpg"
        Image.new("RGB", (80, 80), (40 + i * 40, 90, 140)).save(p)
        photos.append(
            BirdPhoto(
                str(p),
                name,
                datetime(2026, 10, 2, 10, i),
                30.50 + 0.0002 * i,
                114.20,
            )
        )
    fig, ax = plt.subplots(figsize=(6, 10), dpi=100)
    ax.set_xlim(114.199, 114.220)
    ax.set_ylim(30.495, 30.520)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax,
        photos,
        thumb_diameter=80,
        use_gcj=False,
        radius_km=0.1,
        basemap_style="none",
        on_basemap=False,
    )
    sep = _min_thumb_sep_px(80, float(fig.dpi))
    pts = [ax.transData.transform(xy) for xy in layout.displays]
    plt.close(fig)
    assert len(pts) == 3
    for i in range(len(pts)):
        for j in range(i):
            d = (
                (float(pts[i][0]) - float(pts[j][0])) ** 2
                + (float(pts[i][1]) - float(pts[j][1])) ** 2
            ) ** 0.5
            assert d + 1e-3 >= sep


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


def test_occupancy_pad_matches_offsetimage_radius():
    """占位半径不得小于 OffsetImage 实际显示半径，否则右下会被裁切。"""
    from gpx_track.track_map import (
        _cluster_occupancy_pad_px,
        _thumb_display_px,
        _thumb_visual_radius_px,
    )

    dpi, d = 120.0, 120
    assert abs(_thumb_display_px(d, dpi) - 200.0) < 1e-6
    assert abs(_thumb_visual_radius_px(d, dpi) - 100.0) < 1e-6
    assert _cluster_occupancy_pad_px(d, dpi) >= _thumb_visual_radius_px(d, dpi)


def test_markers_visual_extent_stays_inside_axes():
    """鸟图中心加真实显示半径后仍应落在坐标轴内。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _cluster_occupancy_pad_px

    photos = [
        BirdPhoto(f"e{i}.jpg", f"鸟{i}", None, 30.20 + 0.0004 * i, 115.12)
        for i in range(12)
    ]
    fig, ax = plt.subplots(figsize=(9, 16), dpi=120)
    ax.set_xlim(115.06, 115.20)
    ax.set_ylim(30.14, 30.30)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=120, use_gcj=False, radius_km=1.0,
        basemap_style="none", on_basemap=False,
    )
    fig.canvas.draw()
    from gpx_track.track_map import _thumb_offsetimage_overflow_px

    assert layout.displays
    assert _thumb_offsetimage_overflow_px(ax) <= 1.5
    plt.close(fig)


def test_crowded_clusters_stay_inside_axes():
    """多个记录点分列轨迹左右时，真实鸟图窗口盒不得裁出画布。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _thumb_offsetimage_overflow_px

    photos = []
    for k, lon in enumerate((115.100, 115.108, 115.116, 115.124)):
        for i in range(8):
            photos.append(
                BirdPhoto(
                    f"c{k}_{i}.jpg", f"鸟{k}{i}", None,
                    30.200 + 0.0008 * i, lon,
                )
            )
    fig, ax = plt.subplots(figsize=(9, 16), dpi=120)
    ax.set_xlim(115.085, 115.145)
    ax.set_ylim(30.17, 30.25)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=80, use_gcj=False, radius_km=0.2,
        basemap_style="none", on_basemap=False,
        track_xy=[(115.112, 30.18 + 0.001 * t) for t in range(60)],
    )
    fig.canvas.draw()
    assert len(layout.displays) == 32
    assert _thumb_offsetimage_overflow_px(ax) <= 1.5
    plt.close(fig)


def _make_thumb_jpegs(td: Path, n: int, prefix: str = "t"):
    from PIL import Image

    paths = []
    for i in range(n):
        p = td / f"{prefix}{i}.jpg"
        Image.new("RGB", (96, 96), ((40 + i * 13) % 255, 90, 140)).save(p)
        paths.append(p)
    return paths


def test_offsetimage_artists_stay_inside_figure():
    """真实 OffsetImage 窗口盒必须落在 figure 内（savefig 前硬约束）。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _thumb_offsetimage_overflow_px

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_ab_"))
    photos = []
    for i, p in enumerate(_make_thumb_jpegs(td, 12)):
        photos.append(
            BirdPhoto(str(p), f"鸟{i}", None, 30.20 + 0.0004 * i, 115.12)
        )
    fig, ax = plt.subplots(figsize=(9, 16), dpi=120)
    ax.set_xlim(115.08, 115.18)
    ax.set_ylim(30.16, 30.28)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=120, use_gcj=False, radius_km=1.0,
        basemap_style="none", on_basemap=False,
    )
    assert layout.displays
    assert _thumb_offsetimage_overflow_px(ax) <= 1.5
    plt.close(fig)


def test_crowded_offsetimages_stay_inside_figure():
    """多记录点 + 真实鸟图：左/右/下窗口盒不得裁出画布。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _thumb_offsetimage_overflow_px

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_cr_"))
    photos = []
    k = 0
    for lon in (115.100, 115.108, 115.116, 115.124):
        for i in range(8):
            p = _make_thumb_jpegs(td, 1, prefix=f"c{k}_")[0]
            photos.append(
                BirdPhoto(str(p), f"鸟{k}{i}", None, 30.200 + 0.0008 * i, lon)
            )
            k += 1
    fig, ax = plt.subplots(figsize=(9, 16), dpi=120)
    ax.set_xlim(115.09, 115.14)
    ax.set_ylim(30.18, 30.24)
    fig.canvas.draw()
    layout = _add_photo_markers(
        ax, photos, thumb_diameter=80, use_gcj=False, radius_km=0.2,
        basemap_style="none", on_basemap=False,
        track_xy=[(115.112, 30.18 + 0.001 * t) for t in range(60)],
    )
    assert len(layout.displays) == 32
    assert _thumb_offsetimage_overflow_px(ax) <= 1.5
    plt.close(fig)


def _wanghu_sample_paths():
    return (
        Path(r"C:\Users\brigc\Pictures\Photo_Output\classification_20261004_网湖"),
        Path(r"C:\Users\brigc\Downloads\20261004户外步行-网湖.gpx"),
    )


def _load_wanghu_matched_photos():
    from gpx_track.gpx_io import load_gpx
    from gpx_track.gpx_match import match_photos_to_track
    from gpx_track.photo_collect import collect_bird_photos
    from gpx_track.track_map import _photo_path_key

    class_dir, gpx_path = _wanghu_sample_paths()
    track = load_gpx(str(gpx_path))
    photos = collect_bird_photos(str(class_dir), require_gps=False)
    matched = match_photos_to_track(
        [ph.path for ph in photos],
        track,
        max_delta_seconds=3600.0,
        prefer_exif_gps=False,
    )
    by_path = {_photo_path_key(m["path"]): m for m in matched}
    enriched = []
    for ph in photos:
        m = by_path.get(_photo_path_key(ph.path))
        if not m:
            continue
        enriched.append(
            BirdPhoto(
                path=ph.path,
                species_cn=ph.species_cn,
                when=ph.when or m.get("time"),
                when_track=m.get("when_track"),
                lat=m["lat"],
                lon=m["lon"],
                ele=m.get("ele"),
                order_cn=ph.order_cn,
                family_cn=ph.family_cn,
            )
        )
    return track, enriched


def test_wanghu_layout_block_pairwise_overlap():
    """网湖 0.3km：规划阶段图块包围盒两两交叠面积应较小（实测回归）。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        EXPORT_DPI,
        EXPORT_HEIGHT_PX,
        EXPORT_WIDTH_PX,
        _cluster_block_bbox_px,
        _dedupe_photos_by_species_radius,
        _figure_size_inches,
        _plan_cluster_layouts,
        _rects_overlap_area,
        _track_map_thumb_diameters,
    )

    class_dir, gpx_path = _wanghu_sample_paths()
    if not class_dir.is_dir() or not gpx_path.is_file():
        import pytest

        pytest.skip("网湖分类目录或 GPX 不在本机")

    track, matched = _load_wanghu_matched_photos()
    photos = _dedupe_photos_by_species_radius(matched, 0.3)
    fig_w, fig_h = _figure_size_inches(
        EXPORT_WIDTH_PX, EXPORT_HEIGHT_PX, EXPORT_DPI
    )
    thumb_map, _ = _track_map_thumb_diameters(EXPORT_HEIGHT_PX)
    from gpx_track.track_map import _map_xy, _plot_map_ax

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=EXPORT_DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    _plot_map_ax(
        ax,
        track,
        photos,
        "网湖",
        map_title="观鸟记录",
        date_label="2026年10月4日",
        thumb_diameter=thumb_map,
        basemap_style="none",
        map_width_px=EXPORT_WIDTH_PX,
        map_height_px=EXPORT_HEIGHT_PX,
        radius_km=0.3,
    )
    track_xy = [
        _map_xy(p.lon, p.lat, use_gcj=False, east_up=False)
        for p in track[:: max(1, len(track) // 280)]
    ]
    layouts = _plan_cluster_layouts(
        ax,
        photos,
        max_markers=None,
        thumb_diameter=thumb_map,
        use_gcj=False,
        radius_km=0.3,
        east_up=False,
        top_axes=None,
        bottom_axes=None,
        track_xy=track_xy,
    )
    label_fs = 9.0
    bbs = []
    for layout in layouts:
        if not layout.positions:
            continue
        pts_px = [ax.transData.transform(p) for p in layout.positions]
        names = [ph.species_cn for ph, _, _ in layout.group]
        bbs.append(
            _cluster_block_bbox_px(
                pts_px,
                thumb_diameter=thumb_map,
                label_fs=label_fs,
                dpi=EXPORT_DPI,
                species_names=names,
            )
        )
    plt.close(fig)
    overlap = 0.0
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            overlap += _rects_overlap_area(bbs[i], bbs[j], 8.0)
    assert overlap < 2.5e6, f"网湖图块规划交叠过大 {overlap:.0f}px²"


def test_wanghu_radii_thumbs_inside_figure():
    """网湖真数据 0.2 / 0.3 / 1.0 km 出图后鸟图盒 ⊆ 画布。"""
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        EXPORT_DPI,
        EXPORT_HEIGHT_PX,
        EXPORT_WIDTH_PX,
        _dedupe_photos_by_species_radius,
        _figure_size_inches,
        _plot_map_ax,
        _thumb_offsetimage_overflow_px,
        _track_map_thumb_diameters,
    )

    class_dir, gpx_path = _wanghu_sample_paths()
    if not class_dir.is_dir() or not gpx_path.is_file():
        import pytest

        pytest.skip("网湖分类目录或 GPX 不在本机")

    track, matched = _load_wanghu_matched_photos()
    fig_w, fig_h = _figure_size_inches(
        EXPORT_WIDTH_PX, EXPORT_HEIGHT_PX, EXPORT_DPI
    )
    thumb_map, _ = _track_map_thumb_diameters(EXPORT_HEIGHT_PX)
    for radius_km in (0.2, 0.3, 1.0):
        photos = _dedupe_photos_by_species_radius(matched, radius_km)
        assert photos, f"网湖 {radius_km} km 未匹配到鸟图"
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=EXPORT_DPI, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        _plot_map_ax(
            ax,
            track,
            photos,
            "网湖",
            map_title="观鸟记录",
            date_label="2026年10月4日",
            thumb_diameter=thumb_map,
            basemap_style="none",
            map_width_px=EXPORT_WIDTH_PX,
            map_height_px=EXPORT_HEIGHT_PX,
            radius_km=radius_km,
        )
        overflow = _thumb_offsetimage_overflow_px(ax)
        plt.close(fig)
        assert overflow <= 1.5, (
            f"网湖 {radius_km} km 鸟图越界 {overflow:.1f}px（左/右/下）"
        )


def test_reserve_south_lat_keeps_north_and_places_old_south():
    from gpx_track.amap_basemap import reserve_south_lat

    lon0, lon1, lat0, lat1 = 120.0, 121.0, 30.0, 31.0
    a, b, c, d = reserve_south_lat(lon0, lon1, lat0, lat1, 0.2)
    assert a == lon0 and b == lon1 and d == lat1
    assert c < lat0
    assert abs((lat0 - c) / (d - c) - 0.2) < 1e-9


def test_thumb_square_box_uses_full_short_side():
    from gpx_track.track_map import _thumb_square_box

    # 横图：短边 100，应取 100×100 居中，不再只取中心一半
    x0, y0, x1, y1 = _thumb_square_box(200, 100)
    assert (x1 - x0, y1 - y0) == (100, 100)
    assert (x0, y0) == (50, 0)
    x0, y0, x1, y1 = _thumb_square_box(80, 120)
    assert (x1 - x0, y1 - y0) == (80, 80)
    assert (x0, y0) == (0, 20)


def test_elevation_south_reserve_covers_panel_and_thumbs():
    from gpx_track.track_map import ELEV_PANEL_TOP_AXES, _elevation_south_reserve_frac

    frac = _elevation_south_reserve_frac(80, 2560)
    assert frac > ELEV_PANEL_TOP_AXES
    assert frac < 0.40


def test_fit_lon_to_aspect_after_south_reserve():
    from gpx_track.amap_basemap import fit_lon_to_aspect, reserve_south_lat

    aspect = 2160 / 3840
    lon0, lon1 = 120.0, 120.0 + aspect
    lat0, lat1 = 30.0, 31.0
    a, b, c, d = reserve_south_lat(lon0, lon1, lat0, lat1, 0.2)
    assert (b - a) / (d - c) < aspect - 1e-9
    a, b, c, d = fit_lon_to_aspect(a, b, c, d, aspect)
    assert abs((b - a) / (d - c) - aspect) < 1e-9
    assert d == lat1
    assert c < lat0


def test_split_key_place_queries_comma_semicolon_and_coords():
    from gpx_track.track_map import split_key_place_queries

    assert split_key_place_queries("西湖，竹屿湖;观景台") == [
        "西湖",
        "竹屿湖",
        "观景台",
    ]
    assert split_key_place_queries("竹屿湖、观景台、东坪山") == [
        "竹屿湖",
        "观景台",
        "东坪山",
    ]
    assert split_key_place_queries("24.47, 117.94；西湖") == [
        "24.47, 117.94",
        "西湖",
    ]
    assert split_key_place_queries("西湖, 西湖") == ["西湖"]


def test_region_city_tokens_from_xiamen_title():
    from gpx_track.amap_basemap import region_city_tokens

    toks = region_city_tokens("厦门", "思明", "厦门东坪山", "福建")
    assert "厦门" in toks
    assert "厦门东坪山" in toks


def test_pick_geocode_near_map_rejects_far_namesake():
    from gpx_track.amap_basemap import pick_geocode_near_map

    xiamen = (24.48, 118.09)
    local = (24.50, 118.05)
    far = (45.76, 126.64)
    assert pick_geocode_near_map([far, local], xiamen, max_prefer_km=80.0) == local
    assert pick_geocode_near_map([far], xiamen, max_prefer_km=80.0) is None


def test_place_name_match_prefers_reservoir_over_bus_stop():
    from gpx_track.amap_basemap import place_name_match_score

    assert place_name_match_score("东山水库", "东山水库") == 100
    assert place_name_match_score("东山水库", "东山水库公交站") == 45
    assert place_name_match_score("东山水库", "东坪山水库") == 0
    assert place_name_match_score("龟石望顶", "龟石望顶") == 100


def test_pick_amap_place_prefers_reservoir_over_bus_stop():
    from gpx_track.amap_basemap import PlaceHit, pick_amap_place_hit

    center = (24.456, 118.144)
    bus = PlaceHit(24.457, 118.143, "东山水库公交站", "交通设施", "poi_around")
    lake = PlaceHit(24.460, 118.148, "东山水库", "风景名胜;湖泊", "poi_text")
    other = PlaceHit(24.455, 118.140, "东山路", "道路", "geo")
    hit = pick_amap_place_hit(
        "东山水库", [bus, other, lake], center, max_prefer_km=80.0
    )
    assert hit is not None
    assert hit.name == "东山水库"


def test_pick_amap_place_rejects_bus_stop_when_require_strong():
    from gpx_track.amap_basemap import PlaceHit, pick_amap_place_hit

    center = (24.456, 118.144)
    bus = PlaceHit(24.457, 118.143, "东山水库公交站", "交通设施", "poi_around")
    assert (
        pick_amap_place_hit(
            "东山水库",
            [bus],
            center,
            max_prefer_km=80.0,
            require_strong=True,
        )
        is None
    )


def test_pick_amap_place_prefers_closer_exact_namesake():
    from gpx_track.amap_basemap import PlaceHit, pick_amap_place_hit

    center = (24.456, 118.144)
    far = PlaceHit(24.80, 118.10, "东山水库", "风景名胜", "poi_text")
    near = PlaceHit(24.458, 118.145, "东山水库", "风景名胜;湖泊", "poi_text")
    hit = pick_amap_place_hit(
        "东山水库", [far, near], center, max_prefer_km=80.0
    )
    assert hit is near


def test_amap_city_params_drops_province_when_city_exists():
    from gpx_track.amap_basemap import _amap_city_params

    cities = _amap_city_params("厦门", "思明", "厦门东坪山", "福建")
    assert "厦门" in cities
    assert "思明" in cities
    assert "福建" not in cities
    assert "厦门东坪山" not in cities


def test_geocode_key_places_accepts_latlon():
    from gpx_track.track_map import geocode_key_places

    found, failed = geocode_key_places("30.27,120.16")
    assert failed == []
    assert len(found) == 1
    assert abs(found[0].lat - 30.27) < 1e-9
    assert abs(found[0].lon - 120.16) < 1e-9


def test_add_key_place_markers_avoids_thumb_box():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        KeyPlace,
        MapMarkerLayout,
        _add_key_place_markers,
        _rect_overlap_axes_frac,
    )

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    ax.set_xlim(119.7, 119.9)
    ax.set_ylim(25.4, 25.6)
    bird_thumb = (0.08, 0.08, 0.22, 0.18)
    layout = MapMarkerLayout(
        displays=[],
        label_boxes_axes=[],
        thumb_boxes_axes=[bird_thumb],
    )
    layout = _add_key_place_markers(
        ax,
        [KeyPlace("观景台", 25.50, 119.80)],
        layout,
        use_gcj=False,
        thumb_diameter=40,
        basemap_style="none",
        on_basemap=False,
    )
    plt.close(fig)
    assert layout.label_boxes_axes
    pin = layout.thumb_boxes_axes[-1]
    label = layout.label_boxes_axes[-1]
    assert not _rect_overlap_axes_frac(pin, bird_thumb, margin=0.004)
    assert not _rect_overlap_axes_frac(label, bird_thumb, margin=0.004)


def test_add_key_place_markers_shows_all_nearby_names():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        KeyPlace,
        MapMarkerLayout,
        _add_key_place_markers,
        _rect_overlap_axes_frac,
    )

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    ax.set_xlim(119.7, 119.9)
    ax.set_ylim(25.4, 25.6)
    places = [
        KeyPlace("竹屿湖", 25.50, 119.80),
        KeyPlace("观景台", 25.501, 119.801),
        KeyPlace("东坪山", 25.502, 119.802),
    ]
    layout = _add_key_place_markers(
        ax,
        places,
        MapMarkerLayout([], [], []),
        use_gcj=False,
        thumb_diameter=40,
        basemap_style="none",
        on_basemap=False,
    )
    shown = {t.get_text() for t in ax.texts}
    plt.close(fig)
    assert {p.name for p in places} <= shown
    boxes = layout.label_boxes_axes[-3:]
    assert len(boxes) == 3
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _rect_overlap_axes_frac(boxes[i], boxes[j], margin=0.001)


def test_include_nearby_key_places_ignores_far_city():
    from gpx_track.track_map import KeyPlace, _include_nearby_key_places

    bounds = (119.7, 119.9, 25.4, 25.6)
    near = KeyPlace("竹屿湖", 25.50, 119.80)
    far = KeyPlace("北京", 39.90, 116.40)
    out = _include_nearby_key_places(bounds, [near, far])
    assert out is not None
    assert out[0] <= 119.80 <= out[1]
    assert out[2] <= 25.50 <= out[3]
    assert not (out[0] <= 116.40 <= out[1] and out[2] <= 39.90 <= out[3])


def test_partition_key_places_by_view_flags_out_of_range():
    from gpx_track.track_map import (
        KeyPlace,
        format_key_place_alerts,
        partition_key_places_by_view,
    )

    bounds = (119.7, 119.9, 25.4, 25.6)
    near = KeyPlace("竹屿湖", 25.50, 119.80)
    far = KeyPlace("西湖", 30.25, 120.15)
    inside, outside = partition_key_places_by_view(bounds, [near, far])
    assert [p.name for p in inside] == ["竹屿湖"]
    assert [p.name for p in outside] == ["西湖"]
    alert = format_key_place_alerts(
        {"key_places_out_of_range": "西湖 (30.25000, 120.15000)"}
    )
    assert "不在当前地图范围内" in alert
    assert "西湖" in alert


def test_key_place_label_larger_than_species():
    from gpx_track.track_map import (
        _KEY_PLACE_LABEL_HEIGHT_DIV,
        _MAP_SPECIES_LABEL_HEIGHT_DIV,
    )

    assert _KEY_PLACE_LABEL_HEIGHT_DIV < _MAP_SPECIES_LABEL_HEIGHT_DIV


def test_thumb_offset_zoom_uses_point_scale():
    import numpy as np
    from gpx_track.track_map import _thumb_display_px, _thumb_offset_zoom

    arr = np.zeros((100, 100, 4), dtype=np.float32)
    dpi = 120.0
    diameter = 80
    zoom = _thumb_offset_zoom(arr, diameter, dpi)
    displayed_px = 100 * zoom * dpi / 72.0
    assert abs(displayed_px - _thumb_display_px(diameter, dpi)) < 1e-6


def test_separate_positions_px_pushes_apart():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _min_thumb_sep_px, _separate_positions_px

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    sep = _min_thumb_sep_px(40)
    out = _separate_positions_px(ax, [(0.5, 0.5), (0.5, 0.5)], sep)
    p0 = ax.transData.transform(out[0])
    p1 = ax.transData.transform(out[1])
    plt.close(fig)
    dist = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
    assert dist + 1e-6 >= sep


def test_symmetric_top_chrome_places_count_opposite_and_lower():
    from gpx_track.track_map import _try_symmetric_top_chrome

    paired = _try_symmetric_top_chrome(0.22, 0.12, 0.18, 0.08, [], 0.97)
    assert paired is not None
    title, sum_ha, sum_box = paired
    assert {title.ha, sum_ha} == {"left", "right"}
    assert sum_box[3] < title.y_top


def test_summary_uses_largest_empty_when_top_blocked():
    from gpx_track.track_map import _pick_summary_placement

    obstacles = [(0.0, 0.70, 1.0, 1.0)]
    title_box = (0.05, 0.82, 0.30, 0.97)
    _ha, box = _pick_summary_placement(
        0.20,
        0.08,
        obstacles,
        title_ha="left",
        title_y_top=0.97,
        title_box=title_box,
        elev_panel_top=0.17,
    )
    assert box[3] <= 0.70 + 1e-6
    assert box[1] >= 0.17


def test_title_anchor_shifts_away_from_thumbs():
    from gpx_track.track_map import _pick_title_anchor

    # 左上被鸟图占满时，应改到右侧
    obstacles = [(0.02, 0.82, 0.42, 1.0)]
    x, ha, yt = _pick_title_anchor(
        block_w=0.22, block_h=0.12, obstacles=obstacles, y_top=0.97
    )
    assert ha in ("right", "center") or x > 0.45
    assert yt <= 0.97


def test_cjk_text_width_factor_wider_than_digits():
    from gpx_track.track_map import _text_width_factor

    assert _text_width_factor("白鹡鸰") > _text_width_factor("123")


def test_elevation_labels_same_x_do_not_overlap():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        _elev_label_box_display,
        _elev_label_ha_for_marker,
        _layout_elevation_species_labels,
        _rect_overlap_fraction,
    )

    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    markers = [
        (3.0, 40.0, "白鹡鸰"),
        (3.05, 42.0, "暗绿绣眼鸟"),
        (3.1, 38.0, "红隼"),
    ]
    layouts = _layout_elevation_species_labels(
        ax, markers, 8.0, data_x_max=10.0
    )
    boxes = []
    for _ad, _, ld, le, name in layouts:
        ha = _elev_label_ha_for_marker(ld, 10.0)
        boxes.append(_elev_label_box_display(ax, ld, le, name, 8.0, ha=ha))
    plt.close(fig)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert _rect_overlap_fraction(boxes[i], boxes[j]) <= 1e-6


def test_elevation_nearby_labels_first_at_top_later_move_down():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        _elev_label_box_data,
        _elev_label_ha_for_marker,
        _elev_name_size_data,
        _layout_elevation_species_labels,
        _rect_overlap_data,
    )

    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    markers = [
        (3.0, 40.0, "白鹡鸰"),
        (3.08, 42.0, "暗绿绣眼鸟"),
        (3.16, 38.0, "红隼"),
        (8.5, 50.0, "夜鹭"),
    ]
    layouts = _layout_elevation_species_labels(
        ax, markers, 8.0, data_x_max=10.0
    )
    _w, h = _elev_name_size_data(ax, "暗绿绣眼鸟", 8.0)
    boxes = []
    for ad, _ae, ld, le, name in layouts:
        assert abs(ld - ad) < 1e-9
        ha = _elev_label_ha_for_marker(ld, 10.0)
        boxes.append(_elev_label_box_data(ax, ld, le, name, 8.0, ha=ha))
    near = [row for row in layouts if row[0] < 4]
    far = [row for row in layouts if row[0] > 7]
    plt.close(fig)
    near_les = [le for _, _, _, le, _ in near]
    assert near_les[0] > 85.0
    assert near_les[0] > near_les[1] > near_les[2]
    assert near_les[0] - near_les[1] >= h * 0.85
    assert far[0][3] > 85.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _rect_overlap_data(boxes[i], boxes[j])


def test_elevation_labels_far_apart_share_top():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _layout_elevation_species_labels

    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    layouts = _layout_elevation_species_labels(
        ax,
        [(2.0, 40.0, "白鹡鸰"), (8.0, 45.0, "夜鹭")],
        8.0,
        data_x_max=10.0,
    )
    plt.close(fig)
    assert len(layouts) == 2
    assert abs(layouts[0][3] - layouts[1][3]) < 1e-6
    assert layouts[0][3] > 85.0


def test_elev_name_occupancy_is_small_on_inset_panel():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import _elev_name_size_data

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    outer = ax.inset_axes([0.04, 0.03, 0.92, 0.14])
    inner = outer.inset_axes([0.04, 0.1, 0.92, 0.80])
    inner.set_xlim(0, 8)
    inner.set_ylim(0, 80)
    fig.canvas.draw()
    _w, h = _elev_name_size_data(inner, "暗绿绣眼鸟", 7.0)
    plt.close(fig)
    assert h < 80 * 0.18


def test_elevation_many_labels_do_not_share_same_y():
    import matplotlib.pyplot as plt
    from gpx_track.track_map import (
        _elev_label_box_data,
        _elev_label_ha_for_marker,
        _layout_elevation_species_labels,
        _rect_overlap_data,
    )

    fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    names = ["白鹡鸰", "暗绿绣眼鸟", "红隼", "夜鹭", "白鹭", "池鹭"]
    markers = [(3.0 + i * 0.04, 50.0, name) for i, name in enumerate(names)]
    layouts = _layout_elevation_species_labels(
        ax, markers, 8.0, data_x_max=10.0
    )
    boxes = []
    ys = []
    for _ad, _ae, ld, le, name in layouts:
        ha = _elev_label_ha_for_marker(ld, 10.0)
        boxes.append(_elev_label_box_data(ax, ld, le, name, 8.0, ha=ha))
        ys.append(round(le, 4))
    plt.close(fig)
    assert len(set(ys)) == len(ys)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _rect_overlap_data(boxes[i], boxes[j])


def test_track_line_is_solid_on_basemap():
    import matplotlib.pyplot as plt
    from gpx_track.gpx_io import GpxPoint
    from gpx_track.track_map import _draw_track_on_ax

    track = [
        GpxPoint(time=None, lat=25.50, lon=119.78, ele=None),
        GpxPoint(time=None, lat=25.51, lon=119.79, ele=None),
    ]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(119.7, 119.9)
    ax.set_ylim(25.4, 25.6)
    _draw_track_on_ax(ax, track, use_gcj=False, on_basemap=True)
    styles = [ln.get_linestyle() for ln in ax.lines]
    plt.close(fig)
    assert styles
    assert all(s in ("-", "solid") for s in styles)


def _mk_photo(name: str, species: str, when: datetime, lat=25.95, lon=119.36):
    return BirdPhoto(name, species, when, lat, lon)


def test_is_multi_individual_image_token_boundary():
    assert _is_multi_individual_image("DSC01036_01239_all.jpg")
    assert _is_multi_individual_image("x-all.jpeg")
    assert _is_multi_individual_image("ALL.png")
    # 仅独立 token all 命中，ball/wall/install 等不能误判
    assert not _is_multi_individual_image("ball_game.jpg")
    assert not _is_multi_individual_image("wall.jpg")
    assert not _is_multi_individual_image("鸻形目_塍鹬属_inst02_00574.jpg")


def test_dedupe_prefers_single_over_all_at_same_site():
    base = datetime(2026, 9, 26, 10, 0, 0)
    # all 多个体图时间最早（原逻辑会先选中它），单体图随后出现
    photos = [
        _mk_photo("DSC01036_01239_all.jpg", "斑尾塍鹬", base),
        _mk_photo(
            "鸻形目_塍鹬属_斑尾塍鹬_inst02_00574.jpg",
            "斑尾塍鹬",
            base.replace(minute=2),
        ),
        _mk_photo(
            "鸻形目_塍鹬属_斑尾塍鹬_inst03_00566.jpg",
            "斑尾塍鹬",
            base.replace(minute=4),
        ),
    ]
    dropped = []
    picked = _dedupe_photos_by_species_radius(photos, 1.0, dropped=dropped)
    assert len(picked) == 1
    assert not _is_multi_individual_image(picked[0].path)
    assert "inst02" in picked[0].path
    # 被替换的 all 图与后续重复图都应记入 dropped
    assert any("all" in p for p, _ in dropped)
    assert len(dropped) == 2


def test_dedupe_keeps_all_when_only_multi_images():
    base = datetime(2026, 9, 26, 10, 0, 0)
    photos = [
        _mk_photo("DSC01036_01239_all.jpg", "斑尾塍鹬", base),
        _mk_photo("DSC01208_01250_all.jpg", "斑尾塍鹬", base.replace(minute=5)),
    ]
    picked = _dedupe_photos_by_species_radius(photos, 1.0)
    # 该位点只有多个体图时仍保留一张标注，避免该物种在地图上消失
    assert len(picked) == 1
    assert picked[0].path == "DSC01036_01239_all.jpg"


def test_dedupe_all_rule_independent_per_species():
    base = datetime(2026, 9, 26, 10, 0, 0)
    photos = [
        _mk_photo("a_all.jpg", "斑尾塍鹬", base, lat=25.950, lon=119.360),
        _mk_photo("b_inst.jpg", "斑尾塍鹬", base.replace(minute=1), lat=25.950, lon=119.360),
        # 另一物种同位置只有 all 图，应正常保留
        _mk_photo("c_all.jpg", "中杓鹬", base.replace(minute=2), lat=25.950, lon=119.360),
        # 同物种相距 >1km 的位点独立去重
        _mk_photo("d_all.jpg", "斑尾塍鹬", base.replace(minute=3), lat=26.000, lon=119.400),
    ]
    picked = _dedupe_photos_by_species_radius(photos, 1.0)
    paths = {p.path for p in picked}
    assert paths == {"b_inst.jpg", "c_all.jpg", "d_all.jpg"}


def test_dedupe_cluster_anchor_does_not_drift_on_representative_swap():
    """单体图替换 all 代表后簇锚点必须保持在首图位置：
    否则锚点漂移会把围绕原锚点的后续照片挤出簇外，另建第二簇。"""
    base = datetime(2026, 9, 26, 10, 0, 0)
    p0 = (25.950, 119.360)
    # B 距 P0 约 0.99km（同簇，单体替换 all 代表）
    p_b = (25.9415, 119.363)
    # C 距 P0 约 0.26km，但距 B 约 1.2km：锚点漂移时会被误建为第二簇
    p_c = (25.9515, 119.358)
    assert haversine_km(*p0, *p_b) < 1.0
    assert haversine_km(*p0, *p_c) < 0.5
    assert haversine_km(*p_b, *p_c) > 1.0

    photos = [
        _mk_photo("a_all.jpg", "池鹭", base, lat=p0[0], lon=p0[1]),
        _mk_photo("b_inst.jpg", "池鹭", base.replace(minute=2), lat=p_b[0], lon=p_b[1]),
        _mk_photo("c_inst.jpg", "池鹭", base.replace(minute=4), lat=p_c[0], lon=p_c[1]),
    ]
    dropped = []
    picked = _dedupe_photos_by_species_radius(photos, 1.0, dropped=dropped)
    assert len(picked) == 1
    assert picked[0].path == "b_inst.jpg"
    assert any(p == "c_inst.jpg" for p, _ in dropped)


def _wide_ax():
    """构造大视野 ax：80px 鸟图直径对应约 7~8 公里（远大于去重半径），
    用于验证渲染层不会因像素接近而吞并半径外的独立标注点。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(119.25, 119.45)  # 约 20km 宽
    ax.set_ylim(25.85, 26.05)
    return fig, ax


def _render(ax, photos, radius_km, thumb=80):
    return _add_photo_markers(
        ax,
        photos,
        thumb_diameter=thumb,
        use_gcj=False,
        basemap_style="none",
        on_basemap=False,
        radius_km=radius_km,
        resolve_overlaps=False,  # 只验证分组/标注点，排除像素推开干扰
    )


def _anchor_dot_count(ax) -> int:
    """每个布局簇绘制一个带白色描边的定位圆点；缩略图加载失败的 fallback
    散点无白边，据此只统计定位圆点数（即标注点数）。"""
    n = 0
    for c in ax.collections:
        cols = c.get_edgecolors()
        if cols is not None and len(cols) and tuple(cols[0])[:3] == (1.0, 1.0, 1.0):
            n += 1
    return n


def test_markers_strictly_follow_config_radius_not_visual_scale():
    """同物种两张相距约 1.5km：语义去重半径 1km 下为两个独立标注点。
    即使在大视野上它们像素间距不足一个鸟图，也必须各自保留定位圆点，
    不允许渲染层按更宽的视觉半径把它们并成一个点。"""
    import matplotlib.pyplot as plt

    base = datetime(2026, 9, 26, 10, 0, 0)
    p1, p2 = (25.950, 119.360), (25.963, 119.362)
    d = haversine_km(*p1, *p2)
    assert 1.0 < d < 2.0, d
    photos = _dedupe_photos_by_species_radius(
        [
            _mk_photo("a_inst.jpg", "池鹭", base, lat=p1[0], lon=p1[1]),
            _mk_photo("b_inst.jpg", "池鹭", base.replace(minute=30), lat=p2[0], lon=p2[1]),
        ],
        1.0,
    )
    assert len(photos) == 2

    fig, ax = _wide_ax()
    try:
        layout = _render(ax, photos, 1.0)
        assert len(layout.thumb_boxes_axes) == 2
        # 两个独立标注点各有一个定位圆点，不允许按视觉像素并点
        assert _anchor_dot_count(ax) == 2
    finally:
        plt.close(fig)


def test_markers_small_radius_keeps_points_beyond_radius():
    """半径 0.1km 时，相距约 0.15km 的同种照片（主去重已在更小半径合成过的前提下）
    必须仍是两个标注点；半径放大到 0.5km 才合并。"""
    import matplotlib.pyplot as plt

    base = datetime(2026, 9, 26, 10, 0, 0)
    # 纬度差 0.0014 ≈ 0.156km
    p1, p2 = (25.9500, 119.3600), (25.9514, 119.3600)
    d = haversine_km(*p1, *p2)
    assert 0.1 < d < 0.5, d

    fig, ax = _wide_ax()
    try:
        small = _dedupe_photos_by_species_radius(
            [
                _mk_photo("a.jpg", "斑嘴鸭", base, lat=p1[0], lon=p1[1]),
                _mk_photo("b.jpg", "斑嘴鸭", base.replace(minute=5), lat=p2[0], lon=p2[1]),
            ],
            0.1,
        )
        layout_small = _render(ax, small, 0.1)
        assert len(layout_small.thumb_boxes_axes) == 2
        assert _anchor_dot_count(ax) == 2

        ax.cla()
        ax.set_xlim(119.25, 119.45)
        ax.set_ylim(25.85, 26.05)
        large = _dedupe_photos_by_species_radius(
            [
                _mk_photo("a.jpg", "斑嘴鸭", base, lat=p1[0], lon=p1[1]),
                _mk_photo("b.jpg", "斑嘴鸭", base.replace(minute=5), lat=p2[0], lon=p2[1]),
            ],
            0.5,
        )
        layout_large = _render(ax, large, 0.5)
        assert len(layout_large.thumb_boxes_axes) == 1
        assert _anchor_dot_count(ax) == 1
    finally:
        plt.close(fig)


def test_markers_distinct_species_same_site_share_one_dot():
    """不同物种在去重半径内同一位点：各自保留鸟图，但共享一个定位圆点
    （横向网格排布）。"""
    import matplotlib.pyplot as plt

    base = datetime(2026, 9, 26, 10, 0, 0)
    photos = [
        _mk_photo("a_inst.jpg", "池鹭", base, lat=25.950, lon=119.360),
        _mk_photo("b_inst.jpg", "扇尾沙锥", base.replace(minute=1), lat=25.9501, lon=119.3601),
    ]
    fig, ax = _wide_ax()
    try:
        layout = _render(ax, photos, 1.0)
        assert len(layout.thumb_boxes_axes) == 2
        assert _anchor_dot_count(ax) == 1
    finally:
        plt.close(fig)


def test_cluster_sort_groups_same_order_together():
    """同一记录点矩阵内：鸟图按 目→科→种 分类学顺序排列，同目物种相邻，
    不再按拍摄时间混排（物种多时便于查看）。"""
    from gpx_track.photo_collect import photo_taxo_sort_key

    base = datetime(2026, 10, 4, 9, 0, 0)
    # 时间上交错，但分类学上 雁形目(鸭科) 远在 雀形目(鸦科) 之前
    entries = [
        (BirdPhoto("p0.jpg", "喜鹊", base, 30.20, 115.10,
                   order_cn="雀形目", family_cn="鸦科"), 0.1, 0.2),
        (BirdPhoto("p1.jpg", "绿头鸭", base.replace(minute=1), 30.20, 115.10,
                   order_cn="雁形目", family_cn="鸭科"), 0.1, 0.2),
        (BirdPhoto("p2.jpg", "灰喜鹊", base.replace(minute=2), 30.20, 115.10,
                   order_cn="雀形目", family_cn="鸦科"), 0.1, 0.2),
        (BirdPhoto("p3.jpg", "斑嘴鸭", base.replace(minute=3), 30.20, 115.10,
                   order_cn="雁形目", family_cn="鸭科"), 0.1, 0.2),
    ]
    groups = _cluster_photos_by_radius(entries, radius_km=1.0)
    assert len(groups) == 1
    order_in_grid = [t[0].path for t in groups[0]]
    # 名录序号：斑嘴鸭(165) < 绿头鸭(166) < 灰喜鹊(7031) < 喜鹊(7058)
    # 雁形目两张在前且相邻，雀形目两张在后且相邻（不再按拍摄时间交错）
    assert order_in_grid == ["p3.jpg", "p1.jpg", "p2.jpg", "p0.jpg"]
    # 同目内按名录序号：斑嘴鸭 < 绿头鸭
    k1 = photo_taxo_sort_key(entries[1][0])
    k3 = photo_taxo_sort_key(entries[3][0])
    assert k3 < k1


def test_taxonomy_from_classification_path_levels():
    """分类目录 目/科/属/种 层级应解析出目与科。"""
    from gpx_track.photo_collect import _taxonomy_from_classification_path

    root = Path("classification_x")
    p = root / "雁形目" / "鸭科" / "鸭属" / "绿头鸭" / "a.jpg"
    assert _taxonomy_from_classification_path(root, p) == ("雁形目", "鸭科", "绿头鸭")
    # 层级不足时退回末级作为物种名，目/科留空
    p2 = root / "绿头鸭" / "a.jpg"
    assert _taxonomy_from_classification_path(root, p2) == ("", "", "绿头鸭")


def _write_exif_jpg(path: Path, *, when: Optional[datetime] = None,
                    gps: Optional[tuple] = None) -> None:
    """写一张带 EXIF 时间/GPS 的 JPEG（供轨迹图回退标注测试）。"""
    import piexif
    from PIL import Image

    Image.new("RGB", (60, 60), (80, 90, 140)).save(path)
    exif: dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    if when is not None:
        exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = when.strftime(
            "%Y:%m:%d %H:%M:%S"
        ).encode()
    if gps is not None:
        lat, lon = gps

        def _dms(v: float):
            v = abs(v)
            d = int(v)
            m = int((v - d) * 60)
            s = round(((v - d) * 60 - m) * 60 * 100)
            return ((d, 1), (m, 1), (s, 100))

        exif["GPS"][piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
        exif["GPS"][piexif.GPSIFD.GPSLatitude] = _dms(lat)
        exif["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        exif["GPS"][piexif.GPSIFD.GPSLongitude] = _dms(lon)
    piexif.insert(piexif.dump(exif), str(path))


def _fallback_map_fixture(td: Path):
    """构造 GPX + 照片目录：2 张在 GPX 时间范围内，2 张超出范围但 EXIF GPS
    在视野内，1 张超出范围且 EXIF GPS 远在视野外。"""
    photo_root = td / "photos"
    photo_root.mkdir(parents=True)
    # GPX：2026-10-01 08:00-09:00 UTC（EXIF 本地 Asia/Shanghai 16:00-17:00）
    gpx_xml = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
    <trkpt lat="24.5000" lon="118.0000"><time>2026-10-01T08:00:00Z</time></trkpt>
    <trkpt lat="24.5100" lon="118.0100"><time>2026-10-01T08:30:00Z</time></trkpt>
    <trkpt lat="24.5200" lon="118.0200"><time>2026-10-01T09:00:00Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""
    gpx_path = td / "t.gpx"
    gpx_path.write_text(gpx_xml, encoding="utf-8")

    def _mk(species: str, name: str, **kw) -> None:
        d = photo_root / species
        d.mkdir(exist_ok=True)
        _write_exif_jpg(d / name, **kw)

    # 范围内：GPX 插值定位
    _mk("白鹭", "in1.jpg", when=datetime(2026, 10, 1, 16, 10))
    _mk("灰喜鹊", "in2.jpg", when=datetime(2026, 10, 1, 16, 50))
    # 超出 GPX 时间范围，但 EXIF GPS 在地图视野内 → 应补充显示
    _mk("池鹭", "out_in_view.jpg", when=datetime(2026, 10, 1, 21, 0),
        gps=(24.5080, 118.0080))
    # 无拍摄时间（仅有文件 mtime），EXIF GPS 在视野内 → 应补充显示
    _mk("夜鹭", "no_time.jpg", gps=(24.5130, 118.0130))
    # 超出范围且 EXIF GPS 远在视野外 → 不显示
    _mk("苍鹭", "far.jpg", when=datetime(2026, 10, 1, 22, 0),
        gps=(25.5000, 119.0000))
    return photo_root, gpx_path


def test_gpx_mode_exif_fallback_drawn_only_when_within_view():
    """勾选「补充使用 EXIF GPS」时：时间不在 GPX 范围或无拍摄时间的照片，
    带 EXIF GPS 且落在地图范围内的一并显示；视野外的不显示。"""
    import matplotlib
    matplotlib.use("Agg")
    from gpx_track.track_map import generate_track_maps

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_fb_"))
    photo_root, gpx_path = _fallback_map_fixture(td)
    written = generate_track_maps(
        reports_dir=str(td / "reports"),
        gpx_paths=[str(gpx_path)],
        photo_folder=str(photo_root),
        use_gpx_track=True,
        use_exif_gps=True,
        radius_km=1.0,
        include_elevation=False,
        basemap_style="none",
        preview_only=True,
    )
    assert Path(written["track_png"]).is_file()
    # 池鹭（超时） + 夜鹭（无时间） 补充绘制；苍鹭视野外未绘制
    assert written.get("map_exif_fallback_drawn") == "2"
    assert written.get("exif_fallback_out_of_range") == "1"
    assert "far.jpg" in written.get("exif_fallback_out_of_range_paths", "")


def test_gpx_mode_exif_fallback_disabled_when_unchecked():
    """未勾选「补充使用 EXIF GPS」时，GPX 匹配不上的照片直接不绘制。"""
    import matplotlib
    matplotlib.use("Agg")
    from gpx_track.track_map import generate_track_maps

    td = Path(tempfile.mkdtemp(prefix="birdy_tm_fboff_"))
    photo_root, gpx_path = _fallback_map_fixture(td)
    written = generate_track_maps(
        reports_dir=str(td / "reports"),
        gpx_paths=[str(gpx_path)],
        photo_folder=str(photo_root),
        use_gpx_track=True,
        use_exif_gps=False,
        radius_km=1.0,
        include_elevation=False,
        basemap_style="none",
        preview_only=True,
    )
    assert Path(written["track_png"]).is_file()
    assert "map_exif_fallback_drawn" not in written
    # 3 张匹配不上（超时/无时间）且不回退
    assert written.get("skipped_time_mismatch") == "3"
