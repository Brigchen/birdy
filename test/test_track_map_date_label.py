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

    aspect = 1440 / 2560
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
