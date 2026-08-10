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
    _cluster_label_xytext,
    _gps_cluster_key,
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


def test_gps_cluster_key_groups_nearby():
    assert _gps_cluster_key(24.123456, 118.654321) == _gps_cluster_key(
        24.123457, 118.654322
    )


def test_cluster_label_alternates_vertical():
    a = _cluster_label_xytext(0, 10.0, 44)
    b = _cluster_label_xytext(1, 10.0, 44)
    assert a[3] == "bottom" and b[3] == "top"
    assert a[1] > 0 and b[1] < 0
