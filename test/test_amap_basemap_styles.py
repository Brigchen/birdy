# -*- coding: utf-8 -*-
"""底图风格归一化与主题调色单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpx_track.amap_basemap import (  # noqa: E402
    BASEMAP_STYLE_CHOICES,
    _apply_vector_theme_look,
    _compose_tile_layers,
    is_dark_basemap_style,
    is_satellite_basemap_style,
    normalize_basemap_style,
)


def test_normalize_aliases():
    assert normalize_basemap_style("digital") == "normal"
    assert normalize_basemap_style("amap://styles/dark") == "dark"
    assert normalize_basemap_style("sat") == "satellite"
    assert normalize_basemap_style("有路网卫星") == "satellite_roads"
    assert normalize_basemap_style("unknown-xyz") == "normal"


def test_style_catalog_complete():
    ids = [cid for cid, _ in BASEMAP_STYLE_CHOICES]
    assert "normal" in ids
    assert "dark" in ids
    assert "satellite" in ids
    assert "satellite_roads" in ids
    assert len(ids) == 13


def test_dark_and_satellite_flags():
    assert is_dark_basemap_style("dark")
    assert is_dark_basemap_style("darkblue")
    assert is_dark_basemap_style("satellite")
    assert is_dark_basemap_style("satellite_roads")
    assert not is_dark_basemap_style("normal")
    assert not is_dark_basemap_style("fresh")
    assert is_satellite_basemap_style("satellite_roads")
    assert not is_satellite_basemap_style("dark")


def test_theme_look_changes_pixels():
    base = Image.new("RGBA", (32, 32), (180, 200, 160, 255))
    dark = _apply_vector_theme_look(base, "dark")
    fresh = _apply_vector_theme_look(base, "fresh")
    assert dark.size == base.size
    assert not np.array_equal(np.asarray(base), np.asarray(dark))
    assert not np.array_equal(np.asarray(base), np.asarray(fresh))
    assert np.array_equal(
        np.asarray(_apply_vector_theme_look(base, "normal")),
        np.asarray(base),
    )


def test_compose_road_overlay():
    sat = Image.new("RGBA", (16, 16), (40, 80, 40, 255))
    road = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    road.putpixel((8, 8), (255, 255, 255, 200))
    out = _compose_tile_layers([sat, road])
    assert out.getpixel((0, 0))[:3] == (40, 80, 40)
    assert out.getpixel((8, 8))[0] > 150
