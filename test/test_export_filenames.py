#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datetime import date, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.export_filenames import (  # noqa: E402
    checklist_export_slug,
    coords_geo_slug,
    unique_checklist_slug,
)
from record_submit.scan import ChecklistBucket  # noqa: E402


def test_coords_slug():
    assert coords_geo_slug(24.5919, 117.9492) == "N24p5919_E117p9492"
    assert coords_geo_slug(None, None, region_code="CN-FJ") == "region_CN_FJ"


def test_unique_slug_includes_exp():
    b = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=24.5,
        lon=118.0,
        start_time=datetime(2026, 5, 4, 9, 30, 0),
    )
    s = checklist_export_slug(
        b, region_code="CN-FJ", export_moment=datetime(2026, 5, 19, 15, 30, 45)
    )
    assert s == "20260504_0930_N24p5000_E118p0000_exp153045"


def test_unique_no_collision():
    used = set()
    b = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=None,
        lon=None,
        start_time=datetime(2026, 5, 4, 8, 0),
    )
    a = unique_checklist_slug(b, used, export_moment=datetime(2026, 5, 19, 12, 0, 0))
    b2 = unique_checklist_slug(b, used, export_moment=datetime(2026, 5, 19, 12, 0, 0))
    assert a != b2


if __name__ == "__main__":
    test_coords_slug()
    test_unique_slug_includes_exp()
    test_unique_no_collision()
    print("ok")
