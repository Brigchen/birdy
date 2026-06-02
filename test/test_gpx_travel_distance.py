#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpx_track.gpx_io import GpxPoint, load_gpx, load_gpx_many  # noqa: E402
from gpx_track.gpx_match import TrackTimeAlignment  # noqa: E402
from record_submit.gpx_travel_distance import (  # noqa: E402
    GpxTravelDistanceResolver,
    format_distance_miles,
    track_distance_km_in_time_window,
)
from record_submit.scan import ChecklistBucket  # noqa: E402


def test_track_distance_km_in_window():
    align = TrackTimeAlignment(exif_tz="UTC", gpx_tz="UTC")
    t0 = datetime(2026, 5, 4, 8, 0, 0)
    t1 = datetime(2026, 5, 4, 8, 30, 0)
    t2 = datetime(2026, 5, 4, 9, 0, 0)
    track = [
        GpxPoint(t0, 24.5, 118.0),
        GpxPoint(t1, 24.51, 118.0),
        GpxPoint(t2, 24.52, 118.0),
    ]
    km = track_distance_km_in_time_window(
        track, t0, t1, alignment=align
    )
    assert km > 1.0
    mi = format_distance_miles(km)
    assert float(mi) > 0.5


def test_resolver_from_gpx_file():
    gpx_xml = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test">
  <trk><trkseg>
    <trkpt lat="24.5000" lon="118.0000"><time>2026-05-04T08:00:00Z</time></trkpt>
    <trkpt lat="24.5100" lon="118.0000"><time>2026-05-04T08:30:00Z</time></trkpt>
    <trkpt lat="24.5200" lon="118.0000"><time>2026-05-04T09:00:00Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""
    with tempfile.TemporaryDirectory() as td:
        gpx_path = Path(td) / "t.gpx"
        gpx_path.write_text(gpx_xml, encoding="utf-8")
        track = load_gpx(str(gpx_path))
        assert len(track) == 3
        resolver = GpxTravelDistanceResolver(
            str(gpx_path), exif_tz="UTC", gpx_tz="UTC"
        )
        bucket = ChecklistBucket(
            day=date(2026, 5, 4),
            lat=24.5,
            lon=118.0,
            start_time=datetime(2026, 5, 4, 8, 0, 0),
            end_time=datetime(2026, 5, 4, 8, 30, 0),
            species_counts={"a": 1},
        )
        mi = resolver.miles_for_bucket(bucket)
        assert mi is not None
        assert float(mi) > 0.5


def test_load_gpx_many_merges_by_time():
    seg_a = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="24.50" lon="118.00"><time>2026-05-04T08:00:00Z</time></trkpt>
</trkseg></trk></gpx>"""
    seg_b = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="24.52" lon="118.00"><time>2026-05-04T10:00:00Z</time></trkpt>
</trkseg></trk></gpx>"""
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.gpx"
        p2 = Path(td) / "b.gpx"
        p1.write_text(seg_a, encoding="utf-8")
        p2.write_text(seg_b, encoding="utf-8")
        merged = load_gpx_many([str(p1), str(p2)])
        assert len(merged) == 2
        assert merged[0].time and merged[1].time
        assert merged[0].time < merged[1].time


if __name__ == "__main__":
    test_track_distance_km_in_window()
    test_resolver_from_gpx_file()
    test_load_gpx_many_merges_by_time()
    print("ok")
