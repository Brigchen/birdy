#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.ebird_checklist import (  # noqa: E402
    build_checklist_grid,
    default_ebird_checklist_sample_path,
    export_ebird_checklist_files,
    normalize_ebird_state_province,
    write_ebird_checklist_csv,
)
from record_submit.scan import ChecklistBucket  # noqa: E402
from record_submit.taxonomy_cn import ebird_species_cells  # noqa: E402


def test_checklist_csv_matches_sample_layout():
    sample = default_ebird_checklist_sample_path()
    assert sample.is_file()
    bucket = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=24.5919,
        lon=117.9492,
        start_time=datetime(2026, 5, 4, 8, 0, 0),
        species_counts={"麻雀": 2, "红头长尾山雀": 1},
    )
    table = {
        "麻雀": ("Eurasian Tree Sparrow", "Passer montanus"),
        "红头长尾山雀": ("Black-throated Bushtit", "Aegithalos concinnus"),
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "t.csv"
        write_ebird_checklist_csv(
            str(out),
            bucket,
            table,
            sample_path=str(sample),
            location_name="厦门大学翔安校区",
            province_cn="福建",
            city_cn="厦门",
        )
        raw = out.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b'"' not in raw
        text = raw.decode("utf-8")
        assert "\r\n" in text
        with open(out, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows[0]) == 3
        assert rows[0][0] == "" and rows[0][1] == ""
        assert rows[0][2] == "China"
        assert text.splitlines()[0] == ",,China"
        assert not text.splitlines()[0].endswith(",")
        assert rows[1][0] == "Latitude"
        assert rows[1][2] == "24.591900"
        assert rows[3][0] == "Date"
        assert rows[3][2] == "5/4/2026"
        assert rows[4][2] == "8:00 AM"
        # 物种行按中文名排序
        assert rows[14][0] == "Black-throated Tit"
        assert rows[14][1] == "Aegithalos concinnus"
        assert rows[14][2] == "1"
        assert rows[15][0] == "Eurasian Tree Sparrow"
        assert rows[15][1] == "Passer montanus"
        assert rows[15][2] == "2"


def test_export_csv_only():
    bkey = (date(2026, 5, 4), "24.5919", "117.9492")
    buckets = {
        bkey: ChecklistBucket(
            day=date(2026, 5, 4),
            lat=24.5919,
            lon=117.9492,
            start_time=datetime(2026, 5, 4, 8, 0, 0),
            species_counts={"麻雀": 1},
        )
    }
    table = {"麻雀": ("Eurasian Tree Sparrow", "Passer montanus")}
    with tempfile.TemporaryDirectory() as td:
        files = export_ebird_checklist_files(buckets, table, td)
        csv_p = Path(files["ebird_checklist_format_csv"])
        assert csv_p.suffix == ".csv"
        assert "ebird_checklist_" in csv_p.name


def test_state_and_traveling_distance():
    bucket = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=24.5,
        lon=118.0,
        species_counts={"a": 1},
    )
    grid = build_checklist_grid(
        bucket,
        {"a": ("Sp", "")},
        state_province="CN-FJ",
        protocol="Traveling",
    )
    assert len(grid[0]) == 3
    assert grid[5][0] == "State"
    assert grid[5][2] == "FJ"
    assert grid[11][0] == "Dist Traveled (Miles)"
    assert grid[11][2] == "1"


def test_taxonomy_fix_bushtit_to_tit():
    common, sci = ebird_species_cells(
        "Black-throated Bushtit", "Aegithalos concinnus", "红头长尾山雀"
    )
    assert common == "Black-throated Tit"
    assert sci == "Aegithalos concinnus"


if __name__ == "__main__":
    test_checklist_csv_matches_sample_layout()
    test_export_csv_only()
    test_state_and_traveling_distance()
    test_taxonomy_fix_bushtit_to_tit()
    print("ok")
