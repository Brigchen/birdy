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
    default_ebird_checklist_template_path,
    export_ebird_checklist_files,
    write_ebird_checklist_csv,
)
from record_submit.scan import ChecklistBucket  # noqa: E402


def test_checklist_csv_layout():
    tpl = default_ebird_checklist_template_path()
    assert tpl.is_file()
    bucket = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=24.5919,
        lon=117.9492,
        start_time=datetime(2026, 5, 4, 8, 0, 0),
        species_counts={"麻雀": 2},
    )
    table = {"麻雀": ("Eurasian Tree Sparrow", "Passer montanus")}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "t.csv"
        write_ebird_checklist_csv(
            str(out), bucket, table, template_path=str(tpl)
        )
        with open(out, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == ""
        assert "5/4/2026" in rows[3][2]
        assert "Eurasian" in rows[14][0]
        assert rows[14][2].startswith("2")


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
        assert not csv_p.with_suffix(".xls").exists()


def test_build_grid_column_count():
    bucket = ChecklistBucket(
        day=date(2026, 5, 4),
        lat=None,
        lon=None,
        species_counts={"a": 1},
    )
    grid = build_checklist_grid(bucket, {"a": ("Sp", "")})
    assert len(grid[0]) == 3
    assert len(grid) == 15


if __name__ == "__main__":
    test_checklist_csv_layout()
    test_export_csv_only()
    test_build_grid_column_count()
    print("ok")
