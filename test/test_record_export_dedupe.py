#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.dedupe_counts import (  # noqa: E402
    ImageObservation,
    aggregate_species_count,
    should_use_spatial_clustering,
)


def _obs(
    key: str,
    minute: int,
    lat=None,
    lon=None,
    count: int = 1,
) -> ImageObservation:
    return ImageObservation(
        img_key=key,
        path=f"/fake/{key}.jpg",
        dt=datetime(2026, 5, 4, 10, minute, 0),
        lat=lat,
        lon=lon,
        count=count,
    )


def test_spatial_cluster_takes_max_per_batch():
    # 50 m apart → same batch; counts 2 and 5 → 5
    o1 = _obs("a", 0, 24.5, 118.1, count=2)
    o2 = _obs("b", 5, 24.5004, 118.1004, count=5)
    assert should_use_spatial_clustering([o1, o2]) is True
    total = aggregate_species_count(
        [o1, o2],
        count_individuals=True,
        use_spatial=True,
        spatial_threshold_km=0.1,
    )
    assert total == 5


def test_time_cluster_fixed_gps():
    o1 = _obs("a", 0, 24.5, 118.1, count=3)
    o2 = _obs("b", 10, 24.5, 118.1, count=7)
    o3 = _obs("c", 50, 24.5, 118.1, count=2)
    assert should_use_spatial_clustering([o1, o2, o3]) is False
    total = aggregate_species_count(
        [o1, o2, o3],
        count_individuals=True,
        use_spatial=False,
        time_threshold_minutes=30.0,
    )
    assert total == 7 + 2  # 10min batch max 7; 50min separate max 2


def test_no_count_individuals():
    obs = [_obs("a", 0, count=9), _obs("b", 5, count=4)]
    assert aggregate_species_count(obs, count_individuals=False) == 1


def test_spatial_also_merges_by_time_when_gps_drifts():
    """定点观鸟：相距略大于 0.1 km 但在 120 分钟内仍计为 1 批。"""
    o1 = _obs("a", 0, 24.5, 118.1, count=2)
    o2 = _obs("b", 15, 24.502, 118.102, count=4)
    assert should_use_spatial_clustering([o1, o2]) is True
    total = aggregate_species_count(
        [o1, o2],
        count_individuals=True,
        use_spatial=True,
        spatial_threshold_km=0.1,
        time_threshold_minutes=120.0,
    )
    assert total == 4


def test_first_photo_coords_unify_cluster():
    o1 = _obs("a", 0, 24.5, 118.1, count=1)
    o2 = _obs("b", 5, 24.5003, 118.1002, count=2)
    from record_submit.dedupe_counts import apply_first_photo_coords_per_cluster

    out = apply_first_photo_coords_per_cluster(
        [o1, o2], prefer_spatial_gps=True, spatial_threshold_km=0.1
    )
    assert out[0].lat == 24.5 and out[0].lon == 118.1
    assert out[1].lat == 24.5 and out[1].lon == 118.1


def test_source_photo_count_uses_inst_not_file_count():
    from record_submit.scan import (
        _count_from_source_photo,
        _register_archive_on_source_photo,
    )

    tags: set[int] = set()
    _register_archive_on_source_photo(tags, "/x/鹰_inst01_00001.jpg")
    _register_archive_on_source_photo(tags, "/x/鹰_inst02_00002.jpg")
    assert _count_from_source_photo(tags) == 2
    tags2: set[int] = set()
    _register_archive_on_source_photo(tags2, "/x/鹰_inst01_00001.jpg")
    _register_archive_on_source_photo(tags2, "/x/鹰_inst01_00009.jpg")
    assert _count_from_source_photo(tags2) == 1
    tags3: set[int] = set()
    _register_archive_on_source_photo(tags3, "/x/鹰_00003.jpg")
    _register_archive_on_source_photo(tags3, "/x/鹰_00004.jpg")
    assert _count_from_source_photo(tags3) == 1


def test_collapse_buckets_merges_species_counts():
    from datetime import date
    from record_submit.scan import ChecklistBucket, _collapse_buckets_for_export

    b1 = ChecklistBucket(day=date(2026, 5, 4), lat=24.5, lon=118.1, species_counts={"鹰": 2})
    b2 = ChecklistBucket(
        day=date(2026, 5, 5), lat=24.5, lon=118.1, species_counts={"鹰": 2}
    )
    obs = _obs("a", 0, 24.5, 118.1, count=2)
    obs2 = _obs("b", 20, 24.502, 118.102, count=3)
    per = {
        (date(2026, 5, 4), "24.5000", "118.1000"): {
            "鹰": {"a": obs},
        },
        (date(2026, 5, 5), "24.5000", "118.1000"): {
            "鹰": {"b": obs2},
        },
    }
    merged = _collapse_buckets_for_export(
        {
            (date(2026, 5, 4), "24.5000", "118.1000"): b1,
            (date(2026, 5, 5), "24.5000", "118.1000"): b2,
        },
        per,
        count_individuals=True,
        prefer_spatial_gps=True,
        spatial_threshold_km=0.1,
        individual_time_threshold_minutes=120.0,
    )
    assert len(merged) == 1
    only = next(iter(merged.values()))
    # 合并导出：跨日同点 7 日内视为同次活动，20 分钟间隔 + GPS 微移 → 1 批
    assert only.species_counts["鹰"] == 3
    assert only.day_end == date(2026, 5, 5)


if __name__ == "__main__":
    test_source_photo_count_uses_inst_not_file_count()
    test_spatial_cluster_takes_max_per_batch()
    test_spatial_also_merges_by_time_when_gps_drifts()
    test_time_cluster_fixed_gps()
    test_no_count_individuals()
    test_first_photo_coords_unify_cluster()
    test_collapse_buckets_merges_species_counts()
    print("ok")
