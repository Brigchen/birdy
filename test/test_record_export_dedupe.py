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


def test_first_photo_coords_unify_cluster():
    o1 = _obs("a", 0, 24.5, 118.1, count=1)
    o2 = _obs("b", 5, 24.5003, 118.1002, count=2)
    from record_submit.dedupe_counts import apply_first_photo_coords_per_cluster

    out = apply_first_photo_coords_per_cluster(
        [o1, o2], prefer_spatial_gps=True, spatial_threshold_km=0.1
    )
    assert out[0].lat == 24.5 and out[0].lon == 118.1
    assert out[1].lat == 24.5 and out[1].lon == 118.1


if __name__ == "__main__":
    test_spatial_cluster_takes_max_per_batch()
    test_time_cluster_fixed_gps()
    test_no_count_individuals()
    test_first_photo_coords_unify_cluster()
    print("ok")
