# -*- coding: utf-8 -*-
"""连拍保留比例：按全组张数计算；快速模式按抽样比例抽候选。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_grouping import (  # noqa: E402
    compute_burst_keep_count,
    compute_fast_sample_count,
    copy_kept_images_to_screened,
    evenly_spaced_indices,
    normalize_fast_sample_ratio,
    screened_paths_for_kept_images,
)


def test_keep_ratio_point_one_is_ten_percent_of_full_group():
    n = 30
    keep = compute_burst_keep_count(n, 0.1, min_keep=1)
    assert keep == 3
    wrong_on_one_third = compute_burst_keep_count(n // 3, 0.1, min_keep=1)
    assert wrong_on_one_third == 1
    assert keep != wrong_on_one_third


def test_keep_ratio_respects_min_and_cap():
    assert compute_burst_keep_count(8, 0.1, min_keep=2) == 2
    assert compute_burst_keep_count(5, 1.0, min_keep=2) == 5
    assert compute_burst_keep_count(0, 0.1, min_keep=2) == 0


def test_fast_sample_count_default_half():
    n = 30
    keep = compute_burst_keep_count(n, 0.1, min_keep=1)
    sample = compute_fast_sample_count(n, keep)
    assert keep == 3
    assert sample >= keep
    assert sample == 15
    assert compute_fast_sample_count(9, 2) == 5
    assert compute_fast_sample_count(8, 6) == 6


def test_fast_sample_ratio_custom():
    assert normalize_fast_sample_ratio(0.25) == 0.25
    assert normalize_fast_sample_ratio(-1) == 0.0
    assert normalize_fast_sample_ratio(2) == 1.0
    assert normalize_fast_sample_ratio(None) == 0.5
    n, keep = 20, 2
    assert compute_fast_sample_count(n, keep, 0.25) == 5
    assert compute_fast_sample_count(n, keep, 0.0) == keep
    assert compute_fast_sample_count(n, keep, 1.0) == n
    assert compute_fast_sample_count(30, 3, 0.1) == 3


def test_fast_sample_indices_cover_group_and_enough_for_keep():
    n, keep = 30, 3
    sample_n = compute_fast_sample_count(n, keep)
    idx = evenly_spaced_indices(n, sample_n)
    assert len(idx) == sample_n
    assert min(idx) >= 0 and max(idx) < n
    assert len(evenly_spaced_indices(4, 4)) == 4
    assert evenly_spaced_indices(9, 3) == {0, 3, 6}


def test_screened_keeps_raw_bytes(tmp_path: Path):
    src = tmp_path / "in"
    src.mkdir()
    raw = src / "bird.ARW"
    raw.write_bytes(b"raw-payload")
    out = tmp_path / "Screened_images"
    result = {
        "groups": [{"images": [{"path": str(raw), "kept": True}]}],
        "non_burst": [],
    }
    n = copy_kept_images_to_screened(result, str(src), str(out))
    dest = out / "bird.ARW"
    assert n == 1
    assert dest.read_bytes() == b"raw-payload"
    assert not list(out.glob("*.dng"))
    assert not list(out.glob("*.jpg"))
    mapped = screened_paths_for_kept_images(result, str(src), str(out))
    assert Path(mapped[0]).resolve() == dest.resolve()
