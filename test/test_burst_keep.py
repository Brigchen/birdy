# -*- coding: utf-8 -*-
"""连拍保留比例：按全组张数计算，快速模式不得叠在 1/3 采样上。"""

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
    evenly_spaced_indices,
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


def test_fast_sample_count_based_on_full_group_keep_not_n_over_three():
    n = 30
    keep = compute_burst_keep_count(n, 0.1, min_keep=1)
    sample = compute_fast_sample_count(n, keep)
    assert keep == 3
    assert sample >= keep
    assert sample < n // 3 or sample == keep * 2
    assert sample == 6
    assert sample != n // 3


def test_fast_sample_indices_cover_group_and_enough_for_keep():
    n, keep = 30, 3
    sample_n = compute_fast_sample_count(n, keep)
    idx = evenly_spaced_indices(n, sample_n)
    assert len(idx) == sample_n
    assert min(idx) >= 0 and max(idx) < n
    assert len(evenly_spaced_indices(4, 4)) == 4
    assert evenly_spaced_indices(9, 3) == {0, 3, 6}
