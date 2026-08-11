# -*- coding: utf-8 -*-
"""水印：按物种目录随机抽样。"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watermark_generator import sample_images_per_species_dir  # noqa: E402


def test_sample_per_species_dir_caps_and_keeps_small_dirs(tmp_path: Path):
    sp_a = tmp_path / "白鹭"
    sp_b = tmp_path / "苍鹭"
    sp_a.mkdir()
    sp_b.mkdir()
    a_files = [str(sp_a / f"a{i}.jpg") for i in range(5)]
    b_files = [str(sp_b / f"b{i}.jpg") for i in range(2)]
    for p in a_files + b_files:
        Path(p).write_bytes(b"x")

    rng = random.Random(42)
    out = sample_images_per_species_dir(a_files + b_files, 3, rng=rng)
    assert len(out) == 5  # 3 from A + 2 from B
    assert sum(1 for p in out if Path(p).parent.name == "白鹭") == 3
    assert sum(1 for p in out if Path(p).parent.name == "苍鹭") == 2


def test_sample_per_species_dir_disabled_returns_all():
    paths = [
        r"C:\x\白鹭\1.jpg",
        r"C:\x\白鹭\2.jpg",
        r"C:\x\苍鹭\1.jpg",
    ]
    assert sample_images_per_species_dir(paths, 0) == paths
    assert sample_images_per_species_dir(paths, -1) == paths


def test_sample_is_deterministic_with_seed():
    paths = [rf"C:\x\种A\{i}.jpg" for i in range(10)]
    a = sample_images_per_species_dir(paths, 4, rng=random.Random(7))
    b = sample_images_per_species_dir(paths, 4, rng=random.Random(7))
    assert a == b
    assert len(a) == 4
