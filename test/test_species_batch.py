# -*- coding: utf-8 -*-
"""物种识别栏：单独裁切识别的目录解析与线程接口。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from birdy_gui import SpeciesBatchThread, resolve_species_batch_source  # noqa: E402
from PyQt5.QtCore import QThread  # noqa: E402


def test_resolve_species_batch_source_prefers_specified():
    assert (
        resolve_species_batch_source("D:/birds/screened", "D:/birds/raw")
        == "D:/birds/screened"
    )


def test_resolve_species_batch_source_falls_back_to_image_folder():
    assert resolve_species_batch_source("  ", "D:/birds/raw") == "D:/birds/raw"
    assert resolve_species_batch_source("", "") == ""


def test_species_batch_thread_is_stoppable_qthread():
    assert issubclass(SpeciesBatchThread, QThread)
    th = SpeciesBatchThread(
        "in",
        "out",
        {"use_local_model": True, "enable_image_clean_before_species": False},
    )
    assert th.is_running is True
    th.stop()
    assert th.is_running is False
    for name in (
        "progress",
        "log_line",
        "finished_ok",
        "failed",
        "status_updated",
        "eta_checkpoint",
    ):
        assert hasattr(th, name)
